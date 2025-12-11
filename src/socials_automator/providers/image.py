"""Image generation provider with support for multiple backends."""

from __future__ import annotations

import asyncio
import base64
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Callable, Awaitable

import httpx
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import ImageProviderConfig, ProviderConfig, load_provider_config


# Type for AI event callback
AIEventCallback = Callable[[dict[str, Any]], Awaitable[None]] | None


class ImageProvider:
    """Unified image generation provider.

    Supports multiple backends:
    - fal.ai (Flux, SDXL, Nano Banana)
    - Replicate (SDXL, SD3, custom models)
    - OpenAI (DALL-E 3)

    Usage:
        provider = ImageProvider()
        image_bytes = await provider.generate(
            prompt="A minimal tech logo on dark background",
            size=(1080, 1350)  # Instagram portrait
        )

        # Save to file
        await provider.generate_and_save(
            prompt="...",
            output_path=Path("slide_01.jpg")
        )
    """

    # Standard Instagram sizes
    INSTAGRAM_SIZES = {
        "square": (1080, 1080),
        "portrait": (1080, 1350),  # 4:5 ratio - best for carousels
        "story": (1080, 1920),  # 9:16 ratio
    }

    def __init__(self, config: ProviderConfig | None = None, event_callback: AIEventCallback = None):
        """Initialize image provider.

        Args:
            config: Provider configuration. If None, loads from default config file.
            event_callback: Optional callback for AI events (for progress tracking).
        """
        self.config = config or load_provider_config()
        self._current_provider: str | None = None
        self._current_model: str | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._event_callback = event_callback
        self._total_calls = 0
        self._total_cost = 0.0

        # Check for provider override
        self._provider_override: str | None = None
        if isinstance(config, dict) and "image_provider_override" in config:
            self._provider_override = config["image_provider_override"]

    async def _emit_event(self, event: dict[str, Any]) -> None:
        """Emit an AI event if callback is set."""
        if self._event_callback:
            await self._event_callback(event)

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=120.0)
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _get_provider_for_task(self, task: str | None = None) -> tuple[str, ImageProviderConfig]:
        """Get the appropriate provider for a task."""
        if task:
            result = self.config.get_image_provider_for_task(task)
            if result:
                return result

        providers = self.config.get_enabled_image_providers()
        if not providers:
            raise RuntimeError("No image providers are enabled")

        return providers[0]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate(
        self,
        prompt: str,
        size: tuple[int, int] | str = "portrait",
        task: str | None = None,
        negative_prompt: str | None = None,
        style_suffix: str | None = None,
        **kwargs: Any,
    ) -> bytes:
        """Generate an image from a prompt.

        Args:
            prompt: Text description of the image.
            size: Image size as tuple (width, height) or preset name.
            task: Optional task name for provider override.
            negative_prompt: Things to avoid in the image.
            style_suffix: Additional style instructions to append.
            **kwargs: Additional provider-specific arguments.

        Returns:
            Image as bytes (JPEG format).
        """
        # Resolve size preset
        if isinstance(size, str):
            size = self.INSTAGRAM_SIZES.get(size, self.INSTAGRAM_SIZES["portrait"])

        # Build full prompt with style suffix
        full_prompt = prompt
        if style_suffix:
            full_prompt = f"{prompt}, {style_suffix}"

        # Get provider with fallback
        providers = self.config.get_enabled_image_providers()

        # If override is set, prioritize that provider
        if self._provider_override:
            override_name = self._provider_override.lower()
            providers = sorted(providers, key=lambda x: 0 if x[0] == override_name else 1)

        last_error: Exception | None = None
        failed_providers: list[str] = []

        for provider_name, provider_config in providers:
            try:
                self._current_provider = provider_name
                self._current_model = provider_config.model

                # Emit "generating" event with failed providers info
                await self._emit_event({
                    "type": "image_call",
                    "provider": provider_name,
                    "model": provider_config.model,
                    "prompt_preview": full_prompt[:200],
                    "size": size,
                    "task": task,
                    "failed_providers": failed_providers.copy(),
                })

                start_time = time.time()

                if provider_config.type == "fal":
                    image_bytes = await self._generate_fal(
                        provider_config, full_prompt, size, negative_prompt, **kwargs
                    )
                elif provider_config.type == "replicate":
                    image_bytes = await self._generate_replicate(
                        provider_config, full_prompt, size, negative_prompt, **kwargs
                    )
                elif provider_config.type == "openai":
                    image_bytes = await self._generate_openai(
                        provider_config, full_prompt, size, **kwargs
                    )
                else:
                    raise ValueError(f"Unknown provider type: {provider_config.type}")

                duration = time.time() - start_time
                self._total_calls += 1
                cost = provider_config.cost_per_image
                self._total_cost += cost

                # Emit "complete" event
                await self._emit_event({
                    "type": "image_response",
                    "provider": provider_name,
                    "model": provider_config.model,
                    "duration_seconds": duration,
                    "cost_usd": cost,
                    "total_calls": self._total_calls,
                    "total_cost": self._total_cost,
                    "image_size_bytes": len(image_bytes),
                    "failed_providers": failed_providers.copy(),
                })

                return image_bytes

            except Exception as e:
                last_error = e
                failed_providers.append(f"{provider_name}")
                # Emit error event
                await self._emit_event({
                    "type": "image_error",
                    "provider": provider_name,
                    "error": str(e)[:50],
                    "failed_providers": failed_providers.copy(),
                })
                if self.config.provider_settings.fallback_on_error:
                    continue
                raise

        if last_error:
            raise last_error
        raise RuntimeError("No image providers available")

    async def _generate_fal(
        self,
        config: ImageProviderConfig,
        prompt: str,
        size: tuple[int, int],
        negative_prompt: str | None = None,
        **kwargs: Any,
    ) -> bytes:
        """Generate image using fal.ai API."""
        try:
            import fal_client
        except ImportError:
            raise ImportError("fal-client is required for fal.ai provider. Install with: pip install fal-client")

        api_key = config.get_api_key()
        if not api_key:
            raise ValueError("FAL_API_KEY not set")

        # Set API key
        os.environ["FAL_KEY"] = api_key

        # Map size to fal format
        width, height = size
        if height > width:
            image_size = "portrait_4_3"  # Closest to 4:5
        elif width > height:
            image_size = "landscape_4_3"
        else:
            image_size = "square"

        # Build request
        request_data = {
            "prompt": prompt,
            "image_size": image_size,
            **config.settings,
            **kwargs,
        }

        if negative_prompt:
            request_data["negative_prompt"] = negative_prompt

        # Call fal.ai
        result = await asyncio.to_thread(
            fal_client.subscribe,
            config.model,
            arguments=request_data,
        )

        # Get image URL from result
        if "images" in result and len(result["images"]) > 0:
            image_url = result["images"][0]["url"]
        elif "image" in result:
            image_url = result["image"]["url"]
        else:
            raise ValueError(f"Unexpected fal.ai response format: {result}")

        # Download image
        client = await self._get_http_client()
        response = await client.get(image_url)
        response.raise_for_status()

        # Resize to exact dimensions
        return self._resize_image(response.content, size)

    async def _generate_replicate(
        self,
        config: ImageProviderConfig,
        prompt: str,
        size: tuple[int, int],
        negative_prompt: str | None = None,
        **kwargs: Any,
    ) -> bytes:
        """Generate image using Replicate API."""
        try:
            import replicate
        except ImportError:
            raise ImportError("replicate is required. Install with: pip install replicate")

        api_key = config.get_api_key()
        if not api_key:
            raise ValueError("REPLICATE_API_TOKEN not set")

        os.environ["REPLICATE_API_TOKEN"] = api_key

        width, height = size

        # Build input
        input_data = {
            "prompt": prompt,
            "width": width,
            "height": height,
            **config.settings,
            **kwargs,
        }

        if negative_prompt:
            input_data["negative_prompt"] = negative_prompt

        # Run prediction
        output = await asyncio.to_thread(
            replicate.run,
            config.model,
            input=input_data,
        )

        # Output is usually a list of URLs
        if isinstance(output, list) and len(output) > 0:
            image_url = output[0]
        else:
            image_url = output

        # Download image
        client = await self._get_http_client()
        response = await client.get(image_url)
        response.raise_for_status()

        return self._resize_image(response.content, size)

    async def _generate_openai(
        self,
        config: ImageProviderConfig,
        prompt: str,
        size: tuple[int, int],
        **kwargs: Any,
    ) -> bytes:
        """Generate image using OpenAI DALL-E API."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

        api_key = config.get_api_key()
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        client = AsyncOpenAI(api_key=api_key)

        # DALL-E 3 only supports specific sizes
        width, height = size
        if height > width:
            dalle_size = "1024x1792"
        elif width > height:
            dalle_size = "1792x1024"
        else:
            dalle_size = "1024x1024"

        response = await client.images.generate(
            model=config.model,
            prompt=prompt,
            size=dalle_size,
            quality=config.settings.get("quality", "standard"),
            response_format="b64_json",
            n=1,
        )

        # Decode base64 image
        image_data = base64.b64decode(response.data[0].b64_json)

        return self._resize_image(image_data, size)

    def _resize_image(self, image_bytes: bytes, target_size: tuple[int, int]) -> bytes:
        """Resize image to exact target dimensions while preserving aspect ratio.

        Uses cover/crop approach: scales to cover the target area, then center crops.
        """
        img = Image.open(BytesIO(image_bytes))

        # Convert to RGB if necessary
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        target_width, target_height = target_size

        if img.size != target_size:
            # Calculate aspect ratios
            img_ratio = img.width / img.height
            target_ratio = target_width / target_height

            if img_ratio > target_ratio:
                # Image is wider than target - fit height, crop width
                new_height = target_height
                new_width = int(new_height * img_ratio)
            else:
                # Image is taller than target - fit width, crop height
                new_width = target_width
                new_height = int(new_width / img_ratio)

            # Resize maintaining aspect ratio
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Center crop to target size
            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
            img = img.crop((left, top, left + target_width, top + target_height))

        # Save as JPEG
        output = BytesIO()
        img.save(output, format="JPEG", quality=95, optimize=True)
        return output.getvalue()

    async def generate_and_save(
        self,
        prompt: str,
        output_path: Path,
        size: tuple[int, int] | str = "portrait",
        **kwargs: Any,
    ) -> Path:
        """Generate image and save to file.

        Args:
            prompt: Image prompt.
            output_path: Path to save the image.
            size: Image size.
            **kwargs: Additional arguments.

        Returns:
            Path to saved image.
        """
        image_bytes = await self.generate(prompt, size=size, **kwargs)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)

        return output_path

    @property
    def current_provider(self) -> str | None:
        """Get the name of the last used provider."""
        return self._current_provider


# Module-level singleton
_default_provider: ImageProvider | None = None


def get_image_provider(config: ProviderConfig | None = None) -> ImageProvider:
    """Get the default image provider instance.

    Args:
        config: Optional config to use.

    Returns:
        ImageProvider instance.
    """
    global _default_provider
    if _default_provider is None or config is not None:
        _default_provider = ImageProvider(config)
    return _default_provider
