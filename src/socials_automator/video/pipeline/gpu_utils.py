"""GPU utilities for hardware-accelerated video encoding.

Provides GPU detection, NVENC availability checking, and GPU selection.
"""

import subprocess
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUInfo:
    """Information about an available GPU."""

    index: int
    name: str
    memory_total_mb: int
    memory_free_mb: int
    nvenc_supported: bool
    cuda_cores: Optional[int] = None

    def __str__(self) -> str:
        nvenc_status = "[NVENC]" if self.nvenc_supported else "[NO NVENC]"
        return f"GPU {self.index}: {self.name} ({self.memory_total_mb}MB) {nvenc_status}"


class GPUDetector:
    """Detects available GPUs and their capabilities."""

    # GPUs known to support NVENC (GTX 600+ series, RTX series)
    NVENC_SUPPORTED_PATTERNS = [
        r"GTX\s*(6[0-9]{2}|7[0-9]{2}|8[0-9]{2}|9[0-9]{2}|10[0-9]{2}|16[0-9]{2})",
        r"RTX\s*(20[0-9]{2}|30[0-9]{2}|40[0-9]{2}|50[0-9]{2})",
        r"Quadro",
        r"Tesla",
        r"A[0-9]{2,4}",  # A100, A6000, etc.
    ]

    def __init__(self):
        self._gpus: Optional[list[GPUInfo]] = None

    def detect_gpus(self) -> list[GPUInfo]:
        """Detect all available NVIDIA GPUs.

        Returns:
            List of GPUInfo objects for each detected GPU.
        """
        if self._gpus is not None:
            return self._gpus

        self._gpus = []

        try:
            # Use nvidia-smi to get GPU info
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return self._gpus

            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    index = int(parts[0])
                    name = parts[1]
                    memory_total = int(float(parts[2]))
                    memory_free = int(float(parts[3]))

                    # Check if this GPU supports NVENC
                    nvenc_supported = self._check_nvenc_support(name)

                    self._gpus.append(
                        GPUInfo(
                            index=index,
                            name=name,
                            memory_total_mb=memory_total,
                            memory_free_mb=memory_free,
                            nvenc_supported=nvenc_supported,
                        )
                    )

        except FileNotFoundError:
            # nvidia-smi not found - no NVIDIA GPUs
            pass
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

        return self._gpus

    def _check_nvenc_support(self, gpu_name: str) -> bool:
        """Check if a GPU supports NVENC based on its name.

        Args:
            gpu_name: GPU name string from nvidia-smi.

        Returns:
            True if GPU likely supports NVENC.
        """
        for pattern in self.NVENC_SUPPORTED_PATTERNS:
            if re.search(pattern, gpu_name, re.IGNORECASE):
                return True
        return False

    def get_nvenc_gpus(self) -> list[GPUInfo]:
        """Get only GPUs that support NVENC.

        Returns:
            List of GPUInfo for NVENC-capable GPUs.
        """
        return [gpu for gpu in self.detect_gpus() if gpu.nvenc_supported]

    def get_gpu_by_index(self, index: int) -> Optional[GPUInfo]:
        """Get GPU info by index.

        Args:
            index: GPU index (0, 1, etc.)

        Returns:
            GPUInfo if found, None otherwise.
        """
        for gpu in self.detect_gpus():
            if gpu.index == index:
                return gpu
        return None

    def check_ffmpeg_nvenc(self) -> bool:
        """Check if FFmpeg has NVENC support compiled in.

        Returns:
            True if FFmpeg supports h264_nvenc encoder.
        """
        try:
            result = subprocess.run(
                ["ffmpeg", "-encoders"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            return "h264_nvenc" in result.stdout

        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


def select_gpu(
    preferred_index: Optional[int] = None,
    require_nvenc: bool = True,
) -> Optional[GPUInfo]:
    """Select a GPU for encoding.

    Args:
        preferred_index: Preferred GPU index. If None, auto-selects.
        require_nvenc: If True, only consider NVENC-capable GPUs.

    Returns:
        Selected GPUInfo, or None if no suitable GPU found.
    """
    detector = GPUDetector()

    if require_nvenc:
        gpus = detector.get_nvenc_gpus()
    else:
        gpus = detector.detect_gpus()

    if not gpus:
        return None

    # If preferred index specified, try to use it
    if preferred_index is not None:
        for gpu in gpus:
            if gpu.index == preferred_index:
                return gpu
        # Preferred GPU not available/suitable
        return None

    # Auto-select: prefer GPU with most free memory
    return max(gpus, key=lambda g: g.memory_free_mb)


def get_gpu_choices() -> list[tuple[int, str]]:
    """Get list of available GPUs for user selection.

    Returns:
        List of (index, description) tuples.
    """
    detector = GPUDetector()
    gpus = detector.detect_gpus()

    choices = []
    for gpu in gpus:
        nvenc = "NVENC" if gpu.nvenc_supported else "NO NVENC"
        desc = f"{gpu.name} - {gpu.memory_total_mb}MB [{nvenc}]"
        choices.append((gpu.index, desc))

    return choices


def validate_gpu_setup(gpu_index: Optional[int] = None) -> tuple[bool, str, Optional[GPUInfo]]:
    """Validate that GPU acceleration is available.

    Args:
        gpu_index: Optional specific GPU to validate.

    Returns:
        Tuple of (success, message, gpu_info).
    """
    detector = GPUDetector()

    # Check FFmpeg NVENC support
    if not detector.check_ffmpeg_nvenc():
        return False, "FFmpeg does not have NVENC support. Install FFmpeg with NVENC enabled.", None

    # Check for NVENC GPUs
    nvenc_gpus = detector.get_nvenc_gpus()
    if not nvenc_gpus:
        all_gpus = detector.detect_gpus()
        if not all_gpus:
            return False, "No NVIDIA GPUs detected. GPU acceleration requires an NVIDIA GPU.", None
        else:
            gpu_names = ", ".join(g.name for g in all_gpus)
            return False, f"No NVENC-capable GPUs found. Detected GPUs: {gpu_names}", None

    # If specific GPU requested, validate it
    if gpu_index is not None:
        gpu = detector.get_gpu_by_index(gpu_index)
        if gpu is None:
            available = ", ".join(str(g.index) for g in nvenc_gpus)
            return False, f"GPU {gpu_index} not found. Available GPUs: {available}", None
        if not gpu.nvenc_supported:
            return False, f"GPU {gpu_index} ({gpu.name}) does not support NVENC.", None
        return True, f"Using GPU {gpu_index}: {gpu.name}", gpu

    # Auto-select best GPU
    gpu = select_gpu(require_nvenc=True)
    if gpu:
        return True, f"Auto-selected GPU {gpu.index}: {gpu.name}", gpu

    return False, "Could not select a suitable GPU.", None


# Quick test when run directly
if __name__ == "__main__":
    print("GPU Detection Test\n" + "=" * 50)

    detector = GPUDetector()
    gpus = detector.detect_gpus()

    if not gpus:
        print("No NVIDIA GPUs detected.")
    else:
        print(f"Found {len(gpus)} GPU(s):\n")
        for gpu in gpus:
            print(f"  {gpu}")

    print()
    print(f"FFmpeg NVENC support: {detector.check_ffmpeg_nvenc()}")

    print()
    success, message, gpu = validate_gpu_setup()
    print(f"Validation: {message}")
