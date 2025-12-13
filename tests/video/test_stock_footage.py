"""Tests for stock footage module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from socials_automator.video import PexelsConfig, StockFootageError
from socials_automator.video.stock_footage import (
    PexelsClient,
    StockFootageService,
    get_video_dimensions,
    is_9_16,
)


class TestIs916:
    """Tests for 9:16 aspect ratio detection."""

    def test_perfect_ratio(self):
        """Test perfect 9:16 video."""
        video = {
            "video_files": [{"width": 1080, "height": 1920}],
        }
        assert is_9_16(video) is True

    def test_close_ratio(self):
        """Test video close to 9:16."""
        video = {
            "video_files": [{"width": 1000, "height": 1800}],
        }
        # Within tolerance
        assert is_9_16(video) is True

    def test_landscape_video(self):
        """Test landscape video (not 9:16)."""
        video = {
            "video_files": [{"width": 1920, "height": 1080}],
        }
        assert is_9_16(video) is False

    def test_square_video(self):
        """Test square video."""
        video = {
            "video_files": [{"width": 1080, "height": 1080}],
        }
        assert is_9_16(video) is False

    def test_empty_video_files(self):
        """Test with no video files."""
        video = {"video_files": []}
        assert is_9_16(video) is False

    def test_missing_dimensions(self):
        """Test with missing dimensions."""
        video = {"video_files": [{"quality": "hd"}]}
        assert is_9_16(video) is False


class TestGetVideoDimensions:
    """Tests for video dimension extraction."""

    def test_valid_video(self):
        """Test extracting dimensions from valid video."""
        video = {
            "video_files": [{"width": 1080, "height": 1920}],
        }
        width, height = get_video_dimensions(video)
        assert width == 1080
        assert height == 1920

    def test_empty_video_files(self):
        """Test with empty video files."""
        video = {"video_files": []}
        width, height = get_video_dimensions(video)
        assert width == 0
        assert height == 0


class TestPexelsClient:
    """Tests for Pexels API client."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        client = PexelsClient()
        assert client.config.prefer_orientation == "portrait"

    def test_init_custom_config(self, pexels_config):
        """Test initialization with custom config."""
        client = PexelsClient(config=pexels_config)
        assert client.config == pexels_config

    def test_api_key_missing(self, monkeypatch):
        """Test error when API key is missing."""
        monkeypatch.delenv("PEXELS_API_KEY", raising=False)
        client = PexelsClient()
        with pytest.raises(StockFootageError, match="API key not found"):
            _ = client.api_key

    @pytest.mark.asyncio
    async def test_search_videos(self, monkeypatch, mock_pexels_response):
        """Test video search."""
        monkeypatch.setenv("PEXELS_API_KEY", "test_key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(
                return_value=MagicMock(
                    json=MagicMock(return_value=mock_pexels_response),
                    raise_for_status=MagicMock(),
                )
            )
            mock_client.return_value = mock_instance

            client = PexelsClient()
            client._client = mock_instance

            results = await client.search_videos("technology")
            assert results["total_results"] == 1
            assert len(results["videos"]) == 1

    @pytest.mark.asyncio
    async def test_close_client(self, monkeypatch):
        """Test closing the client."""
        monkeypatch.setenv("PEXELS_API_KEY", "test_key")

        client = PexelsClient()
        mock_client = AsyncMock()
        client._client = mock_client

        await client.close()
        mock_client.aclose.assert_called_once()
        assert client._client is None


class TestStockFootageService:
    """Tests for stock footage service."""

    def test_init(self, pexels_config):
        """Test service initialization."""
        service = StockFootageService(config=pexels_config)
        assert service.config == pexels_config

    def test_select_best_duration(self):
        """Test video selection by duration."""
        service = StockFootageService()

        videos = [
            {"duration": 10, "id": 1},
            {"duration": 15, "id": 2},
            {"duration": 7, "id": 3},
        ]

        # Should select video closest to target
        result = service._select_best_duration(videos, target_duration=9)
        assert result["id"] == 1  # 10 seconds, closest to 9 (diff=1 vs diff=2 for id=3)

        result = service._select_best_duration(videos, target_duration=14)
        assert result["id"] == 2  # 15 seconds, closest to 14

    def test_select_best_duration_empty(self):
        """Test with empty video list."""
        service = StockFootageService()
        result = service._select_best_duration([], target_duration=10)
        assert result is None


class TestStockFootageIntegration:
    """Integration tests (require API key)."""

    @pytest.mark.skipif(
        True,
        reason="Requires PEXELS_API_KEY and network access",
    )
    @pytest.mark.asyncio
    async def test_find_video(self, temp_dir, monkeypatch):
        """Test finding and downloading a video."""
        # Set API key for testing
        monkeypatch.setenv("PEXELS_API_KEY", "your_test_key")

        service = StockFootageService()
        try:
            clip = await service.find_video(
                keywords=["technology", "abstract"],
                target_duration=10.0,
                scene_index=1,
                output_dir=temp_dir,
            )
            assert clip.path.exists()
            assert clip.duration_seconds > 0
        finally:
            await service.close()
