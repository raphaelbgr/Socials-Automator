"""Data models for Instagram posting."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import os


class InstagramPostStatus(str, Enum):
    """Status of an Instagram posting operation."""
    PENDING = "pending"
    UPLOADING = "uploading"
    CREATING_CONTAINERS = "creating_containers"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    FAILED = "failed"


@dataclass
class InstagramConfig:
    """Instagram API configuration loaded from environment."""
    instagram_user_id: str
    access_token: str
    cloudinary_cloud_name: str
    cloudinary_api_key: str
    cloudinary_api_secret: str

    # API settings
    api_version: str = "v21.0"

    @classmethod
    def from_env(cls) -> "InstagramConfig":
        """Load configuration from environment variables."""
        instagram_user_id = os.getenv("INSTAGRAM_USER_ID")
        access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN")
        cloudinary_cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
        cloudinary_api_key = os.getenv("CLOUDINARY_API_KEY")
        cloudinary_api_secret = os.getenv("CLOUDINARY_API_SECRET")

        missing = []
        if not instagram_user_id:
            missing.append("INSTAGRAM_USER_ID")
        if not access_token:
            missing.append("INSTAGRAM_ACCESS_TOKEN")
        if not cloudinary_cloud_name:
            missing.append("CLOUDINARY_CLOUD_NAME")
        if not cloudinary_api_key:
            missing.append("CLOUDINARY_API_KEY")
        if not cloudinary_api_secret:
            missing.append("CLOUDINARY_API_SECRET")

        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Please add them to your .env file. See README for setup instructions."
            )

        return cls(
            instagram_user_id=instagram_user_id,
            access_token=access_token,
            cloudinary_cloud_name=cloudinary_cloud_name,
            cloudinary_api_key=cloudinary_api_key,
            cloudinary_api_secret=cloudinary_api_secret,
        )

    def is_configured(self) -> bool:
        """Check if all required fields are set."""
        return all([
            self.instagram_user_id,
            self.access_token,
            self.cloudinary_cloud_name,
            self.cloudinary_api_key,
            self.cloudinary_api_secret,
        ])


@dataclass
class InstagramProgress:
    """Progress tracking for Instagram publishing."""
    status: InstagramPostStatus = InstagramPostStatus.PENDING
    current_step: str = "Initializing..."
    progress_percent: float = 0.0

    # Upload tracking
    images_uploaded: int = 0
    total_images: int = 0
    image_urls: list[str] = field(default_factory=list)

    # Container tracking
    containers_created: int = 0
    container_ids: list[str] = field(default_factory=list)
    carousel_container_id: str | None = None

    # Result
    error: str | None = None


@dataclass
class InstagramPublishResult:
    """Result of publishing to Instagram."""
    success: bool
    media_id: str | None = None
    permalink: str | None = None
    error_message: str | None = None

    # Details
    container_ids: list[str] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)
    published_at: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "media_id": self.media_id,
            "permalink": self.permalink,
            "error_message": self.error_message,
            "published_at": self.published_at.isoformat() if self.published_at else None,
        }
