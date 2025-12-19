"""Caption audit and sync utilities for Instagram captions.

This module provides tools to:
1. Audit posted reels for caption issues (log-based or API-verified)
2. Sync actual Instagram captions to local metadata
3. Generate fix reports for reels with missing/mismatched captions

Usage:
    from socials_automator.utils.caption_audit import (
        # Audit tools
        CaptionAuditor,
        CaptionIssue,
        CaptionAuditResult,
        # Sync tools
        CaptionSyncer,
        SyncResult,
        sync_profile_captions,
    )

    # Quick audit (log-based)
    result = audit_profile_captions("my-profile")

    # Sync captions from Instagram to local metadata
    syncer = CaptionSyncer(access_token="...")
    result = await syncer.sync_profile("my-profile")

    # Generate fix report for reels with empty Instagram captions
    report = syncer.generate_fix_report(result)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class CaptionIssue:
    """A single caption issue detected during audit."""

    reel_path: Path
    reel_name: str
    permalink: str
    issue_type: str  # "rate_limit", "api_error", "mismatch", "empty", "unknown"
    issue_detail: str
    uploaded_at: Optional[str]
    local_caption: str
    instagram_caption: Optional[str] = None  # Only populated with API verification


@dataclass
class CaptionAuditResult:
    """Result of caption audit operation."""

    profile: str
    total_reels_scanned: int
    issues_found: int
    issues: List[CaptionIssue] = field(default_factory=list)
    report_path: Optional[Path] = None


@dataclass
class LogError:
    """An error entry parsed from Instagram API log."""

    timestamp: str
    user_id: Optional[str]
    container_id: Optional[str]
    error_type: str  # "rate_limit", "api_error"
    detail: str


# =============================================================================
# CAPTION AUDITOR CLASS
# =============================================================================


class CaptionAuditor:
    """Auditor for detecting caption issues in posted reels.

    Combines log-based detection with optional API verification for
    comprehensive caption issue detection.

    Example:
        auditor = CaptionAuditor(
            profile_path=Path("profiles/news.but.quick"),
            log_path=Path("logs/instagram_api.log"),
        )

        # Quick log-based audit
        result = auditor.audit()

        # Thorough API-verified audit
        result = await auditor.audit_with_verification(access_token)
    """

    # Error patterns to detect in logs
    RATE_LIMIT_PATTERNS = [
        "rate limit",
        "too many actions",
        "2207042",
        "media publish limit",
    ]

    def __init__(
        self,
        profile_path: Path,
        log_path: Optional[Path] = None,
        profile_name: Optional[str] = None,
    ):
        """Initialize the auditor.

        Args:
            profile_path: Path to the profile directory.
            log_path: Path to instagram_api.log. Defaults to logs/instagram_api.log.
            profile_name: Display name for the profile. Defaults to folder name.
        """
        self.profile_path = profile_path
        self.log_path = log_path or Path.cwd() / "logs" / "instagram_api.log"
        self.profile_name = profile_name or profile_path.name

        self._log_errors: Optional[Dict[str, List[LogError]]] = None

    # =========================================================================
    # LOG PARSING
    # =========================================================================

    def parse_log_errors(self) -> Dict[str, List[LogError]]:
        """Parse Instagram API log for upload errors.

        Returns:
            Dict with 'by_container' and 'by_timestamp' mappings to errors.
        """
        if self._log_errors is not None:
            return self._log_errors

        errors_by_container: Dict[str, List[LogError]] = {}
        errors_by_timestamp: Dict[str, List[LogError]] = {}

        if not self.log_path.exists():
            self._log_errors = {
                "by_container": errors_by_container,
                "by_timestamp": errors_by_timestamp,
            }
            return self._log_errors

        current_session_user = None
        current_container_id = None
        current_timestamp = None

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                # Parse timestamp
                ts_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                if ts_match:
                    current_timestamp = ts_match.group(1)

                # Track session user
                if "NEW SESSION" in line and "Instagram User ID:" in line:
                    match = re.search(r"User ID: (\d+)", line)
                    if match:
                        current_session_user = match.group(1)

                # Track container creation
                if "Reel container created:" in line:
                    match = re.search(r"container created: (\d+)", line)
                    if match:
                        current_container_id = match.group(1)

                # Detect rate limit errors
                line_lower = line.lower()
                is_rate_limit = any(p in line_lower for p in self.RATE_LIMIT_PATTERNS)

                if "ERROR" in line and is_rate_limit:
                    error = LogError(
                        timestamp=current_timestamp or "",
                        user_id=current_session_user,
                        container_id=current_container_id,
                        error_type="rate_limit",
                        detail=line.strip()[:200],
                    )

                    if current_container_id:
                        if current_container_id not in errors_by_container:
                            errors_by_container[current_container_id] = []
                        errors_by_container[current_container_id].append(error)

                    if current_timestamp:
                        if current_timestamp not in errors_by_timestamp:
                            errors_by_timestamp[current_timestamp] = []
                        errors_by_timestamp[current_timestamp].append(error)

                # Detect other API errors
                elif "ERROR" in line and "API CALL" in line:
                    error = LogError(
                        timestamp=current_timestamp or "",
                        user_id=current_session_user,
                        container_id=current_container_id,
                        error_type="api_error",
                        detail=line.strip()[:200],
                    )

                    if current_container_id:
                        if current_container_id not in errors_by_container:
                            errors_by_container[current_container_id] = []
                        errors_by_container[current_container_id].append(error)

        self._log_errors = {
            "by_container": errors_by_container,
            "by_timestamp": errors_by_timestamp,
        }
        return self._log_errors

    # =========================================================================
    # REEL SCANNING
    # =========================================================================

    def find_posted_reels(self) -> List[Path]:
        """Find all posted reel directories.

        Returns:
            List of paths to posted reel folders.
        """
        reels_dir = self.profile_path / "reels"
        if not reels_dir.exists():
            return []

        posted_reels = []

        for year_dir in reels_dir.glob("*"):
            if not (year_dir.is_dir() and year_dir.name.isdigit()):
                continue

            for month_dir in year_dir.glob("*"):
                if not (month_dir.is_dir() and month_dir.name.isdigit()):
                    continue

                posted_dir = month_dir / "posted"
                if not posted_dir.exists():
                    continue

                for reel_dir in posted_dir.glob("*"):
                    if reel_dir.is_dir():
                        posted_reels.append(reel_dir)

        return sorted(posted_reels)

    def _load_reel_metadata(self, reel_path: Path) -> Optional[dict]:
        """Load metadata.json from a reel folder."""
        metadata_path = reel_path / "metadata.json"
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _load_reel_caption(self, reel_path: Path) -> str:
        """Load caption+hashtags.txt from a reel folder."""
        caption_path = reel_path / "caption+hashtags.txt"
        if not caption_path.exists():
            return ""

        try:
            with open(caption_path, encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""

    # =========================================================================
    # ISSUE DETECTION
    # =========================================================================

    def _check_reel_for_issues(
        self,
        reel_path: Path,
        log_errors: Dict[str, List[LogError]],
    ) -> Optional[CaptionIssue]:
        """Check a single reel for caption issues.

        Args:
            reel_path: Path to the reel folder.
            log_errors: Parsed log errors from parse_log_errors().

        Returns:
            CaptionIssue if an issue was detected, None otherwise.
        """
        metadata = self._load_reel_metadata(reel_path)
        if metadata is None:
            return None

        # Get Instagram info
        platform_status = metadata.get("platform_status", {})
        ig_status = platform_status.get("instagram", {})

        if not ig_status.get("uploaded"):
            return None  # Not uploaded to Instagram

        permalink = ig_status.get("permalink", "")
        media_id = ig_status.get("media_id", "")
        uploaded_at = ig_status.get("uploaded_at", "")

        # Load local caption
        local_caption = self._load_reel_caption(reel_path)

        # Check for issues
        errors_by_container = log_errors.get("by_container", {})
        errors_by_timestamp = log_errors.get("by_timestamp", {})

        issue_found = False
        issue_type = "unknown"
        issue_detail = ""

        # Check 1: Error by media container ID
        if media_id in errors_by_container:
            error = errors_by_container[media_id][0]
            issue_found = True
            issue_type = error.error_type
            issue_detail = f"Error during upload: {error.detail}"

        # Check 2: Error by timestamp (within 5 minute window)
        if not issue_found and uploaded_at:
            try:
                upload_dt = datetime.fromisoformat(uploaded_at.replace("Z", ""))

                for ts, errors in errors_by_timestamp.items():
                    try:
                        error_dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                        time_diff = abs((upload_dt - error_dt).total_seconds())

                        if time_diff < 300:  # 5 minutes
                            issue_found = True
                            issue_type = errors[0].error_type
                            issue_detail = f"Error near upload time ({ts}): {errors[0].detail}"
                            break
                    except Exception:
                        continue
            except Exception:
                pass

        # Check 3: Empty local caption
        if not issue_found and not local_caption:
            issue_found = True
            issue_type = "empty"
            issue_detail = "Local caption file is empty"

        if issue_found:
            return CaptionIssue(
                reel_path=reel_path,
                reel_name=reel_path.name,
                permalink=permalink,
                issue_type=issue_type,
                issue_detail=issue_detail,
                uploaded_at=uploaded_at,
                local_caption=local_caption,
            )

        return None

    # =========================================================================
    # AUDIT METHODS
    # =========================================================================

    def audit(self) -> CaptionAuditResult:
        """Run log-based caption audit.

        Fast audit that detects issues by cross-referencing posted reels
        with error entries in the Instagram API log.

        Returns:
            CaptionAuditResult with detected issues.
        """
        log_errors = self.parse_log_errors()
        posted_reels = self.find_posted_reels()

        issues = []
        for reel_path in posted_reels:
            issue = self._check_reel_for_issues(reel_path, log_errors)
            if issue:
                issues.append(issue)

        return CaptionAuditResult(
            profile=self.profile_name,
            total_reels_scanned=len(posted_reels),
            issues_found=len(issues),
            issues=issues,
        )

    async def verify_caption_via_api(
        self,
        media_id: str,
        access_token: str,
    ) -> Optional[str]:
        """Fetch actual caption from Instagram API.

        Args:
            media_id: Instagram media ID.
            access_token: Instagram access token.

        Returns:
            Caption text or None if fetch failed.
        """
        import aiohttp

        try:
            url = f"https://graph.facebook.com/v21.0/{media_id}"
            params = {
                "fields": "caption",
                "access_token": access_token,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("caption", "")
        except Exception:
            pass

        return None

    async def audit_with_verification(
        self,
        access_token: str,
        progress_callback=None,
    ) -> CaptionAuditResult:
        """Run audit with API verification.

        Performs log-based detection first, then verifies detected issues
        by fetching actual captions from Instagram API.

        Args:
            access_token: Instagram access token for API calls.
            progress_callback: Optional callback(current, total, reel_name).

        Returns:
            CaptionAuditResult with verified issues.
        """
        # First run log-based audit
        result = self.audit()

        if not result.issues:
            return result

        # Verify each issue via API
        for idx, issue in enumerate(result.issues):
            if progress_callback:
                progress_callback(idx + 1, len(result.issues), issue.reel_name)

            metadata = self._load_reel_metadata(issue.reel_path)
            if metadata:
                media_id = metadata.get("platform_status", {}).get("instagram", {}).get("media_id")
                if media_id:
                    caption = await self.verify_caption_via_api(media_id, access_token)
                    issue.instagram_caption = caption

                    # Update issue type based on verification
                    if caption is not None:
                        if not caption:
                            issue.issue_type = "mismatch"
                            issue.issue_detail = "Instagram has empty caption (verified via API)"
                        elif caption != issue.local_caption:
                            issue.issue_type = "mismatch"
                            issue.issue_detail = "Instagram caption differs from local"

        return result

    # =========================================================================
    # REPORT GENERATION
    # =========================================================================

    def generate_report(
        self,
        result: CaptionAuditResult,
        output_dir: Path,
    ) -> Path:
        """Generate markdown report with fix instructions.

        Args:
            result: Audit result with issues.
            output_dir: Directory to write report.

        Returns:
            Path to generated report file.
        """
        report_path = output_dir / "fix_captions.md"

        lines = [
            f"# Reels Missing Captions - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"**Profile**: {result.profile}",
            f"**Total Reels Scanned**: {result.total_reels_scanned}",
            f"**Issues Found**: {result.issues_found}",
            "",
            "---",
            "",
        ]

        if not result.issues:
            lines.append("No caption issues detected!")
        else:
            lines.append("## Reels to Fix")
            lines.append("")
            lines.append("Click the URL, then edit the reel description and paste the caption below.")
            lines.append("")

            for idx, issue in enumerate(result.issues, 1):
                lines.append(f"### {idx}. {issue.reel_name}")
                lines.append("")
                lines.append(f"- **URL**: {issue.permalink}")
                lines.append(f"- **Issue**: {issue.issue_type} - {issue.issue_detail}")
                lines.append(f"- **Uploaded**: {issue.uploaded_at or 'Unknown'}")

                if issue.instagram_caption is not None:
                    lines.append(f"- **Current Instagram Caption**: {len(issue.instagram_caption)} chars")

                lines.append("")
                lines.append("**Caption to paste:**")
                lines.append("")
                lines.append("```")
                lines.append(issue.local_caption if issue.local_caption else "(No local caption found)")
                lines.append("```")
                lines.append("")
                lines.append("---")
                lines.append("")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        result.report_path = report_path
        return report_path


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def audit_profile_captions(
    profile_name: str,
    profiles_dir: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> CaptionAuditResult:
    """Convenience function to audit a profile's captions.

    Args:
        profile_name: Name of the profile folder.
        profiles_dir: Base profiles directory. Defaults to cwd/profiles.
        log_path: Path to instagram_api.log. Defaults to cwd/logs/instagram_api.log.

    Returns:
        CaptionAuditResult with detected issues.

    Example:
        result = audit_profile_captions("news.but.quick")
        print(f"Found {result.issues_found} issues")
    """
    if profiles_dir is None:
        profiles_dir = Path.cwd() / "profiles"

    profile_path = profiles_dir / profile_name

    auditor = CaptionAuditor(
        profile_path=profile_path,
        log_path=log_path,
        profile_name=profile_name,
    )

    return auditor.audit()


async def audit_and_verify_captions(
    profile_name: str,
    access_token: str,
    profiles_dir: Optional[Path] = None,
    log_path: Optional[Path] = None,
    progress_callback=None,
) -> CaptionAuditResult:
    """Convenience function to audit and verify captions via API.

    Args:
        profile_name: Name of the profile folder.
        access_token: Instagram access token for API verification.
        profiles_dir: Base profiles directory. Defaults to cwd/profiles.
        log_path: Path to instagram_api.log. Defaults to cwd/logs/instagram_api.log.
        progress_callback: Optional callback(current, total, reel_name).

    Returns:
        CaptionAuditResult with verified issues.

    Example:
        result = await audit_and_verify_captions(
            "news.but.quick",
            access_token=os.getenv("INSTAGRAM_ACCESS_TOKEN"),
        )
    """
    if profiles_dir is None:
        profiles_dir = Path.cwd() / "profiles"

    profile_path = profiles_dir / profile_name

    auditor = CaptionAuditor(
        profile_path=profile_path,
        log_path=log_path,
        profile_name=profile_name,
    )

    return await auditor.audit_with_verification(
        access_token=access_token,
        progress_callback=progress_callback,
    )


def generate_caption_fix_report(
    profile_name: str,
    result: CaptionAuditResult,
    profiles_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Generate fix report in the profile's posted folder.

    Args:
        profile_name: Name of the profile folder.
        result: Audit result with issues.
        profiles_dir: Base profiles directory. Defaults to cwd/profiles.

    Returns:
        Path to generated report, or None if no posted folder found.

    Example:
        result = audit_profile_captions("ai.for.mortals")
        if result.issues:
            report = generate_caption_fix_report("ai.for.mortals", result)
            print(f"Report: {report}")
    """
    if profiles_dir is None:
        profiles_dir = Path.cwd() / "profiles"

    profile_path = profiles_dir / profile_name
    reels_dir = profile_path / "reels"

    if not reels_dir.exists():
        return None

    # Find the most recent posted directory
    posted_dirs = []
    for year_dir in reels_dir.glob("*"):
        if year_dir.is_dir() and year_dir.name.isdigit():
            for month_dir in year_dir.glob("*"):
                if month_dir.is_dir() and month_dir.name.isdigit():
                    posted_dir = month_dir / "posted"
                    if posted_dir.exists():
                        posted_dirs.append(posted_dir)

    if not posted_dirs:
        return None

    output_dir = sorted(posted_dirs)[-1]

    auditor = CaptionAuditor(profile_path=profile_path, profile_name=profile_name)
    return auditor.generate_report(result, output_dir)


# =============================================================================
# CAPTION SYNC DATA CLASSES
# =============================================================================


@dataclass
class ReelSyncStatus:
    """Status of a single reel after caption sync."""

    reel_path: Path
    reel_name: str
    media_id: str
    permalink: str
    local_caption: str
    instagram_caption: Optional[str]  # None if fetch failed
    status: str  # "synced", "empty", "mismatch", "error", "skipped"
    error: Optional[str] = None


@dataclass
class SyncResult:
    """Result of caption sync operation."""

    profile: str
    total_reels: int
    synced: int
    empty_captions: int  # Reels with empty Instagram caption
    mismatched: int  # Reels where Instagram != local
    errors: int
    skipped: int  # Reels without media_id
    reels: List[ReelSyncStatus] = field(default_factory=list)
    report_path: Optional[Path] = None


# =============================================================================
# CAPTION SYNCER CLASS
# =============================================================================


class CaptionSyncer:
    """Syncs actual Instagram captions to local metadata.

    Fetches the real caption from Instagram for each posted reel and
    stores it in metadata.json under `instagram.actual_caption`.

    Example:
        syncer = CaptionSyncer(access_token=os.getenv("INSTAGRAM_ACCESS_TOKEN"))

        # Sync a single profile
        result = await syncer.sync_profile("ai.for.mortals")
        print(f"Empty captions: {result.empty_captions}")

        # Sync all profiles
        results = await syncer.sync_all_profiles()

        # Generate fix report
        if result.empty_captions > 0:
            report = syncer.generate_fix_report(result, output_dir)
    """

    def __init__(
        self,
        access_token: str,
        profiles_dir: Optional[Path] = None,
        request_delay: float = 0.5,
    ):
        """Initialize the syncer.

        Args:
            access_token: Instagram Graph API access token.
            profiles_dir: Base profiles directory. Defaults to cwd/profiles.
            request_delay: Delay between API requests in seconds.
        """
        self.access_token = access_token
        self.profiles_dir = profiles_dir or Path.cwd() / "profiles"
        self.request_delay = request_delay

    async def fetch_instagram_caption(self, media_id: str) -> tuple[Optional[str], Optional[str]]:
        """Fetch caption from Instagram API.

        Args:
            media_id: Instagram media ID.

        Returns:
            Tuple of (caption, error). Caption is None if fetch failed.
        """
        import aiohttp
        import asyncio

        try:
            url = f"https://graph.facebook.com/v21.0/{media_id}"
            params = {
                "fields": "caption",
                "access_token": self.access_token,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Instagram returns empty string if no caption, or the caption
                        return data.get("caption", ""), None
                    else:
                        error_data = await response.json()
                        error_msg = error_data.get("error", {}).get("message", f"HTTP {response.status}")
                        return None, error_msg

        except Exception as e:
            return None, str(e)

    def _find_posted_reels(self, profile_path: Path) -> List[Path]:
        """Find all posted reel directories for a profile."""
        reels_dir = profile_path / "reels"
        if not reels_dir.exists():
            return []

        posted_reels = []

        for year_dir in reels_dir.glob("*"):
            if not (year_dir.is_dir() and year_dir.name.isdigit()):
                continue

            for month_dir in year_dir.glob("*"):
                if not (month_dir.is_dir() and month_dir.name.isdigit()):
                    continue

                posted_dir = month_dir / "posted"
                if not posted_dir.exists():
                    continue

                for reel_dir in posted_dir.glob("*"):
                    if reel_dir.is_dir() and (reel_dir / "metadata.json").exists():
                        posted_reels.append(reel_dir)

        return sorted(posted_reels)

    def _load_metadata(self, reel_path: Path) -> Optional[dict]:
        """Load metadata.json from reel folder."""
        metadata_path = reel_path / "metadata.json"
        try:
            with open(metadata_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_metadata(self, reel_path: Path, metadata: dict) -> bool:
        """Save metadata.json to reel folder."""
        metadata_path = reel_path / "metadata.json"
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    def _load_local_caption(self, reel_path: Path) -> str:
        """Load local caption from caption+hashtags.txt."""
        caption_path = reel_path / "caption+hashtags.txt"
        if not caption_path.exists():
            return ""
        try:
            with open(caption_path, encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""

    async def sync_reel(self, reel_path: Path) -> ReelSyncStatus:
        """Sync caption for a single reel.

        Args:
            reel_path: Path to the reel folder.

        Returns:
            ReelSyncStatus with sync result.
        """
        import asyncio

        reel_name = reel_path.name
        metadata = self._load_metadata(reel_path)

        if metadata is None:
            return ReelSyncStatus(
                reel_path=reel_path,
                reel_name=reel_name,
                media_id="",
                permalink="",
                local_caption="",
                instagram_caption=None,
                status="error",
                error="Could not load metadata.json",
            )

        # Get Instagram info
        platform_status = metadata.get("platform_status", {})
        ig_status = platform_status.get("instagram", {})

        media_id = ig_status.get("media_id", "")
        permalink = ig_status.get("permalink", "")

        if not media_id:
            return ReelSyncStatus(
                reel_path=reel_path,
                reel_name=reel_name,
                media_id="",
                permalink=permalink,
                local_caption="",
                instagram_caption=None,
                status="skipped",
                error="No media_id in metadata",
            )

        # Load local caption
        local_caption = self._load_local_caption(reel_path)

        # Fetch from Instagram
        instagram_caption, error = await self.fetch_instagram_caption(media_id)

        if error:
            return ReelSyncStatus(
                reel_path=reel_path,
                reel_name=reel_name,
                media_id=media_id,
                permalink=permalink,
                local_caption=local_caption,
                instagram_caption=None,
                status="error",
                error=error,
            )

        # Update metadata with actual caption
        if "instagram" not in metadata:
            metadata["instagram"] = {}
        metadata["instagram"]["actual_caption"] = instagram_caption
        metadata["instagram"]["caption_synced_at"] = datetime.now().isoformat()

        self._save_metadata(reel_path, metadata)

        # Determine status
        if not instagram_caption:
            status = "empty"
        elif instagram_caption != local_caption:
            status = "mismatch"
        else:
            status = "synced"

        # Add delay to avoid rate limiting
        await asyncio.sleep(self.request_delay)

        return ReelSyncStatus(
            reel_path=reel_path,
            reel_name=reel_name,
            media_id=media_id,
            permalink=permalink,
            local_caption=local_caption,
            instagram_caption=instagram_caption,
            status=status,
        )

    async def sync_profile(
        self,
        profile_name: str,
        progress_callback=None,
    ) -> SyncResult:
        """Sync captions for all reels in a profile.

        Args:
            profile_name: Name of the profile folder.
            progress_callback: Optional callback(current, total, reel_name).

        Returns:
            SyncResult with all reel statuses.
        """
        profile_path = self.profiles_dir / profile_name

        if not profile_path.exists():
            return SyncResult(
                profile=profile_name,
                total_reels=0,
                synced=0,
                empty_captions=0,
                mismatched=0,
                errors=0,
                skipped=0,
            )

        posted_reels = self._find_posted_reels(profile_path)
        total = len(posted_reels)

        reels = []
        for idx, reel_path in enumerate(posted_reels):
            if progress_callback:
                progress_callback(idx + 1, total, reel_path.name)

            status = await self.sync_reel(reel_path)
            reels.append(status)

        # Calculate stats
        synced = sum(1 for r in reels if r.status == "synced")
        empty_captions = sum(1 for r in reels if r.status == "empty")
        mismatched = sum(1 for r in reels if r.status == "mismatch")
        errors = sum(1 for r in reels if r.status == "error")
        skipped = sum(1 for r in reels if r.status == "skipped")

        return SyncResult(
            profile=profile_name,
            total_reels=total,
            synced=synced,
            empty_captions=empty_captions,
            mismatched=mismatched,
            errors=errors,
            skipped=skipped,
            reels=reels,
        )

    async def sync_all_profiles(
        self,
        progress_callback=None,
    ) -> List[SyncResult]:
        """Sync captions for all profiles.

        Args:
            progress_callback: Optional callback(profile_name, current, total).

        Returns:
            List of SyncResult for each profile.
        """
        results = []

        if not self.profiles_dir.exists():
            return results

        profiles = [d for d in self.profiles_dir.iterdir() if d.is_dir()]
        total = len(profiles)

        for idx, profile_path in enumerate(profiles):
            if progress_callback:
                progress_callback(profile_path.name, idx + 1, total)

            result = await self.sync_profile(profile_path.name)
            if result.total_reels > 0:
                results.append(result)

        return results

    def generate_fix_report(
        self,
        result: SyncResult,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Generate markdown report for reels needing caption fixes.

        Args:
            result: SyncResult from sync operation.
            output_dir: Directory to write report. Defaults to profile's posted folder.

        Returns:
            Path to generated report.
        """
        # Find output directory
        if output_dir is None:
            profile_path = self.profiles_dir / result.profile
            reels_dir = profile_path / "reels"

            posted_dirs = []
            for year_dir in reels_dir.glob("*"):
                if year_dir.is_dir() and year_dir.name.isdigit():
                    for month_dir in year_dir.glob("*"):
                        if month_dir.is_dir() and month_dir.name.isdigit():
                            posted_dir = month_dir / "posted"
                            if posted_dir.exists():
                                posted_dirs.append(posted_dir)

            output_dir = sorted(posted_dirs)[-1] if posted_dirs else profile_path

        report_path = output_dir / "fix_captions.md"

        # Filter to only problematic reels
        issues = [r for r in result.reels if r.status in ("empty", "mismatch")]

        lines = [
            f"# Reels Needing Caption Fixes - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"**Profile**: {result.profile}",
            f"**Total Reels Synced**: {result.total_reels}",
            f"**Empty Captions**: {result.empty_captions}",
            f"**Mismatched Captions**: {result.mismatched}",
            "",
            "---",
            "",
        ]

        if not issues:
            lines.append("All captions are correct! No fixes needed.")
        else:
            lines.append("## Reels to Fix")
            lines.append("")
            lines.append("Click each URL, edit the reel, and paste the caption below.")
            lines.append("")

            for idx, reel in enumerate(issues, 1):
                status_emoji = "[EMPTY]" if reel.status == "empty" else "[MISMATCH]"
                lines.append(f"### {idx}. {status_emoji} {reel.reel_name}")
                lines.append("")
                lines.append(f"- **URL**: {reel.permalink}")
                lines.append(f"- **Status**: {reel.status}")

                if reel.instagram_caption:
                    preview = reel.instagram_caption[:100] + "..." if len(reel.instagram_caption) > 100 else reel.instagram_caption
                    lines.append(f"- **Current IG Caption**: {len(reel.instagram_caption)} chars - \"{preview}\"")
                else:
                    lines.append("- **Current IG Caption**: (empty)")

                lines.append("")
                lines.append("**Caption to paste:**")
                lines.append("")
                lines.append("```")
                lines.append(reel.local_caption if reel.local_caption else "(No local caption found)")
                lines.append("```")
                lines.append("")
                lines.append("---")
                lines.append("")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        result.report_path = report_path
        return report_path


# =============================================================================
# CAPTION SYNC CONVENIENCE FUNCTIONS
# =============================================================================


async def sync_profile_captions(
    profile_name: str,
    access_token: str,
    profiles_dir: Optional[Path] = None,
    progress_callback=None,
) -> SyncResult:
    """Convenience function to sync captions for a profile.

    Args:
        profile_name: Name of the profile folder.
        access_token: Instagram Graph API access token.
        profiles_dir: Base profiles directory. Defaults to cwd/profiles.
        progress_callback: Optional callback(current, total, reel_name).

    Returns:
        SyncResult with all reel statuses.

    Example:
        result = await sync_profile_captions(
            "ai.for.mortals",
            access_token=os.getenv("INSTAGRAM_ACCESS_TOKEN"),
        )
        print(f"Empty captions: {result.empty_captions}")
    """
    syncer = CaptionSyncer(
        access_token=access_token,
        profiles_dir=profiles_dir,
    )

    return await syncer.sync_profile(
        profile_name=profile_name,
        progress_callback=progress_callback,
    )


async def sync_all_captions(
    access_token: str,
    profiles_dir: Optional[Path] = None,
    progress_callback=None,
) -> List[SyncResult]:
    """Convenience function to sync captions for all profiles.

    Args:
        access_token: Instagram Graph API access token.
        profiles_dir: Base profiles directory. Defaults to cwd/profiles.
        progress_callback: Optional callback(profile_name, current, total).

    Returns:
        List of SyncResult for each profile.

    Example:
        results = await sync_all_captions(
            access_token=os.getenv("INSTAGRAM_ACCESS_TOKEN"),
        )
        for result in results:
            print(f"{result.profile}: {result.empty_captions} empty")
    """
    syncer = CaptionSyncer(
        access_token=access_token,
        profiles_dir=profiles_dir,
    )

    return await syncer.sync_all_profiles(progress_callback=progress_callback)
