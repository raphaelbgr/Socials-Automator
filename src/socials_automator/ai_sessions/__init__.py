"""AI Sessions module for profile-scoped conversation context.

Provides persistent AI conversation memory per profile, enabling:
- Multi-turn conversation context across reel/post generations
- Learning from feedback (quality tracking)
- Constraint generation from history (avoid repetition)
- Provider performance tracking per profile

Usage:
    from socials_automator.ai_sessions import ProfileAISession, ScriptSession

    # Create session for a profile
    session = ScriptSession(profile_path=Path("profiles/ai.for.mortals"))

    # Generate with accumulated context
    script = await session.generate_script(
        topic="ChatGPT productivity",
        target_duration=60,
    )

    # Record feedback after generation
    session.add_feedback(quality="accepted", notes="Good hook variety")
"""

from .profile_session import ProfileAISession
from .script_session import ScriptSession
from .storage import SessionStorage

__all__ = [
    "ProfileAISession",
    "ScriptSession",
    "SessionStorage",
]
