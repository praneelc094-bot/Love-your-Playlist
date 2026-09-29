"""
Recommendation & Model Bridge for Person B.

Connects the app pipeline to Person A's taste profile builder (model/profile.py)
and ranking model (model/rank.py), allowing dynamic lyric vs. audio vibe weighting.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure model/ is on python path
MODEL_DIR = Path(__file__).parent.parent / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

try:
    from profile import build_profile
    from rank import rank as model_rank
    _HAS_REAL_MODEL = True
except ImportError:
    from app.dummy_ranker import dummy_rank

    def build_profile(user_tracks: list[dict[str, Any]]) -> dict[str, Any]:
        return {"track_count": len(user_tracks), "is_dummy": True}

    def model_rank(profile, candidate_tracks, n=25, **kwargs):
        return dummy_rank(profile, candidate_tracks, n)

    _HAS_REAL_MODEL = False


def generate_recommendations(
    user_tracks: list[dict[str, Any]],
    candidate_tracks: list[dict[str, Any]],
    n: int = 25,
    *,
    lyric_weight: float = 0.5,
    audio_weight: float = 0.5,
    diversity_penalty: float = 0.2,
    use_dummy: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    End-to-end recommendation function:
      1. Builds the user taste profile from user_tracks.
      2. Ranks candidate_tracks against the taste profile.
      3. Returns (profile, ranked_tracks).
    """
    if not user_tracks:
        raise ValueError("Cannot generate recommendations without user tracks.")

    if not candidate_tracks:
        return {}, []

    if use_dummy or not _HAS_REAL_MODEL:
        from app.dummy_ranker import dummy_rank
        profile = build_profile(user_tracks)
        ranked = dummy_rank(profile, candidate_tracks, n=n)
        return profile, ranked

    # 1. Build Taste Profile
    profile = build_profile(user_tracks)

    # 2. Rank Candidates
    ranked_tracks = model_rank(
        profile,
        candidate_tracks,
        n=n,
        lyric_weight=lyric_weight,
        audio_weight=audio_weight,
        diversity_penalty=diversity_penalty,
    )

    return profile, ranked_tracks
