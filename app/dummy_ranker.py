"""
Dummy Ranker for Phase 1 App Development (Person B).

Allows Person B to build and test the full Spotify pipeline and UI
independently of Person A's model training/feature-scaling progress.
"""

from __future__ import annotations

import random
from typing import Any


def dummy_rank(
    profile: dict[str, Any] | None,
    candidate_tracks: list[dict[str, Any]],
    n: int = 25,
) -> list[dict[str, Any]]:
    """
    Shuffles candidates and returns top n tracks with placeholder scores.
    Conforms to the shared model interface in docs/model-interface.md.
    """
    if not candidate_tracks:
        return []

    shuffled = list(candidate_tracks)
    random.seed(42)
    random.shuffle(shuffled)

    results = []
    for i, track in enumerate(shuffled[:n]):
        t_copy = dict(track)
        t_copy["score"] = round(1.0 - (i * 0.03), 2)
        t_copy["lyric_similarity"] = round(0.5 + (random.random() * 0.4), 2)
        t_copy["audio_similarity"] = round(0.5 + (random.random() * 0.4), 2)
        t_copy["match_reason"] = "Dummy baseline recommendation"
        results.append(t_copy)

    return results
