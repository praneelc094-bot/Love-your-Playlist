# Model interface (what app/ calls into model/)

```python
def build_profile(user_tracks: list[dict]) -> dict:
    """user_tracks: list of tracks in the shared track format (the user's top
    tracks / library). Returns a taste profile — a weighted-average embedding
    plus preferred audio-feature ranges."""

def rank(profile: dict, candidate_tracks: list[dict], n: int) -> list[dict]:
    """Ranks candidate_tracks by similarity to profile, applies a diversity
    penalty, and returns the top n tracks (same shape as candidate_tracks,
    in ranked order)."""
```

Person B can start against a dummy `rank()` that shuffles the candidates, then
swap in the real implementation from `model/` in Phase 3 without touching the
rest of the app.
