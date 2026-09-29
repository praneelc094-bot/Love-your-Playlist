"""
Unit and Integration Tests for Love Your Playlist (Person A + Person B).
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add project root and model to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from app.auth import get_mock_client, MockSpotifyClient
from app.library import (
    get_user_profile_summary,
    fetch_user_top_tracks,
    fetch_user_top_artists,
    extract_top_genres,
    enrich_tracks_with_shared_format,
    spotify_track_to_shared_format,
)
from app.candidates import (
    build_search_queries,
    generate_candidate_pool,
    MOOD_PRESETS,
)
from app.dummy_ranker import dummy_rank
from app.ranker import generate_recommendations
from app.playlist import (
    create_spotify_playlist,
    add_tracks_to_playlist,
    export_ranked_playlist_to_spotify,
)
from model.profile import build_profile
from model.rank import rank as model_rank


class TestPersonBPipeline(unittest.TestCase):
    def setUp(self):
        self.mock_sp = get_mock_client("Test User")

    def test_mock_auth_and_profile(self):
        profile = get_user_profile_summary(self.mock_sp)
        self.assertEqual(profile["id"], "demo_user_101")
        self.assertEqual(profile["display_name"], "Test User")

    def test_library_fetching_and_conversion(self):
        raw_tracks = fetch_user_top_tracks(self.mock_sp, limit=10)
        self.assertGreater(len(raw_tracks), 0)

        artists = fetch_user_top_artists(self.mock_sp, limit=10)
        self.assertGreater(len(artists), 0)

        genres = extract_top_genres(artists)
        self.assertIsInstance(genres, list)

        # Test conversion to shared format
        shared_tracks = enrich_tracks_with_shared_format(raw_tracks, lyrics_lookup=False)
        self.assertEqual(len(shared_tracks), len(raw_tracks))
        first = shared_tracks[0]
        self.assertIn("spotify_id", first)
        self.assertIn("title", first)
        self.assertIn("artist", first)
        self.assertIn("audio_features", first)
        self.assertIn("audio_features_normalized", first)
        self.assertEqual(len(first["audio_features_normalized"]), 12)

    def test_candidate_generation(self):
        raw_tracks = fetch_user_top_tracks(self.mock_sp, limit=5)
        artists = fetch_user_top_artists(self.mock_sp, limit=5)
        genres = extract_top_genres(artists)
        user_tracks = enrich_tracks_with_shared_format(raw_tracks, lyrics_lookup=False)

        candidates = generate_candidate_pool(
            self.mock_sp,
            user_tracks,
            artists,
            genres,
            target_pool_size=15,
            mood_preset="Chill & Cozy",
            lyrics_lookup=False,
        )
        self.assertGreater(len(candidates), 0)
        user_ids = {t["spotify_id"] for t in user_tracks}
        for cand in candidates:
            self.assertNotIn(cand["spotify_id"], user_ids)

    def test_model_profile_and_ranking(self):
        user_tracks = [
            {
                "spotify_id": f"ut_{i}",
                "title": f"User Song {i}",
                "artist": "User Artist",
                "genre": "pop",
                "lyrics_text": "happy summer sunshine vibes",
                "audio_features": {"energy": 0.8, "danceability": 0.7},
                "audio_features_normalized": np.full(12, 0.7, dtype=np.float32),
                "lyric_embedding": np.ones(384, dtype=np.float32),
                "has_lyrics": True,
            }
            for i in range(5)
        ]

        profile = build_profile(user_tracks)
        self.assertIsNotNone(profile["lyric_embedding"])
        self.assertEqual(len(profile["audio_features_mean"]), 12)
        self.assertEqual(profile["track_count"], 5)

        candidates = [
            {
                "spotify_id": f"cand_{i}",
                "title": f"Cand Song {i}",
                "artist": f"Artist {i % 3}",
                "genre": "pop",
                "lyrics_text": "summer sunshine celebration" if i % 2 == 0 else None,
                "audio_features": {"energy": 0.75, "danceability": 0.7},
                "audio_features_normalized": np.full(12, 0.65 + (i * 0.02), dtype=np.float32),
                "lyric_embedding": np.ones(384, dtype=np.float32) if i % 2 == 0 else None,
                "has_lyrics": i % 2 == 0,
            }
            for i in range(10)
        ]

        ranked = model_rank(profile, candidates, n=5, lyric_weight=0.5, audio_weight=0.5)
        self.assertEqual(len(ranked), 5)
        self.assertIn("score", ranked[0])
        self.assertIn("match_reason", ranked[0])
        self.assertGreaterEqual(ranked[0]["score"], ranked[-1]["score"])

    def test_playlist_creation_and_export(self):
        ranked = [
            {
                "spotify_id": f"t_{i}",
                "title": f"Song {i}",
                "artist": "Artist",
                "score": 0.9,
                "match_reason": "Vibe match",
            }
            for i in range(8)
        ]

        res = export_ranked_playlist_to_spotify(
            self.mock_sp,
            user_id="demo_user_101",
            ranked_tracks=ranked,
            playlist_name="Test Mix",
        )
        self.assertEqual(res["tracks_added"], 8)
        self.assertIn("https://open.spotify.com/playlist/", res["url"])


if __name__ == "__main__":
    unittest.main()
