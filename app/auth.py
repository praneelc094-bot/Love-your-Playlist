"""
Spotify Authentication Module for Person B.

Handles:
- Spotify OAuth 2.0 Authorization Code Flow via Spotipy
- Token caching and refresh
- Environment variable configuration (.env)
- Realistic Mock Spotify Client for instant testing and demo mode
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Required Spotify scopes for reading top tracks, library, and creating playlists
SPOTIFY_SCOPES = [
    "user-top-read",
    "user-library-read",
    "playlist-modify-public",
    "playlist-modify-private",
    "user-read-private",
    "user-read-email",
]

DEFAULT_REDIRECT_URI = "http://127.0.0.1:8888/callback"
DEFAULT_CACHE_PATH = Path(__file__).parent / ".cache"


def get_spotify_oauth(
    client_id: str | None = None,
    client_secret: str | None = None,
    redirect_uri: str | None = None,
    cache_path: str | Path | None = None,
):
    """
    Creates a SpotifyOAuth object configured with the required scopes.
    Falls back to environment variables SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET,
    and SPOTIPY_REDIRECT_URI if parameters are omitted.
    """
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth
    except ImportError:
        raise ImportError(
            "spotipy is not installed. Run: pip install spotipy"
        )

    cid = client_id or os.getenv("SPOTIPY_CLIENT_ID")
    csecret = client_secret or os.getenv("SPOTIPY_CLIENT_SECRET")
    r_uri = redirect_uri or os.getenv("SPOTIPY_REDIRECT_URI") or DEFAULT_REDIRECT_URI
    c_path = str(cache_path or DEFAULT_CACHE_PATH)

    if not cid or not csecret:
        return None

    return SpotifyOAuth(
        client_id=cid,
        client_secret=csecret,
        redirect_uri=r_uri,
        scope=" ".join(SPOTIFY_SCOPES),
        cache_path=c_path,
        open_browser=False,
    )


def get_spotify_client(
    client_id: str | None = None,
    client_secret: str | None = None,
    redirect_uri: str | None = None,
    auth_code: str | None = None,
):
    """
    Returns an authenticated spotipy.Spotify client if valid cached tokens
    or credentials exist, otherwise None.
    """
    try:
        import spotipy
    except ImportError:
        return None

    sp_oauth = get_spotify_oauth(client_id, client_secret, redirect_uri)
    if not sp_oauth:
        return None

    if auth_code:
        token_info = sp_oauth.get_access_token(auth_code, as_dict=True)
        if token_info and "access_token" in token_info:
            return spotipy.Spotify(auth=token_info["access_token"])

    # Try cached token
    token_info = sp_oauth.validate_token(sp_oauth.cache_handler.get_cached_token())
    if token_info:
        return spotipy.Spotify(auth=token_info["access_token"])

    return None


class MockSpotifyClient:
    """
    High-fidelity Mock Spotify Client for Demo Mode and offline testing.
    Provides realistic Spotify API responses for:
      - current_user()
      - current_user_top_tracks()
      - current_user_top_artists()
      - current_user_saved_tracks()
      - search()
      - user_playlist_create()
      - playlist_add_items()
    """

    def __init__(self, user_name: str = "Alex (Demo Listener)"):
        self.user_name = user_name
        self.user_id = "demo_user_101"
        self._created_playlists: list[dict[str, Any]] = []

    def current_user(self) -> dict[str, Any]:
        return {
            "id": self.user_id,
            "display_name": self.user_name,
            "email": "demo.user@example.com",
            "country": "US",
            "product": "premium",
            "followers": {"total": 42},
            "images": [
                {
                    "url": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=256&q=80",
                    "height": 256,
                    "width": 256,
                }
            ],
        }

    def current_user_top_artists(
        self, limit: int = 20, time_range: str = "medium_term"
    ) -> dict[str, Any]:
        artists = [
            {"id": "a1", "name": "Coldplay", "genres": ["permanent wave", "pop rock", "post-grunge"]},
            {"id": "a2", "name": "Taylor Swift", "genres": ["pop", "country pop"]},
            {"id": "a3", "name": "Arctic Monkeys", "genres": ["indie rock", "garage rock", "modern rock"]},
            {"id": "a4", "name": "The Weeknd", "genres": ["canadian pop", "pop", "r&b"]},
            {"id": "a5", "name": "Billie Eilish", "genres": ["art pop", "electropop", "indie pop"]},
            {"id": "a6", "name": "Dua Lipa", "genres": ["dance pop", "pop", "uk pop"]},
            {"id": "a7", "name": "Tame Impala", "genres": ["australian psych", "neo-psychedelic", "indie pop"]},
            {"id": "a8", "name": "Lana Del Rey", "genres": ["art pop", "sad indie", "dream pop"]},
            {"id": "a9", "name": "Arijit Singh", "genres": ["filmi", "bollywood", "desi pop"]},
            {"id": "a10", "name": "Post Malone", "genres": ["dfw rap", "melodic rap", "pop"]},
        ]
        return {"items": artists[:limit], "total": len(artists)}

    def current_user_top_tracks(
        self, limit: int = 50, time_range: str = "medium_term"
    ) -> dict[str, Any]:
        tracks = [
            {"id": "t1", "name": "Yellow", "artists": [{"name": "Coldplay"}], "album": {"name": "Parachutes", "images": [{"url": "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=300"}]}, "popularity": 88, "duration_ms": 269453},
            {"id": "t2", "name": "Cardigan", "artists": [{"name": "Taylor Swift"}], "album": {"name": "folklore", "images": [{"url": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300"}]}, "popularity": 85, "duration_ms": 239560},
            {"id": "t3", "name": "Do I Wanna Know?", "artists": [{"name": "Arctic Monkeys"}], "album": {"name": "AM", "images": [{"url": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=300"}]}, "popularity": 89, "duration_ms": 272386},
            {"id": "t4", "name": "Blinding Lights", "artists": [{"name": "The Weeknd"}], "album": {"name": "After Hours", "images": [{"url": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=300"}]}, "popularity": 92, "duration_ms": 200040},
            {"id": "t5", "name": "ocean eyes", "artists": [{"name": "Billie Eilish"}], "album": {"name": "dont smile at me", "images": [{"url": "https://images.unsplash.com/photo-1518609878373-06d740f60d8b?w=300"}]}, "popularity": 83, "duration_ms": 200379},
            {"id": "t6", "name": "Levitating", "artists": [{"name": "Dua Lipa"}], "album": {"name": "Future Nostalgia", "images": [{"url": "https://images.unsplash.com/photo-1487180144351-b8472da7d491?w=300"}]}, "popularity": 86, "duration_ms": 203807},
            {"id": "t7", "name": "The Less I Know The Better", "artists": [{"name": "Tame Impala"}], "album": {"name": "Currents", "images": [{"url": "https://images.unsplash.com/photo-1459749411175-04bf5292ceea?w=300"}]}, "popularity": 87, "duration_ms": 216320},
            {"id": "t8", "name": "Video Games", "artists": [{"name": "Lana Del Rey"}], "album": {"name": "Born to Die", "images": [{"url": "https://images.unsplash.com/photo-1501386761578-eac5c94b800a?w=300"}]}, "popularity": 80, "duration_ms": 282000},
            {"id": "t9", "name": "Tum Hi Ho", "artists": [{"name": "Arijit Singh"}], "album": {"name": "Aashiqui 2", "images": [{"url": "https://images.unsplash.com/photo-1526478806334-5fd488fcaabc?w=300"}]}, "popularity": 82, "duration_ms": 262000},
            {"id": "t10", "name": "Circles", "artists": [{"name": "Post Malone"}], "album": {"name": "Hollywood's Bleeding", "images": [{"url": "https://images.unsplash.com/photo-1445985543470-41fdd5c31447?w=300"}]}, "popularity": 88, "duration_ms": 215280},
            {"id": "t11", "name": "Viva La Vida", "artists": [{"name": "Coldplay"}], "album": {"name": "Viva La Vida", "images": [{"url": "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=300"}]}, "popularity": 89, "duration_ms": 242000},
            {"id": "t12", "name": "Cruel Summer", "artists": [{"name": "Taylor Swift"}], "album": {"name": "Lover", "images": [{"url": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300"}]}, "popularity": 93, "duration_ms": 178000},
            {"id": "t13", "name": "505", "artists": [{"name": "Arctic Monkeys"}], "album": {"name": "Favourite Worst Nightmare", "images": [{"url": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=300"}]}, "popularity": 88, "duration_ms": 253000},
            {"id": "t14", "name": "Save Your Tears", "artists": [{"name": "The Weeknd"}], "album": {"name": "After Hours", "images": [{"url": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=300"}]}, "popularity": 90, "duration_ms": 215000},
            {"id": "t15", "name": "bad guy", "artists": [{"name": "Billie Eilish"}], "album": {"name": "WHEN WE ALL FALL ASLEEP", "images": [{"url": "https://images.unsplash.com/photo-1518609878373-06d740f60d8b?w=300"}]}, "popularity": 87, "duration_ms": 194000},
        ]
        return {"items": tracks[:limit], "total": len(tracks)}

    def current_user_saved_tracks(self, limit: int = 50) -> dict[str, Any]:
        top = self.current_user_top_tracks(limit=limit)
        items = [{"track": t} for t in top["items"]]
        return {"items": items, "total": len(items)}

    def search(
        self, q: str, type: str = "track", limit: int = 10
    ) -> dict[str, Any]:
        candidates_db = [
            {"id": "cand_1", "name": "Fix You", "artists": [{"name": "Coldplay"}], "album": {"name": "X&Y", "images": [{"url": "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=300"}]}},
            {"id": "cand_2", "name": "August", "artists": [{"name": "Taylor Swift"}], "album": {"name": "folklore", "images": [{"url": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300"}]}},
            {"id": "cand_3", "name": "Starboy", "artists": [{"name": "The Weeknd"}], "album": {"name": "Starboy", "images": [{"url": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=300"}]}},
            {"id": "cand_4", "name": "Let It Happen", "artists": [{"name": "Tame Impala"}], "album": {"name": "Currents", "images": [{"url": "https://images.unsplash.com/photo-1459749411175-04bf5292ceea?w=300"}]}},
            {"id": "cand_5", "name": "Summertime Sadness", "artists": [{"name": "Lana Del Rey"}], "album": {"name": "Born to Die", "images": [{"url": "https://images.unsplash.com/photo-1501386761578-eac5c94b800a?w=300"}]}},
            {"id": "cand_6", "name": "R U Mine?", "artists": [{"name": "Arctic Monkeys"}], "album": {"name": "AM", "images": [{"url": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=300"}]}},
            {"id": "cand_7", "name": "Everything I Wanted", "artists": [{"name": "Billie Eilish"}], "album": {"name": "Single", "images": [{"url": "https://images.unsplash.com/photo-1518609878373-06d740f60d8b?w=300"}]}},
            {"id": "cand_8", "name": "Don't Start Now", "artists": [{"name": "Dua Lipa"}], "album": {"name": "Future Nostalgia", "images": [{"url": "https://images.unsplash.com/photo-1487180144351-b8472da7d491?w=300"}]}},
            {"id": "cand_9", "name": "Kesariya", "artists": [{"name": "Arijit Singh"}], "album": {"name": "Brahmastra", "images": [{"url": "https://images.unsplash.com/photo-1526478806334-5fd488fcaabc?w=300"}]}},
            {"id": "cand_10", "name": "Sunflower", "artists": [{"name": "Post Malone"}], "album": {"name": "Spider-Man Soundtrack", "images": [{"url": "https://images.unsplash.com/photo-1445985543470-41fdd5c31447?w=300"}]}},
            {"id": "cand_11", "name": "Midnight City", "artists": [{"name": "M83"}], "album": {"name": "Hurry Up, We're Dreaming", "images": [{"url": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=300"}]}},
            {"id": "cand_12", "name": "Somebody Else", "artists": [{"name": "The 1975"}], "album": {"name": "I like it when you sleep", "images": [{"url": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300"}]}},
            {"id": "cand_13", "name": "Sweater Weather", "artists": [{"name": "The Neighbourhood"}], "album": {"name": "I Love You.", "images": [{"url": "https://images.unsplash.com/photo-1518609878373-06d740f60d8b?w=300"}]}},
            {"id": "cand_14", "name": "Heat Waves", "artists": [{"name": "Glass Animals"}], "album": {"name": "Dreamland", "images": [{"url": "https://images.unsplash.com/photo-1459749411175-04bf5292ceea?w=300"}]}},
            {"id": "cand_15", "name": "As It Was", "artists": [{"name": "Harry Styles"}], "album": {"name": "Harry's House", "images": [{"url": "https://images.unsplash.com/photo-1501386761578-eac5c94b800a?w=300"}]}},
            {"id": "cand_16", "name": "Style", "artists": [{"name": "Taylor Swift"}], "album": {"name": "1989", "images": [{"url": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300"}]}},
            {"id": "cand_17", "name": "Fluorescent Adolescent", "artists": [{"name": "Arctic Monkeys"}], "album": {"name": "Favourite Worst Nightmare", "images": [{"url": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=300"}]}},
            {"id": "cand_18", "name": "In Your Eyes", "artists": [{"name": "The Weeknd"}], "album": {"name": "After Hours", "images": [{"url": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=300"}]}},
            {"id": "cand_19", "name": "Happier Than Ever", "artists": [{"name": "Billie Eilish"}], "album": {"name": "Happier Than Ever", "images": [{"url": "https://images.unsplash.com/photo-1518609878373-06d740f60d8b?w=300"}]}},
            {"id": "cand_20", "name": "Sparks", "artists": [{"name": "Coldplay"}], "album": {"name": "Parachutes", "images": [{"url": "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=300"}]}},
        ]
        q_lower = q.lower()
        matched = [
            t
            for t in candidates_db
            if any(word in t["name"].lower() or word in t["artists"][0]["name"].lower() for word in q_lower.split())
        ]
        if not matched:
            matched = random.sample(candidates_db, min(limit, len(candidates_db)))

        return {"tracks": {"items": matched[:limit], "total": len(matched)}}

    def user_playlist_create(
        self,
        user: str,
        name: str,
        public: bool = True,
        collaborative: bool = False,
        description: str = "",
    ) -> dict[str, Any]:
        playlist_id = f"demo_pl_{len(self._created_playlists) + 1}_{random.randint(1000, 9999)}"
        playlist_obj = {
            "id": playlist_id,
            "name": name,
            "public": public,
            "collaborative": collaborative,
            "description": description,
            "owner": {"id": user, "display_name": self.user_name},
            "external_urls": {"spotify": f"https://open.spotify.com/playlist/{playlist_id}"},
            "tracks": {"total": 0, "items": []},
            "uri": f"spotify:playlist:{playlist_id}",
        }
        self._created_playlists.append(playlist_obj)
        return playlist_obj

    def playlist_add_items(
        self, playlist_id: str, items: list[str], position: int | None = None
    ) -> dict[str, Any]:
        for pl in self._created_playlists:
            if pl["id"] == playlist_id:
                pl["tracks"]["total"] += len(items)
                return {"snapshot_id": f"snapshot_{random.randint(100, 999)}"}
        return {"snapshot_id": "snapshot_demo"}

    def user_playlists(self, user: str, limit: int = 50) -> dict[str, Any]:
        return {"items": self._created_playlists[:limit], "total": len(self._created_playlists)}


def get_mock_client(user_name: str = "Demo User") -> MockSpotifyClient:
    """Returns a ready-to-use MockSpotifyClient instance."""
    return MockSpotifyClient(user_name=user_name)
