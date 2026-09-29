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

# Required Spotify scopes for reading top tracks, library, playlists, and creating playlists
SPOTIFY_SCOPES = [
    "user-top-read",
    "user-library-read",
    "playlist-read-private",
    "playlist-read-collaborative",
    "user-read-recently-played",
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
            {"id": "4gzpq5YKM9GaeEk69fC6xw", "name": "Coldplay", "genres": ["permanent wave", "pop rock", "post-grunge"]},
            {"id": "06HL4z0CvFAxyc27GXpf02", "name": "Taylor Swift", "genres": ["pop", "country pop"]},
            {"id": "7Ln80taz6rY9g5JuqkiDCY", "name": "Arctic Monkeys", "genres": ["indie rock", "garage rock", "modern rock"]},
            {"id": "1Xyo4u8uXC1ZmMpatF05PJ", "name": "The Weeknd", "genres": ["canadian pop", "pop", "r&b"]},
            {"id": "6qqNVTkY8uBg9cP3Jd7DAH", "name": "Billie Eilish", "genres": ["art pop", "electropop", "indie pop"]},
            {"id": "6M2wZ9GZgrQXHCFfjv46we", "name": "Dua Lipa", "genres": ["dance pop", "pop", "uk pop"]},
            {"id": "5INjqkS1o8h1gGI4G294dG", "name": "Tame Impala", "genres": ["australian psych", "neo-psychedelic", "indie pop"]},
            {"id": "00FQb4jTyendwdQGpyWDbr", "name": "Lana Del Rey", "genres": ["art pop", "sad indie", "dream pop"]},
            {"id": "4YRxDV8wJFhgIMRBXxeWOP", "name": "Arijit Singh", "genres": ["filmi", "bollywood", "desi pop"]},
            {"id": "246dkjvS1zLTtiykYqagQB", "name": "Post Malone", "genres": ["dfw rap", "melodic rap", "pop"]},
        ]
        return {"items": artists[:limit], "total": len(artists)}

    def current_user_top_tracks(
        self, limit: int = 50, time_range: str = "medium_term"
    ) -> dict[str, Any]:
        tracks = [
            {"id": "3AJwUDP919kvQ9QcozQPxg", "name": "Yellow", "artists": [{"name": "Coldplay"}], "album": {"name": "Parachutes", "images": [{"url": "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=300"}]}, "popularity": 88, "duration_ms": 269453},
            {"id": "4R2kfaDFwwAvFmV4llofDX", "name": "Cardigan", "artists": [{"name": "Taylor Swift"}], "album": {"name": "folklore", "images": [{"url": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300"}]}, "popularity": 85, "duration_ms": 239560},
            {"id": "5FVd6KXrgO9B3JPmC8OPst", "name": "Do I Wanna Know?", "artists": [{"name": "Arctic Monkeys"}], "album": {"name": "AM", "images": [{"url": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=300"}]}, "popularity": 89, "duration_ms": 272386},
            {"id": "0VjIjW4GlUZAMYd2vXMi3b", "name": "Blinding Lights", "artists": [{"name": "The Weeknd"}], "album": {"name": "After Hours", "images": [{"url": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=300"}]}, "popularity": 92, "duration_ms": 200040},
            {"id": "7hDVYcQq6MxkdWweuCtlZq", "name": "ocean eyes", "artists": [{"name": "Billie Eilish"}], "album": {"name": "dont smile at me", "images": [{"url": "https://images.unsplash.com/photo-1518609878373-06d740f60d8b?w=300"}]}, "popularity": 83, "duration_ms": 200379},
            {"id": "463CkQjx2Zk1yXoBuierM9", "name": "Levitating", "artists": [{"name": "Dua Lipa"}], "album": {"name": "Future Nostalgia", "images": [{"url": "https://images.unsplash.com/photo-1487180144351-b8472da7d491?w=300"}]}, "popularity": 86, "duration_ms": 203807},
            {"id": "6K4t31amVTZDgR3sKmwUJJ", "name": "The Less I Know The Better", "artists": [{"name": "Tame Impala"}], "album": {"name": "Currents", "images": [{"url": "https://images.unsplash.com/photo-1459749411175-04bf5292ceea?w=300"}]}, "popularity": 87, "duration_ms": 216320},
            {"id": "2NYGgN617d5n347sZ66236", "name": "Video Games", "artists": [{"name": "Lana Del Rey"}], "album": {"name": "Born to Die", "images": [{"url": "https://images.unsplash.com/photo-1501386761578-eac5c94b800a?w=300"}]}, "popularity": 80, "duration_ms": 282000},
            {"id": "56zZ48jdyY2oDXHVRY2K5g", "name": "Tum Hi Ho", "artists": [{"name": "Arijit Singh"}], "album": {"name": "Aashiqui 2", "images": [{"url": "https://images.unsplash.com/photo-1526478806334-5fd488fcaabc?w=300"}]}, "popularity": 82, "duration_ms": 262000},
            {"id": "21jGcNKet2qwijlDFuPiPb", "name": "Circles", "artists": [{"name": "Post Malone"}], "album": {"name": "Hollywood's Bleeding", "images": [{"url": "https://images.unsplash.com/photo-1445985543470-41fdd5c31447?w=300"}]}, "popularity": 88, "duration_ms": 215280},
            {"id": "1mea3bSkSGXuIRvnydlB5b", "name": "Viva La Vida", "artists": [{"name": "Coldplay"}], "album": {"name": "Viva La Vida", "images": [{"url": "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=300"}]}, "popularity": 89, "duration_ms": 242000},
            {"id": "1BxfuPKGuaTgP7aM0XbdMe", "name": "Cruel Summer", "artists": [{"name": "Taylor Swift"}], "album": {"name": "Lover", "images": [{"url": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300"}]}, "popularity": 93, "duration_ms": 178000},
            {"id": "0BxE48rCVxDnnRwwD0eAaD", "name": "505", "artists": [{"name": "Arctic Monkeys"}], "album": {"name": "Favourite Worst Nightmare", "images": [{"url": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=300"}]}, "popularity": 88, "duration_ms": 253000},
            {"id": "5QO79kh1waicV47BqGRIO3", "name": "Save Your Tears", "artists": [{"name": "The Weeknd"}], "album": {"name": "After Hours", "images": [{"url": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=300"}]}, "popularity": 90, "duration_ms": 215000},
            {"id": "2Fxmhks0bxGSBdJ92vM42m", "name": "bad guy", "artists": [{"name": "Billie Eilish"}], "album": {"name": "WHEN WE ALL FALL ASLEEP", "images": [{"url": "https://images.unsplash.com/photo-1518609878373-06d740f60d8b?w=300"}]}, "popularity": 87, "duration_ms": 194000},
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
            {"id": "7LVHVU3tWfcxj5aiPFEW4Q", "name": "Fix You", "artists": [{"name": "Coldplay"}], "album": {"name": "X&Y", "images": [{"url": "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=300"}]}},
            {"id": "3EdSembr1bA6eOCu999YpM", "name": "August", "artists": [{"name": "Taylor Swift"}], "album": {"name": "folklore", "images": [{"url": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300"}]}},
            {"id": "7MXVkk9YM5IZxh0VU69um0", "name": "Starboy", "artists": [{"name": "The Weeknd"}], "album": {"name": "Starboy", "images": [{"url": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=300"}]}},
            {"id": "2X485T9Z5Ly0xyaghY73bv", "name": "Let It Happen", "artists": [{"name": "Tame Impala"}], "album": {"name": "Currents", "images": [{"url": "https://images.unsplash.com/photo-1459749411175-04bf5292ceea?w=300"}]}},
            {"id": "2dBwB6KUHB76GYanBk79bS", "name": "Summertime Sadness", "artists": [{"name": "Lana Del Rey"}], "album": {"name": "Born to Die", "images": [{"url": "https://images.unsplash.com/photo-1501386761578-eac5c94b800a?w=300"}]}},
            {"id": "2AT8i7KDTpe0pPieSnLrg2", "name": "R U Mine?", "artists": [{"name": "Arctic Monkeys"}], "album": {"name": "AM", "images": [{"url": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=300"}]}},
            {"id": "3ZCTVFBt2Br03R4RqlxExQ", "name": "Everything I Wanted", "artists": [{"name": "Billie Eilish"}], "album": {"name": "Single", "images": [{"url": "https://images.unsplash.com/photo-1518609878373-06d740f60d8b?w=300"}]}},
            {"id": "3PfIrDoz19wz7qK7tYeu62", "name": "Don't Start Now", "artists": [{"name": "Dua Lipa"}], "album": {"name": "Future Nostalgia", "images": [{"url": "https://images.unsplash.com/photo-1487180144351-b8472da7d491?w=300"}]}},
            {"id": "6AQbmUe0Qwf5PZvg4hm5ez", "name": "Kesariya", "artists": [{"name": "Arijit Singh"}], "album": {"name": "Brahmastra", "images": [{"url": "https://images.unsplash.com/photo-1526478806334-5fd488fcaabc?w=300"}]}},
            {"id": "3KkXRkHbMCARz0aVfEt68P", "name": "Sunflower", "artists": [{"name": "Post Malone"}], "album": {"name": "Spider-Man Soundtrack", "images": [{"url": "https://images.unsplash.com/photo-1445985543470-41fdd5c31447?w=300"}]}},
            {"id": "6GyFP1nfCDB8lbD2bG0Hq9", "name": "Midnight City", "artists": [{"name": "M83"}], "album": {"name": "Hurry Up, We're Dreaming", "images": [{"url": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=300"}]}},
            {"id": "4m0q0xQ2R518296k1xT17Q", "name": "Somebody Else", "artists": [{"name": "The 1975"}], "album": {"name": "I like it when you sleep", "images": [{"url": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300"}]}},
            {"id": "2QjOHCTQ1Jl3zawyva09GQ", "name": "Sweater Weather", "artists": [{"name": "The Neighbourhood"}], "album": {"name": "I Love You.", "images": [{"url": "https://images.unsplash.com/photo-1518609878373-06d740f60d8b?w=300"}]}},
            {"id": "02MWAaffLxlfxAUY7c5dvx", "name": "Heat Waves", "artists": [{"name": "Glass Animals"}], "album": {"name": "Dreamland", "images": [{"url": "https://images.unsplash.com/photo-1459749411175-04bf5292ceea?w=300"}]}},
            {"id": "4LRPiXqCikLlN15c3yZAea", "name": "As It Was", "artists": [{"name": "Harry Styles"}], "album": {"name": "Harry's House", "images": [{"url": "https://images.unsplash.com/photo-1501386761578-eac5c94b800a?w=300"}]}},
            {"id": "0ug5KNajqBz3urNdteOmRa", "name": "Style", "artists": [{"name": "Taylor Swift"}], "album": {"name": "1989", "images": [{"url": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300"}]}},
            {"id": "2HfSV0sR2M4fFw6s3b192n", "name": "Fluorescent Adolescent", "artists": [{"name": "Arctic Monkeys"}], "album": {"name": "Favourite Worst Nightmare", "images": [{"url": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=300"}]}},
            {"id": "7szuecWvxnjufLfdvdo9dC", "name": "In Your Eyes", "artists": [{"name": "The Weeknd"}], "album": {"name": "After Hours", "images": [{"url": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=300"}]}},
            {"id": "4RVwu0g322qgAzOiTWJFDU", "name": "Happier Than Ever", "artists": [{"name": "Billie Eilish"}], "album": {"name": "Happier Than Ever", "images": [{"url": "https://images.unsplash.com/photo-1518609878373-06d740f60d8b?w=300"}]}},
            {"id": "7D0RhFdozyUMYe5qAmvPgQ", "name": "Sparks", "artists": [{"name": "Coldplay"}], "album": {"name": "Parachutes", "images": [{"url": "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=300"}]}},
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

    def current_user_playlists(self, limit: int = 50) -> dict[str, Any]:
        dummy_pl = [
            {"id": "demo_favs_1", "name": "My Favorites", "tracks": {"total": 15}}
        ] + self._created_playlists
        return {"items": dummy_pl[:limit], "total": len(dummy_pl)}

    def playlist_tracks(self, playlist_id: str, limit: int = 50) -> dict[str, Any]:
        top = self.current_user_top_tracks(limit=limit)
        items = [{"track": t} for t in top["items"]]
        return {"items": items, "total": len(items)}

    def current_user_recently_played(self, limit: int = 50) -> dict[str, Any]:
        top = self.current_user_top_tracks(limit=limit)
        items = [{"track": t} for t in top["items"]]
        return {"items": items, "total": len(items)}


def get_mock_client(user_name: str = "Demo User") -> MockSpotifyClient:
    """Returns a ready-to-use MockSpotifyClient instance."""
    return MockSpotifyClient(user_name=user_name)
