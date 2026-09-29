"""
Command-Line Interface for Love Your Playlist (Person B).

Enables running the full pipeline from terminal:
    python -m app.cli --demo --count 20 --lyric-weight 0.6 --mood "Chill & Cozy"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Safe console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODEL_DIR = PROJECT_ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from app.auth import get_mock_client, get_spotify_client
from app.candidates import generate_candidate_pool
from app.library import (
    extract_top_genres,
    fetch_user_top_artists,
    fetch_user_top_tracks,
    get_user_profile_summary,
    enrich_tracks_with_shared_format,
)
from app.playlist import export_ranked_playlist_to_spotify
from app.ranker import generate_recommendations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Love Your Playlist - AI Taste & Lyrics Playlist Generator"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in Demo Mode with simulated Spotify data (no API keys required)",
    )
    parser.add_argument(
        "--time-range",
        choices=["short_term", "medium_term", "long_term"],
        default="medium_term",
        help="Listening history time range to analyze (default: medium_term)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of recommended tracks to generate (default: 20)",
    )
    parser.add_argument(
        "--lyric-weight",
        type=float,
        default=0.5,
        help="Weight for lyric semantic matching [0.0 - 1.0] (default: 0.5)",
    )
    parser.add_argument(
        "--mood",
        type=str,
        default="Current Taste (Default)",
        help="Mood preset (e.g. 'Chill & Cozy', 'High Energy Workout', 'Late Night Drive')",
    )
    parser.add_argument(
        "--create-playlist",
        action="store_true",
        help="Automatically create and populate the playlist in Spotify",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Love Your Playlist Mix",
        help="Name of the playlist to create",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🎧 LOVE YOUR PLAYLIST — AI PLAYLIST GENERATOR")
    print("=" * 60)

    # 1. Initialize Client
    if args.demo:
        print("🟢 Running in DEMO Mode (simulated user)")
        sp = get_mock_client("Demo Listener")
    else:
        print("🔐 Initializing Live Spotify Authentication...")
        sp = get_spotify_client()
        if not sp:
            print("❌ No valid Spotify credentials found. Please set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET, or run with --demo.")
            sys.exit(1)

    profile = get_user_profile_summary(sp)
    print(f"👤 Connected User: {profile['display_name']} ({profile['id']})")

    # 2. Ingest Taste History
    print(f"\n📥 Fetching listening history ({args.time_range})...")
    raw_tracks = fetch_user_top_tracks(sp, time_range=args.time_range, limit=50)
    top_artists = fetch_user_top_artists(sp, time_range=args.time_range, limit=20)
    top_genres = extract_top_genres(top_artists)

    user_tracks = enrich_tracks_with_shared_format(raw_tracks, lyrics_lookup=True)
    print(f"✅ Analyzed {len(user_tracks)} top tracks across {len(top_genres)} genres.")
    if top_genres:
        print(f"   Top Genres: {', '.join([g[0] for g in top_genres[:5]])}")

    # 3. Generate Candidates
    print(f"\n🔍 Searching candidate pool (Mood: {args.mood})...")
    candidates = generate_candidate_pool(
        sp,
        user_tracks,
        top_artists,
        top_genres,
        target_pool_size=100,
        mood_preset=args.mood,
        lyrics_lookup=True,
    )
    print(f"✅ Found {len(candidates)} fresh candidate tracks.")

    # 4. Rank with Model
    print(f"\n🧠 Ranking candidates (Lyrics: {args.lyric_weight:.0%}, Audio: {1.0 - args.lyric_weight:.0%})...")
    taste_profile, ranked = generate_recommendations(
        user_tracks,
        candidates,
        n=args.count,
        lyric_weight=args.lyric_weight,
        audio_weight=1.0 - args.lyric_weight,
    )

    print("\n" + "=" * 60)
    print(f"🎶 TOP {len(ranked)} RECOMMENDED TRACKS")
    print("=" * 60)
    for idx, t in enumerate(ranked, 1):
        score_pct = f"{t.get('score', 0):.0%}"
        print(f"{idx:2d}. {t['title']} — {t['artist']}")
        print(f"    Match: {score_pct} | Reason: {t.get('match_reason', '')}")
    print("=" * 60)

    # 5. Optional Playlist Creation
    if args.create_playlist:
        print(f"\n🚀 Creating Spotify Playlist: '{args.name}'...")
        res = export_ranked_playlist_to_spotify(
            sp,
            user_id=profile["id"],
            ranked_tracks=ranked,
            playlist_name=args.name,
            lyric_weight=args.lyric_weight,
            audio_weight=1.0 - args.lyric_weight,
        )
        print(f"🎉 Playlist Created Successfully!")
        print(f"   URL: {res['url']}")
        print(f"   Tracks Added: {res['tracks_added']}")


if __name__ == "__main__":
    main()
