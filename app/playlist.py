"""
Spotify Playlist Management Module for Person B.

Handles:
- Creating playlists via Spotify Web API (POST /v1/users/{user_id}/playlists)
- Populating playlists with ranked track URIs (batched in chunks of 100)
- Generating informative playlist descriptions based on taste profile weights
"""

from __future__ import annotations

from typing import Any


def create_spotify_playlist(
    sp,
    user_id: str,
    name: str = "Love Your Playlist Mix",
    description: str = "Personalized playlist curated by Love Your Playlist AI based on your lyrics and tune taste.",
    public: bool = True,
) -> dict[str, Any]:
    """
    Creates a new Spotify playlist on the user's account.
    Returns the created playlist dictionary.
    """
    try:
        playlist = sp.user_playlist_create(
            user=user_id,
            name=name,
            public=public,
            description=description,
        )
        return playlist
    except Exception as e:
        raise RuntimeError(f"Failed to create Spotify playlist: {e}")


def add_tracks_to_playlist(
    sp,
    playlist_id: str,
    track_ids_or_uris: list[str],
    chunk_size: int = 100,
) -> int:
    """
    Adds a list of track IDs or URIs to the specified playlist in batches.
    Spotify API limits track additions to 100 items per call.
    Returns the total number of tracks successfully added.
    """
    if not track_ids_or_uris:
        return 0

    formatted_uris: list[str] = []
    for item in track_ids_or_uris:
        if not item:
            continue
        if item.startswith("spotify:track:"):
            formatted_uris.append(item)
        else:
            formatted_uris.append(f"spotify:track:{item}")

    added_count = 0
    for i in range(0, len(formatted_uris), chunk_size):
        chunk = formatted_uris[i : i + chunk_size]
        try:
            sp.playlist_add_items(playlist_id=playlist_id, items=chunk)
            added_count += len(chunk)
        except Exception as e:
            print(f"Error adding tracks chunk ({i}-{i+len(chunk)}): {e}")

    return added_count


def export_ranked_playlist_to_spotify(
    sp,
    user_id: str,
    ranked_tracks: list[dict[str, Any]],
    playlist_name: str = "Love Your Playlist Mix",
    description: str | None = None,
    public: bool = True,
    lyric_weight: float = 0.5,
    audio_weight: float = 0.5,
) -> dict[str, Any]:
    """
    Complete flow to create a playlist and add ranked tracks.
    Returns summary dict with playlist_id, name, url, uri, and tracks_count.
    """
    if not ranked_tracks:
        raise ValueError("Cannot export an empty ranked track list.")

    if not description:
        lyric_pct = int(lyric_weight * 100)
        audio_pct = int(audio_weight * 100)
        description = (
            f"Crafted by Love Your Playlist AI • "
            f"Taste Blend: {lyric_pct}% Lyric Themes, {audio_pct}% Audio Vibes • "
            f"{len(ranked_tracks)} songs"
        )

    # 1. Create Playlist
    playlist = create_spotify_playlist(
        sp,
        user_id=user_id,
        name=playlist_name,
        description=description,
        public=public,
    )
    playlist_id = playlist["id"]
    external_url = playlist.get("external_urls", {}).get("spotify", f"https://open.spotify.com/playlist/{playlist_id}")

    # 2. Extract valid Spotify track IDs
    track_ids = [
        t["spotify_id"]
        for t in ranked_tracks
        if t.get("spotify_id") and str(t["spotify_id"]).strip()
    ]

    # 3. Add tracks in batches
    added_count = add_tracks_to_playlist(sp, playlist_id, track_ids)

    return {
        "id": playlist_id,
        "name": playlist_name,
        "url": external_url,
        "uri": playlist.get("uri", f"spotify:playlist:{playlist_id}"),
        "tracks_added": added_count,
        "description": description,
    }
