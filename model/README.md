# model/ — Person A: Data & Model

## Phase 1 goal
A script that turns a list of tracks into the shared track format
(`../docs/track-format.md`), running entirely offline from Spotify.

## Layout
```
data/                 gitignored — datasets and caches live here
lyrics.py             fetch_lyrics(artist, title) -> str | None, with local caching
audio_features.py     get_audio_features(track) -> dict
build.py              build_track(spotify_id, title, artist) -> dict (shared format)
embeddings.py         lyric embeddings + audio-feature normalization
profile.py            build_profile(user_tracks) -> profile
rank.py                rank(profile, candidate_tracks, n) -> ranked tracks
requirements.txt
```

## Step-by-step (Week 1)
1. Set up venv + `requirements.txt` (pandas, requests, sentence-transformers,
   librosa, scikit-learn, tqdm).
2. Download a Kaggle Spotify tracks dataset into `data/`, sample 5k-20k tracks.
3. Build `lyrics.py` against LRCLIB, with a local cache (SQLite/JSON) and rate
   limiting. Measure lyrics coverage on the sample.
4. Stub `audio_features.py` — for dataset tracks, read the existing columns; for
   live tracks (later), wrap ReccoBeats or librosa behind the same function
   signature.
5. Write `build.py` to assemble the shared track format, embed lyrics with
   `sentence-transformers`, normalize audio features, save the scaler.
6. Sanity check: nearest neighbours of a few known tracks should make sense.
   Export a 50-track sample in the shared format for Person B.

See the full plan discussed in chat for day-by-day detail.
