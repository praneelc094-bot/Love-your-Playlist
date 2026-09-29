# `app/` — Person B: Spotify & App

This directory contains Person B's complete implementation for the **Love Your Playlist** project:
- Spotify OAuth 2.0 Authorization Flow & Instant Demo Simulator
- User Top Tracks & Library Ingestion pipeline
- Search-based Candidate Pool Generator (adapted for Spotify's 2026 API changes)
- Model Bridge & Dynamic Recommendation Ranker
- Spotify Playlist Creation and Batch Track Importer
- Modern Interactive Streamlit Web UI & CLI

---

## Structure

```
app/
├── auth.py             # Spotify OAuth authorization code flow & MockSpotifyClient
├── library.py          # Top tracks, artists, genres & shared track format converter
├── candidates.py       # 2026 API-compliant multi-angle search candidate pool generator
├── dummy_ranker.py     # Baseline ranker conforming to docs/model-interface.md
├── ranker.py           # Seamless bridge to Person A's model (profile.py & rank.py)
├── playlist.py         # POST /v1/users/{id}/playlists & batch track adding
├── ui.py               # Full-featured Spotify-themed Streamlit Web App
├── cli.py              # Terminal CLI runner for headless execution
├── requirements.txt    # Dependencies for the app
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r app/requirements.txt
```

### 2. Run the Interactive Web UI

You can start the app immediately in **Instant Demo Mode** (no Spotify developer credentials needed) or with your own credentials:

```bash
streamlit run app/ui.py
```

### 3. Run from Command Line (CLI)

```bash
# Run in Instant Demo Mode
python -m app.cli --demo --count 20 --lyric-weight 0.6 --mood "Chill & Cozy"

# Run with real Spotify account and create a playlist
python -m app.cli --count 25 --lyric-weight 0.5 --create-playlist --name "My Taste Mix"
```

---

## Spotify Developer App Setup (For Live Mode)

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) and log in.
2. Click **Create an App**.
3. Set the **Redirect URI** to:
   ```
   http://127.0.0.1:8888/callback
   ```
4. Copy your **Client ID** and **Client Secret**.
5. Create a `.env` file in the project root:
   ```ini
   SPOTIPY_CLIENT_ID=your_client_id_here
   SPOTIPY_CLIENT_SECRET=your_client_secret_here
   SPOTIPY_REDIRECT_URI=http://127.0.0.1:8888/callback
   ```

---

## Addressing Spotify's 2026 API Restrictions

As of February 2026:
- Spotify removed the `/v1/recommendations`, Audio Features, and Related Artists endpoints for new applications.
- Spotify search `/v1/search` is capped at 10 results per query.

**How Person B solves this:**
- `candidates.py` runs dozens of targeted search queries across the user's top artists, top genres, mood presets, and release year windows.
- De-duplicates candidate tracks and filters out songs the user has already listened to.
- Uses `model/lyrics.py` (LRCLIB) for lyrics semantics and `model/embeddings.py` for feature normalization and embeddings.
