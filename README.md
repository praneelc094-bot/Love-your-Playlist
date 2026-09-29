# Love Your Playlist

An AI model and application that reads a Spotify user's taste — lyric themes, tune/audio features, and listening history — and generates a customized playlist matching their unique vibe.

---

## Architecture & Team Split

- **Person A — Data & Model (`model/`)**:
  - Offline dataset ingestion (`prepare_dataset.py`, `sample_tracks.csv`)
  - Lyrics fetching & caching (`lyrics.py` via LRCLIB)
  - Semantic lyric embeddings with Sentence-Transformers (`embeddings.py`)
  - Audio feature scaling & normalization (`embeddings.py`)
  - Taste profile builder (`profile.py`)
  - Similarity ranking with cosine similarity and diversity penalty (`rank.py`)

- **Person B — Spotify & App (`app/`)**:
  - Spotify OAuth 2.0 Authorization Code Flow & Mock Simulator (`auth.py`)
  - Top tracks & library ingestion pipeline (`library.py`)
  - Multi-query candidate pool generator adapted for 2026 API limits (`candidates.py`)
  - Baseline dummy ranker (`dummy_ranker.py`)
  - Model integration bridge (`ranker.py`)
  - Spotify playlist creator and batch track exporter (`playlist.py`)
  - Interactive Spotify-themed Streamlit Web App (`ui.py`)
  - Headless CLI runner (`cli.py`)

---

## Project Structure

```
├── docs/
│   ├── track-format.md         # Agreed shared track data schema
│   └── model-interface.md      # Agreed model interface contracts
├── model/                      # Person A deliverables
│   ├── lyrics.py
│   ├── prepare_dataset.py
│   ├── check_lyrics_coverage.py
│   ├── fetch_all_lyrics.py
│   ├── embeddings.py
│   ├── build.py
│   ├── profile.py
│   ├── rank.py
│   ├── requirements.txt
│   └── README.md
├── app/                        # Person B deliverables
│   ├── auth.py
│   ├── library.py
│   ├── candidates.py
│   ├── dummy_ranker.py
│   ├── ranker.py
│   ├── playlist.py
│   ├── ui.py
│   ├── cli.py
│   ├── requirements.txt
│   └── README.md
├── .env.example
├── .gitignore
└── README.md
```

---

## How to Run

### 1. Web UI (Streamlit)

```bash
pip install -r app/requirements.txt
streamlit run app/ui.py
```

*You can use Instant Demo Mode (no API key required) or connect your live Spotify account.*

### 2. Command Line (CLI)

```bash
# Run in Instant Demo Mode
python -m app.cli --demo --count 20 --lyric-weight 0.6 --mood "Chill & Cozy" --create-playlist

# Run with Live Spotify Account
python -m app.cli --time-range medium_term --count 25 --lyric-weight 0.5 --create-playlist --name "My Taste Mix"
```
