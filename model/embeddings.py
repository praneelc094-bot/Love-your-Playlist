"""
Turns raw track data into the numeric pieces the model needs:
  - a lyric embedding (semantic meaning of the words)
  - a normalized audio-feature vector (tempo, energy, valence, ...)

Both pieces get concatenated later (in build.py) into one track vector.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = Path(__file__).parent / "data"
SCALER_PATH = DATA_DIR / "audio_feature_scaler.joblib"

# Order matters — this is the fixed column order used everywhere downstream.
AUDIO_FEATURE_COLS = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]

_LYRIC_MODEL_NAME = "all-MiniLM-L6-v2"
_lyric_model: SentenceTransformer | None = None


def get_lyric_model() -> SentenceTransformer:
    """Lazy-loaded singleton — the model is ~80MB, don't reload it per call."""
    global _lyric_model
    if _lyric_model is None:
        _lyric_model = SentenceTransformer(_LYRIC_MODEL_NAME)
    return _lyric_model


def embed_lyrics(lyrics_text: str | None) -> np.ndarray | None:
    """Returns a 384-dim embedding of the lyrics, or None if there's no text."""
    if not lyrics_text or not lyrics_text.strip():
        return None
    model = get_lyric_model()
    # truncate very long lyrics — the model has a token limit anyway, and the
    # chorus/opening verses carry most of a song's semantic signature
    truncated = lyrics_text[:2000]
    embedding = model.encode(truncated, normalize_embeddings=True)
    return embedding.astype(np.float32)


def fit_audio_scaler(audio_feature_rows: np.ndarray) -> MinMaxScaler:
    """Fits a MinMaxScaler on the full dataset's audio features and saves it.
    Must be fit once on the whole corpus, then reused (never refit) for every
    later normalize_audio_features() call — otherwise the same raw tempo
    would map to different normalized values at different times."""
    scaler = MinMaxScaler()
    scaler.fit(audio_feature_rows)
    DATA_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    return scaler


def load_audio_scaler() -> MinMaxScaler:
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"{SCALER_PATH} not found — run fit_audio_scaler() on the full "
            "dataset first (build.py does this automatically)."
        )
    return joblib.load(SCALER_PATH)


def normalize_audio_features(raw_features: dict, scaler: MinMaxScaler) -> np.ndarray:
    """raw_features: dict with the keys in AUDIO_FEATURE_COLS (raw Spotify-style
    values). Returns a normalized vector in the same column order."""
    row = np.array([[raw_features[col] for col in AUDIO_FEATURE_COLS]])
    return scaler.transform(row)[0].astype(np.float32)
