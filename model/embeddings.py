"""
Turns raw track data into the numeric pieces the model needs:
  - a lyric embedding (semantic meaning of the words)
  - a normalized audio-feature vector (tempo, energy, valence, ...)

Both pieces get concatenated later (in build.py) into one track vector.
"""

from __future__ import annotations

from pathlib import Path

try:
    import joblib
except ImportError:
    joblib = None

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    class MinMaxScaler:
        def fit(self, X):
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
            self.range_ = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_)
            return self

        def transform(self, X):
            return (X - getattr(self, "min_", 0.0)) / getattr(self, "range_", 1.0)

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
_lyric_model = None


def get_lyric_model():
    """Lazy-loaded singleton — the model is ~80MB, don't reload it per call."""
    global _lyric_model
    if _lyric_model is None and SentenceTransformer is not None:
        try:
            _lyric_model = SentenceTransformer(_LYRIC_MODEL_NAME)
        except Exception:
            _lyric_model = None
    return _lyric_model


def embed_lyrics(lyrics_text: str | None) -> np.ndarray | None:
    """Returns a 384-dim embedding of the lyrics, or None if there's no text."""
    if not lyrics_text or not lyrics_text.strip():
        return None
    model = get_lyric_model()
    if model is None:
        vec = np.zeros(384, dtype=np.float32)
        words = lyrics_text.lower().split()
        for i, w in enumerate(words[:100]):
            idx = abs(hash(w)) % 384
            vec[idx] += 1.0 / (i + 1.0)
        norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32) if norm > 0 else vec

    # truncate very long lyrics — the model has a token limit anyway
    truncated = lyrics_text[:2000]
    embedding = model.encode(truncated, normalize_embeddings=True)
    return embedding.astype(np.float32)


def fit_audio_scaler(audio_feature_rows: np.ndarray) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(audio_feature_rows)
    DATA_DIR.mkdir(exist_ok=True)
    if joblib is not None:
        joblib.dump(scaler, SCALER_PATH)
    return scaler


def load_audio_scaler() -> MinMaxScaler:
    if joblib is not None and SCALER_PATH.exists():
        return joblib.load(SCALER_PATH)
    return MinMaxScaler()


def normalize_audio_features(raw_features: dict, scaler: MinMaxScaler) -> np.ndarray:
    """raw_features: dict with the keys in AUDIO_FEATURE_COLS (raw Spotify-style
    values). Returns a normalized vector in the same column order."""
    row = np.array([[raw_features[col] for col in AUDIO_FEATURE_COLS]])
    return scaler.transform(row)[0].astype(np.float32)
