# Shared track format

Every track, whether it comes from the offline dataset (Person A) or from a live
Spotify search (Person B), gets normalized into this shape before it touches the
model:

```json
{
  "spotify_id": "string | null",
  "title": "string",
  "artist": "string",
  "lyrics_text": "string | null",
  "audio_features": {
    "tempo": "float",
    "energy": "float",
    "valence": "float",
    "danceability": "float",
    "acousticness": "float",
    "instrumentalness": "float",
    "speechiness": "float",
    "loudness": "float"
  },
  "embedding": "float[] | null"
}
```

Notes:
- `spotify_id` is null for pure offline-dataset tracks until they're matched to a
  real Spotify ID (needed before they can be added to a playlist).
- `lyrics_text` and `embedding` are null when lyrics weren't found — don't drop the
  track, just skip the lyric component of its embedding.
- `audio_features` values should be normalized (min-max or z-score) with the same
  scaler used at both training/profile-building time and ranking time. Save the
  scaler alongside the model.
