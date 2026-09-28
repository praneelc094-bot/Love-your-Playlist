# Love your Playlist

A model that reads a Spotify user's taste — lyrics, tune/audio features, and listening
history — and generates a playlist that matches it.

## Team split

- **Person A — Data & Model** (`model/`): lyrics fetching, audio features, embeddings,
  taste profile, ranking.
- **Person B — Spotify & App** (`app/`): Spotify auth, top tracks/library, candidate
  track search, playlist creation, UI.

## Shared interface

Both sides agree on one track format and one model interface so each side can work
independently. See [`docs/track-format.md`](docs/track-format.md) and
[`docs/model-interface.md`](docs/model-interface.md).

## Status

- [ ] Phase 1 — Foundations (offline dataset, lyrics fetcher, audio features)
- [ ] Phase 2 — Core logic (embeddings, taste profile, ranking, app pipeline + UI)
- [ ] Phase 3 — Integration (real model plugged into the app)
- [ ] Phase 4 — Polish (docs, demo, terms check)

## Repo layout

```
model/     Person A — data collection and taste/ranking model
app/       Person B — Spotify auth, candidate search, playlist creation, UI
docs/      Shared contracts both sides code against
```
