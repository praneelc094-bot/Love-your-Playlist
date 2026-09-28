# app/ — Person B: Spotify & App

## Phase 1 goal
Spotify login, top-tracks/library fetch, a search-based candidate fetcher, and a
tested `POST /me/playlists` call.

## Layout
```
auth.py            Spotify Authorization Code flow
library.py          fetch top tracks / saved library, converted to shared track format
candidates.py        search-based candidate fetcher (10-result cap workaround)
playlist.py          create playlist, add tracks
ui/                   Streamlit or Flask app
requirements.txt
```

See the full plan discussed in chat for day-by-day detail. Note: Spotify's Web API
removed Recommendations/Related Artists/Audio Features for new apps, caps search at
10 results, and requires the app owner to have Premium (Feb 2026 Dev Mode changes) —
`candidates.py` works around the search cap by running many small queries across
the user's top artists/genres instead of relying on `/recommendations`.
