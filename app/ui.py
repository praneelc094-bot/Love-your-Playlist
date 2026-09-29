"""
Streamlit Web Interface for Love Your Playlist (Person B).

Interactive Web Application featuring:
- Spotify OAuth Login & Instant Demo Mode (no credentials needed to test)
- Real-time Taste Profile Visualizer (Radar Charts, Genre breakdowns, Top tracks)
- Playlist Studio with customizable Lyric vs. Tune Weight Sliders & Mood Presets
- Candidate track generation & multi-factor ranking
- One-click Spotify Playlist Creation & direct Spotify Web Player links
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="Love Your Playlist",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODEL_DIR = PROJECT_ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from app.auth import (
    DEFAULT_REDIRECT_URI,
    get_mock_client,
    get_spotify_client,
    get_spotify_oauth,
)
from app.candidates import MOOD_PRESETS, generate_candidate_pool
from app.library import (
    extract_top_genres,
    fetch_user_saved_tracks,
    fetch_user_top_artists,
    fetch_user_top_tracks,
    get_user_profile_summary,
    enrich_tracks_with_shared_format,
)
from app.playlist import export_ranked_playlist_to_spotify
from app.ranker import generate_recommendations

# Custom CSS for Spotify-like dark aesthetic
st.markdown(
    """
    <style>
    .main {
        background-color: #121212;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #1DB954;
        color: #000000;
        font-weight: 700;
        border-radius: 500px;
        border: none;
        padding: 0.5rem 2rem;
        transition: transform 0.2s ease, background-color 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1ed760;
        transform: scale(1.03);
        color: #000000;
    }
    .track-card {
        background-color: #181818;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        border: 1px solid #282828;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .track-card:hover {
        background-color: #282828;
    }
    .match-badge {
        background-color: rgba(29, 185, 84, 0.2);
        color: #1DB954;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .genre-chip {
        display: inline-block;
        background-color: #282828;
        color: #e0e0e0;
        padding: 4px 10px;
        border-radius: 16px;
        margin-right: 6px;
        margin-bottom: 6px;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    if "sp_client" not in st.session_state:
        st.session_state.sp_client = None
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = None
    if "user_tracks" not in st.session_state:
        st.session_state.user_tracks = []
    if "top_artists" not in st.session_state:
        st.session_state.top_artists = []
    if "top_genres" not in st.session_state:
        st.session_state.top_genres = []
    if "ranked_tracks" not in st.session_state:
        st.session_state.ranked_tracks = []
    if "taste_profile" not in st.session_state:
        st.session_state.taste_profile = None
    if "playlist_created" not in st.session_state:
        st.session_state.playlist_created = None
    if "is_demo" not in st.session_state:
        st.session_state.is_demo = True


init_session_state()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("🎧 Love Your Playlist")
    st.caption("AI-Powered Taste & Lyric Matching")

    mode = st.radio(
        "Connection Mode:",
        ["✨ Instant Demo Mode (No Setup)", "🔐 Live Spotify Account"],
        index=0 if st.session_state.is_demo else 1,
    )

    if mode == "✨ Instant Demo Mode (No Setup)":
        st.session_state.is_demo = True
        st.session_state.sp_client = get_mock_client("Alex (Demo Listener)")
        st.success("🟢 Demo Mode Active: Connected as Alex")
    else:
        st.session_state.is_demo = False
        st.subheader("Spotify API Credentials")
        client_id = st.text_input(
            "Client ID",
            value=os.getenv("SPOTIPY_CLIENT_ID", ""),
            type="password",
        )
        client_secret = st.text_input(
            "Client Secret",
            value=os.getenv("SPOTIPY_CLIENT_SECRET", ""),
            type="password",
        )
        redirect_uri = st.text_input(
            "Redirect URI",
            value=os.getenv("SPOTIPY_REDIRECT_URI", DEFAULT_REDIRECT_URI),
        )

        if client_id and client_secret:
            sp_oauth = get_spotify_oauth(client_id, client_secret, redirect_uri)
            auth_url = sp_oauth.get_authorize_url()

            st.markdown(
                f'<a href="{auth_url}" target="_blank"><button style="width:100%; background:#1DB954; color:#000; font-weight:bold; border-radius:20px; padding:8px; border:none; cursor:pointer;">1. Authorize on Spotify</button></a>',
                unsafe_allow_html=True,
            )

            auth_code = st.text_input(
                "2. Paste the redirect URL or Code:",
                help="After authorizing, copy the full URL you were redirected to.",
            )

            if st.button("Connect Account"):
                if auth_code:
                    code = sp_oauth.parse_response_code(auth_code) if "?" in auth_code else auth_code
                    sp = get_spotify_client(client_id, client_secret, redirect_uri, auth_code=code)
                    if sp:
                        st.session_state.sp_client = sp
                        st.success("Connected to Spotify successfully!")
                        st.rerun()
                    else:
                        st.error("Authentication failed. Please verify the code.")
        else:
            st.info("💡 Tip: Provide Client ID and Secret in `.env` or paste above.")

    st.markdown("---")
    st.markdown(
        "**About This Project**\n\n"
        "Built with Python, Sentence-Transformers, Spotipy & Streamlit.\n\n"
        "• **Person A:** Lyrics, Embeddings & Taste Ranking Model\n"
        "• **Person B:** Spotify Auth, Candidate Engine & Web UI"
    )

# ----------------- MAIN VIEW -----------------

sp = st.session_state.sp_client

if not sp:
    st.warning("👈 Please select Instant Demo Mode or connect your Spotify Account in the sidebar.")
    st.stop()

# Fetch and store profile
if not st.session_state.user_profile:
    with st.spinner("Loading Spotify Profile..."):
        st.session_state.user_profile = get_user_profile_summary(sp)

profile_info = st.session_state.user_profile

# Header Section
col_head1, col_head2 = st.columns([1, 6])
with col_head1:
    if profile_info.get("image_url"):
        st.image(profile_info["image_url"], width=90)
    else:
        st.markdown("## 🎵")
with col_head2:
    st.title(f"Welcome, {profile_info.get('display_name', 'Listener')}!")
    st.caption(f"Account: {profile_info.get('product', 'standard').capitalize()} Tier • Mode: {'Demo' if st.session_state.is_demo else 'Live'}")

st.markdown("---")

tab_taste, tab_generate, tab_playlist = st.tabs([
    "📊 1. Your Taste Profile",
    "🎛️ 2. Playlist Studio",
    "🎶 3. Generated Playlist & Export",
])

# ----------------- TAB 1: YOUR TASTE PROFILE -----------------
with tab_taste:
    st.subheader("Listening History & Preferences")

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        time_range = st.selectbox(
            "Analyze Listening History Over:",
            options=["medium_term", "short_term", "long_term"],
            format_func=lambda x: {
                "short_term": "Recent Taste (Last 4 Weeks)",
                "medium_term": "Core Taste (Last 6 Months)",
                "long_term": "All-Time Favorites (Years)",
            }[x],
        )

    with col_t2:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Taste Data") or not st.session_state.user_tracks:
            with st.spinner("Fetching top tracks and generating embeddings..."):
                raw_top_tracks = fetch_user_top_tracks(sp, time_range=time_range, limit=50)
                st.session_state.top_artists = fetch_user_top_artists(sp, time_range=time_range, limit=20)
                st.session_state.top_genres = extract_top_genres(st.session_state.top_artists)
                st.session_state.user_tracks = enrich_tracks_with_shared_format(
                    raw_top_tracks, lyrics_lookup=True
                )
                st.success(f"Loaded {len(st.session_state.user_tracks)} top tracks!")

    if st.session_state.user_tracks:
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Top Tracks Analyzed", len(st.session_state.user_tracks))
        m2.metric("Top Artists", len(st.session_state.top_artists))
        m3.metric("Top Genres", len(st.session_state.top_genres))
        lyrics_count = sum(1 for t in st.session_state.user_tracks if t.get("has_lyrics"))
        m4.metric("Lyrics Embedded", f"{lyrics_count}/{len(st.session_state.user_tracks)}")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### 🎸 Top Genres")
            if st.session_state.top_genres:
                chips_html = "".join([
                    f'<span class="genre-chip">{genre.title()} ({count})</span>'
                    for genre, count in st.session_state.top_genres[:12]
                ])
                st.markdown(chips_html, unsafe_allow_html=True)
            else:
                st.write("No genre metadata available.")

            st.markdown("#### 🎤 Top Artists")
            artist_names = [a.get("name") for a in st.session_state.top_artists[:8]]
            st.write(", ".join(artist_names) if artist_names else "None found")

        with col_right:
            st.markdown("#### 🎼 Audio Vibe Radar")
            features_to_plot = ["energy", "danceability", "valence", "acousticness", "liveness", "speechiness"]
            feature_avgs = {}
            for feat in features_to_plot:
                vals = [
                    t["audio_features"].get(feat, 0.5)
                    for t in st.session_state.user_tracks
                    if "audio_features" in t
                ]
                feature_avgs[feat.capitalize()] = np.mean(vals) if vals else 0.5

            try:
                import plotly.graph_objects as go
                fig = go.Figure(
                    data=go.Scatterpolar(
                        r=list(feature_avgs.values()),
                        theta=list(feature_avgs.keys()),
                        fill="toself",
                        fillcolor="rgba(29, 185, 84, 0.4)",
                        line=dict(color="#1DB954", width=2),
                    )
                )
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=30, r=30, t=20, b=20),
                    height=260,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.bar_chart(feature_avgs)

        with st.expander("👀 View Your Top Tracks"):
            for i, track in enumerate(st.session_state.user_tracks[:10], 1):
                col_i1, col_i2, col_i3 = st.columns([1, 6, 3])
                with col_i1:
                    if track.get("image_url"):
                        st.image(track["image_url"], width=45)
                    else:
                        st.write("🎵")
                with col_i2:
                    st.markdown(f"**{i}. {track['title']}** — {track['artist']}")
                    if track.get("lyrics_text"):
                        snippet = track["lyrics_text"].replace("\n", " ")[:60]
                        st.caption(f'💬 "{snippet}..."')
                with col_i3:
                    st.caption(f"⚡ Energy: {track['audio_features']['energy']:.2f} | 💃 Dance: {track['audio_features']['danceability']:.2f}")

# ----------------- TAB 2: PLAYLIST STUDIO -----------------
with tab_generate:
    st.subheader("Customize & Generate New Playlist")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("#### 🎚️ Matching Balance")
        lyric_pct = st.slider(
            "Lyrics Meaning vs. Audio Vibe Weight:",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            format="%d%%",
            help="100% means pure lyric & thematic matching. 0% means pure audio rhythm/tempo matching.",
        )
        audio_pct = 100 - lyric_pct
        st.caption(f"**Weight:** {lyric_pct}% Lyrics Match • {audio_pct}% Audio Tune Match")

        playlist_size = st.slider(
            "Playlist Length (Tracks):",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
        )

    with col_g2:
        st.markdown("#### 🎭 Mood & Discovery Depth")
        mood_selection = st.selectbox(
            "Target Mood / Activity:",
            options=list(MOOD_PRESETS.keys()),
            index=0,
        )

        candidate_pool_size = st.select_slider(
            "Candidate Search Pool Depth:",
            options=[50, 75, 100, 150, 200],
            value=100,
            help="Higher depth searches more tracks across Spotify for higher quality recommendations.",
        )

        diversity = st.slider(
            "Artist Diversity Factor:",
            min_value=0.0,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Penalizes repeated artists so the playlist has variety.",
        )

    st.markdown("---")

    if st.button("⚡ Generate Playlist Now", use_container_width=True):
        if not st.session_state.user_tracks:
            st.error("Please load your taste data in Tab 1 first.")
        else:
            progress_bar = st.progress(0, text="Initializing generation pipeline...")

            progress_bar.progress(20, text="🔍 Searching candidate pool across top artists, genres and mood...")
            candidates = generate_candidate_pool(
                sp,
                st.session_state.user_tracks,
                st.session_state.top_artists,
                st.session_state.top_genres,
                target_pool_size=candidate_pool_size,
                mood_preset=mood_selection,
                lyrics_lookup=True,
            )

            progress_bar.progress(60, text="🧠 Building taste profile & computing semantic similarity...")
            profile, ranked = generate_recommendations(
                st.session_state.user_tracks,
                candidates,
                n=playlist_size,
                lyric_weight=lyric_pct / 100.0,
                audio_weight=audio_pct / 100.0,
                diversity_penalty=diversity,
            )

            progress_bar.progress(100, text="✅ Playlist Generation Complete!")
            st.session_state.taste_profile = profile
            st.session_state.ranked_tracks = ranked
            st.session_state.playlist_created = None
            st.success(f"Generated {len(ranked)} perfectly matched tracks! Check Tab 3 to preview and export.")

# ----------------- TAB 3: PLAYLIST PREVIEW & EXPORT -----------------
with tab_playlist:
    st.subheader("Your Curated Playlist")

    if not st.session_state.ranked_tracks:
        st.info("No playlist generated yet. Click 'Generate Playlist Now' in Tab 2!")
    else:
        ranked = st.session_state.ranked_tracks

        col_meta1, col_meta2 = st.columns([3, 2])
        with col_meta1:
            pl_name = st.text_input(
                "Playlist Title:",
                value=f"Love Your Playlist • {mood_selection if 'mood_selection' in locals() else 'My Blend'}",
            )
            pl_desc = st.text_area(
                "Description:",
                value=f"AI-crafted playlist curated to your unique taste. Matches lyric themes ({lyric_pct if 'lyric_pct' in locals() else 50}%) and audio vibes ({audio_pct if 'audio_pct' in locals() else 50}%).",
                height=70,
            )
        with col_meta2:
            st.write("")
            st.write("")
            is_public = st.checkbox("Make Playlist Public", value=True)
            if st.button("🚀 Create Playlist in Spotify", use_container_width=True):
                with st.spinner("Creating playlist and adding tracks..."):
                    res = export_ranked_playlist_to_spotify(
                        sp,
                        user_id=profile_info.get("id", "me"),
                        ranked_tracks=ranked,
                        playlist_name=pl_name,
                        description=pl_desc,
                        public=is_public,
                        lyric_weight=(lyric_pct if "lyric_pct" in locals() else 50) / 100.0,
                        audio_weight=(audio_pct if "audio_pct" in locals() else 50) / 100.0,
                    )
                    st.session_state.playlist_created = res
                    st.balloons()

        if st.session_state.playlist_created:
            res = st.session_state.playlist_created
            st.success(f"🎉 Playlist Created! **{res['name']}** ({res['tracks_added']} tracks added)")
            st.markdown(
                f'👉 **[Open Playlist in Spotify]({res["url"]})**',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### Tracklist & Match Insights")

        for idx, track in enumerate(ranked, 1):
            with st.container():
                c1, c2, c3, c4 = st.columns([1, 6, 3, 2])
                with c1:
                    if track.get("image_url"):
                        st.image(track["image_url"], width=50)
                    else:
                        st.write("🎵")
                with c2:
                    st.markdown(f"**{idx}. {track['title']}** — *{track['artist']}*")
                    if track.get("lyrics_text"):
                        snippet = track["lyrics_text"].replace("\n", " ")[:70]
                        st.caption(f'💬 "{snippet}..."')
                with c3:
                    st.markdown(f'<span class="match-badge">Match: {track.get("score", 0.8):.0%}</span>', unsafe_allow_html=True)
                    st.caption(track.get("match_reason", "Taste match"))
                with c4:
                    if track.get("spotify_id"):
                        link = f"https://open.spotify.com/track/{track['spotify_id']}"
                        st.markdown(f"[Listen on Spotify]({link})")
                st.markdown("<hr style='margin: 4px 0; opacity: 0.1;' />", unsafe_allow_html=True)
