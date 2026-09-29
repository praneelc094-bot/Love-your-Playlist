"""
Love Your Playlist - Web Application.
Ultra-modern, premium Spotify-inspired interface with seamless OAuth & Google/Apple login support.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Love Your Playlist",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add paths
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
    fetch_user_top_artists,
    fetch_user_top_tracks,
    get_user_profile_summary,
    enrich_tracks_with_shared_format,
)
from app.playlist import export_ranked_playlist_to_spotify
from app.ranker import generate_recommendations

# High-end styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(20, 30, 25, 0.9) 0%, rgba(10, 10, 12, 1) 90%);
        color: #FFFFFF;
    }

    /* Gradient Hero Text */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1DB954 0%, #00F5D4 50%, #FFFFFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
    }
    
    .hero-subtitle {
        color: #A0A0A5;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    /* Glassmorphism Stat Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), border-color 0.3s ease;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(29, 185, 84, 0.4);
        box-shadow: 0 12px 36px 0 rgba(29, 185, 84, 0.15);
    }
    .metric-num {
        font-size: 1.9rem;
        font-weight: 800;
        color: #1DB954;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        color: #8E8E93;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Glowing Action Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1DB954 0%, #17A34A 100%);
        color: #000000;
        font-weight: 700;
        font-size: 1rem;
        border-radius: 500px;
        border: none;
        padding: 0.65rem 2.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(29, 185, 84, 0.35);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1ed760 0%, #22c55e 100%);
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 28px rgba(29, 185, 84, 0.55);
        color: #000000;
    }

    /* Track Cards */
    .track-row {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 12px 18px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.25s ease;
    }
    .track-row:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(29, 185, 84, 0.3);
        transform: translateX(4px);
    }

    .match-pill {
        background: linear-gradient(135deg, rgba(29, 185, 84, 0.15) 0%, rgba(6, 182, 212, 0.15) 100%);
        color: #1DB954;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid rgba(29, 185, 84, 0.3);
        display: inline-block;
    }

    .genre-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #D1D5DB;
        padding: 5px 12px;
        border-radius: 20px;
        margin: 4px 4px 4px 0;
        font-size: 0.82rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .genre-tag:hover {
        background: rgba(29, 185, 84, 0.15);
        border-color: rgba(29, 185, 84, 0.4);
        color: #FFFFFF;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.02);
        padding: 6px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        color: #9CA3AF;
        padding: 8px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(29, 185, 84, 0.18) !important;
        color: #1DB954 !important;
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

# Auto-capture OAuth code if redirected directly to Streamlit
if "code" in st.query_params:
    oauth_code = st.query_params.get("code")
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI", DEFAULT_REDIRECT_URI)
    if client_id and client_secret:
        sp_client = get_spotify_client(client_id, client_secret, redirect_uri, auth_code=oauth_code)
        if sp_client:
            st.session_state.sp_client = sp_client
            st.session_state.is_demo = False
            st.query_params.clear()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### 🎧 **Audio Settings**")

    mode = st.radio(
        "Mode:",
        ["✨ Instant Demo", "🔐 Connect Spotify Account"],
        index=0 if st.session_state.is_demo else 1,
    )

    if mode == "✨ Instant Demo":
        st.session_state.is_demo = True
        st.session_state.sp_client = get_mock_client("Alex")
        st.success("🟢 Demo Active: Connected as Alex")
    else:
        st.session_state.is_demo = False
        st.caption("Spotify Developer Credentials:")
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
                f'<a href="{auth_url}" target="_blank">'
                f'<button style="width:100%; background:linear-gradient(135deg, #10B981 0%, #059669 100%); color:#FFFFFF; font-weight:700; border-radius:24px; padding:10px; border:none; cursor:pointer; margin-top:8px; box-shadow:0 4px 15px rgba(16,185,129,0.3);">'
                f'1. Authorize via Spotify'
                f'</button></a>',
                unsafe_allow_html=True,
            )

            st.caption("🔒 Opens Spotify's secure authentication portal.")

            auth_code = st.text_input(
                "2. Paste Redirect URL:",
                help="Copy the redirected URL from your browser address bar after authorizing.",
            )

            if st.button("Connect Account"):
                if auth_code:
                    code = sp_oauth.parse_response_code(auth_code) if "?" in auth_code else auth_code
                    sp = get_spotify_client(client_id, client_secret, redirect_uri, auth_code=code)
                    if sp:
                        st.session_state.sp_client = sp
                        st.success("Connected successfully!")
                        st.rerun()
                    else:
                        st.error("Authentication failed. Please verify the URL.")


sp = st.session_state.sp_client

if not sp:
    st.warning("👈 Select Demo Mode or Connect Spotify in the sidebar to begin.")
    st.stop()

# Load profile
if not st.session_state.user_profile:
    with st.spinner("Loading Profile..."):
        st.session_state.user_profile = get_user_profile_summary(sp)

profile_info = st.session_state.user_profile

# ----------------- MAIN HERO -----------------
col_h1, col_h2 = st.columns([1, 7])
with col_h1:
    if profile_info.get("image_url"):
        st.markdown(
            f'<img src="{profile_info["image_url"]}" style="width:80px; height:80px; border-radius:50%; border:2px solid #1DB954; box-shadow:0 0 20px rgba(29,185,84,0.3); object-fit:cover;">',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="width:75px; height:75px; border-radius:50%; background:rgba(29,185,84,0.2); display:flex; align-items:center; justify-content:center; font-size:2rem; border:2px solid #1DB954;">🎵</div>',
            unsafe_allow_html=True,
        )

with col_h2:
    st.markdown(f'<div class="hero-title">Welcome back, {profile_info.get("display_name", "Listener")}!</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Intelligent playlist curation tailored to your lyric taste and musical vibe.</div>', unsafe_allow_html=True)

# ----------------- NAVIGATION TABS -----------------
tab_taste, tab_studio, tab_result = st.tabs([
    "📊 Taste Profile",
    "🎛️ Playlist Studio",
    "🎶 Curated Playlist",
])

# ----------------- TAB 1: TASTE PROFILE -----------------
with tab_taste:
    c_sel1, c_sel2 = st.columns([3, 1])
    with c_sel1:
        time_range = st.selectbox(
            "Listening Period:",
            options=["medium_term", "short_term", "long_term"],
            format_func=lambda x: {
                "short_term": "Recent Favorites (Last 4 Weeks)",
                "medium_term": "Core Rotation (Last 6 Months)",
                "long_term": "All-Time Classics (Years)",
            }[x],
        )
    with c_sel2:
        st.write("")
        st.write("")
        if st.button("⚡ Sync Taste Data") or not st.session_state.user_tracks:
            with st.spinner("Analyzing music taste..."):
                raw_top_tracks = fetch_user_top_tracks(sp, time_range=time_range, limit=50)
                st.session_state.top_artists = fetch_user_top_artists(sp, time_range=time_range, limit=20)
                st.session_state.top_genres = extract_top_genres(st.session_state.top_artists)
                st.session_state.user_tracks = enrich_tracks_with_shared_format(
                    raw_top_tracks, lyrics_lookup=True
                )
                st.success("Taste data synced!")

    if st.session_state.user_tracks:
        st.write("")
        # Stat Cards
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-num">{len(st.session_state.user_tracks)}</div><div class="metric-label">Analyzed Tracks</div></div>',
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-num">{len(st.session_state.top_artists)}</div><div class="metric-label">Top Artists</div></div>',
                unsafe_allow_html=True,
            )
        with s3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-num">{len(st.session_state.top_genres)}</div><div class="metric-label">Distinct Genres</div></div>',
                unsafe_allow_html=True,
            )
        with s4:
            lyrics_count = sum(1 for t in st.session_state.user_tracks if t.get("has_lyrics"))
            pct = int((lyrics_count / max(1, len(st.session_state.user_tracks))) * 100)
            st.markdown(
                f'<div class="metric-card"><div class="metric-num">{pct}%</div><div class="metric-label">Lyric Coverage</div></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        st.write("")

        col_radar, col_tags = st.columns([1, 1])

        with col_radar:
            st.markdown("##### 🧬 Audio Vibe Profile")
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
                        fillcolor="rgba(29, 185, 84, 0.25)",
                        line=dict(color="#1DB954", width=2.5),
                        marker=dict(color="#00F5D4", size=6),
                    )
                )
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, linecolor="rgba(255,255,255,0.1)"),
                        angularaxis=dict(color="#E5E7EB"),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=40, r=40, t=20, b=20),
                    height=280,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.bar_chart(feature_avgs)

        with col_tags:
            st.markdown("##### 🎸 Top Genres")
            if st.session_state.top_genres:
                chips = "".join([
                    f'<span class="genre-tag">{genre.title()} • {count}</span>'
                    for genre, count in st.session_state.top_genres[:12]
                ])
                st.markdown(chips, unsafe_allow_html=True)

            st.write("")
            st.markdown("##### 🎤 Top Artists in Rotation")
            artists_str = " • ".join([a.get("name") for a in st.session_state.top_artists[:8]])
            st.markdown(f'<div style="color:#9CA3AF; font-size:0.95rem; line-height:1.6;">{artists_str}</div>', unsafe_allow_html=True)


# ----------------- TAB 2: PLAYLIST STUDIO -----------------
with tab_studio:
    st.markdown("### 🎛️ **Playlist Studio**")
    st.caption("Tune the AI parameters to craft your ideal mix.")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("##### ⚖️ Balance: Lyrics vs. Audio Vibe")
        lyric_pct = st.slider(
            "Lyric Meaning vs. Tune Rhythm Weight:",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            format="%d%%",
        )
        audio_pct = 100 - lyric_pct
        st.markdown(
            f'<div style="display:flex; justify-content:space-between; color:#9CA3AF; font-size:0.9rem; margin-top:-8px;">'
            f'<span>💬 <b>{lyric_pct}%</b> Lyric Match</span>'
            f'<span>🎵 <b>{audio_pct}%</b> Vibe Match</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.write("")
        playlist_size = st.slider(
            "Playlist Length (Songs):",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
        )

    with col_s2:
        st.markdown("##### 🎭 Mood & Atmosphere")
        mood_selection = st.selectbox(
            "Select Vibe Preset:",
            options=list(MOOD_PRESETS.keys()),
            index=0,
        )

        candidate_depth = st.select_slider(
            "Discovery Depth (Search Pool):",
            options=[50, 75, 100, 150, 200],
            value=100,
            help="Deeper search explores wider artist networks across Spotify.",
        )

        diversity = st.slider(
            "Artist Variety:",
            min_value=0.0,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Higher values guarantee a broader variety of artists.",
        )

    st.write("")
    st.markdown("---")

    if st.button("🚀 Generate Curated Playlist", use_container_width=True):
        if not st.session_state.user_tracks:
            st.error("Please load your taste data in the Taste Profile tab first.")
        else:
            prog = st.progress(0, text="Searching catalog...")

            prog.progress(25, text="🔍 Scanning Spotify catalog for candidate tracks...")
            candidates = generate_candidate_pool(
                sp,
                st.session_state.user_tracks,
                st.session_state.top_artists,
                st.session_state.top_genres,
                target_pool_size=candidate_depth,
                mood_preset=mood_selection,
                lyrics_lookup=True,
            )

            prog.progress(65, text="🧠 Computing lyric semantics & audio vibe match...")
            profile, ranked = generate_recommendations(
                st.session_state.user_tracks,
                candidates,
                n=playlist_size,
                lyric_weight=lyric_pct / 100.0,
                audio_weight=audio_pct / 100.0,
                diversity_penalty=diversity,
            )

            prog.progress(100, text="✨ Playlist Ready!")
            st.session_state.taste_profile = profile
            st.session_state.ranked_tracks = ranked
            st.session_state.playlist_created = None
            st.success(f"Generated {len(ranked)} tracks! Head to the Curated Playlist tab to preview.")


# ----------------- TAB 3: CURATED PLAYLIST -----------------
with tab_result:
    if not st.session_state.ranked_tracks:
        st.info("No playlist generated yet. Configure your preferences and click 'Generate Curated Playlist' in the Studio tab!")
    else:
        ranked = st.session_state.ranked_tracks

        col_p1, col_p2 = st.columns([3, 2])
        with col_p1:
            pl_name = st.text_input(
                "Playlist Title:",
                value=f"Love Your Playlist • {mood_selection if 'mood_selection' in locals() else 'Personal Blend'}",
            )
            pl_desc = st.text_area(
                "Description:",
                value=f"Curated mix matching {lyric_pct if 'lyric_pct' in locals() else 50}% lyric themes and {audio_pct if 'audio_pct' in locals() else 50}% audio vibe.",
                height=65,
            )
        with col_p2:
            st.write("")
            st.write("")
            is_public = st.checkbox("Public Playlist", value=True)
            if st.button("💚 Export to Spotify Account", use_container_width=True):
                with st.spinner("Creating playlist in Spotify..."):
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
            st.success(f"🎉 Playlist Created! **{res['name']}** ({res['tracks_added']} tracks)")
            st.markdown(
                f'👉 **[Open in Spotify Web Player]({res["url"]})**',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(f"#### 🎵 Tracklist ({len(ranked)} Songs)")

        for idx, track in enumerate(ranked, 1):
            col_cover, col_info, col_score, col_link = st.columns([1, 6, 3, 2])

            with col_cover:
                if track.get("image_url"):
                    st.markdown(
                        f'<img src="{track["image_url"]}" style="width:48px; height:48px; border-radius:8px; object-fit:cover;">',
                        unsafe_allow_html=True,
                    )
                else:
                    st.write("🎵")

            with col_info:
                st.markdown(f"**{idx}. {track['title']}** — *{track['artist']}*")
                if track.get("lyrics_text"):
                    snippet = track["lyrics_text"].replace("\n", " ")[:65]
                    st.caption(f'"{snippet}..."')

            with col_score:
                score_val = track.get("score", 0.85)
                pct_str = f"{int(score_val * 100)}%" if score_val <= 1.0 else f"{int(score_val)}%"
                st.markdown(f'<span class="match-pill">{pct_str} Match</span>', unsafe_allow_html=True)
                st.caption(track.get("match_reason", "Vibe Match"))

            with col_link:
                if track.get("spotify_id"):
                    track_url = f"https://open.spotify.com/track/{track['spotify_id']}"
                    st.markdown(f"[▶ Listen on Spotify]({track_url})")

            st.markdown("<hr style='margin:4px 0; border:none; border-top:1px solid rgba(255,255,255,0.06);' />", unsafe_allow_html=True)
