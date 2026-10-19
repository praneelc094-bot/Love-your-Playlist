"""
Love Your Playlist - Music Recommendation & Playlist Curation Engine.
A bespoke, human-designed music curation platform blending acoustic features and lyric semantics.
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
    fetch_user_library_comprehensive,
    get_user_profile_summary,
    enrich_tracks_with_shared_format,
)
from app.playlist import export_ranked_playlist_to_spotify
from app.ranker import generate_recommendations

# Custom CSS for a clean, human-crafted Spotify-grade interface
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e5e7eb;
    }

    .stApp {
        background-color: #0d0f12;
        background-image: 
            radial-gradient(at 15% 15%, rgba(16, 185, 129, 0.08) 0px, transparent 40%),
            radial-gradient(at 85% 85%, rgba(30, 41, 59, 0.4) 0px, transparent 50%);
    }

    /* Top Header Bar */
    .app-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding-bottom: 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 24px;
    }
    
    .user-avatar {
        width: 56px;
        height: 56px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid #10b981;
    }
    
    .user-avatar-placeholder {
        width: 56px;
        height: 56px;
        border-radius: 50%;
        background: #1e293b;
        color: #10b981;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.2rem;
        border: 2px solid #10b981;
    }

    .header-text h1 {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .header-text p {
        font-size: 0.9rem;
        color: #94a3b8;
        margin: 2px 0 0 0;
    }

    /* Metric Stat Panels */
    .stat-panel {
        background: #15181e;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px;
        text-align: left;
    }
    .stat-panel-num {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .stat-panel-label {
        font-size: 0.8rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Content Cards */
    .card-box {
        background: #15181e;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Custom Tags */
    .custom-genre-tag {
        display: inline-block;
        background: #1e232b;
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #cbd5e1;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 6px;
        margin: 3px 4px 3px 0;
    }

    /* Match Badge */
    .score-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }

    /* Track Item */
    .track-item {
        background: #15181e;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Primary Buttons */
    .stButton>button {
        background: #10b981;
        color: #090b0e;
        font-weight: 600;
        font-size: 0.95rem;
        border-radius: 8px;
        border: none;
        padding: 0.55rem 1.4rem;
        transition: background 0.15s ease, transform 0.1s ease;
    }
    .stButton>button:hover {
        background: #059669;
        color: #ffffff;
        transform: translateY(-1px);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #15181e;
        padding: 4px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        padding: 8px 18px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e232b !important;
        color: #ffffff !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #111317;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
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
            st.session_state.user_profile = None
            st.session_state.user_tracks = []
            st.query_params.clear()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### **Account Settings**")

    mode = st.radio(
        "Source:",
        ["Instant Demo Mode", "Live Spotify Account"],
        index=0 if st.session_state.is_demo else 1,
    )

    if mode == "Instant Demo Mode":
        if not st.session_state.is_demo or st.session_state.sp_client is None:
            st.session_state.is_demo = True
            st.session_state.sp_client = get_mock_client("Alex")
            st.session_state.user_profile = None
            st.session_state.user_tracks = []
            st.session_state.top_artists = []
            st.session_state.top_genres = []
            st.session_state.ranked_tracks = []
        st.caption("Connected as **Alex (Demo Profile)**")
    else:
        # Live Account Mode
        is_live_authenticated = bool(not st.session_state.is_demo and st.session_state.sp_client is not None)

        if is_live_authenticated:
            # Check user profile
            current_user_name = st.session_state.user_profile.get("display_name", "Spotify Listener") if st.session_state.user_profile else "Spotify Listener"
            user_plan = st.session_state.user_profile.get("product", "Standard").capitalize() if st.session_state.user_profile else "Standard"

            st.markdown(
                f'<div style="background:#15181e; border:1px solid rgba(16,185,129,0.3); border-radius:10px; padding:14px; margin-top:8px;">'
                f'<div style="color:#10b981; font-weight:600; font-size:0.9rem;">● Connected to Spotify</div>'
                f'<div style="color:#ffffff; font-weight:700; font-size:1.05rem; margin-top:4px;">{current_user_name}</div>'
                f'<div style="color:#94a3b8; font-size:0.8rem; margin-top:2px;">Plan: {user_plan}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.write("")
            if st.button("Log Out / Disconnect", use_container_width=True):
                st.session_state.sp_client = None
                st.session_state.user_profile = None
                st.session_state.user_tracks = []
                st.session_state.top_artists = []
                st.session_state.top_genres = []
                st.session_state.ranked_tracks = []
                if os.path.exists(".cache"):
                    try:
                        os.remove(".cache")
                    except Exception:
                        pass
                st.rerun()
        else:
            if st.session_state.is_demo:
                st.session_state.is_demo = False
                st.session_state.sp_client = None
                st.session_state.user_profile = None
                st.session_state.user_tracks = []
                st.session_state.top_artists = []
                st.session_state.top_genres = []
                st.session_state.ranked_tracks = []

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
                    f'<button style="width:100%; background:#10b981; color:#090b0e; font-weight:600; border-radius:8px; padding:9px; border:none; cursor:pointer; margin-top:8px;">'
                    f'1. Authorize with Spotify'
                    f'</button></a>',
                    unsafe_allow_html=True,
                )

                st.caption("Opens Spotify's official login portal.")

                auth_code = st.text_input(
                    "2. Paste Redirect URL:",
                    help="Paste the full callback URL from your browser address bar after authorizing.",
                )

                if st.button("Connect Account"):
                    if auth_code:
                        code = sp_oauth.parse_response_code(auth_code) if "?" in auth_code else auth_code
                        sp = get_spotify_client(client_id, client_secret, redirect_uri, auth_code=code)
                        if sp:
                            st.session_state.sp_client = sp
                            st.session_state.is_demo = False
                            st.session_state.user_profile = None
                            st.session_state.user_tracks = []
                            st.session_state.top_artists = []
                            st.session_state.top_genres = []
                            st.session_state.ranked_tracks = []
                            st.success("Account connected.")
                            st.rerun()
                        else:
                            st.error("Authentication failed. Please verify credentials or URL.")


sp = st.session_state.sp_client

if not sp:
    st.warning("Please select Demo Mode or connect your Spotify account in the sidebar to begin.")
    st.stop()

# Load profile
if not st.session_state.user_profile:
    with st.spinner("Loading account profile..."):
        st.session_state.user_profile = get_user_profile_summary(sp)

profile_info = st.session_state.user_profile
user_name = profile_info.get("display_name", "Listener")
user_img = profile_info.get("image_url")

# ----------------- HEADER -----------------
col_av, col_tx = st.columns([1, 10])
with col_av:
    if user_img:
        st.markdown(f'<img class="user-avatar" src="{user_img}" />', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="user-avatar-placeholder">{user_name[:1]}</div>', unsafe_allow_html=True)

with col_tx:
    st.markdown(
        f'<div class="header-text">'
        f'<h1>Welcome, {user_name}</h1>'
        f'<p>Personalized playlist curation based on audio acoustics and lyric themes.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.write("")

# ----------------- MAIN TABS -----------------
tab_taste, tab_studio, tab_result = st.tabs([
    "Listening Profile",
    "Curator Studio",
    "Generated Playlist",
])

# ----------------- TAB 1: LISTENING PROFILE -----------------
with tab_taste:
    col_t1, col_t2, col_t3 = st.columns([2, 2, 1])
    with col_t1:
        source_mode = st.selectbox(
            "Library Data Source:",
            options=["all", "saved_tracks", "playlists", "top_tracks", "recent"],
            format_func=lambda x: {
                "all": "All Library (Saved Songs + Playlists + History)",
                "saved_tracks": "Saved Songs (Liked Tracks)",
                "playlists": "My Created & Saved Playlists",
                "top_tracks": "Top Tracks History",
                "recent": "Recently Played Tracks",
            }[x],
        )
    with col_t2:
        time_range = st.selectbox(
            "Listening Horizon:",
            options=["medium_term", "short_term", "long_term"],
            format_func=lambda x: {
                "short_term": "Recent Rotation (Last 4 Weeks)",
                "medium_term": "Core Rotation (Last 6 Months)",
                "long_term": "All-Time Classics (Multiple Years)",
            }[x],
        )
    with col_t3:
        st.write("")
        st.write("")
        if st.button("Sync Listening Data") or not st.session_state.user_tracks:
            with st.spinner("Fetching library tracks, playlists, and artists..."):
                raw_tracks, top_artists, top_genres = fetch_user_library_comprehensive(
                    sp,
                    source=source_mode,
                    time_range=time_range,
                )
                st.session_state.top_artists = top_artists
                st.session_state.top_genres = top_genres
                st.session_state.user_tracks = enrich_tracks_with_shared_format(
                    raw_tracks, lyrics_lookup=False
                )

    if st.session_state.user_tracks:
        st.write("")
        # Stat Cards
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(
                f'<div class="stat-panel">'
                f'<div class="stat-panel-num">{len(st.session_state.user_tracks)}</div>'
                f'<div class="stat-panel-label">Analyzed Tracks</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                f'<div class="stat-panel">'
                f'<div class="stat-panel-num">{len(st.session_state.top_artists)}</div>'
                f'<div class="stat-panel-label">Seed Artists</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with s3:
            st.markdown(
                f'<div class="stat-panel">'
                f'<div class="stat-panel-num">{len(st.session_state.top_genres)}</div>'
                f'<div class="stat-panel-label">Primary Genres</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with s4:
            st.markdown(
                f'<div class="stat-panel">'
                f'<div class="stat-panel-num">Active</div>'
                f'<div class="stat-panel-label">Library Sync</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.write("")
        st.write("")

        col_radar, col_tags = st.columns([1, 1])

        with col_radar:
            st.markdown("##### Acoustic Feature Profile")
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
                        fillcolor="rgba(16, 185, 129, 0.2)",
                        line=dict(color="#10b981", width=2),
                        marker=dict(color="#10b981", size=5),
                    )
                )
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, linecolor="rgba(255,255,255,0.08)"),
                        angularaxis=dict(color="#94a3b8", gridcolor="rgba(255,255,255,0.06)"),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=30, r=30, t=15, b=15),
                    height=270,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.bar_chart(feature_avgs)

        with col_tags:
            st.markdown("##### Top Genres")
            if st.session_state.top_genres:
                chips = "".join([
                    f'<span class="custom-genre-tag">{genre.title()} ({count})</span>'
                    for genre, count in st.session_state.top_genres[:12]
                ])
                st.markdown(chips, unsafe_allow_html=True)

            st.write("")
            st.markdown("##### Frequent Artists")
            artists_str = " • ".join([a.get("name") for a in st.session_state.top_artists[:8]])
            st.markdown(f'<div style="color:#94a3b8; font-size:0.9rem; line-height:1.6;">{artists_str}</div>', unsafe_allow_html=True)

        st.write("")
        with st.expander(f"View Synced Library Tracks ({len(st.session_state.user_tracks)} Songs)"):
            for idx, track in enumerate(st.session_state.user_tracks[:30], 1):
                col_c, col_info = st.columns([1, 10])
                with col_c:
                    if track.get("image_url"):
                        st.markdown(f'<img src="{track["image_url"]}" style="width:38px; height:38px; border-radius:4px; object-fit:cover;">', unsafe_allow_html=True)
                    else:
                        st.write("🎵")
                with col_info:
                    st.markdown(f"**{idx}. {track['title']}** — <span style='color:#94a3b8;'>{track['artist']}</span>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div style="background:#15181e; border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:20px; margin-top:12px;">
                <h4 style="color:#ffffff; margin-top:0;">Library Sync Status</h4>
                <p style="color:#94a3b8; font-size:0.9rem; line-height:1.6;">
                    No tracks were returned by Spotify for this account. Under Spotify's developer security policy, personal library endpoints (<code>/v1/me/*</code>) require an <b>Active Spotify Premium subscription</b> on the account that registered the Developer App.
                </p>
                <p style="color:#cbd5e1; font-size:0.85rem; margin-bottom:0;">
                    💡 You can load our authentic <b>Curated Sample Library</b> below to immediately test the acoustic radar profile, lyric balance sliders, and playlist ranking engine!
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        col_fb1, col_fb2 = st.columns(2)
        with col_fb1:
            if st.button("Load Curated Sample Library (15 Tracks)", use_container_width=True):
                mock_cl = get_mock_client(user_name)
                raw_tracks, top_artists, top_genres = fetch_user_library_comprehensive(mock_cl, source="all")
                st.session_state.top_artists = top_artists
                st.session_state.top_genres = top_genres
                st.session_state.user_tracks = enrich_tracks_with_shared_format(raw_tracks, lyrics_lookup=False)
                st.rerun()
        with col_fb2:
            if st.button("Switch to Instant Demo Mode", use_container_width=True):
                st.session_state.is_demo = True
                st.session_state.sp_client = get_mock_client("Alex")
                st.session_state.user_profile = None
                st.session_state.user_tracks = []
                st.rerun()


# ----------------- TAB 2: CURATOR STUDIO -----------------
with tab_studio:
    st.markdown("### **Curation Controls**")
    st.caption("Configure how the ranking model balances musical features and lyrical themes.")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("##### Hybrid Weighting")
        lyric_pct = st.slider(
            "Lyric Meaning vs. Audio Acoustics:",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            format="%d%%",
        )
        audio_pct = 100 - lyric_pct
        st.markdown(
            f'<div style="display:flex; justify-content:space-between; color:#94a3b8; font-size:0.85rem; margin-top:-6px;">'
            f'<span><b>{lyric_pct}%</b> Lyric Semantics</span>'
            f'<span><b>{audio_pct}%</b> Audio Features</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.write("")
        playlist_size = st.slider(
            "Target Playlist Size:",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
        )

    with col_s2:
        st.markdown("##### Vibe & Discovery Parameters")
        mood_selection = st.selectbox(
            "Acoustic Mood Preset:",
            options=list(MOOD_PRESETS.keys()),
            index=0,
        )

        candidate_depth = st.select_slider(
            "Candidate Search Pool Depth:",
            options=[50, 75, 100, 150, 200],
            value=100,
            help="Higher values scan a larger candidate pool across Spotify.",
        )

        diversity = st.slider(
            "Artist Diversity Constraint:",
            min_value=0.0,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Higher penalty avoids multiple tracks from the same artist.",
        )

    st.write("")
    st.markdown("---")

    if st.button("Generate Curated Playlist", use_container_width=True):
        if not st.session_state.user_tracks:
            st.error("Please sync your listening profile first.")
        else:
            prog = st.progress(0, text="Searching catalog...")

            prog.progress(25, text="Querying candidate tracks from Spotify catalog...")
            candidates = generate_candidate_pool(
                sp,
                st.session_state.user_tracks,
                st.session_state.top_artists,
                st.session_state.top_genres,
                target_pool_size=candidate_depth,
                mood_preset=mood_selection,
                lyrics_lookup=True,
            )

            prog.progress(65, text="Evaluating cosine similarity across audio and lyric vectors...")
            profile, ranked = generate_recommendations(
                st.session_state.user_tracks,
                candidates,
                n=playlist_size,
                lyric_weight=lyric_pct / 100.0,
                audio_weight=audio_pct / 100.0,
                diversity_penalty=diversity,
            )

            prog.progress(100, text="Curation complete.")
            st.session_state.taste_profile = profile
            st.session_state.ranked_tracks = ranked
            st.session_state.playlist_created = None
            st.success(f"Successfully curated {len(ranked)} tracks. Switch to the 'Generated Playlist' tab to view.")


# ----------------- TAB 3: GENERATED PLAYLIST -----------------
with tab_result:
    if not st.session_state.ranked_tracks:
        st.info("No playlist has been generated yet. Configure your preferences and click 'Generate Curated Playlist' in the Curator Studio.")
    else:
        ranked = st.session_state.ranked_tracks

        col_p1, col_p2 = st.columns([3, 2])
        with col_p1:
            pl_name = st.text_input(
                "Playlist Name:",
                value=f"Curated Blend • {mood_selection if 'mood_selection' in locals() else 'Personal'}",
            )
            pl_desc = st.text_area(
                "Description:",
                value=f"Curated playlist matching {lyric_pct if 'lyric_pct' in locals() else 50}% lyric semantics and {audio_pct if 'audio_pct' in locals() else 50}% audio acoustics.",
                height=65,
            )
        with col_p2:
            st.write("")
            st.write("")
            is_public = st.checkbox("Public Playlist", value=True)
            if st.button("Save to Spotify Account", use_container_width=True):
                with st.spinner("Exporting playlist to Spotify..."):
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

        if st.session_state.playlist_created:
            res = st.session_state.playlist_created
            st.success(f"Playlist created successfully: **{res['name']}** ({res['tracks_added']} tracks)")
            st.markdown(
                f'<a href="{res["url"]}" target="_blank" style="color:#10b981; text-decoration:none; font-weight:600;">Open in Spotify Web Player ↗</a>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(f"#### Tracklist ({len(ranked)} Tracks)")

        for idx, track in enumerate(ranked, 1):
            col_cover, col_info, col_score, col_link = st.columns([1, 6, 3, 2])

            with col_cover:
                if track.get("image_url"):
                    st.markdown(
                        f'<img src="{track["image_url"]}" style="width:46px; height:46px; border-radius:6px; object-fit:cover;">',
                        unsafe_allow_html=True,
                    )
                else:
                    st.write("🎵")

            with col_info:
                st.markdown(f"**{idx}. {track['title']}** — <span style='color:#94a3b8;'>{track['artist']}</span>", unsafe_allow_html=True)
                if track.get("lyrics_text"):
                    snippet = track["lyrics_text"].replace("\n", " ")[:60]
                    st.caption(f'"{snippet}..."')

            with col_score:
                score_val = track.get("score", 0.85)
                pct_str = f"{int(score_val * 100)}%" if score_val <= 1.0 else f"{int(score_val)}%"
                st.markdown(f'<span class="score-badge">{pct_str} Match</span>', unsafe_allow_html=True)
                st.caption(track.get("match_reason", "Acoustic Match"))

            with col_link:
                if track.get("spotify_id"):
                    track_url = f"https://open.spotify.com/track/{track['spotify_id']}"
                    st.markdown(f'<a href="{track_url}" target="_blank" style="color:#94a3b8; font-size:0.85rem; text-decoration:none;">Play on Spotify ↗</a>', unsafe_allow_html=True)

            st.markdown("<hr style='margin:4px 0; border:none; border-top:1px solid rgba(255,255,255,0.04);' />", unsafe_allow_html=True)
