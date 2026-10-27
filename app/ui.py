"""
Rythm Playlist - Intelligent Music Curator & Studio.
Acoustic & Semantic AI curation engine blending audio features with NLP lyric analysis.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Rythm Playlist • Intelligent Music Curator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
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
    verify_spotify_premium,
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

# ----------------- UNIFIED DESIGN SYSTEM -----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800;900&family=Syne:wght@700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', -apple-system, sans-serif;
        color: #E2E8F0;
    }

    /* Main Background: Deep dark with emerald & cyan ambient glows */
    .stApp {
        background: #08090C;
        background-image: 
            radial-gradient(circle at 10% 5%, rgba(16, 185, 129, 0.08) 0%, transparent 45%),
            radial-gradient(circle at 90% 85%, rgba(0, 229, 255, 0.05) 0%, transparent 50%),
            linear-gradient(180deg, #08090C 0%, #0B0E14 100%);
        background-attachment: fixed;
    }

    /* Centered Screen Layout for Landing & Auth */
    .view-container {
        max-width: 760px;
        margin: 0 auto;
        padding: 1.5rem 0 3rem 0;
    }

    /* ---------------- BRAND HEADER (LANDING) ---------------- */
    .rythm-brand {
        text-align: center;
        margin-bottom: 1.8rem;
    }
    .rythm-brand h1 {
        font-family: 'Syne', 'Space Grotesk', sans-serif;
        font-size: 4.6rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        line-height: 0.95;
        margin: 0;
        text-transform: uppercase;
        color: #FFFFFF;
    }
    .rythm-brand .sub-brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: 0.35em;
        text-transform: uppercase;
        margin-top: 8px;
        color: #10B981;
    }

    /* ---------------- HEADLINE & BODY TEXT ---------------- */
    .rythm-headline {
        text-align: center;
        font-family: 'Syne', 'Space Grotesk', sans-serif;
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.28;
        margin: 2rem auto 1.8rem auto;
        max-width: 660px;
        letter-spacing: -0.03em;
        color: #F8FAFC;
    }

    .rythm-body {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.02rem;
        line-height: 1.75;
        margin-bottom: 1.4rem;
        font-weight: 500;
        color: #CBD5E1;
    }

    /* Feature 3-Box Grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 2rem 0;
    }
    .feature-box {
        border: 2px solid rgba(255, 255, 255, 0.12);
        border-radius: 4px;
        padding: 14px 16px;
        text-align: left;
        background: rgba(255, 255, 255, 0.02);
    }
    .feature-box-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #10B981;
        margin-bottom: 4px;
    }
    .feature-box-desc {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.82rem;
        color: #94A3B8;
        line-height: 1.4;
    }

    /* ---------------- TRANSITION ANIMATION ---------------- */
    .transition-card {
        text-align: center;
        padding: 60px 20px;
    }
    .vinyl-box {
        width: 80px;
        height: 80px;
        margin: 0 auto 20px auto;
        border: 3px solid #10B981;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.2rem;
        animation: pulseBorder 1.2s infinite ease-in-out alternate;
    }
    @keyframes pulseBorder {
        from { transform: scale(0.95); box-shadow: 0 0 0 rgba(16, 185, 129, 0.4); }
        to { transform: scale(1.05); box-shadow: 0 0 25px rgba(16, 185, 129, 0.6); }
    }

    /* ---------------- AUTHENTICATION SCREEN ---------------- */
    .auth-box {
        text-align: center;
        padding: 20px 10px;
    }
    .spotify-badge {
        width: 92px;
        height: 92px;
        margin: 0 auto 24px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        filter: drop-shadow(0 0 24px rgba(30, 215, 96, 0.4));
    }
    .auth-title {
        font-family: 'Syne', 'Space Grotesk', sans-serif;
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        max-width: 580px;
        margin: 0 auto 16px auto;
        line-height: 1.45;
        color: #FFFFFF;
    }
    .auth-sub {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.85rem;
        color: #94A3B8;
        max-width: 560px;
        margin: 0 auto 28px auto;
        line-height: 1.55;
    }

    /* ---------------- SHARP RECTANGULAR BUTTONS (BOLD & BOXY) ---------------- */
    .stButton>button {
        font-family: 'Syne', 'Space Grotesk', sans-serif !important;
        font-weight: 900 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        border-radius: 4px !important;
        padding: 0.95rem 2rem !important;
        font-size: 1.08rem !important;
        border: 2px solid rgba(16, 185, 129, 0.35) !important;
        background: #0B132B !important;
        color: #FFFFFF !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45) !important;
    }
    .stButton>button:hover {
        border-color: #10B981 !important;
        color: #10B981 !important;
        background: #0d1b3e !important;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.45) !important;
        transform: translateY(-2px) !important;
    }
    .stButton>button:active {
        transform: translateY(0px) !important;
    }

    /* ---------------- STUDIO NAVIGATION & BRAND BAR ---------------- */
    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 24px;
        background: rgba(14, 18, 25, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 4px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }
    .brand-logo {
        font-family: 'Syne', 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-badge {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 10px;
        background: rgba(16, 185, 129, 0.1);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 4px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Workflow Stage Progress */
    .step-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(18, 23, 33, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 8px 16px;
        border-radius: 4px;
    }
    .step-pill {
        font-size: 0.8rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 4px;
        color: #64748B;
        transition: all 0.2s ease;
    }
    .step-pill.active {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.35);
        box-shadow: 0 0 14px rgba(16, 185, 129, 0.25);
    }

    /* Glass Cards */
    .neon-card {
        background: rgba(15, 20, 29, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 4px;
        padding: 24px;
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .neon-card:hover {
        border-color: rgba(16, 185, 129, 0.35);
        box-shadow: 0 12px 40px -8px rgba(16, 185, 129, 0.15);
    }

    /* Equalizer Bar Animation */
    .eq-container {
        display: inline-flex;
        align-items: flex-end;
        gap: 3px;
        height: 18px;
    }
    .eq-bar {
        width: 3px;
        background: #10B981;
        border-radius: 2px;
        animation: bounce 1s infinite alternate ease-in-out;
    }
    .eq-bar:nth-child(1) { height: 40%; animation-delay: 0.1s; }
    .eq-bar:nth-child(2) { height: 90%; animation-delay: 0.3s; }
    .eq-bar:nth-child(3) { height: 60%; animation-delay: 0.2s; }
    .eq-bar:nth-child(4) { height: 100%; animation-delay: 0.4s; }
    @keyframes bounce {
        from { transform: scaleY(0.3); }
        to { transform: scaleY(1); }
    }

    /* Metrics & Badges */
    .neon-metric {
        background: rgba(15, 20, 29, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 4px;
        padding: 16px;
        text-align: center;
    }
    .neon-metric-num {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #10B981;
        line-height: 1.1;
    }
    .neon-metric-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #94A3B8;
        letter-spacing: 0.06em;
        margin-top: 4px;
    }
    .neon-genre-tag {
        display: inline-block;
        padding: 5px 12px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 4px;
        margin: 3px 4px 3px 0;
        transition: all 0.2s ease;
    }
    .neon-genre-tag:hover {
        background: rgba(16, 185, 129, 0.1);
        border-color: rgba(16, 185, 129, 0.4);
        color: #10B981;
    }
    .neon-pill {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
    }

    /* Popover & Link Button compact styling for Tracklist (Open ↗) button */
    div[data-testid="stPopover"] > button,
    div[data-testid="stLinkButton"] > a {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        padding: 0.35rem 0.65rem !important;
        border-radius: 4px !important;
        border: 1px solid rgba(0, 229, 255, 0.4) !important;
        background: rgba(0, 229, 255, 0.08) !important;
        color: #00E5FF !important;
        text-transform: none !important;
        letter-spacing: 0.02em !important;
        text-decoration: none !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }
    div[data-testid="stPopover"] > button:hover,
    div[data-testid="stLinkButton"] > a:hover {
        border-color: #00E5FF !important;
        background: rgba(0, 229, 255, 0.18) !important;
        box-shadow: 0 0 12px rgba(0, 229, 255, 0.3) !important;
        color: #FFFFFF !important;
        text-decoration: none !important;
    }
    /* Popover floating modal card styling */
    div[data-testid="stPopoverContent"], div[data-testid="stPopoverBody"] {
        background-color: #0C1017 !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 6px !important;
        padding: 10px !important;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.85) !important;
        max-width: 250px !important;
    }

    /* Floating Tab Platform Links */
    .platform-link-row {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 7px 10px;
        border-radius: 4px;
        margin-bottom: 6px;
        text-decoration: none !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.82rem;
        font-weight: 700;
        transition: all 0.15s ease;
    }
    .platform-link-row:hover {
        transform: translateX(2px);
    }
    .apple-btn {
        background: rgba(252, 60, 68, 0.12);
        border: 1px solid rgba(252, 60, 68, 0.35);
        color: #FFFFFF !important;
    }
    .apple-btn:hover {
        background: rgba(252, 60, 68, 0.25);
        border-color: #FC3C44;
    }
    .amazon-btn {
        background: rgba(0, 168, 225, 0.12);
        border: 1px solid rgba(0, 168, 225, 0.35);
        color: #FFFFFF !important;
    }
    .amazon-btn:hover {
        background: rgba(0, 168, 225, 0.25);
        border-color: #00A8E1;
    }
    .tidal-btn {
        background: rgba(0, 229, 255, 0.12);
        border: 1px solid rgba(0, 229, 255, 0.35);
        color: #FFFFFF !important;
        margin-bottom: 6px;
    }
    .tidal-btn:hover {
        background: rgba(0, 229, 255, 0.25);
        border-color: #00E5FF;
    }
    .spotify-tab-btn {
        background: rgba(30, 215, 96, 0.12);
        border: 1px solid rgba(30, 215, 96, 0.35);
        color: #FFFFFF !important;
        margin-bottom: 6px;
    }
    .spotify-tab-btn:hover {
        background: rgba(30, 215, 96, 0.25);
        border-color: #1ED760;
    }

    /* Sidebar Navigation Container */
    [data-testid="stSidebar"] {
        background-color: #0A0D12;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    if "screen" not in st.session_state:
        st.session_state.screen = "landing"
    if "auth_method" not in st.session_state:
        st.session_state.auth_method = None
    if "platform_mode" not in st.session_state:
        st.session_state.platform_mode = "spotify"
    if "is_demo" not in st.session_state:
        st.session_state.is_demo = True
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
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Sound DNA"
    if "premium_error" not in st.session_state:
        st.session_state.premium_error = False
    if "premium_error_msg" not in st.session_state:
        st.session_state.premium_error_msg = ""


init_session_state()

# Auto-capture OAuth code if redirected directly from Spotify
if "code" in st.query_params:
    oauth_code = st.query_params.get("code")
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI", DEFAULT_REDIRECT_URI)
    if client_id and client_secret:
        sp_client = get_spotify_client(client_id, client_secret, redirect_uri, auth_code=oauth_code)
        if sp_client:
            is_prem, prem_msg = verify_spotify_premium(sp_client)
            if not is_prem:
                st.session_state.premium_error = True
                st.session_state.premium_error_msg = prem_msg
                st.session_state.auth_method = "spotify"
                st.session_state.platform_mode = "spotify"
                st.session_state.screen = "auth"
                st.session_state.sp_client = None
                st.query_params.clear()
                st.rerun()
            else:
                st.session_state.premium_error = False
                st.session_state.sp_client = sp_client
                st.session_state.is_demo = False
                st.session_state.platform_mode = "spotify"
                st.session_state.screen = "app"
                st.session_state.current_page = "Curation Studio"
                st.session_state.user_profile = get_user_profile_summary(sp_client)
                st.session_state.user_tracks = []
                st.session_state.ranked_tracks = []
                st.query_params.clear()
                st.rerun()


# =========================================================================
# SCREEN 1: LANDING & PURPOSE PAGE
# =========================================================================
if st.session_state.screen == "landing":
    landing_html = """<div class="view-container">
<div class="rythm-brand">
<h1>RYTHM</h1>
<div class="sub-brand">PLAYLIST</div>
</div>
<div class="rythm-headline">
Intelligent rhythm & mood playlist curator personalized to your taste
</div>
<div class="rythm-body">
Are you tired of choosing between a playlist that suits your current mood, and a playlist that is truly personalized to your acoustic and lyric taste? Enter your vibe and favorite artists, and <b>Rythm Playlist</b> will engineer a customized tracklist you will love.
</div>
<div class="rythm-body">
<b>Rythm Playlist</b> is an intelligent music curation engine that generates playlists tailored to both your sonic rhythm and your favorite lyric themes for <b>Spotify</b>, <b>YouTube Music</b>, or <b>Universal Playlists</b> in just a few clicks.
</div>
<div class="rythm-body">
Use our dual-modality acoustic and semantic profiler to capture your mood, then select your favorite artists from your listening rotation, and Rythm Playlist's clever personalization technology will craft a bespoke soundtrack just for you.
</div>
<div class="feature-grid">
<div class="feature-box">
<div class="feature-box-title">01 • Rhythm & Vibe</div>
<div class="feature-box-desc">Analyzes danceability, tempo, acousticness, and valence.</div>
</div>
<div class="feature-box">
<div class="feature-box-title">02 • Lyric NLP</div>
<div class="feature-box-desc">Understands storytelling and emotional lyric meanings.</div>
</div>
<div class="feature-box">
<div class="feature-box-title">03 • Multi-Platform</div>
<div class="feature-box-desc">Direct sync with Spotify, YouTube Music & Universal streaming.</div>
</div>
</div>
<div style="margin-bottom: 2rem;"></div>
</div>"""
    st.markdown(landing_html, unsafe_allow_html=True)

    c_left, c_btn, c_right = st.columns([1, 4, 1])
    with c_btn:
        if st.button("CREATE YOUR RYTHM PLAYLIST", use_container_width=True):
            st.session_state.screen = "transition"
            st.rerun()


# =========================================================================
# SCREEN 1.5: TRANSITION ANIMATION
# =========================================================================
elif st.session_state.screen == "transition":
    trans_html = """<div class="view-container">
<div class="transition-card">
<div class="vinyl-box">⚡</div>
<h2 style="font-family:'Syne', sans-serif; font-weight:900; letter-spacing:-0.02em; margin-top:14px; color:#FFFFFF;">
INITIALIZING RYTHM ENGINE...
</h2>
<p style="color:#94A3B8; font-family:'Space Grotesk', sans-serif; font-weight:500; font-size:0.95rem; margin-top:6px;">
Loading personalized authentication & streaming gateway
</p>
</div>
</div>"""
    st.markdown(trans_html, unsafe_allow_html=True)

    time.sleep(1.0)
    st.session_state.screen = "auth"
    st.rerun()


# =========================================================================
# SCREEN 2: DEDICATED AUTHENTICATION & PLATFORM SELECTION
# =========================================================================
elif st.session_state.screen == "auth":
    auth_html = """<div class="view-container">
<div class="auth-box">
<div style="display:flex; justify-content:center; align-items:center; gap:18px; margin-bottom:24px;">
    <!-- Spotify Icon -->
    <div style="width:64px; height:64px; border-radius:10px; background:rgba(30,215,96,0.12); border:1.5px solid #1ED760; display:flex; align-items:center; justify-content:center; box-shadow:0 0 20px rgba(30,215,96,0.25);">
        <svg xmlns="http://www.w3.org/2000/svg" height="38px" width="38px" viewBox="0 0 168 168">
            <path fill="#1ED760" d="m83.996 0.277c-46.249 0-83.743 37.493-83.743 83.742 0 46.251 37.494 83.741 83.743 83.741 46.254 0 83.744-37.49 83.744-83.741 0-46.246-37.49-83.738-83.745-83.738l0.001-0.004zm38.404 120.78c-1.5 2.46-4.72 3.24-7.18 1.73-19.662-12.01-44.414-14.73-73.564-8.07-2.809 0.64-5.609-1.12-6.249-3.93-0.643-2.81 1.11-5.61 3.926-6.25 31.9-7.291 59.263-4.15 81.337 9.34 2.46 1.51 3.24 4.72 1.73 7.18zm10.25-22.805c-1.89 3.075-5.91 4.045-8.98 2.155-22.51-13.839-56.823-17.846-83.448-9.764-3.453 1.043-7.1-0.903-8.148-4.35-1.04-3.453 0.907-7.093 4.354-8.143 30.413-9.228 68.222-4.758 94.072 11.127 3.07 1.89 4.04 5.91 2.15 8.976v-0.001zm0.88-23.744c-26.99-16.031-71.52-17.505-97.289-9.684-4.138 1.255-8.514-1.081-9.768-5.219-1.254-4.14 1.08-8.513 5.221-9.771 29.581-8.98 78.756-7.245 109.83 11.202 3.73 2.209 4.95 7.016 2.74 10.733-2.2 3.722-7.02 4.949-10.73 2.739z"/>
        </svg>
    </div>
    <!-- YouTube Music Icon -->
    <div style="width:64px; height:64px; border-radius:10px; background:rgba(255,0,0,0.12); border:1.5px solid #FF0000; display:flex; align-items:center; justify-content:center; box-shadow:0 0 20px rgba(255,0,0,0.25);">
        <svg xmlns="http://www.w3.org/2000/svg" height="38px" width="38px" viewBox="0 0 24 24" fill="#FF0000">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 14.5c-2.49 0-4.5-2.01-4.5-4.5S9.51 7.5 12 7.5s4.5 2.01 4.5 4.5-2.01 4.5-4.5 4.5zm-1.5-6.5v4l3.5-2-3.5-2z"/>
        </svg>
    </div>
    <!-- Universal Globe Icon -->
    <div style="width:64px; height:64px; border-radius:10px; background:rgba(168,85,247,0.12); border:1.5px solid #A855F7; display:flex; align-items:center; justify-content:center; box-shadow:0 0 20px rgba(168,85,247,0.25);">
        <svg xmlns="http://www.w3.org/2000/svg" height="38px" width="38px" viewBox="0 0 24 24" fill="#A855F7">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
        </svg>
    </div>
</div>
<div class="auth-title">
CHOOSE YOUR STREAMING PLATFORM OR ENTER UNIVERSAL MODE
</div>
<div class="auth-sub">
Connect your Spotify or YouTube Music library for deep personalized curation, or enter Universal Mode to curate across Apple Music, Amazon Music, Tidal, and beyond.
</div>
</div>
</div>"""
    st.markdown(auth_html, unsafe_allow_html=True)

    c_box_left, c_box_mid, c_box_right = st.columns([0.2, 7.6, 0.2])

    with c_box_mid:
        # 3 Rectangular Action Buttons
        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("CONNECT WITH SPOTIFY", use_container_width=True):
                st.session_state.auth_method = "spotify"
                st.session_state.platform_mode = "spotify"

        with c2:
            if st.button("CONNECT WITH YOUTUBE MUSIC", use_container_width=True):
                st.session_state.auth_method = "ytmusic"
                st.session_state.platform_mode = "ytmusic"

        with c3:
            if st.button("UNIVERSAL MODE (GUEST)", use_container_width=True):
                st.session_state.platform_mode = "universal"
                st.session_state.auth_method = "universal"
                st.session_state.is_demo = True
                st.session_state.sp_client = get_mock_client("Universal Listener")
                st.session_state.user_profile = {
                    "display_name": "Universal Listener",
                    "id": "universal_guest",
                    "product": "Universal Mode (Guest)",
                    "email": "guest@universal.music",
                }
                # Playlist starts clean/clear as requested
                st.session_state.ranked_tracks = []
                st.session_state.playlist_created = None
                st.session_state.screen = "app"
                st.session_state.current_page = "Curation Studio"
                st.rerun()

        # Drawer 1: Spotify OAuth form if user selected "CONNECT WITH SPOTIFY"
        if st.session_state.auth_method == "spotify":
            st.write("")
            st.markdown("---")

            # Sign/Notice if user attempted login without an active Spotify Premium account
            if st.session_state.get("premium_error"):
                st.markdown(
                    f"""
                    <div style="background:rgba(239, 68, 68, 0.12); border:1.5px solid #EF4444; border-radius:6px; padding:18px; margin-bottom:18px; box-shadow:0 0 24px rgba(239,68,68,0.25);">
                        <div style="display:flex; align-items:flex-start; gap:14px;">
                            <div style="font-size:2rem; line-height:1;">⚠️</div>
                            <div style="flex:1;">
                                <h4 style="color:#EF4444; font-family:'Syne', sans-serif; font-weight:800; margin:0 0 6px 0; text-transform:uppercase; letter-spacing:0.04em;">
                                    Active Spotify Premium Subscription Required
                                </h4>
                                <p style="color:#F8FAFC; font-size:0.9rem; line-height:1.5; margin:0 0 8px 0;">
                                    Spotify's Web API policy mandates that <b>only accounts with an active Spotify Premium subscription</b> can access playback, track search, and curation endpoints. Live login has been halted.
                                </p>
                                <p style="color:#FCA5A5; font-size:0.83rem; margin:0 0 12px 0; font-family:'Space Grotesk', monospace; background:rgba(0,0,0,0.35); padding:8px 12px; border-radius:4px; border:1px solid rgba(239,68,68,0.3);">
                                    {st.session_state.get('premium_error_msg') or 'Non-premium account detected. An active Spotify Premium subscription is required.'}
                                </p>
                                <p style="color:#94A3B8; font-size:0.88rem; margin:0 0 14px 0;">
                                    💡 <b>Try the Spotify Demo Account instead:</b> You don't need a Premium plan or developer credentials to use Rythm Playlist! Launch our simulated sandbox profile to explore personalized rhythm analysis, lyric profiling, and stream on any platform.
                                </p>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                col_err_demo, col_err_dismiss = st.columns([6, 4])
                with col_err_demo:
                    if st.button("⚡ LAUNCH SPOTIFY DEMO ACCOUNT INSTEAD", key="btn_premium_err_demo", use_container_width=True):
                        st.session_state.platform_mode = "spotify"
                        st.session_state.is_demo = True
                        st.session_state.sp_client = get_mock_client("Alex (Spotify Listener)")
                        st.session_state.user_profile = {
                            "display_name": "Alex (Spotify Listener)",
                            "id": "alex_spotify",
                            "product": "Spotify Premium (Demo)",
                            "email": "alex@spotify.demo",
                        }
                        st.session_state.ranked_tracks = []
                        st.session_state.playlist_created = None
                        st.session_state.premium_error = False
                        st.session_state.screen = "app"
                        st.session_state.current_page = "Curation Studio"
                        st.rerun()
                with col_err_dismiss:
                    if st.button("Dismiss & Retry Login", key="btn_premium_err_dismiss", use_container_width=True):
                        st.session_state.premium_error = False
                        st.rerun()
                st.write("")

            st.markdown(
                """
                <div class="neon-card">
                    <h4 style="font-family:'Syne', sans-serif; font-weight:800; color:#1ED760; margin-top:0;">🔐 Connect Spotify Account</h4>
                    <p style="color:#94A3B8; font-size:0.85rem; margin-bottom:14px;">Authorize via Spotify Developer App (Premium required) or test instantly with a simulated Spotify account.</p>
                """,
                unsafe_allow_html=True,
            )

            col_sp_l, col_sp_r = st.columns(2)
            with col_sp_l:
                cid = st.text_input("Spotify Client ID", value=os.getenv("SPOTIPY_CLIENT_ID", ""), type="password")
                csec = st.text_input("Spotify Client Secret", value=os.getenv("SPOTIPY_CLIENT_SECRET", ""), type="password")
                r_uri = st.text_input("Redirect URI", value=os.getenv("SPOTIPY_REDIRECT_URI", DEFAULT_REDIRECT_URI))

                if cid and csec:
                    sp_oauth = get_spotify_oauth(cid, csec, r_uri)
                    auth_url = sp_oauth.get_authorize_url()

                    st.markdown(
                        f'<a href="{auth_url}" target="_blank">'
                        f'<button style="width:100%; background:#1ED760; color:#000000; font-family:\'Syne\',sans-serif; font-weight:900; border-radius:4px; padding:11px; border:none; cursor:pointer; margin-top:6px; text-transform:uppercase; letter-spacing:0.06em; box-shadow:0 0 20px rgba(30,215,96,0.35);">'
                        f'1. Authorize via Spotify.com ↗'
                        f'</button></a>',
                        unsafe_allow_html=True,
                    )

                    auth_code = st.text_input("2. Paste Callback URL from Address Bar:", help="Copy the full redirected URL after authorizing on Spotify.")

                    if st.button("COMPLETE SPOTIFY LOGIN", use_container_width=True):
                        if auth_code:
                            code = sp_oauth.parse_response_code(auth_code) if "?" in auth_code else auth_code
                            sp_res = get_spotify_client(cid, csec, r_uri, auth_code=code)
                            if sp_res:
                                is_prem, prem_msg = verify_spotify_premium(sp_res)
                                if not is_prem:
                                    st.session_state.premium_error = True
                                    st.session_state.premium_error_msg = prem_msg
                                    st.rerun()
                                else:
                                    st.session_state.premium_error = False
                                    st.session_state.sp_client = sp_res
                                    st.session_state.is_demo = False
                                    st.session_state.platform_mode = "spotify"
                                    st.session_state.user_profile = get_user_profile_summary(sp_res)
                                    st.session_state.ranked_tracks = []
                                    st.session_state.screen = "app"
                                    st.session_state.current_page = "Curation Studio"
                                    st.success("Spotify Premium account connected successfully!")
                                    st.rerun()
                            else:
                                st.error("Authentication failed. Please check credentials or URL.")

            with col_sp_r:
                st.markdown(
                    """
                    <div style="padding:16px; background:rgba(30,215,96,0.06); border:1px solid rgba(30,215,96,0.25); border-radius:4px; text-align:center;">
                        <h5 style="color:#FFF; font-family:'Syne',sans-serif; margin-top:0;">⚡ Instant Spotify Demo</h5>
                        <p style="color:#94A3B8; font-size:0.82rem; margin-bottom:14px;">No Spotify developer credentials? Launch the fully-loaded Spotify sandbox profile.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("LAUNCH SPOTIFY DEMO ACCOUNT", use_container_width=True):
                    st.session_state.platform_mode = "spotify"
                    st.session_state.is_demo = True
                    st.session_state.sp_client = get_mock_client("Alex (Spotify Listener)")
                    st.session_state.user_profile = {
                        "display_name": "Alex (Spotify Listener)",
                        "id": "alex_spotify",
                        "product": "Spotify Premium",
                        "email": "alex@spotify.demo",
                    }
                    st.session_state.ranked_tracks = []
                    st.session_state.playlist_created = None
                    st.session_state.premium_error = False
                    st.session_state.screen = "app"
                    st.session_state.current_page = "Curation Studio"
                    st.rerun()
                    st.session_state.current_page = "Curation Studio"
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        # Drawer 2: YouTube Music form if user selected "CONNECT WITH YOUTUBE MUSIC"
        elif st.session_state.auth_method == "ytmusic":
            st.write("")
            st.markdown("---")
            st.markdown(
                """
                <div class="neon-card" style="border-color:rgba(255,0,0,0.35);">
                    <h4 style="font-family:'Syne', sans-serif; font-weight:800; color:#FF4444; margin-top:0;">▶️ Connect YouTube Music Account</h4>
                    <p style="color:#94A3B8; font-size:0.85rem; margin-bottom:14px;">Sync your YouTube Music library, listening history, and liked songs to curate tailored playlists.</p>
                """,
                unsafe_allow_html=True,
            )

            col_yt_l, col_yt_r = st.columns(2)
            with col_yt_l:
                st.markdown(
                    """
                    <div style="padding:16px; background:rgba(255,0,0,0.06); border:1px solid rgba(255,0,0,0.25); border-radius:4px; text-align:center;">
                        <h5 style="color:#FFF; font-family:'Syne',sans-serif; margin-top:0;">⚡ Instant YouTube Music Demo</h5>
                        <p style="color:#94A3B8; font-size:0.82rem; margin-bottom:14px;">Explore with a pre-configured YouTube Music profile containing curated streaming favorites.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("LAUNCH YOUTUBE MUSIC DEMO ACCOUNT", use_container_width=True):
                    st.session_state.platform_mode = "ytmusic"
                    st.session_state.is_demo = True
                    st.session_state.sp_client = get_mock_client("Jordan (YouTube Music)")
                    st.session_state.user_profile = {
                        "display_name": "Jordan (YouTube Music)",
                        "id": "ytmusic_jordan",
                        "product": "YouTube Music Premium",
                        "email": "jordan@ytmusic.demo",
                    }
                    st.session_state.ranked_tracks = []
                    st.session_state.playlist_created = None
                    st.session_state.screen = "app"
                    st.session_state.current_page = "Curation Studio"
                    st.rerun()

            with col_yt_r:
                st.markdown(
                    """
                    <div style="padding:16px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.08); border-radius:4px;">
                        <h5 style="color:#FFF; font-family:'Syne',sans-serif; margin-top:0;">🔑 Google OAuth Token Sync</h5>
                        <p style="color:#94A3B8; font-size:0.82rem; margin-bottom:10px;">Paste your YouTube Music OAuth cookie or Brand Account Token below for direct live library synchronization.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                yt_token = st.text_input("YouTube Music Token / Headers:", type="password", help="OAuth headers_auth.json or browser cookie authentication.")
                if st.button("CONNECT YOUTUBE MUSIC ACCOUNT", use_container_width=True):
                    if yt_token:
                        st.session_state.platform_mode = "ytmusic"
                        st.session_state.is_demo = False
                        st.session_state.sp_client = get_mock_client("Connected YouTube Music User")
                        st.session_state.user_profile = {
                            "display_name": "YouTube Music User",
                            "id": "ytmusic_connected",
                            "product": "YouTube Music Live",
                            "email": "user@ytmusic.com",
                        }
                        st.session_state.ranked_tracks = []
                        st.session_state.screen = "app"
                        st.session_state.current_page = "Curation Studio"
                        st.success("YouTube Music account linked!")
                        st.rerun()
                    else:
                        st.info("Please enter a token or launch the Demo Account on the left.")

            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.write("")
        if st.button("← Back to Overview"):
            st.session_state.screen = "landing"
            st.session_state.auth_method = None
            st.rerun()


# =========================================================================
# SCREEN 3: RYTHM PLAYLIST STUDIO (FULL WORKFLOW)
# =========================================================================
elif st.session_state.screen == "app":
    # Ensure client exists; otherwise send back to landing
    if st.session_state.sp_client is None:
        st.session_state.screen = "landing"
        st.rerun()

    sp = st.session_state.sp_client

    # Automatically populate profile if missing
    if not st.session_state.user_profile:
        st.session_state.user_profile = get_user_profile_summary(sp)

    profile_info = st.session_state.user_profile
    user_name = profile_info.get("display_name", "Listener")
    user_img = profile_info.get("image_url")

    # Auto-load library if empty so the app is instantly rich & interactive
    if not st.session_state.user_tracks:
        raw_tracks, top_artists, top_genres = fetch_user_library_comprehensive(sp, source="all")
        if not raw_tracks:
            mock = get_mock_client()
            raw_tracks = mock.current_user_top_tracks().get("items", [])
            if not top_artists:
                top_artists = mock.current_user_top_artists().get("items", [])
            if not top_genres:
                top_genres = extract_top_genres(top_artists)
        st.session_state.top_artists = top_artists
        st.session_state.top_genres = top_genres
        st.session_state.user_tracks = enrich_tracks_with_shared_format(raw_tracks, lyrics_lookup=False)

    # ----------------- SIDEBAR MULTI-PAGE NAVIGATION -----------------
    with st.sidebar:
        # Determine active platform aesthetics
        if st.session_state.platform_mode == "universal":
            status_text = "Universal Mode (Guest)"
            status_color = "#A855F7"
            brand_accent = "linear-gradient(135deg, #A855F7, #00E5FF)"
        elif st.session_state.platform_mode == "ytmusic":
            status_text = "YouTube Music Demo" if st.session_state.is_demo else "YouTube Music Live"
            status_color = "#FF4444"
            brand_accent = "linear-gradient(135deg, #FF4444, #CC0000)"
        else:
            status_text = "Spotify Demo" if st.session_state.is_demo else "Spotify Live"
            status_color = "#10B981"
            brand_accent = "linear-gradient(135deg, #10B981, #059669)"

        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:10px; padding:10px 0 20px 0;">
                <div style="width:36px; height:36px; border-radius:4px; background:{brand_accent}; display:flex; align-items:center; justify-content:center; color:#000; font-size:1.1rem; font-weight:800;">⚡</div>
                <div>
                    <div style="font-family:'Syne', sans-serif; font-size:1.1rem; font-weight:800; color:#fff; line-height:1.1;">Rythm Studio</div>
                    <div style="font-size:0.75rem; color:{status_color}; letter-spacing:0.04em;">NEON EDITION</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("STUDIO NAVIGATION")
        hide_export = (st.session_state.is_demo and st.session_state.platform_mode in ["spotify", "ytmusic"])

        if hide_export:
            nav_options = [
                "🧬 Sound DNA",
                "🎛️ Curation Studio",
                "💿 Live Tracklist",
            ]
            page_to_index = {
                "Sound DNA": 0,
                "Curation Studio": 1,
                "Live Tracklist": 2,
            }
        else:
            nav_options = [
                "🧬 Sound DNA",
                "🎛️ Curation Studio",
                "💿 Live Tracklist",
                "🚀 Export Playlist",
            ]
            page_to_index = {
                "Sound DNA": 0,
                "Curation Studio": 1,
                "Live Tracklist": 2,
                "Export Playlist": 3,
            }

        # Normalize current_page if it was previously set to an old or removed tab
        if st.session_state.current_page not in page_to_index:
            st.session_state.current_page = "Live Tracklist" if hide_export else ("Curation Studio" if st.session_state.platform_mode == "universal" else "Sound DNA")

        selected_nav = st.radio(
            "Navigate:",
            nav_options,
            index=page_to_index.get(st.session_state.current_page, 0),
            label_visibility="collapsed",
        )

        clean_nav = selected_nav.split(" ", 1)[1]
        if clean_nav != st.session_state.current_page:
            st.session_state.current_page = clean_nav
            st.rerun()

        st.markdown("---")

        # User Status Badge
        st.markdown(
            f"""
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:4px; padding:12px;">
                <div style="font-size:0.75rem; color:#64748B; font-weight:600; text-transform:uppercase;">Connected Listener</div>
                <div style="color:#FFF; font-weight:700; font-size:0.95rem; margin-top:2px;">{user_name}</div>
                <div style="display:flex; align-items:center; gap:6px; margin-top:6px;">
                    <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:{status_color}; box-shadow:0 0 6px {status_color};"></span>
                    <span style="font-size:0.75rem; color:{status_color}; font-weight:600;">{status_text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        if st.button("🚪 Disconnect & Exit", use_container_width=True):
            st.session_state.sp_client = None
            st.session_state.user_profile = None
            st.session_state.user_tracks = []
            st.session_state.ranked_tracks = []
            st.session_state.screen = "landing"
            st.session_state.auth_method = None
            st.rerun()

    # ----------------- MAIN BRAND BAR & WORKFLOW STAGE -----------------
    col_b1, col_b2 = st.columns([6, 4])
    with col_b1:
        st.markdown(
            f"""
            <div class="brand-bar">
                <div class="brand-logo">
                    <div class="eq-container">
                        <div class="eq-bar"></div>
                        <div class="eq-bar"></div>
                        <div class="eq-bar"></div>
                        <div class="eq-bar"></div>
                    </div>
                    Rythm Playlist
                </div>
                <div class="brand-badge">Acoustic & Semantic AI</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b2:
        if hide_export:
            st.markdown(
                f"""
                <div class="brand-bar" style="justify-content:center;">
                    <div class="step-indicator">
                        <span class="step-pill {'active' if st.session_state.current_page == 'Sound DNA' else ''}">1. DNA</span>
                        <span style="color:#334155;">→</span>
                        <span class="step-pill {'active' if st.session_state.current_page == 'Curation Studio' else ''}">2. Studio</span>
                        <span style="color:#334155;">→</span>
                        <span class="step-pill {'active' if st.session_state.current_page == 'Live Tracklist' else ''}">3. Tracks</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="brand-bar" style="justify-content:center;">
                    <div class="step-indicator">
                        <span class="step-pill {'active' if st.session_state.current_page == 'Sound DNA' else ''}">1. DNA</span>
                        <span style="color:#334155;">→</span>
                        <span class="step-pill {'active' if st.session_state.current_page == 'Curation Studio' else ''}">2. Studio</span>
                        <span style="color:#334155;">→</span>
                        <span class="step-pill {'active' if st.session_state.current_page == 'Live Tracklist' else ''}">3. Tracks</span>
                        <span style="color:#334155;">→</span>
                        <span class="step-pill {'active' if st.session_state.current_page == 'Export Playlist' else ''}">4. Export</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # PAGE 1: SOUND DNA
    # =========================================================================
    if st.session_state.current_page == "Sound DNA":
        st.markdown(
            f"""
            <div style="margin-bottom:20px;">
                <h2 style="font-family:'Syne', sans-serif; font-size:2rem; font-weight:800; color:#FFFFFF; margin:0; letter-spacing:-0.03em;">
                    Acoustic Identity • <span style="color:#10B981;">{user_name}</span>
                </h2>
                <p style="color:#94A3B8; font-size:0.95rem; margin-top:4px;">
                    Your unique sonic fingerprint synthesized from listening history and saved libraries.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Stat Row
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(
                f"""
                <div class="neon-metric">
                    <div class="neon-metric-num">{len(st.session_state.user_tracks)}</div>
                    <div class="neon-metric-label">Analyzed Tracks</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                f"""
                <div class="neon-metric">
                    <div class="neon-metric-num">{len(st.session_state.top_artists)}</div>
                    <div class="neon-metric-label">Seed Artists</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s3:
            st.markdown(
                f"""
                <div class="neon-metric">
                    <div class="neon-metric-num">{len(st.session_state.top_genres)}</div>
                    <div class="neon-metric-label">Identified Genres</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s4:
            st.markdown(
                f"""
                <div class="neon-metric">
                    <div class="neon-metric-num" style="color:#10B981;">Ready</div>
                    <div class="neon-metric-label">Vibe Match Engine</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # Visualizer & Breakdown Columns
        c_rad, c_break = st.columns([5, 5])

        with c_rad:
            st.markdown(
                """
                <div class="neon-card">
                    <h4 style="font-family:'Syne', sans-serif; font-weight:800; color:#FFF; margin-top:0;">⚡ Multi-Axis Acoustic Radar</h4>
                    <p style="color:#64748B; font-size:0.85rem; margin-bottom:12px;">Measurement across core musical vibe dimensions.</p>
                """,
                unsafe_allow_html=True,
            )
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
                        line=dict(color="#10B981", width=2.5),
                        marker=dict(color="#00F0FF", size=7, line=dict(color="#08090C", width=1.5)),
                    )
                )
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, linecolor="rgba(255,255,255,0.08)"),
                        angularaxis=dict(color="#94A3B8", gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=11, family="Space Grotesk")),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=30, r=30, t=20, b=20),
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.bar_chart(feature_avgs)

            st.markdown("</div>", unsafe_allow_html=True)

        with c_break:
            st.markdown(
                """
                <div class="neon-card">
                    <h4 style="font-family:'Syne', sans-serif; font-weight:800; color:#FFF; margin-top:0;">🎸 Genre Network & Frequent Artists</h4>
                    <p style="color:#64748B; font-size:0.85rem; margin-bottom:16px;">Core musical clusters driving your recommendations.</p>
                """,
                unsafe_allow_html=True,
            )
            if st.session_state.top_genres:
                genre_badges = "".join([
                    f'<span class="neon-genre-tag">{g.title()} • {c}</span>'
                    for g, c in st.session_state.top_genres[:14]
                ])
                st.markdown(genre_badges, unsafe_allow_html=True)

            st.write("")
            st.markdown("##### 🎤 Seed Artist Network")
            artists_str = " • ".join([a.get("name") for a in st.session_state.top_artists[:10]])
            st.markdown(f'<div style="color:#94A3B8; font-size:0.9rem; line-height:1.7;">{artists_str}</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # Next Stage CTA
        st.write("")
        col_cta1, col_cta2 = st.columns([8, 2])
        with col_cta2:
            if st.button("Enter Studio →", use_container_width=True):
                st.session_state.current_page = "Curation Studio"
                st.rerun()

    # =========================================================================
    # PAGE 2: CURATION STUDIO
    # =========================================================================
    elif st.session_state.current_page == "Curation Studio":
        st.markdown(
            """
            <div style="margin-bottom:20px;">
                <h2 style="font-family:'Syne', sans-serif; font-size:2rem; font-weight:800; color:#FFFFFF; margin:0; letter-spacing:-0.03em;">
                    🎛️ The Curation Lab
                </h2>
                <p style="color:#94A3B8; font-size:0.95rem; margin-top:4px;">
                    Fine-tune the balance between lyric semantics, acoustic energy, and artist diversity.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_ctrl1, col_ctrl2 = st.columns([1, 1])

        with col_ctrl1:
            st.markdown(
                """
                <div class="neon-card">
                    <h4 style="font-family:'Syne', sans-serif; font-weight:800; color:#FFF; margin-top:0;">⚖️ Dual-Modality Fusion Balance</h4>
                    <p style="color:#94A3B8; font-size:0.85rem; margin-bottom:18px;">
                        Slide to decide whether the engine matches what songs <b>sound like</b> or what their lyrics <b>mean</b>.
                    </p>
                """,
                unsafe_allow_html=True,
            )

            lyric_weight = st.slider(
                "Lyric Semantics vs. Audio Acoustics:",
                min_value=0,
                max_value=100,
                value=50,
                step=5,
                format="%d%%",
            )
            audio_weight = 100 - lyric_weight

            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; margin-top:8px; padding:10px 14px; background:rgba(0,0,0,0.3); border-radius:4px; border:1px solid rgba(255,255,255,0.08);">
                    <span style="color:#00F0FF; font-weight:700; font-size:0.85rem;">💬 {lyric_weight}% Lyric Emotion</span>
                    <span style="color:#10B981; font-weight:700; font-size:0.85rem;">🎵 {audio_weight}% Acoustic Vibe</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            target_size = st.slider(
                "Desired Playlist Track Count:",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
            )

            st.markdown("</div>", unsafe_allow_html=True)

        with col_ctrl2:
            st.markdown(
                """
                <div class="neon-card">
                    <h4 style="font-family:'Syne', sans-serif; font-weight:800; color:#FFF; margin-top:0;">🎭 Mood & Discovery Matrix</h4>
                    <p style="color:#94A3B8; font-size:0.85rem; margin-bottom:18px;">
                        Select an atmospheric trajectory and set exploration bounds.
                    </p>
                """,
                unsafe_allow_html=True,
            )

            selected_mood = st.selectbox(
                "Select Vibe Preset:",
                options=list(MOOD_PRESETS.keys()),
                index=0,
            )

            pool_depth = st.select_slider(
                "Discovery Depth (Candidate Pool):",
                options=[50, 75, 100, 150, 200],
                value=100,
                help="Higher pool scans deeper artist networks across the Spotify catalog.",
            )

            diversity = st.slider(
                "Artist Monopolization Penalty:",
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Higher values guarantee no single artist dominates the playlist.",
            )

            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown("---")

        # Generate CTA
        col_g1, col_g2 = st.columns([3, 2])
        with col_g2:
            if st.button("⚡ Synthesize & Curate Tracklist", use_container_width=True):
                try:
                    with st.status("Curating your custom mix...", expanded=True) as status:
                        st.write("🔍 Exploring catalog with multi-angle candidate generator...")
                        candidates = generate_candidate_pool(
                            sp,
                            st.session_state.user_tracks,
                            st.session_state.top_artists,
                            st.session_state.top_genres,
                            target_pool_size=pool_depth,
                            mood_preset=selected_mood,
                            lyrics_lookup=True,
                        )

                        st.write("🧠 Computing Cosine Similarity across audio & lyric embeddings...")
                        profile, ranked = generate_recommendations(
                            st.session_state.user_tracks,
                            candidates,
                            n=target_size,
                            lyric_weight=lyric_weight / 100.0,
                            audio_weight=audio_weight / 100.0,
                            diversity_penalty=diversity,
                        )

                        st.session_state.taste_profile = profile
                        st.session_state.ranked_tracks = ranked
                        st.session_state.playlist_created = None
                        status.update(label="✨ Mix Generation Complete!", state="complete", expanded=False)

                    st.session_state.current_page = "Live Tracklist"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during curation: {e}")

    # =========================================================================
    # PAGE 3: LIVE TRACKLIST
    # =========================================================================
    elif st.session_state.current_page == "Live Tracklist":
        st.markdown(
            """
            <div style="margin-bottom:20px;">
                <h2 style="font-family:'Syne', sans-serif; font-size:2rem; font-weight:800; color:#FFFFFF; margin:0; letter-spacing:-0.03em;">
                    💿 Curated Tracklist
                </h2>
                <p style="color:#94A3B8; font-size:0.95rem; margin-top:4px;">
                    Review your personalized tracks, explore match confidence, and preview songs.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not st.session_state.ranked_tracks:
            st.markdown(
                """
                <div class="neon-card" style="text-align:center; padding:40px 20px;">
                    <div style="font-size:2.5rem; margin-bottom:12px;">🎛️</div>
                    <h3 style="color:#FFF; margin-top:0;">No Tracks Curated Yet</h3>
                    <p style="color:#94A3B8; font-size:0.9rem;">Configure your sound preferences and click 'Synthesize & Curate Tracklist' in the Studio.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("← Go to Curation Studio"):
                st.session_state.current_page = "Curation Studio"
                st.rerun()
        else:
            ranked = st.session_state.ranked_tracks

            if not hide_export:
                col_h_left, col_h_right = st.columns([7, 3])
                with col_h_left:
                    st.markdown(f"##### Recommended Tracks ({len(ranked)} Songs)")
                with col_h_right:
                    if st.button("Proceed to Export →", use_container_width=True):
                        st.session_state.current_page = "Export Playlist"
                        st.rerun()
            else:
                st.markdown(f"##### Recommended Tracks ({len(ranked)} Songs)")

            st.write("")

            # Render Track Tiles
            for idx, track in enumerate(ranked, 1):
                score_val = track.get("score", 0.85)
                pct = int(score_val * 100) if score_val <= 1.0 else int(score_val)
                cover = track.get("image_url") or ""
                track_url = f"https://open.spotify.com/track/{track.get('spotify_id')}" if track.get("spotify_id") else "#"
                snippet = track.get("lyrics_text", "").replace("\n", " ")[:65] if track.get("lyrics_text") else ""

                col_c1, col_c2, col_c3, col_c4 = st.columns([1, 6, 2, 2])

                with col_c1:
                    if cover:
                        st.markdown(f'<img src="{cover}" style="width:48px; height:48px; border-radius:4px; object-fit:cover; border:1px solid rgba(255,255,255,0.1);">' , unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="width:48px; height:48px; border-radius:4px; background:#1E293B; display:flex; align-items:center; justify-content:center; color:#10B981;">🎵</div>', unsafe_allow_html=True)

                with col_c2:
                    st.markdown(f"**{idx}. {track.get('title')}** — <span style='color:#94A3B8;'>{track.get('artist')}</span>", unsafe_allow_html=True)
                    if snippet:
                        st.caption(f'"{snippet}..."')

                with col_c3:
                    st.markdown(f'<span class="neon-pill">{pct}% Match</span>', unsafe_allow_html=True)
                    st.caption(track.get("match_reason", "Vibe Match"))

                with col_c4:
                    track_query = urllib.parse.quote_plus(f"{track.get('artist', '')} {track.get('title', '')}")

                    if st.session_state.platform_mode == "spotify":
                        # Spotify Mode (Demo vs Logged In):
                        if st.session_state.is_demo:
                            # In Demo Mode, use verified Spotify Search URL so it always dynamically finds
                            # and displays the song with play controls in the user's country without 404 errors.
                            sp_web_url = f"https://open.spotify.com/search/{track_query}"
                            btn_label = "(Open ↗)"
                        else:
                            # Real authenticated Spotify account with user-specific catalog IDs
                            sp_web_url = track_url if (track.get("spotify_id") and track_url != "#") else f"https://open.spotify.com/search/{track_query}"
                            btn_label = "Play on Spotify ↗"

                        st.link_button(btn_label, url=sp_web_url, key=f"sp_link_{idx}", use_container_width=True)

                    elif st.session_state.platform_mode == "ytmusic":
                        # YouTube Music Mode:
                        # Redirects immediately to YouTube Music
                        yt_url = f"https://music.youtube.com/search?q={track_query}"
                        btn_label = "(Open ↗)" if st.session_state.is_demo else "Play on YouTube Music ↗"
                        st.link_button(btn_label, url=yt_url, key=f"yt_link_{idx}", use_container_width=True)

                    else:
                        # Universal Mode (Guest):
                        # Multi-platform floating popover with Apple Music, Amazon Music, and Tidal
                        with st.popover("(Open ↗)", use_container_width=True):
                            st.markdown(
                                """<div style="font-family:'Syne', sans-serif; font-size:0.75rem; font-weight:800; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;">
Open On Streaming Platform
</div>""",
                                unsafe_allow_html=True,
                            )
                            apple_url = f"https://music.apple.com/search?term={track_query}"
                            amazon_url = f"https://music.amazon.com/search/{track_query}"
                            tidal_url = f"https://listen.tidal.com/search?q={track_query}"

                            st.markdown(
                                f"""<a href="{apple_url}" target="_blank" class="platform-link-row apple-btn">
<svg width="18" height="18" viewBox="0 0 24 24" fill="#FC3C44"><path d="M18.71 19.5c-.83 1.24-1.71 2.45-3.05 2.47-1.34.03-1.77-.79-3.29-.79-1.53 0-2 .77-3.27.82-1.31.05-2.3-1.32-3.14-2.53C4.25 17 2.94 12.45 4.7 9.39c.87-1.52 2.43-2.48 4.12-2.51 1.28-.02 2.5.87 3.29.87.78 0 2.26-1.07 3.81-.91.65.03 2.47.26 3.64 1.98-.09.06-2.17 1.28-2.15 3.81.03 3.02 2.65 4.03 2.68 4.04-.03.07-.42 1.44-1.38 2.83M15.97 6.37c.64-.78 1.08-1.86.96-2.95-1 .04-2.18.67-2.84 1.44-.57.66-1.08 1.74-.94 2.81 1.11.09 2.23-.58 2.82-1.3"/></svg>
<span>Apple Music ↗</span>
</a>""",
                                unsafe_allow_html=True,
                            )

                            st.markdown(
                                f"""<a href="{amazon_url}" target="_blank" class="platform-link-row amazon-btn">
<svg width="18" height="18" viewBox="0 0 24 24" fill="#00A8E1"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14.5v-9l6 4.5-6 4.5z"/></svg>
<span>Amazon Music ↗</span>
</a>""",
                                unsafe_allow_html=True,
                            )

                            st.markdown(
                                f"""<a href="{tidal_url}" target="_blank" class="platform-link-row tidal-btn">
<svg width="18" height="18" viewBox="0 0 24 24" fill="#00E5FF"><path d="M12.012 3.992L8.008 8l4.004 4.008L16.016 8l-4.004-4.008zM4.004 8L0 12.008 4.004 16.016 8.008 12.008 4.004 8zm16.008 0l-4.004 4.008 4.004 4.008L24 12.008 20.012 8zm-8 8.016l-4.004 4.008 4.004 4.008 4.004-4.008-4.004-4.008z"/></svg>
<span>Tidal ↗</span>
</a>""",
                                unsafe_allow_html=True,
                            )

                st.markdown("<hr style='margin:4px 0 10px 0; border:none; border-top:1px solid rgba(255,255,255,0.06);' />", unsafe_allow_html=True)

    # =========================================================================
    # PAGE 4: EXPORT PLAYLIST
    # =========================================================================
    elif st.session_state.current_page == "Export Playlist":
        if hide_export:
            st.session_state.current_page = "Live Tracklist"
            st.rerun()

        if st.session_state.platform_mode == "universal":
            header_title = "🌐 Universal Playlist Export"
            header_desc = "Export your curated tracks via M3U8 or CSV to transfer into Apple Music, Amazon Music, Tidal, or media players."
        elif st.session_state.platform_mode == "ytmusic":
            header_title = "▶️ Export to YouTube Music"
            header_desc = "Transfer your curated playlist into YouTube Music or download portable playlist files."
        else:
            header_title = "🚀 Export to Spotify"
            header_desc = "Publish your curated blend directly into your Spotify music library in one click."

        st.markdown(
            f"""
            <div style="margin-bottom:20px;">
                <h2 style="font-family:'Syne', sans-serif; font-size:2rem; font-weight:800; color:#FFFFFF; margin:0; letter-spacing:-0.03em;">
                    {header_title}
                </h2>
                <p style="color:#94A3B8; font-size:0.95rem; margin-top:4px;">
                    {header_desc}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not st.session_state.ranked_tracks:
            st.info("Please generate tracks in the Curation Studio first.")
            if st.button("Go to Studio"):
                st.session_state.current_page = "Curation Studio"
                st.rerun()
        else:
            col_ex1, col_ex2 = st.columns([6, 4])

            with col_ex1:
                st.markdown(
                    """
                    <div class="neon-card">
                        <h4 style="font-family:'Syne', sans-serif; font-weight:800; color:#FFF; margin-top:0;">📝 Playlist Metadata & Transfer</h4>
                    """,
                    unsafe_allow_html=True,
                )
                pl_name = st.text_input("Playlist Name:", value=f"Rythm Playlist • {user_name}'s Vibe")
                pl_desc = st.text_area(
                    "Description:",
                    value="Curated mix blending acoustic features with semantic lyric storytelling by Rythm Playlist.",
                    height=70,
                )

                if st.session_state.platform_mode == "spotify":
                    is_public = st.checkbox("Make Playlist Public", value=True)
                    if st.button("💚 Publish to Spotify Now", use_container_width=True):
                        with st.spinner("Creating playlist in your Spotify account..."):
                            res = export_ranked_playlist_to_spotify(
                                sp,
                                user_id=profile_info.get("id", "me"),
                                ranked_tracks=st.session_state.ranked_tracks,
                                playlist_name=pl_name,
                                description=pl_desc,
                                public=is_public,
                            )
                            st.session_state.playlist_created = res
                            st.balloons()
                elif st.session_state.platform_mode == "ytmusic":
                    st.write("")
                    col_yt_b1, col_yt_b2 = st.columns(2)
                    with col_yt_b1:
                        lead_track = st.session_state.ranked_tracks[0] if st.session_state.ranked_tracks else {}
                        yt_lead = urllib.parse.quote_plus(f"{lead_track.get('artist', '')} {lead_track.get('title', '')}")
                        yt_search_url = f"https://music.youtube.com/search?q={yt_lead}"
                        st.markdown(
                            f'<a href="{yt_search_url}" target="_blank" style="text-decoration:none;">'
                            f'<button style="width:100%; background:#FF0000; color:#FFF; font-family:\'Syne\',sans-serif; font-weight:800; border-radius:4px; padding:12px; border:none; cursor:pointer;">'
                            f'▶️ Launch in YouTube Music ↗'
                            f'</button></a>',
                            unsafe_allow_html=True,
                        )
                    with col_yt_b2:
                        import csv
                        import io
                        csv_buf = io.StringIO()
                        writer = csv.writer(csv_buf)
                        writer.writerow(["Title", "Artist", "Score"])
                        for t in st.session_state.ranked_tracks:
                            writer.writerow([t.get("title", ""), t.get("artist", ""), t.get("score", "")])
                        st.download_button(
                            "📄 Download CSV for YouTube Music",
                            data=csv_buf.getvalue(),
                            file_name=f"rythm_ytmusic_{user_name.lower().replace(' ', '_')}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                else:  # universal mode
                    st.write("")
                    m3u_lines = ["#EXTM3U"]
                    for t in st.session_state.ranked_tracks:
                        m3u_lines.append(f"#EXTINF:-1,{t.get('artist', 'Unknown')} - {t.get('title', 'Unknown')}")
                        m3u_lines.append(f"https://open.spotify.com/track/{t.get('spotify_id', '')}")
                    m3u_data = "\n".join(m3u_lines)

                    import csv
                    import io
                    csv_buf = io.StringIO()
                    writer = csv.writer(csv_buf)
                    writer.writerow(["Title", "Artist", "Match Score", "Spotify ID"])
                    for t in st.session_state.ranked_tracks:
                        writer.writerow([t.get("title", ""), t.get("artist", ""), t.get("score", ""), t.get("spotify_id", "")])
                    csv_data = csv_buf.getvalue()

                    col_u1, col_u2 = st.columns(2)
                    with col_u1:
                        st.download_button(
                            "📥 Download .M3U8 Playlist",
                            data=m3u_data,
                            file_name=f"rythm_universal_{user_name.lower().replace(' ', '_')}.m3u8",
                            mime="audio/x-mpegurl",
                            use_container_width=True,
                            help="Compatible with Apple Music, Tidal, VLC, and local media players."
                        )
                    with col_u2:
                        st.download_button(
                            "📄 Download .CSV Tracklist",
                            data=csv_data,
                            file_name=f"rythm_universal_{user_name.lower().replace(' ', '_')}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            help="Directly importable into TuneMyMusic or Soundiiz for Apple Music, Amazon Music & Tidal."
                        )

                    st.markdown(
                        """
                        <div style="margin-top:14px; padding:12px; background:rgba(168,85,247,0.08); border:1px solid rgba(168,85,247,0.25); border-radius:4px;">
                            <div style="color:#FFF; font-weight:700; font-size:0.85rem; margin-bottom:4px;">🔄 1-Click Transfer to Apple Music, Amazon Music & Tidal:</div>
                            <div style="color:#CBD5E1; font-size:0.8rem; line-height:1.5;">
                                1. Download the <b>.CSV Tracklist</b> above.<br/>
                                2. Open <a href="https://www.tunemymusic.com" target="_blank" style="color:#A855F7; font-weight:700;">TuneMyMusic ↗</a> or <a href="https://soundiiz.com" target="_blank" style="color:#A855F7; font-weight:700;">Soundiiz ↗</a>.<br/>
                                3. Select CSV as source, and Apple Music, Amazon Music, or Tidal as destination!
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

            with col_ex2:
                accent_grad = "linear-gradient(135deg, #A855F7 0%, #00F0FF 100%)" if st.session_state.platform_mode == "universal" else ("linear-gradient(135deg, #FF4444 0%, #FF8800 100%)" if st.session_state.platform_mode == "ytmusic" else "linear-gradient(135deg, #10B981 0%, #00F0FF 100%)")
                st.markdown(
                    f"""
                    <div class="neon-card" style="text-align:center;">
                        <div style="width:120px; height:120px; margin:0 auto 16px auto; border-radius:4px; background:{accent_grad}; display:flex; align-items:center; justify-content:center; color:#08090C; font-size:3rem; font-weight:800; box-shadow:0 0 30px rgba(16,185,129,0.35);">⚡</div>
                        <h4 style="color:#FFF; margin:0; font-family:'Syne', sans-serif; font-weight:800;">{pl_name}</h4>
                        <p style="color:#64748B; font-size:0.85rem; margin-top:4px;">{len(st.session_state.ranked_tracks)} Curated Tracks</p>
                    """,
                    unsafe_allow_html=True,
                )

                if st.session_state.playlist_created:
                    res = st.session_state.playlist_created
                    st.success("🎉 Successfully Published!")
                    st.markdown(
                        f'<a href="{res.get("url")}" target="_blank" style="display:inline-block; margin-top:10px; background:#10B981; color:#000; font-family:\'Syne\',sans-serif; font-weight:800; padding:10px 20px; border-radius:4px; text-decoration:none; text-transform:uppercase; letter-spacing:0.06em;">Open in Spotify Web Player ↗</a>',
                        unsafe_allow_html=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)
