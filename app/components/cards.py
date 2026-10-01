"""Reusable KPI and insight cards for CricketIQ."""

import streamlit as st
from typing import Optional

def render_metric_card(
    title: str,
    value: str,
    subtitle: Optional[str] = None,
    delta: Optional[str] = None,
    delta_color: str = "normal"
) -> None:
    """Renders a custom HTML KPI metric card."""
    card_html = f"""
    <div class="cricketiq-card">
        <div class="card-title">{title}</div>
        <div class="card-value">{value}</div>
        {f'<div class="card-subtitle">{subtitle}</div>' if subtitle else ''}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def render_trend_badge(trend: str) -> str:
    """Returns HTML badge for statistical form trend."""
    trend_lower = trend.lower()
    badge_class = f"badge-{trend_lower}" if trend_lower in ["improving", "declining", "stable", "volatile"] else "badge-stable"
    return f'<span class="{badge_class}">{trend.upper()}</span>'

def render_sample_warning(message: str = "Limited historical sample — interpret cautiously.") -> None:
    """Renders an analytical sample size caution box."""
    st.markdown(
        f'<div class="sample-warning">⚠️ <strong>Notice:</strong> {message}</div>',
        unsafe_allow_html=True
    )

def render_hero_banner(title: str, subtitle: str, badge: Optional[str] = None) -> None:
    """Renders a sleek executive hero banner at the top of a page."""
    badge_html = f'<span class="pill-tag pill-blue" style="vertical-align: middle; margin-left: 8px;">{badge}</span>' if badge else ''
    banner_html = f"""
    <div class="hero-banner">
        <div class="hero-title">{title} {badge_html}</div>
        <div class="hero-subtitle">{subtitle}</div>
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)

def render_player_header(name: str, role: str, hand: str, style: str, trend: str) -> None:
    """Renders a high-end visual dossier header for an athlete."""
    trend_badge = render_trend_badge(trend)
    header_html = f"""
    <div class="hero-banner" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #60A5FA; font-weight: 700; margin-bottom: 2px;">Player Intelligence Dossier</div>
            <div style="font-size: 2.1rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.03em; line-height: 1.1;">{name}</div>
            <div style="display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap;">
                <span class="pill-tag pill-blue">🏏 {role}</span>
                <span class="pill-tag pill-purple">🧤 {hand}</span>
                <span class="pill-tag pill-amber">🎯 {style}</span>
            </div>
        </div>
        <div style="text-align: right; background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 0.75rem 1.25rem;">
            <div style="font-size: 0.7rem; text-transform: uppercase; color: #94A3B8; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 4px;">Form Trajectory</div>
            <div>{trend_badge}</div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def render_versus_hero(batter: str, bowler: str, dominance: str, balls: int, runs: int, dismissals: int) -> None:
    """Renders a high-impact Versus confrontation card."""
    dom_color = "#34D399" if "Batter" in dominance else ("#F87171" if "Bowler" in dominance else "#60A5FA")
    html = f"""
    <div class="versus-container">
        <div class="versus-fighter" style="text-align: left;">
            <div style="font-size: 0.75rem; color: #60A5FA; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Batter</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #F8FAFC;">{batter}</div>
            <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">{runs} Runs off {balls} balls</div>
        </div>
        <div class="versus-center">
            <div style="font-size: 1.8rem; font-weight: 900; letter-spacing: -1px; color: #EF4444;">VS</div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {dom_color}; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px;">{dominance}</div>
        </div>
        <div class="versus-fighter" style="text-align: right;">
            <div style="font-size: 0.75rem; color: #F59E0B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Bowler</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #F8FAFC;">{bowler}</div>
            <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">{dismissals} Dismissals</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
