"""Interactive Plotly Visualization Library for CricketIQ."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import networkx as nx

CHART_THEME = "plotly_dark"
COLOR_PRIMARY = "#3B82F6"
COLOR_SECONDARY = "#10B981"
COLOR_ACCENT = "#F59E0B"
COLOR_DANGER = "#EF4444"

def plot_skill_radar(skill_dict: Dict[str, float], title: str = "Player Skill Profile") -> go.Figure:
    """Renders a closed radar/polar chart of a player's skill dimensions."""
    categories = list(skill_dict.keys())
    values = list(skill_dict.values())
    
    # Close the loop
    categories += [categories[0]]
    values += [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.25)',
        line=dict(color=COLOR_PRIMARY, width=2.5),
        name='Skill Score'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10, color="#9CA3AF"),
                gridcolor="#374151"
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#E5E7EB", weight="bold"),
                gridcolor="#374151"
            ),
            bgcolor="rgba(17, 24, 39, 0.8)"
        ),
        showlegend=False,
        template=CHART_THEME,
        margin=dict(l=40, r=40, t=40, b=40),
        height=380,
        title=dict(text=title, font=dict(size=14, color="#E5E7EB"))
    )
    return fig

def plot_multi_radar(radar_dict: Dict[str, Dict[str, float]]) -> go.Figure:
    """Overlays skill profiles for up to 4 players."""
    colors = ["#3B82F6", "#10B981", "#F59E0B", "#EC4899"]
    fig = go.Figure()

    for idx, (player, skills) in enumerate(radar_dict.items()):
        cats = list(skills.keys()) + [list(skills.keys())[0]]
        vals = list(skills.values()) + [list(skills.values())[0]]
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=cats,
            fill='toself',
            fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}',
            line=dict(color=color, width=2),
            name=player
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#374151"),
            angularaxis=dict(tickfont=dict(size=11, color="#E5E7EB"), gridcolor="#374151"),
            bgcolor="rgba(17, 24, 39, 0.8)"
        ),
        template=CHART_THEME,
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def plot_form_trajectory(scores: List[int], dates: List[str], player_name: str, career_avg: float) -> go.Figure:
    """Renders recent match scores with trendline and career baseline."""
    n = len(scores)
    x = list(range(1, n + 1))
    
    fig = go.Figure()
    
    # Career average baseline
    fig.add_hline(
        y=career_avg,
        line_dash="dot",
        line_color="#9CA3AF",
        annotation_text=f"Career Avg ({career_avg})",
        annotation_position="bottom right"
    )

    # Raw match scores bar
    fig.add_trace(go.Bar(
        x=x,
        y=scores,
        name="Match Score",
        marker_color="rgba(59, 130, 246, 0.5)",
        hovertext=[f"Match {i} | Date: {d} | Score: {s}" for i, (d, s) in enumerate(zip(dates, scores), 1)],
        hoverinfo="text"
    ))

    # Rolling trendline if >= 3 matches
    if n >= 3:
        rolling = pd.Series(scores).rolling(3, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=x,
            y=rolling,
            mode="lines+markers",
            name="3-Match Moving Avg",
            line=dict(color=COLOR_SECONDARY, width=3)
        ))

    fig.update_layout(
        title=f"Recent Form Trajectory ({player_name})",
        xaxis=dict(title="Recent Matches (Chronological)", tickmode="linear", tick0=1, dtick=1),
        yaxis=dict(title="Score (Runs / Wickets)"),
        template=CHART_THEME,
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_phase_breakdown(phase_df: pd.DataFrame) -> go.Figure:
    """Renders horizontal comparison of Strike Rates across Powerplay, Middle, Death."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=phase_df["phase"],
        x=phase_df["strike_rate"],
        orientation='h',
        marker=dict(
            color=phase_df["strike_rate"],
            colorscale="Viridis",
            showscale=False
        ),
        text=[f"{sr:.1f} SR ({b} balls)" for sr, b in zip(phase_df["strike_rate"], phase_df["balls"])],
        textposition="inside"
    ))

    fig.update_layout(
        title="Phase Strike Rates (PP vs Middle vs Death)",
        xaxis=dict(title="Strike Rate"),
        yaxis=dict(autorange="reversed"),
        template=CHART_THEME,
        height=260,
        margin=dict(l=40, r=20, t=40, b=30)
    )
    return fig

def plot_pace_vs_spin(df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing performance vs Pace vs Spin."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["bowler_type"],
        y=df["strike_rate"],
        name="Strike Rate",
        marker_color=["#3B82F6", "#F59E0B"],
        text=[f"{sr:.1f} SR" for sr in df["strike_rate"]],
        textposition="outside"
    ))

    fig.update_layout(
        title="Strike Rate vs Pace vs Spin",
        yaxis=dict(title="Strike Rate", range=[0, max(df["strike_rate"].max() * 1.25, 160)]),
        template=CHART_THEME,
        height=280,
        margin=dict(l=40, r=20, t=40, b=30)
    )
    return fig

def plot_shap_waterfall_bars(shap_res: Dict[str, Any]) -> go.Figure:
    """Horizontal bar chart showing local SHAP factor contributions."""
    factors = shap_res.get("factors", [])
    if not factors:
        return go.Figure()

    names = [f["feature"] for f in factors][::-1]
    impacts = [f["impact"] for f in factors][::-1]
    colors = ["#10B981" if imp >= 0 else "#EF4444" for imp in impacts]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names,
        x=impacts,
        orientation='h',
        marker_color=colors,
        text=[f"{imp:+.2f}" for imp in impacts],
        textposition="outside"
    ))

    fig.update_layout(
        title=f"SHAP Local Feature Attribution (Base: {shap_res.get('base_value', 0)} -> Pred: {shap_res.get('predicted_value', 0)})",
        xaxis=dict(title="Impact on Predicted Runs"),
        template=CHART_THEME,
        height=280,
        margin=dict(l=120, r=40, t=40, b=30)
    )
    return fig

def plot_player_embeddings_2d(emb_df: pd.DataFrame) -> go.Figure:
    """Interactive 2D PCA scatter plot colored by Archetype clusters."""
    fig = px.scatter(
        emb_df,
        x="pc1",
        y="pc2",
        color="archetype",
        hover_name="player",
        hover_data=["run_scoring", "boundary_ability", "strike_rotation", "death_skill"],
        title="2D Player Skill Manifold & Discovered Archetypes (PCA)",
        template=CHART_THEME,
        height=520
    )
    fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=1, color="white")))
    fig.update_layout(
        xaxis=dict(title="Principal Component 1 (Scoring & Aggression)"),
        yaxis=dict(title="Principal Component 2 (Phase & Rotation)"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    return fig

def plot_matchup_network(G: nx.DiGraph, center_player: str) -> go.Figure:
    """Interactive Plotly network diagram of bowler vs batter interactions."""
    pos = nx.spring_layout(G, seed=42)
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#4B5563'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []
    node_colors = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        if node == center_player:
            node_colors.append("#F59E0B")
        else:
            node_colors.append("#3B82F6")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            size=18,
            color=node_colors,
            line_width=2,
            line_color="white"
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Head-to-Head Interaction Network ({center_player})",
            showlegend=False,
            template=CHART_THEME,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=450
        )
    )
    return fig
