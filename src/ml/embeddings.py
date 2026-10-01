"""2D Player Embeddings using Principal Component Analysis (PCA)."""

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.database.connection import engine
from src.analytics.skill import SkillModel
from src.ml.clustering import PlayerClusteringEngine

class PlayerEmbeddingEngine:
    """Projects high-dimensional cricket skill spaces into 2D manifolds for visual exploration."""

    @staticmethod
    def get_2d_player_embeddings(min_balls: int = 40) -> pd.DataFrame:
        """Computes 2D PCA projection for all qualifying players with archetypes."""
        cluster_res = PlayerClusteringEngine.cluster_batters(min_balls=min_balls)
        if not cluster_res or "skills_df" not in cluster_res:
            return pd.DataFrame()

        skills_df = cluster_res["skills_df"].copy()
        feature_cols = [c for c in skills_df.columns if c not in ["cluster", "archetype"]]

        X = skills_df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)

        emb_df = pd.DataFrame({
            "player": skills_df.index,
            "pc1": np.round(coords[:, 0], 2),
            "pc2": np.round(coords[:, 1], 2),
            "cluster": skills_df["cluster"].values,
            "archetype": skills_df["archetype"].values,
            "run_scoring": skills_df["Run Scoring"].values,
            "boundary_ability": skills_df["Boundary Ability"].values,
            "strike_rotation": skills_df["Strike Rotation"].values,
            "death_skill": skills_df["Death Skill"].values
        })

        var_explained = [round(float(v) * 100.0, 1) for v in pca.explained_variance_ratio_]
        emb_df.attrs["variance_explained"] = var_explained

        return emb_df
