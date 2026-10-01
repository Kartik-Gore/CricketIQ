"""Unsupervised Player Archetype Clustering Engine."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

from src.database.connection import engine
from src.analytics.skill import SkillModel
from src.database.queries import CricketIQQueries
from src.ml.model_registry import ModelRegistry
from src.utils.logger import logger

class PlayerClusteringEngine:
    """Discovers player archetypes using unsupervised clustering on standardized skill profiles."""

    @staticmethod
    def cluster_batters(min_balls: int = 40) -> Dict[str, Any]:
        """Clusters all qualifying batters into statistical archetypes."""
        query = f"""
            SELECT batter as player, count(*) as balls
            FROM deliveries
            GROUP BY batter
            HAVING balls >= {min_balls}
        """
        qualifying = pd.read_sql(query, engine)["player"].tolist()
        if len(qualifying) < 10:
            return {}

        skill_records = []
        valid_players = []
        for p in qualifying:
            vec = SkillModel.get_batting_skill_vector(p)
            if vec and vec.get("Run Scoring", 0) > 0:
                skill_records.append(vec)
                valid_players.append(p)

        df_skills = pd.DataFrame(skill_records, index=valid_players)
        feature_cols = list(df_skills.columns)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_skills)

        # Determine optimal clusters between 3 and 5 via Silhouette Score
        best_k = 4
        best_score = -1.0
        for k in range(3, min(6, len(valid_players))):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_k = k

        # Final clustering
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        df_skills["cluster"] = kmeans.fit_predict(X_scaled)

        # Generate data-grounded archetype labels from cluster centroids
        centroids = df_skills.groupby("cluster")[feature_cols].mean()
        cluster_labels = {}
        for c in range(best_k):
            row = centroids.loc[c]
            # Archetype rules based on centroid dominance
            if row["Powerplay Skill"] > 60 and row["Boundary Ability"] > 55:
                label = "Aggressive Opener / Powerplay Striker"
            elif row["Death Skill"] > 60 and row["Boundary Ability"] > 55:
                label = "Death-Overs Finisher / Power Hitter"
            elif row["Run Scoring"] > 65 and row["Strike Rotation"] > 60:
                label = "Top-Order Anchor / Accumulator"
            elif row["Spin Skill"] > 55:
                label = "Middle-Order Spin Specialist"
            else:
                label = f"Dynamic Batter (Archetype {c+1})"
            cluster_labels[c] = label

        df_skills["archetype"] = df_skills["cluster"].map(cluster_labels)

        # Calculate metrics
        sil = round(float(silhouette_score(X_scaled, df_skills["cluster"])), 3)
        db = round(float(davies_bouldin_score(X_scaled, df_skills["cluster"])), 3)

        result = {
            "player_archetypes": df_skills[["cluster", "archetype"]].to_dict(orient="index"),
            "centroids": centroids.to_dict(orient="index"),
            "archetype_names": cluster_labels,
            "silhouette_score": sil,
            "davies_bouldin_index": db,
            "total_players_clustered": len(valid_players),
            "scaler": scaler,
            "kmeans": kmeans,
            "skills_df": df_skills
        }

        # Cache/Save to registry
        ModelRegistry.save_model(
            model_name="player_clusters",
            model_obj={
                "kmeans": kmeans,
                "scaler": scaler,
                "archetype_names": cluster_labels,
                "player_archetypes": result["player_archetypes"]
            },
            features=feature_cols,
            target="archetype_cluster",
            metrics={"silhouette_score": sil, "davies_bouldin": db}
        )

        return result
