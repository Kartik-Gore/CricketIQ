"""Player Similarity Search Engine using Normalized Cosine Metrics."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from src.database.connection import engine
from src.analytics.skill import SkillModel
from src.database.queries import CricketIQQueries

class PlayerSimilarityEngine:
    """Finds statistically similar players using cosine similarity across normalized skill spaces."""

    @staticmethod
    def find_similar_players(player_name: str, top_n: int = 4) -> List[Dict[str, Any]]:
        """Identifies most comparable players in the database."""
        # 1. Fetch target player skill vector
        target_vec = SkillModel.get_batting_skill_vector(player_name)
        if not target_vec or sum(target_vec.values()) == 0:
            return []

        # 2. Fetch all qualifying players
        query = """
            SELECT batter as player, count(*) as balls
            FROM deliveries
            GROUP BY batter
            HAVING balls >= 35
        """
        qualifying = pd.read_sql(query, engine)["player"].tolist()
        if player_name not in qualifying:
            qualifying.append(player_name)

        players_list = []
        vectors = []
        for p in qualifying:
            vec = SkillModel.get_batting_skill_vector(p)
            if vec and vec.get("Run Scoring", 0) > 0:
                players_list.append(p)
                vectors.append(list(vec.values()))

        if len(players_list) < 2:
            return []

        skill_keys = list(target_vec.keys())
        df_vecs = pd.DataFrame(vectors, index=players_list, columns=skill_keys)

        # Standardize for distance / similarity
        scaler = StandardScaler()
        scaled_vecs = scaler.fit_transform(df_vecs)

        # Calculate cosine similarity matrix
        sim_matrix = cosine_similarity(scaled_vecs)
        sim_df = pd.DataFrame(sim_matrix, index=players_list, columns=players_list)

        if player_name not in sim_df.index:
            return []

        similar_scores = sim_df[player_name].drop(player_name).sort_values(ascending=False).head(top_n)

        results = []
        for peer, sim_score in similar_scores.items():
            # Percentage similarity (mapped to ~50% - 99%)
            norm_sim = round(float(np.clip((sim_score + 1.0) / 2.0 * 100.0, 40.0, 99.5)), 1)
            peer_vec = df_vecs.loc[peer].to_dict()

            # Find top matching dimensions and divergence
            diffs = {k: abs(target_vec[k] - peer_vec[k]) for k in skill_keys}
            closest_traits = sorted(diffs.items(), key=lambda x: x[1])[:2]
            furthest_traits = sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:2]

            results.append({
                "player": peer,
                "similarity_score": norm_sim,
                "raw_cosine": round(float(sim_score), 3),
                "key_similarities": [f"{k} (diff {v:.1f})" for k, v in closest_traits],
                "key_differences": [f"{k} (diff {v:.1f})" for k, v in furthest_traits],
                "peer_skills": peer_vec
            })

        return results
