"""Unit tests for ML models, prediction intervals, SHAP, and similarity."""

import pytest
from src.ml.batting_model import BattingModelTrainer
from src.ml.bowling_model import BowlingModelTrainer
from src.ml.similarity import PlayerSimilarityEngine
from src.explainability.shap_analysis import SHAPExplanationEngine
from src.simulation.what_if import ScenarioSimulator

def test_batting_prediction():
    pred = BattingModelTrainer.predict_runs("V Kohli")
    assert "expected_runs" in pred
    assert pred["expected_runs"] > 0
    assert pred["lower_bound"] <= pred["expected_runs"] <= pred["upper_bound"]

def test_bowling_prediction():
    pred = BowlingModelTrainer.predict_wickets("JJ Bumrah")
    assert "expected_wickets" in pred
    assert pred["expected_wickets"] >= 0
    assert pred["lower_bound"] <= pred["expected_wickets"] <= pred["upper_bound"]

def test_shap_explanation():
    pred = BattingModelTrainer.predict_runs("V Kohli")
    shap_res = SHAPExplanationEngine.explain_batting_prediction(pred["input_features"])
    assert "base_value" in shap_res
    assert "factors" in shap_res
    assert len(shap_res["factors"]) > 0

def test_similarity_search():
    sims = PlayerSimilarityEngine.find_similar_players("V Kohli", top_n=2)
    assert len(sims) == 2
    assert "similarity_score" in sims[0]

def test_scenario_simulator():
    sim = ScenarioSimulator.simulate_batting_scenario(
        "V Kohli", venue="Wankhede Stadium, Mumbai", opposition="Mumbai Indians"
    )
    assert sim["expected_runs"] > 0
    assert "range_str" in sim
