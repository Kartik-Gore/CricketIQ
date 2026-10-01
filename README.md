# 🏏 CricketIQ: Advanced Cricket Intelligence Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn & XGBoost](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest-green.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP%20TreeExplainer-red.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**CricketIQ** is an enterprise-grade cricket analytics, player intelligence, and predictive simulation platform built in Python. Designed for team analysts, high-performance coaches, scouts, and cricket data scientists, CricketIQ transforms historical ball-by-ball delivery feeds into actionable micro-matchup insights, multidimensional player skill vectors, dynamic form trajectories, and explainable machine learning predictions.

---

## 1. Project Overview
Traditional cricket dashboards are often passive record books that aggregate basic counting statistics (career runs, wickets, raw strike rates) without contextual grounding. CricketIQ goes beyond static statistics by modeling the nuanced realities of T20 cricket:
- **Phase dynamics:** How batter scoring velocity shifts across Powerplay (overs 0–5), Middle (overs 6–14), and Death (overs 15–19).
- **Match pressure proxies:** Quantifying the situational leverage based on Required Run Rate (RRR), wickets remaining, and chase target.
- **Matchup micro-interactions:** Head-to-head batter vs. bowler encounters, pace vs. spin proficiency, and handedness splits.
- **Explainable forecasting:** Predicting expected runs and wickets with residual quantile prediction intervals and game-theoretic SHAP local attributions.

---

## 2. Problem Statement
Cricket performance is non-linear and intensely context-dependent. A batter scoring 35 runs off 20 balls in a high-pressure death-overs chase against elite pace bowling is significantly more impactful than scoring 45 runs off 40 balls on a flat batting pitch with no scoreboard pressure. Existing platforms frequently suffer from:
1. **Lack of contextual adjustment:** Ignoring stadium dimensions, pitch friction, and bowling quality.
2. **Data leakage in predictive models:** Training time-series models on shuffled data containing future matches.
3. **Black-box opacity:** Producing point predictions without confidence intervals or explainability.
4. **Simplistic form definitions:** Relying solely on raw 5-match moving averages without exponential time-decay.

---

## 3. Objectives
- **End-to-End Modular Architecture:** Establish separate, testable layers for ingestion, validation, relational database storage, feature extraction, ML training, explainability, and user presentation.
- **Strict Data Quality & Governance:** Provide automated validation gates that flag anomalous deliveries, duplicate records, and impossible scoring events.
- **Zero Data Leakage:** Implement strictly chronological, time-aware feature generation and temporal train/validation/test splits.
- **Transparent AI:** Ensure every predictive estimate is accompanied by uncertainty intervals and game-theoretic SHAP attributions.
- **Interactive Decision Support:** Deliver an intuitive command center with counterfactual scenario simulation (What-If analysis) and NetworkX matchup interaction graphs.

---

## 4. Key Features
- **🏏 360-Degree Player Dossier:** 10-dimensional radar skill profile, career records, pace vs. spin breakdowns, and top nemesis tracking.
- **📈 Dynamic Form Modeling:** Exponentially time-weighted recent performance with statistical trend classification (*Improving*, *Stable*, *Declining*, *Volatile*).
- **⚔️ Matchup Engine & Network Graph:** Head-to-head batter vs. bowler historical encounters with sample size alerts and directed NetworkX interaction graphs.
- **🏟️ Venue Intelligence & Stadium Index:** Stadium rankings by Venue Difficulty Index (VDI), 1st vs. 2nd innings run averages, and pace/spin friction bias.
- **🛡️ Opposition Matrix:** Detailed historical performance breakdown across all franchise teams.
- **🧩 Unsupervised Player Archetypes:** K-Means clustering on standardized skill vectors with optimal cluster selection via Silhouette Score and Davies-Bouldin Index.
- **🧬 Player Similarity Search:** Cosine similarity engine identifying statistically comparable cricketers and their key overlapping/divergent traits.
- **🔮 Prediction Lab & Uncertainty Intervals:** Calibrated machine learning regressions (XGBoost, Random Forest, Ridge) providing point forecasts and 80% prediction bounds.
- **🧠 SHAP Explainable AI:** Game-theoretic TreeExplainer isolating positive and negative feature impacts for every prediction.
- **🎲 What-If Scenario Simulator:** Interactive counterfactual tool allowing analysts to toggle venue, opposition, batting position, and match pressure to evaluate performance sensitivity.
- **📂 Interactive Data Explorer:** Live table inspector with column filters, search queries, metadata diagnostics, and CSV export.

---

## 5. System Architecture
```
                        CRICKET DATA ARCHIVES (Cricsheet / Flat CSV)
                                           │
                                           ▼
                                [ Data Ingestion Layer ]
                           (Polymorphic Data Adapters)
                                           │
                                           ▼
                             [ Data Quality Engine ]
                        (Schema Validation & Anomaly Gates)
                                           │
                                           ▼
                               [ Relational Database ]
                           (SQLite / SQLAlchemy Engine)
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
        [ Feature Engineering ]                        [ Machine Learning Pipeline ]
     - Batting & Bowling Splits                     - Chronological Temporal Splits
     - Dynamic Form (exp decay)                     - XGBoost / Random Forest
     - Pressure & Context Indices                   - Prediction Quantile Bounds
     - 10-D Skill Vectors                           - SHAP TreeExplainer Local/Global
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           ▼
                             [ Streamlit Analytics UI ]
                     - Command Center         - Matchup Network
                     - Player Dossiers        - Archetypes & Embeddings
                     - Similarity Engine      - What-If Simulator
```

---

## 6. Technology Stack
- **Core Runtime:** Python 3.11+
- **Data Engineering & Manipulation:** Pandas, NumPy, SciPy
- **Database & Storage:** SQLite, SQLAlchemy
- **Machine Learning & Modeling:** Scikit-Learn, XGBoost, LightGBM
- **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations)
- **Graph & Network Analytics:** NetworkX
- **Interactive Visualization:** Plotly Graph Objects & Express, Matplotlib, Seaborn
- **Frontend Dashboard:** Streamlit
- **Model Serialization:** Joblib
- **Testing & Quality Assurance:** Pytest

---

## 7. Dataset Specifications
CricketIQ natively interfaces with official ball-by-ball datasets from [Cricsheet](https://cricsheet.org/):
- **Corpus Size:** 300+ matches (scalable to all 1,244+ IPL matches from 2008–2024), ~72,000+ ball-by-ball deliveries.
- **Granular Fields:** `match_id`, `season`, `date`, `venue`, `innings`, `over`, `ball_number`, `batter`, `bowler`, `batter_runs`, `extras`, `total_runs`, `wides`, `noballs`, `byes`, `legbyes`, `is_legal`, `is_wicket`, `is_bowler_wicket`, `dismissal_kind`, `phase`, `pressure_index`.

---

## 8. Data Leakage Prevention (Strict Temporal Splitting)
To maintain mathematical integrity and prevent predictive over-optimism:
1. **Prior-Match Rolling Windows:** All features used to train ML models (such as career batting average, rolling strike rate, and recent form) are generated strictly using deliveries bowled *before* the match under prediction (`shift(1)` operations).
2. **Temporal Splitting:** Training and testing partitions are partitioned strictly chronologically based on match dates (80% earliest matches for training, 20% most recent for evaluation).
3. **No Target Leakage:** In-game outcomes (post-match totals, win margins) are never exposed to the feature store during inference.

---

## 9. Mathematical Formulas & Methodology

### Dynamic Form Score
Recent form is computed using an exponential decay function prioritizing recent matches over distant historical games:
$$w_i = \exp(-\lambda \cdot \Delta t_i)$$
$$\text{Form Score} = \frac{\sum_{i=1}^{N} w_i \cdot \text{Score}_i}{\sum_{i=1}^{N} w_i}$$
*Where $\lambda$ is the configurable decay factor (default: $0.05$), and $\Delta t_i$ is match age.*

### Consistency Score
Derived from the Coefficient of Variation ($CV = \sigma / \mu$):
$$\text{Consistency Score} = \max\left(0, \min\left(100, 100 \times \left(1 - \frac{\min(CV, 1.5)}{1.5}\right)\right)\right)$$

### Match Pressure Index
A normalized situational index ($0 \le P \le 100$) reflecting scoreboard leverage:
- **Chasing (2nd Innings):** Incorporates Required Run Rate ($RRR$), Current Run Rate ($CRR$), wickets lost, and remaining deliveries.
- **Defending (1st Innings):** Incorporates wickets lost relative to phase par and acceleration urgency.

### Contextual Performance Index (CPI)
$$\text{CPI} = 100 \times \left(\frac{\text{Actual Runs}}{\max(1, \text{Expected Context Runs})}\right)$$

---

## 10. Project Structure
```
cricketiq/
├── app/
│   ├── main.py                     # Streamlit application entrypoint
│   ├── pages/                      # 12 Modular analytics views
│   │   ├── overview.py             # Command Center
│   │   ├── player_intelligence.py  # 360-Degree Player Dossier
│   │   ├── player_comparison.py    # Multi-player comparison
│   │   ├── matchup_analyzer.py     # Micro-matchups & NetworkX
│   │   ├── form_analysis.py        # Dynamic form models
│   │   ├── player_similarity.py    # Cosine similarity search
│   │   ├── player_archetypes.py    # Archetypes & 2D PCA
│   │   ├── venue_intelligence.py   # Stadium difficulty index
│   │   ├── opposition_analysis.py  # Franchise matrix
│   │   ├── prediction.py           # ML prediction intervals
│   │   ├── explainability.py       # SHAP local/global attributions
│   │   ├── what_if_simulator.py    # Counterfactual simulator
│   │   └── data_explorer.py        # SQL table explorer & CSV export
│   ├── components/                 # Reusable UI widgets
│   │   ├── sidebar.py
│   │   ├── cards.py
│   │   ├── charts.py
│   │   ├── tables.py
│   │   └── filters.py
│   └── styles/
│       └── style.css               # Professional dark sports theme
│
├── src/
│   ├── config.py                   # Centralized configuration
│   ├── data/                       # Ingestion, validation, cleaning
│   │   ├── ingestion.py
│   │   ├── cleaning.py
│   │   ├── validation.py
│   │   └── transformation.py
│   ├── database/                   # Database engine & queries
│   │   ├── connection.py
│   │   ├── models.py
│   │   ├── schema.py
│   │   └── queries.py
│   ├── features/                   # Feature extraction modules
│   │   ├── batting_features.py
│   │   ├── bowling_features.py
│   │   ├── matchup_features.py
│   │   ├── venue_features.py
│   │   ├── form_features.py
│   │   ├── pressure_features.py
│   │   └── context_features.py
│   ├── analytics/                  # Analytical engines
│   │   ├── player_analysis.py
│   │   ├── comparison.py
│   │   ├── form.py
│   │   ├── venue.py
│   │   ├── opposition.py
│   │   ├── matchup.py
│   │   └── skill.py
│   ├── ml/                         # Machine learning & clustering
│   │   ├── preprocessing.py
│   │   ├── batting_model.py
│   │   ├── bowling_model.py
│   │   ├── clustering.py
│   │   ├── similarity.py
│   │   ├── embeddings.py
│   │   └── model_registry.py
│   ├── explainability/             # SHAP XAI engine
│   │   └── shap_analysis.py
│   ├── simulation/                 # What-if scenario modeling
│   │   └── what_if.py
│   └── utils/                      # Utilities & constants
│       ├── logger.py
│       ├── metrics.py
│       ├── helpers.py
│       └── constants.py
│
├── data/                           # Data storage directories
│   ├── raw/
│   ├── processed/
│   ├── external/
│   └── sample/
│
├── models/                         # Serialized joblib models & metadata
│   ├── batting_runs_model.joblib
│   ├── bowling_wickets_model.joblib
│   ├── player_clusters.joblib
│   └── metadata.json
│
├── tests/                          # 24 Automated unit tests
│   ├── test_utils.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_analytics.py
│   └── test_models.py
│
├── scripts/                        # CLI execution tools
│   ├── ingest_data.py
│   ├── build_database.py
│   ├── generate_features.py
│   ├── train_models.py
│   └── evaluate_models.py
│
├── requirements.txt
├── .gitignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── README.md
└── LICENSE
```

---

## 11. Installation & Setup

### 1. Clone & Set Up Virtual Environment
```bash
# Clone the repository
git clone https://github.com/your-username/cricketiq.git
cd cricketiq

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Build Database & Ingest Data
```bash
python scripts/build_database.py --limit 300
```
*(Ingests 300 matches, cleans records, enriches pressure & phase features, and generates SQLite database at `data/processed/cricketiq.db` in ~5 seconds).*

### 3. Train Machine Learning Models
```bash
python scripts/train_models.py
```
*(Executes time-split training across Ridge, Random Forest, and XGBoost, registers best models, performs archetype clustering, and writes `models/metadata.json`).*

### 4. Evaluate Models & Inspect Telemetry
```bash
python scripts/evaluate_models.py
```

### 5. Launch Interactive Dashboard
```bash
streamlit run app/main.py
```
*Access the application in your browser at `http://localhost:8501`.*

---

## 12. Verification & Testing
CricketIQ includes a comprehensive test suite covering statistical formulas, edge cases (zero balls, zero dismissals), schema validation, feature extraction, and ML pipelines.

Run all tests via pytest:
```bash
pytest -v
```
Output:
```
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
collected 24 items

tests/test_analytics.py::test_player_profile PASSED                      [  4%]
tests/test_analytics.py::test_matchup_engine PASSED                      [  8%]
tests/test_analytics.py::test_player_comparison PASSED                   [ 12%]
tests/test_analytics.py::test_venue_summary PASSED                       [ 16%]
tests/test_data.py::test_data_validator_mapping PASSED                   [ 20%]
tests/test_data.py::test_data_validator_quality_check PASSED             [ 25%]
tests/test_data.py::test_cleaner_legal_deliveries PASSED                 [ 29%]
tests/test_features.py::test_batting_features_existing_player PASSED     [ 33%]
tests/test_features.py::test_batting_features_nonexistent_player PASSED  [ 37%]
tests/test_features.py::test_bowling_features_existing_player PASSED     [ 41%]
tests/test_features.py::test_form_feature_extractor PASSED               [ 45%]
tests/test_features.py::test_contextual_performance_index PASSED         [ 50%]
tests/test_models.py::test_batting_prediction PASSED                     [ 54%]
tests/test_models.py::test_bowling_prediction PASSED                     [ 58%]
tests/test_models.py::test_shap_explanation PASSED                       [ 62%]
tests/test_models.py::test_similarity_search PASSED                      [ 66%]
tests/test_models.py::test_scenario_simulator PASSED                     [ 70%]
tests/test_utils.py::test_batting_average_standard PASSED                [ 75%]
tests/test_utils.py::test_batting_average_zero_dismissals PASSED         [ 79%]
tests/test_utils.py::test_strike_rate PASSED                             [ 83%]
tests/test_utils.py::test_bowling_economy PASSED                         [ 87%]
tests/test_utils.py::test_bowling_average_and_strike_rate PASSED         [ 91%]
tests/test_utils.py::test_consistency_score PASSED                       [ 95%]
tests/test_utils.py::test_pressure_index_bounds PASSED                   [100%]

============================= 24 passed in 6.94s ==============================
```

---

## 13. Docker Deployment
CricketIQ includes multi-stage containerization configurations:

```bash
# Build and run using Docker Compose
docker-compose up --build -d

# Open browser at http://localhost:8501
```

---

## 14. Real-World Limitations
1. **Weather & Pitch Deterioration:** Ball-by-ball feeds do not record atmospheric humidity, dew conditions, or pitch crack progression during the match.
2. **Field Placement Data:** Tracking precise field settings (deep mid-wicket, fine leg inside the ring) requires computer vision / Hawk-Eye feeds not available in public ball-by-ball text datasets.
3. **Small Sample Warnings:** Head-to-head encounters with fewer than 12 balls should be interpreted cautiously due to natural cricket variance.

---

## 15. Future Roadmap
- Integration with live ball-by-ball WebSocket APIs.
- Hawk-Eye tracking data integration for 3D wagon wheels and pitch maps.
- Bayesian hierarchical models for latent skill estimation under extreme sample scarcity.
- Automated scouting PDF dossier generator with report export.

---

## 16. License
This project is licensed under the [MIT License](LICENSE).

## 17. Author
- **Kartik Gore**
- 📧 Email: [kartikgore39@gmail.com](mailto:kartikgore39@gmail.com)