# nba-game-predictor

A production-grade, modular machine learning system for predicting NBA game "Over/Under" totals using XGBoost and automated data pipelines.

## Project Overview

**nba-game-predictor** bridges the gap between historical data analysis and real-time inference. Unlike simple analysis scripts, this project works as a complete **MLOps lifecycle**:
1.  **Ingests** historical data (2008-2025) to build a robust foundation.
2.  **Updates Daily** by hitting the NBA API to fetch yesterday's results and update rolling statistics.
3.  **Predicts** tonight's game totals against Vegas lines using an XGBoost regressor.

Designed to demonstrate **software engineering best practices** in a data science context, featuring secure database patterns, type hinting, and modular architecture.

---

## Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Language** | Python 3.10+ | Core programming language with type hints |
| **ML Framework** | XGBoost 2.0+ | Gradient boosting for regression |
| **ML Utilities** | scikit-learn 1.3+ | GridSearchCV, TimeSeriesSplit, preprocessing |
| **Data Processing** | pandas 2.0+ | DataFrame manipulation & analysis |
| **Numerical Computing** | NumPy 1.24+ | Array operations & mathematics |
| **Database** | SQLite 3 | Local data warehouse (built into Python) |
| **Serialization** | joblib 1.3+ | store complete model |
| **API Client** | nba_api 1.4+ | Fetching live NBA data |

---

### Key Features & Engineering Highlights

*   **State-of-the-art Modeling**:
    *   Utilizes **XGBoost (Extreme Gradient Boosting)** for superior performance on tabular sports data.
    *   Implements **Time-Series Cross-Validation** to prevent data leakage (training on past, testing on future).
    *   **GridSearchCV** with 512 parameter combinations for optimal hyperparameter tuning.
*   **Automated Data Pipeline**:
    *   **Daily ETL Workflow**: Automated scripts fetch live game data, sanitize inputs, and update the local SQLite warehouse.
    *   **Advanced Feature Engineering**: 40+ dynamic features including rolling averages (3/5/10/15 games) calculated with strict time-series splits to prevent lookahead bias, pace estimation, and rest-days/back-to-back impact.
    *   **Consistency Checks**: Ensures training features exactly match inference features to avoid training-serving skew.
*   **Enterprise-Grade Security**:
    *   **SQL Injection Prevention**: Custom query builders use rigorous parameter binding and column whitelisting.
    *   **Secure Serialization**: Replaces `pickle` with `joblib` to prevent arbitrary code execution vulnerabilities.
    *   **Input Validation**: Strict typing and allow-lists for all external data sources.
*   **Clean Architecture**:
    *   Separation of concerns: `src/database` (Data Layer), `src/features` (Logic Layer), `src/api_client` (Integration Layer).
    *   Fully type-hinted Python 3.12 codebase.
    *   Production-ready logging and error handling.

---

## Architecture

The project is organized as a modular Python package:

```
nba-game-predictor/
├── data/                    # Data storage (ignored by git)
│   ├── nba.db               # SQLite database
│   └── *.csv                # Raw historical data
├── models/                  # Trained ML models
│   ├── nba_model.joblib     # Trained XGBoost model
│   └── imputer.joblib       # Feature imputer for inference
├── scripts/                 # Executable scripts
│   └── seed_database.py     # Initializes DB from CSV
├── src/                     # Application source code
│   ├── api_client.py        # NBA API interface
│   ├── config.py            # Configuration & Security settings
│   ├── database.py          # Secure database operations
│   ├── daily_update.py      # Daily data fetching & updates
│   ├── evaluate_model.py    # Betting simulation & ROI analysis
│   ├── features.py          # Feature engineering logic
│   ├── nba_data.py          # Data processing pipeline
│   ├── predictor.py         # Inference engine
│   ├── today_predictions.py # Predict today's games
│   ├── train_model.py       # Model training with GridSearchCV
│   └── utils.py             # Shared utilities
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Complete Setup Guide

Follow these steps to set up the project from scratch and make your first prediction.

### Prerequisites

- **Python 3.10+** (Python 3.12 recommended)
- **pip** (Python package manager)
- **Git** (for cloning the repository)

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/YourUsername/nba-game-predictor.git
cd nba-game-predictor
```

---

### Step 2: Create a Virtual Environment (Recommended)

Create an isolated Python environment to avoid dependency conflicts:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

This installs:
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning utilities
- `xgboost` - Gradient boosting model
- `joblib` - Secure model serialization
- `nba_api` - NBA data fetching

---

### Step 4: Prepare the Historical Data

Place your historical NBA data CSV file in the `data/` folder:

```
data/nba_2008-2025.csv
```

The CSV should contain columns like:
- `season`, `date`, `away`, `home`, `score_away`, `score_home`, `total` (Vegas line)

---

### Step 5: Seed the Database

Initialize the SQLite database and import historical data:

```bash
python scripts/seed_database.py
```

**What this does:**
- Creates `data/nba.db` SQLite database
- Creates tables: `games`, `team_stats`, `scheduled_games`, `predictions`
- Imports all historical games from the CSV
- You should see: `"Imported X games into the database"`

---

### Step 6: Train the Model

Train the XGBoost model with hyperparameter optimization:

```bash
python -m src.train_model
```

**What this does:**
- Loads all games from the database
- Engineers 40+ features (rolling averages, pace, trends, etc.)
- Runs **GridSearchCV** with 512 parameter combinations
- Uses **TimeSeriesSplit** (5 folds) to prevent data leakage
- Saves the best model to `models/nba_model.joblib`
- Saves the imputer to `models/imputer.joblib`

**Expected output:**
```
Best parameters: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 400, ...}
Best CV Score (neg MAE): -14.27
Test MAE: 14.27 points
Model saved to models/nba_model.joblib
```

**Note:** Training takes 10-30 minutes depending on your hardware.

---

### Step 7: Evaluate Model Performance (Optional)

Run the betting simulation to analyze profitability:

```bash
python -m src.evaluate_model
```

**What this shows:**
- Overall win rate and ROI
- Performance at different edge thresholds (1-10 points)
- Monthly breakdown of results
- Identifies the optimal betting strategy

---

## Making Predictions

### Option 1: Predict a Single Game

Predict the total for a specific matchup:

```bash
python -m src.today_predictions --away lal --home bos --vegas 225.5
```

**Example output:**
```
LAL @ BOS
  Predicted Total: 214.6
  Vegas Total:     225.5
  Edge:            -10.9
  Recommendation:  UNDER
  HIGH CONFIDENCE - Strong edge detected!
```

### Option 2: Predict Today's Games (Interactive)

Fetch today's scheduled games from the NBA API and make predictions:

```bash
python -m src.today_predictions
```

**This will:**
1. Fetch today's games from the NBA API
2. Display all matchups
3. Prompt you to enter Vegas O/U lines
4. Generate predictions for each game
5. Highlight high-confidence picks (5+ point edge)

### Option 3: Quick Mode (No Interaction)

Fetch and predict without entering Vegas lines:

```bash
python -m src.today_predictions --quick
```

---

## Daily Updates

Keep your database current with the latest game results:

```bash
python -m src.daily_update
```

**What this does:**
- Fetches yesterday's completed games from the NBA API
- Inserts new game results into the database
- Updates rolling team statistics
- Updates prediction results with actual totals

### Backfill Missing Games

If you missed several days, backfill a date range:

```bash
python -m src.daily_update --backfill --start-date 2026-01-01 --end-date 2026-01-14
```

---

## Team Abbreviations

Use these abbreviations when making predictions:

| Team | Abbrev | Team | Abbrev |
|------|--------|------|--------|
| Atlanta Hawks | `atl` | Milwaukee Bucks | `mil` |
| Boston Celtics | `bos` | Minnesota Timberwolves | `min` |
| Brooklyn Nets | `bkn` | New Orleans Pelicans | `no` |
| Charlotte Hornets | `cha` | New York Knicks | `ny` |
| Chicago Bulls | `chi` | Oklahoma City Thunder | `okc` |
| Cleveland Cavaliers | `cle` | Orlando Magic | `orl` |
| Dallas Mavericks | `dal` | Philadelphia 76ers | `phi` |
| Denver Nuggets | `den` | Phoenix Suns | `phx` |
| Detroit Pistons | `det` | Portland Trail Blazers | `por` |
| Golden State Warriors | `gs` | Sacramento Kings | `sac` |
| Houston Rockets | `hou` | San Antonio Spurs | `sa` |
| Indiana Pacers | `ind` | Toronto Raptors | `tor` |
| LA Clippers | `lac` | Utah Jazz | `utah` |
| Los Angeles Lakers | `lal` | Washington Wizards | `wsh` |
| Memphis Grizzlies | `mem` | | |
| Miami Heat | `mia` | | |

 **Note:** `uta` and `was` also work as aliases for Utah and Washington.

---

## Training Workflow & Model Selection

### Methodology

1.  **Data Processing**:
    Raw data is processed through `src/nba_data.py`, generating 40+ features like rolling scoring averages, pace estimates, and trend indicators. Future data (like the actual game score) is strictly separated from input features.

2.  **Model Comparison**:
    We benchmarked multiple algorithms:
    *   **Ridge Regression**: Linear baseline model
    *   **Random Forest**: Ensemble of decision trees
    *   **XGBoost (Champion)**: Outperformed others by capturing non-linear relationships

3.  **Hyperparameter Tuning (GridSearchCV)**:
    512 parameter combinations tested:
    ```python
    param_grid = {
        'n_estimators': [200, 300, 400, 500],
        'learning_rate': [0.01, 0.02, 0.03, 0.05],
        'max_depth': [4, 5, 6, 7],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.75, 0.85],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 2]
    }
    ```

4.  **Time-Series Cross-Validation**:
    Uses `TimeSeriesSplit` with 5 folds to ensure we always train on past data and test on future data—simulating real betting conditions.

5.  **Betting Profitability Analysis**:
    Model evaluation focuses on ROI, not just accuracy. Our analysis shows:
    - **5+ point edge**: ~60% win rate, +5% ROI
    - **Higher edges = more profitable** but fewer opportunities

---

## Security Measures

*   **No Pickles**: We use `joblib` instead of `pickle` for model serialization to prevent arbitrary code execution vulnerabilities.
*   **SQL Injection Prevention**: All database queries use parameterized queries and column whitelisting.
*   **Input Sanitization**: All team abbreviations are validated against strict allow-lists.
*   **Path Validation**: Model loading validates paths are within the trusted `models/` directory.

---

## Quick Start Summary

```bash
# 1. Clone and setup
git clone https://github.com/YourUsername/nba-game-predictor.git
cd nba-game-predictor
python -m venv venv && venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Seed database (place CSV in data/ first)
python scripts/seed_database.py

# 3. Train model (takes 10-30 min)
python -m src.train_model

# 4. Make predictions!
python -m src.today_predictions --away lal --home bos --vegas 225

# 5. Keep database updated daily
python -m src.daily_update
```

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive license that allows you to use, modify, and distribute this software freely, with the only requirement being that the license and copyright notice are preserved.

**Disclaimer:** This software is provided "as is" without warranty. The predictions are for educational and entertainment purposes only. Always gamble responsibly.
