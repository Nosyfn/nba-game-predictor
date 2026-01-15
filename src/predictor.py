"""
Prediction service for nba-game-predictor
Makes predictions for today's games using the trained model and current team features.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

# Use joblib instead of pickle for safer serialization of sklearn models
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Warning: joblib not installed. Run: pip install joblib")

from .database import (
    get_all_games,
    get_latest_team_stats,
    get_scheduled_games,
    add_scheduled_game,
    save_prediction,
    get_team_recent_games
)
from .api_client import check_api_available, get_todays_games, normalize_team_abbrev
from .features import (
    calculate_team_stats,
    calculate_rolling_features,
    calculate_pace_features,
    calculate_trend_features,
    calculate_season_features,
    calculate_matchup_features,
    create_over_under_features
)
from .config import (
    PROJECT_ROOT,
    TEAM_FULL_NAMES,
    EXCLUDE_FROM_FEATURES,
    ROLLING_WINDOWS,
    MODEL_DIR,
    ALLOWED_TEAMS
)


# Path to saved model (must be within MODEL_DIR for security)
MODEL_PATH = MODEL_DIR / "nba_model.joblib"
# Legacy pickle path (for migration, will be converted to joblib)
LEGACY_MODEL_PATH = MODEL_DIR / "nba_model.pkl"
# Path to saved imputer
IMPUTER_PATH = MODEL_DIR / "imputer.joblib"


def _validate_model_path(model_path) -> bool:
    """
    Validate that the model path is within the allowed MODEL_DIR.
    Prevents path traversal attacks.
    
    Parameters:
        model_path: Path to validate
        
    Returns:
        True if path is safe, False otherwise
    """
    try:
        # Resolve to absolute path and check it's within MODEL_DIR
        resolved_path = model_path.resolve()
        model_dir_resolved = MODEL_DIR.resolve()
        
        # Check that the model path is within the model directory
        return str(resolved_path).startswith(str(model_dir_resolved))
    except Exception:
        return False


def load_model(model_path=None):
    """
    Securely load the trained ML model from disk.
    
    Parameters:
        model_path: Optional custom path (must be within MODEL_DIR)
        
    Returns:
        Trained model object or None if not found/invalid
    """
    if not JOBLIB_AVAILABLE:
        print("Error: joblib is required for secure model loading")
        print("Install with: pip install joblib")
        return None
    
    # Use default path if not specified
    if model_path is None:
        model_path = MODEL_PATH
    
    # Security: Validate the path is within allowed directory
    if not _validate_model_path(model_path):
        print(f"Security Error: Model path must be within {MODEL_DIR}")
        return None
    
    # Check for joblib model first
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            print(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    # Check for legacy pickle model and offer migration
    if LEGACY_MODEL_PATH.exists():
        print(f"Found legacy pickle model at {LEGACY_MODEL_PATH}")
        print("For security, please migrate to joblib format using save_model()")
        print("Legacy pickle files can execute arbitrary code if tampered with.")
        return None
    
    print(f"No trained model found at {model_path}")
    print("Please train a model first using the training script.")
    return None


def save_model(model, model_path=None):
    """
    Securely save a trained model to disk using joblib.
    
    Parameters:
        model: Trained model object
        model_path: Optional custom path (must be within MODEL_DIR)
        
    Returns:
        True if saved successfully, False otherwise
    """
    if not JOBLIB_AVAILABLE:
        print("Error: joblib is required for model saving")
        return False
    
    if model_path is None:
        model_path = MODEL_PATH
    
    # Security: Validate the path is within allowed directory
    if not _validate_model_path(model_path):
        print(f"Security Error: Model path must be within {MODEL_DIR}")
        return False
    
    try:
        joblib.dump(model, model_path)
        print(f"Model saved securely to {model_path}")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False


def validate_team(team: str) -> bool:
    return team.lower() in ALLOWED_TEAMS


def get_team_features_from_db(team: str) -> Optional[Dict[str, Any]]:
    """
    Get the current feature values for a team from the database.
    
    Parameters:
        team: Team abbreviation
        
    Returns:
        Dictionary of feature values or None
    """
    # Validate team input
    if not validate_team(team):
        print(f"Invalid team abbreviation: {team}")
        return None
    
    stats = get_latest_team_stats(team)
    
    if stats is None:
        print(f"No stats found for team: {team}")
        return None
    
    return stats


def build_prediction_features(away_team: str, home_team: str) -> Optional[pd.DataFrame]:
    """
    Build feature vector for a matchup between two teams.
    
    Parameters:
        away_team: Away team abbreviation
        home_team: Home team abbreviation
        
    Returns:
        DataFrame with feature values for this matchup
    """
    # Always use the full feature engineering pipeline to ensure
    # column names match exactly what the model was trained on
    return build_features_from_recent_games(away_team, home_team)


def build_features_from_recent_games(away_team: str, home_team: str) -> Optional[pd.DataFrame]:
    """
    Build prediction features by processing recent games from the database.
    Uses the EXACT same feature engineering pipeline as training to ensure
    column names and calculations match.
    
    Parameters:
        away_team: Away team abbreviation
        home_team: Home team abbreviation
        
    Returns:
        DataFrame with features (single row for this matchup)
    """
    # Get all games from database
    all_games = get_all_games()
    
    if all_games.empty:
        print("No games in database")
        return None
    
    # Run the SAME feature engineering pipeline as training (nba_data.py)
    # Step 1-5: Calculate team-level stats
    team_games = calculate_team_stats(all_games)
    team_games = calculate_rolling_features(team_games)
    team_games = calculate_pace_features(team_games)
    team_games = calculate_trend_features(team_games)
    team_games = calculate_season_features(team_games)
    
    # Step 6: Create matchup features (adds home_/away_ prefixes)
    processed_df = calculate_matchup_features(all_games, team_games)
    
    # Step 7: Create over/under specific features (combined_scoring, expected_total, etc.)
    processed_df = create_over_under_features(processed_df)
    
    # Get the latest game for the matchup involving both teams
    # We need to find a recent game to use as a "template" for this matchup
    # Then update the away/home features to reflect the actual teams playing
    
    # Get latest stats for each team from team_games
    away_data = team_games[team_games['team'] == away_team].copy()
    home_data = team_games[team_games['team'] == home_team].copy()
    
    if away_data.empty or home_data.empty:
        print(f"Insufficient data for {away_team} or {home_team}")
        return None
    
    away_latest = away_data.sort_values('date').iloc[-1]
    home_latest = home_data.sort_values('date').iloc[-1]
    
    # Build matchup features matching the training column structure
    features = {}
    
    # Get feature columns from processed_df to understand structure
    sample_row = processed_df.iloc[-1]
    
    # Add away team features
    for col in sample_row.index:
        if col.startswith('away_'):
            base_col = col[5:]  # Remove 'away_' prefix
            if base_col in away_latest.index and pd.notna(away_latest[base_col]):
                features[col] = away_latest[base_col]
            elif col in sample_row.index:
                features[col] = np.nan  # Will be imputed
    
    # Add home team features
    for col in sample_row.index:
        if col.startswith('home_'):
            base_col = col[5:]  # Remove 'home_' prefix
            if base_col in home_latest.index and pd.notna(home_latest[base_col]):
                features[col] = home_latest[base_col]
            elif col in sample_row.index:
                features[col] = np.nan  # Will be imputed
    
    # Calculate combined features (from create_over_under_features)
    for window in [3, 5, 10, 15]:
        # Combined scoring
        home_scored = features.get(f'home_points_scored_roll_{window}')
        away_scored = features.get(f'away_points_scored_roll_{window}')
        if home_scored is not None and away_scored is not None:
            features[f'combined_scoring_{window}'] = home_scored + away_scored
        
        # Combined defense
        home_allowed = features.get(f'home_points_allowed_roll_{window}')
        away_allowed = features.get(f'away_points_allowed_roll_{window}')
        if home_allowed is not None and away_allowed is not None:
            features[f'combined_defense_{window}'] = home_allowed + away_allowed
        
        # Expected total
        if all(v is not None for v in [home_scored, away_allowed, away_scored, home_allowed]):
            features[f'expected_total_{window}'] = (
                (home_scored + away_allowed) / 2 +
                (away_scored + home_allowed) / 2
            )
        
        # Pace matchup
        home_pace = features.get(f'home_pace_{window}')
        away_pace = features.get(f'away_pace_{window}')
        if home_pace is not None and away_pace is not None:
            features[f'pace_matchup_{window}'] = (home_pace + away_pace) / 2
        
        # Combined volatility
        home_std = features.get(f'home_total_points_std_{window}')
        away_std = features.get(f'away_total_points_std_{window}')
        if home_std is not None and away_std is not None:
            features[f'combined_volatility_{window}'] = (home_std + away_std) / 2
    
    # Rest features
    home_rest = features.get('home_days_rest', 2)
    away_rest = features.get('away_days_rest', 2)
    features['total_rest'] = home_rest + away_rest
    features['rest_diff'] = home_rest - away_rest
    features['both_rested'] = int(home_rest >= 2 and away_rest >= 2)
    
    home_b2b = features.get('home_is_b2b', 0)
    away_b2b = features.get('away_is_b2b', 0)
    features['both_b2b'] = int(home_b2b + away_b2b == 2)
    
    # Trend combinations
    home_trend = features.get('home_scoring_trend')
    away_trend = features.get('away_scoring_trend')
    if home_trend is not None and away_trend is not None:
        features['combined_scoring_trend'] = home_trend + away_trend
    
    home_total_trend = features.get('home_total_trend')
    away_total_trend = features.get('away_total_trend')
    if home_total_trend is not None and away_total_trend is not None:
        features['combined_total_trend'] = home_total_trend + away_total_trend
    
    # Season phase
    home_game_num = features.get('home_game_num', 40)
    away_game_num = features.get('away_game_num', 40)
    features['home_early_season'] = int(home_game_num <= 25)
    features['away_early_season'] = int(away_game_num <= 25)
    
    return pd.DataFrame([features])


def load_imputer():
    """Load the saved imputer for preprocessing features."""
    if not JOBLIB_AVAILABLE:
        return None
    
    if IMPUTER_PATH.exists():
        try:
            return joblib.load(IMPUTER_PATH)
        except Exception as e:
            print(f"Error loading imputer: {e}")
            return None
    return None


def predict_game(away_team: str, home_team: str, 
                 model=None, vegas_total: float = None) -> Dict[str, Any]:
    """
    Make a prediction for a single game.
    
    Parameters:
        away_team: Away team abbreviation
        home_team: Home team abbreviation
        model: Trained model (loaded if not provided)
        vegas_total: Vegas total line if available
        
    Returns:
        Dictionary with prediction results
    """
    if model is None:
        model = load_model()
        if model is None:
            return {'error': 'No model available'}
            
    # Load imputer
    imputer = load_imputer()
    
    # Build features
    features = build_prediction_features(away_team, home_team)
    
    if features is None:
        return {'error': 'Could not build features for this matchup'}
    
    # Ensure all column names are strings (fixes imputer compatibility)
    features.columns = features.columns.astype(str)
    
    # Get the feature columns the model was trained on
    if hasattr(model, 'feature_names_in_'):
        expected_features = [str(f) for f in model.feature_names_in_]
        
        # Ensure we have all expected features (add missing columns as NaN)
        for col in expected_features:
            if col not in features.columns:
                features[col] = np.nan
        
        # Reorder to match expected columns only
        features = features[expected_features]
        
        # Convert all columns to numeric, coercing errors to NaN
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')
        
        # Apply imputer to handle ALL NaN values (both from missing columns AND missing data)
        if imputer is not None:
            try:
                features_imputed = imputer.transform(features)
                features = pd.DataFrame(
                    features_imputed,
                    columns=expected_features,
                    index=features.index
                )
            except Exception as e:
                print(f"Warning: Imputer failed ({e}), falling back to fillna(0)")
                features = features.fillna(0)
        else:
            # Fallback when imputer is not available
            print("Warning: No imputer found, filling NaN with 0")
            features = features.fillna(0)
    else:
        # Convert to numeric first
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')
        features = features.fillna(0)
    
    # Make prediction
    try:
        predicted_total = model.predict(features)[0]
    except Exception as e:
        return {'error': f'Prediction failed: {e}'}
    
    result = {
        'away_team': away_team,
        'home_team': home_team,
        'away_name': TEAM_FULL_NAMES.get(away_team, away_team),
        'home_name': TEAM_FULL_NAMES.get(home_team, home_team),
        'predicted_total': round(predicted_total, 1),
        'vegas_total': vegas_total,
    }
    
    if vegas_total is not None:
        edge = round(predicted_total - vegas_total, 1)
        result['difference'] = edge
        result['edge'] = edge  # Alias for difference
        result['recommendation'] = 'OVER' if predicted_total > vegas_total else 'UNDER'
    
    return result


def predict_todays_games(model=None) -> List[Dict[str, Any]]:
    """
    Make predictions for all of today's scheduled games.
    
    Parameters:
        model: Trained model (loaded if not provided)
        
    Returns:
        List of prediction dictionaries
    """
    print("\n" + "=" * 60)
    print(f"nba-game-predictor Predictions for {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    if model is None:
        model = load_model()
        if model is None:
            return []
    
    # Try to get today's games from API
    predictions = []
    
    if check_api_available():
        print("\nFetching today's games from API...")
        today_games = get_todays_games()
        
        if not today_games.empty:
            for _, game in today_games.iterrows():
                # We'd need to map team IDs to abbreviations
                # For now, this requires the games to have team abbreviations
                pass
    
    # Check scheduled games table in database
    today = datetime.now().strftime('%Y-%m-%d')
    scheduled = get_scheduled_games(today)
    
    if not scheduled.empty:
        print(f"\nFound {len(scheduled)} scheduled games in database")
        
        for _, game in scheduled.iterrows():
            prediction = predict_game(
                away_team=game['away'],
                home_team=game['home'],
                model=model,
                vegas_total=game.get('vegas_total')
            )
            
            if 'error' not in prediction:
                predictions.append(prediction)
                
                # Save prediction to database
                save_prediction(
                    game_date=today,
                    away=game['away'],
                    home=game['home'],
                    predicted_total=prediction['predicted_total'],
                    vegas_total=game.get('vegas_total'),
                    model_version='v1.0'
                )
    else:
        print("\nNo scheduled games found. Add games using add_scheduled_game()")
        print("Example: add_scheduled_game('2025-01-13', 'bos', 'lal', vegas_total=225.5)")
    
    return predictions


def display_predictions(predictions: List[Dict[str, Any]]) -> None:
    """
    Display predictions in a formatted table.
    
    Parameters:
        predictions: List of prediction dictionaries
    """
    if not predictions:
        print("\nNo predictions to display")
        return
    
    print("\n" + "-" * 80)
    print(f"{'Matchup':<40} {'Predicted':>10} {'Vegas':>10} {'Diff':>8} {'Rec':>8}")
    print("-" * 80)
    
    for p in predictions:
        matchup = f"{p['away_name']} @ {p['home_name']}"
        if len(matchup) > 38:
            matchup = matchup[:35] + "..."
        
        vegas = f"{p['vegas_total']:.1f}" if p.get('vegas_total') else "N/A"
        diff = f"{p['difference']:+.1f}" if p.get('difference') else ""
        rec = p.get('recommendation', '')
        
        print(f"{matchup:<40} {p['predicted_total']:>10.1f} {vegas:>10} {diff:>8} {rec:>8}")
    
    print("-" * 80)


def add_games_for_prediction(games: List[Dict[str, str]]) -> None:
    """
    Add games to the scheduled_games table for prediction.
    
    Parameters:
        games: List of dicts with 'away', 'home', and optional 'vegas_total'
    """
    today = datetime.now().strftime('%Y-%m-%d')
    
    for game in games:
        add_scheduled_game(
            game_date=today,
            away=game['away'],
            home=game['home'],
            vegas_total=game.get('vegas_total')
        )
        print(f"Added: {game['away']} @ {game['home']}")


if __name__ == "__main__":
    # Example usage
    print("nba-game-predictor Prediction Service")
    print("-" * 40)
    
    # Load model
    model = load_model()
    
    if model:
        # Make predictions for today's games
        predictions = predict_todays_games(model)
        display_predictions(predictions)
    else:
        print("\nNo model found. Please train a model first.")
        print("\nTo test feature generation, you can try:")
        print("  features = build_prediction_features('bos', 'lal')")
        print("  print(features)")
