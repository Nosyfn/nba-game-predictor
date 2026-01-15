"""
Train and evaluate models for NBA Over/Under prediction.
Compares Linear Regression and XGBoost using Time-Series Cross-Validation.
Implements GridSearchCV for hyperparameter tuning.

"""
import sys
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROJECT_ROOT, TRAINING_SEASONS
from src.utils import fetch_nba_data, fetch_nba_data_from_db
from src.nba_data import process_nba_data
from src.predictor import save_model


def get_data():
    """Load and process data from database or CSV."""
    print("Loading data...")
    # Try DB first, then CSV
    df = fetch_nba_data_from_db()
    
    if df is None:
        csv_path = PROJECT_ROOT / "data" / "nba_2008-2025.csv"
        if csv_path.exists():
            print(f"Database empty, loading from {csv_path}")
            df = fetch_nba_data(str(csv_path))
        else:
            print("No data found. Please run seed_database.py")
            return None
            
    # Process features
    print("Processing features...")
    processed_df = process_nba_data(df)
    return processed_df


def train_and_evaluate(df, target_col='actual_total', test_seasons=[2024, 2025], 
                       skip_tuning=False):
    """
    Train models using Time-Series Cross-Validation.
    
    Parameters:
        df: Processed DataFrame with features
        target_col: Target column name
        test_seasons: Seasons to hold out for final testing
        skip_tuning: If True, skip GridSearchCV (faster but uses default params)
    """
    # Sort by date to ensure time-series integrity
    df = df.sort_values('date').reset_index(drop=True)
    
    # Feature columns (exclude non-features)
    from src.config import EXCLUDE_FROM_FEATURES
    feature_cols = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]
    
    print(f"\nTraining on {len(feature_cols)} features...")
    print(f"Target: {target_col}")
    
    # Prepare data
    X = df[feature_cols]
    y = df[target_col]
    
    # Time Series Split for validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    
    # 1. Baseline: Ridge Regression
    
    print("\n" + "=" * 60)
    print("Model 1: Ridge Regression (Baseline)")
    print("=" * 60)
    
    lr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0))
    ])
    
    lr_scores = []
    for train_index, val_index in tscv.split(X):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        
        lr_pipeline.fit(X_train_fold, y_train_fold)
        preds = lr_pipeline.predict(X_val_fold)
        mae = mean_absolute_error(y_val_fold, preds)
        lr_scores.append(mae)
        
    print(f"Average MAE (Cross-Val): {np.mean(lr_scores):.4f}")
    
    
    #XGBoost with Hyperparameter Tuning
    
    print("\n" + "=" * 60)
    print("Model 2: XGBoost with GridSearchCV")
    print("=" * 60)
    
    # Split into Train (Historic) vs Test (Recent) BEFORE tuning
    # This ensures test set is never seen during hyperparameter selection
    mask_test = df['season'].isin(test_seasons)
    X_train = X[~mask_test]
    y_train = y[~mask_test]
    X_test = X[mask_test]
    y_test = y[mask_test]
    
    print(f"Training set: {len(X_train)} games (excludes seasons {test_seasons})")
    print(f"Test set: {len(X_test)} games (seasons {test_seasons})")
    
    # Handle NaN values in training data
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    if skip_tuning:
        # Use reasonable defaults without tuning
        print("\n[Skipping GridSearch - using default hyperparameters]")
        best_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        print(f"Default params: {best_params}")
    else:
        # COMPREHENSIVE GRID SEARCH: Find optimal hyperparameters
        print("\n" + "=" * 60)
        print("COMPREHENSIVE HYPERPARAMETER TUNING")
        print("=" * 60)
        print("This process may take 30-60 minutes depending on hardware.")
        
        # Hyperparameter search space
        param_grid = {
            'n_estimators': [400, 600, 800, 1000],         
            'learning_rate': [0.01, 0.03, 0.05, 0.08],     
            'max_depth': [4, 5, 6, 7],                     
            'subsample': [0.75, 0.85],                      
            'colsample_bytree': [0.75, 0.85],               
            'min_child_weight': [1, 3],                     
            'reg_lambda': [1, 3],                           
        }
        
        total_combinations = 4 * 4 * 4 * 2 * 2 * 2 * 2
        print(f"Searching {total_combinations} parameter combinations...")
        print(f"With 5-fold CV = {total_combinations * 5} total model fits")
        print(f"Estimated time: 45-60 minutes\n")
        
        # Base XGBoost model with early stopping capability
        xgb_base = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            verbosity=0  # Suppress XGBoost warnings during grid search
        )
        
        # Use MAE as scoring (negative because sklearn maximizes)
        mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        
        # GridSearchCV with Time Series Split
        grid_search = GridSearchCV(
            estimator=xgb_base,
            param_grid=param_grid,
            cv=tscv,
            scoring=mae_scorer,
            verbose=2,           # More detailed progress output
            n_jobs=-1,           # Parallelize across parameter combinations
            return_train_score=True
        )
        
        # Fit GridSearch on training data only
        grid_search.fit(X_train_imputed, y_train)
        
        # Best parameters found
        best_params = grid_search.best_params_
        best_cv_score = -grid_search.best_score_  # Negate back to positive MAE
        
        print("\n" + "-" * 40)
        print("GRID SEARCH RESULTS")
        print("-" * 40)
        print(f"Best MAE (CV): {best_cv_score:.4f}")
        print(f"Best Parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        # Show top 5 parameter combinations
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df['mean_test_mae'] = -results_df['mean_test_score']
        results_df = results_df.sort_values('mean_test_mae')
        
        print("\nTop 5 Parameter Combinations:")
        for i, row in results_df.head(5).iterrows():
            print(f"  MAE: {row['mean_test_mae']:.4f} | "
                  f"n_est={row['param_n_estimators']}, "
                  f"lr={row['param_learning_rate']}, "
                  f"depth={row['param_max_depth']}")
    
    #Train Final Model with Best Parameters
    
    print("\n" + "=" * 60)
    print("Training Final Model with Best Parameters")
    print("=" * 60)
    
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        subsample=best_params.get('subsample', 0.8),
        colsample_bytree=best_params.get('colsample_bytree', 0.8),
        random_state=42,
        n_jobs=-1
    )
    
    # Train on full training set
    final_model.fit(X_train_imputed, y_train)
    
    
    #Evaluate on Held-Out Test Set
    
    print("\n" + "=" * 60)
    print(f"Final Evaluation on Test Set (Seasons {test_seasons})")
    print("=" * 60)
    
    test_preds = final_model.predict(X_test_imputed)
    
    final_mae = mean_absolute_error(y_test, test_preds)
    final_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    final_r2 = r2_score(y_test, test_preds)
    
    print(f"\nTest Set Results:")
    print(f"  MAE:  {final_mae:.4f} points")
    print(f"  RMSE: {final_rmse:.4f} points")
    print(f"  R²:   {final_r2:.4f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    print(f"  On average, predictions are off by {final_mae:.1f} points")
    print(f"  The model explains {final_r2*100:.1f}% of the variance in game totals")
    
    
    #Feature Importance
    
    print("\n" + "=" * 60)
    print("Top 15 Most Important Features")
    print("=" * 60)
    
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importance.head(15).iterrows():
        print(f"  {row['importance']:.4f} | {row['feature']}")
    
    
    #Save Model
    
    print("\n" + "=" * 60)
    print("Saving Model")
    print("=" * 60)
    
    save_model(final_model)
    
    # Also save the imputer for inference
    
    imputer_path = PROJECT_ROOT / "models" / "imputer.joblib"
    joblib.dump(imputer, imputer_path)
    print(f"Imputer saved to {imputer_path}")
    
    return final_model, best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train NBA Over/Under prediction model')
    parser.add_argument('--skip-tuning', action='store_true',
                        help='Skip GridSearchCV (faster, uses default params)')
    args = parser.parse_args()
    
    df = get_data()
    if df is not None:
        train_and_evaluate(df, skip_tuning=args.skip_tuning)
