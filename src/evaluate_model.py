"""
Evaluate trained model with betting simulation metrics.
Shows profitability analysis, not just MAE.

Usage:
    python src/evaluate_model.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROJECT_ROOT, EXCLUDE_FROM_FEATURES
from src.utils import fetch_nba_data_from_db
from src.nba_data import process_nba_data
from src.predictor import load_model, load_imputer


def evaluate_betting_performance(test_seasons=[2024, 2025]):
    """
    Evaluate model performance with betting-specific metrics.
    """
    print("=" * 70)
    print("nba-game-predictor BETTING PERFORMANCE EVALUATION")
    print("=" * 70)
    
    # Load model
    model = load_model()
    if model is None:
        print("No model found. Please train first.")
        return
    
    imputer = load_imputer()
    
    # Load and process data
    print("\nLoading data...")
    df = fetch_nba_data_from_db()
    if df is None:
        print("No data in database.")
        return
    
    print("Processing features...")
    processed_df = process_nba_data(df)
    processed_df = processed_df.sort_values('date').reset_index(drop=True)
    
    # Get feature columns
    feature_cols = [c for c in processed_df.columns if c not in EXCLUDE_FROM_FEATURES]
    
    # Filter to test seasons
    test_df = processed_df[processed_df['season'].isin(test_seasons)].copy()
    print(f"\nEvaluating on {len(test_df)} games (Seasons {test_seasons})")
    
    # Prepare features
    X_test = test_df[feature_cols]
    y_test = test_df['actual_total']
    vegas_lines = test_df['total']  # Vegas O/U line
    
    # Handle NaN values
    if imputer is not None:
        X_test_imputed = pd.DataFrame(
            imputer.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
    else:
        X_test_imputed = X_test.fillna(0)
    
    # Make predictions
    predictions = model.predict(X_test_imputed)
    
    
    # STANDARD METRICS
    print("\n" + "=" * 70)
    print("STANDARD METRICS")
    print("=" * 70)
    
    mae = np.mean(np.abs(predictions - y_test))
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    
    print(f"MAE:  {mae:.2f} points (average prediction error)")
    print(f"RMSE: {rmse:.2f} points")
    

    # BETTING SIMULATION
    print("\n" + "=" * 70)
    print("BETTING SIMULATION (All Games)")
    print("=" * 70)
    
    # Determine model's bet: OVER if prediction > vegas line
    model_says_over = predictions > vegas_lines.values
    actual_went_over = y_test.values > vegas_lines.values
    
    # Did the model win?
    model_correct = model_says_over == actual_went_over
    
    # Handle pushes (prediction = actual = line)
    pushes = y_test.values == vegas_lines.values
    
    total_bets = len(test_df)
    wins = model_correct.sum()
    losses = total_bets - wins - pushes.sum()
    push_count = pushes.sum()
    
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    
    print(f"\nResults betting EVERY game:")
    print(f"  Wins:   {wins}")
    print(f"  Losses: {losses}")
    print(f"  Pushes: {push_count}")
    print(f"  Win Rate: {win_rate:.1f}%")
    
    # ROI Calculation (assuming -110 odds, bet $110 to win $100)
    profit = (wins * 100) - (losses * 110)
    total_wagered = (wins + losses) * 110
    roi = (profit / total_wagered) * 100 if total_wagered > 0 else 0
    
    print(f"\nAssuming $110 per bet (-110 odds):")
    print(f"  Total Wagered: ${total_wagered:,.0f}")
    print(f"  Profit/Loss:   ${profit:+,.0f}")
    print(f"  ROI:           {roi:+.1f}%")
    
    # Break-even analysis
    print(f"\nBreak-even win rate: 52.4% (due to -110 vig)")
    if win_rate > 52.4:
        print(f"  [PASS] Model BEATS break-even by {win_rate - 52.4:.1f}%")
    else:
        print(f"  [FAIL] Model is {52.4 - win_rate:.1f}% BELOW break-even")
    
  
    print("\n" + "=" * 70)
    print("EDGE BETTING (Only bet when model disagrees with Vegas)")
    print("=" * 70)
    
    for min_edge in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        # Only bet when model differs from Vegas by at least min_edge points
        model_diff = predictions - vegas_lines.values
        confident_bets = np.abs(model_diff) >= min_edge
        
        if confident_bets.sum() == 0:
            continue
        
        edge_over = model_says_over[confident_bets]
        edge_actual = actual_went_over[confident_bets]
        edge_correct = edge_over == edge_actual
        
        edge_wins = edge_correct.sum()
        edge_total = confident_bets.sum()
        edge_losses = edge_total - edge_wins
        edge_win_rate = edge_wins / edge_total * 100
        
        edge_profit = (edge_wins * 100) - (edge_losses * 110)
        edge_wagered = edge_total * 110
        edge_roi = (edge_profit / edge_wagered) * 100 if edge_wagered > 0 else 0
        
        profitable = "[+]" if edge_roi > 0 else "[-]"
        
        print(f"\nMin Edge: {min_edge}+ points")
        print(f"  Bets: {edge_total} | Wins: {edge_wins} | Win Rate: {edge_win_rate:.1f}%")
        print(f"  {profitable} ROI: {edge_roi:+.1f}% | Profit: ${edge_profit:+,.0f}")
    

    print("\n" + "=" * 70)
    print("OVER vs UNDER BREAKDOWN")
    print("=" * 70)
    
    # OVER bets
    over_bets = model_says_over
    over_correct = (over_bets & actual_went_over)
    over_wins = over_correct.sum()
    over_total = over_bets.sum()
    over_win_rate = over_wins / over_total * 100 if over_total > 0 else 0
    
    print(f"\nOVER bets:")
    print(f"  Total: {over_total} | Wins: {over_wins} | Win Rate: {over_win_rate:.1f}%")
    
    # UNDER bets
    under_bets = ~model_says_over
    under_correct = (under_bets & ~actual_went_over)
    under_wins = under_correct.sum()
    under_total = under_bets.sum()
    under_win_rate = under_wins / under_total * 100 if under_total > 0 else 0
    
    print(f"\nUNDER bets:")
    print(f"  Total: {under_total} | Wins: {under_wins} | Win Rate: {under_win_rate:.1f}%")
    
    print("\n" + "=" * 70)
    print("MONTHLY PERFORMANCE")
    print("=" * 70)
    
    test_df = test_df.copy()
    test_df['prediction'] = predictions
    test_df['correct'] = model_correct
    test_df['month'] = pd.to_datetime(test_df['date']).dt.to_period('M')
    
    monthly = test_df.groupby('month').agg({
        'correct': ['sum', 'count']
    })
    monthly.columns = ['wins', 'total']
    monthly['win_rate'] = monthly['wins'] / monthly['total'] * 100
    monthly['profit'] = (monthly['wins'] * 100) - ((monthly['total'] - monthly['wins']) * 110)
    
    print(f"\n{'Month':<10} {'Bets':>6} {'Wins':>6} {'Win%':>8} {'Profit':>10}")
    print("-" * 45)
    for month, row in monthly.iterrows():
        profit_str = f"${row['profit']:+,.0f}"
        print(f"{str(month):<10} {row['total']:>6.0f} {row['wins']:>6.0f} {row['win_rate']:>7.1f}% {profit_str:>10}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_betting_performance()
