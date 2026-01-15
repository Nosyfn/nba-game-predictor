"""
Today's Predictions Script for nba-game-predictor
Fetches today's scheduled games and makes predictions.
"""
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path

# Allow running as a script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    __package__ = "src"

from .api_client import get_todays_games, normalize_team_abbrev, get_all_nba_teams
from .predictor import predict_game, load_model
from .database import add_scheduled_game, save_prediction


def get_team_abbrev_by_id(team_id: int) -> str:
    """
    Get team abbreviation from team ID.
    
    Parameters:
        team_id: NBA team ID
        
    Returns:
        Team abbreviation in our format (lowercase)
    """
    teams = get_all_nba_teams()
    for team in teams:
        if team['id'] == team_id:
            return normalize_team_abbrev(team['abbreviation'])
    return None


def fetch_todays_matchups() -> List[Dict[str, Any]]:
    """
    Fetch today's scheduled games from NBA API.
    
    Returns:
        List of matchups with away_team, home_team, game_time
    """
    print(f"\nFetching games for {datetime.now().strftime('%Y-%m-%d')}...")
    
    games_df = get_todays_games()
    
    if games_df.empty:
        print("No games scheduled for today.")
        return []
    
    matchups = []
    for _, game in games_df.iterrows():
        away_team = get_team_abbrev_by_id(game['away_team_id'])
        home_team = get_team_abbrev_by_id(game['home_team_id'])
        
        if away_team and home_team:
            matchups.append({
                'away_team': away_team,
                'home_team': home_team,
                'game_id': game['game_id'],
                'game_time': game.get('game_time', 'TBD'),
                'game_status': game.get('game_status', 1),
                'vegas_total': None  # Must be added manually
            })
    
    return matchups


def display_todays_games(matchups: List[Dict[str, Any]]) -> None:
    """
    Display today's games in a formatted table.
    """
    if not matchups:
        print("\nNo games to display.")
        return
    
    print(f"\n{'='*60}")
    print(f"TODAY'S NBA GAMES - {datetime.now().strftime('%A, %B %d, %Y')}")
    print(f"{'='*60}")
    print(f"{'#':<3} {'Away':<6} {'@':<3} {'Home':<6} {'Time':<15} {'Vegas O/U':<10}")
    print("-" * 60)
    
    for i, game in enumerate(matchups, 1):
        vegas = game.get('vegas_total') or 'N/A'
        print(f"{i:<3} {game['away_team'].upper():<6} {'@':<3} {game['home_team'].upper():<6} "
              f"{game['game_time']:<15} {vegas:<10}")
    
    print("-" * 60)


def add_vegas_totals(matchups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Interactively add Vegas totals to matchups.
    
    Parameters:
        matchups: List of game matchups
        
    Returns:
        Updated matchups with Vegas totals
    """
    print("\nEnter Vegas O/U totals (or press Enter to skip):")
    print("-" * 40)
    
    for i, game in enumerate(matchups):
        prompt = f"{game['away_team'].upper()} @ {game['home_team'].upper()} O/U: "
        user_input = input(prompt).strip()
        
        if user_input:
            try:
                matchups[i]['vegas_total'] = float(user_input)
            except ValueError:
                print(f"  Invalid number, skipping...")
    
    return matchups


def predict_todays_games(matchups: List[Dict[str, Any]], 
                         save_to_db: bool = True) -> List[Dict[str, Any]]:
    """
    Make predictions for today's games.
    
    Parameters:
        matchups: List of game matchups with vegas_total
        save_to_db: Whether to save predictions to database
        
    Returns:
        List of predictions
    """
    print(f"\n{'='*60}")
    print("PREDICTIONS")
    print(f"{'='*60}")
    
    model = load_model()
    if model is None:
        print("Error: No trained model found. Run train_model.py first.")
        return []
    
    predictions = []
    today = datetime.now().strftime('%Y-%m-%d')
    
    for game in matchups:
        result = predict_game(
            away_team=game['away_team'],
            home_team=game['home_team'],
            vegas_total=game.get('vegas_total')
        )
        
        if 'error' in result:
            print(f"\n{game['away_team'].upper()} @ {game['home_team'].upper()}: Error - {result['error']}")
            continue
        
        predictions.append(result)
        
        # Display prediction
        print(f"\n{game['away_team'].upper()} @ {game['home_team'].upper()}")
        print(f"  Predicted Total: {result['predicted_total']:.1f}")
        
        if game.get('vegas_total'):
            print(f"  Vegas Total:     {game['vegas_total']}")
            print(f"  Edge:            {result.get('edge', 0):+.1f} points")
            print(f"  Recommendation:  {result.get('recommendation', 'N/A')}")
            
            # Betting advice based on edge
            edge = abs(result.get('edge', 0))
            if edge >= 5:
                print(f"  HIGH CONFIDENCE - Strong edge detected!")
            elif edge >= 3:
                print(f"  MODERATE edge")
            else:
                print(f"  Small edge - consider skipping")
        
        # Save to database
        if save_to_db and game.get('vegas_total'):
            try:
                # Add to scheduled games
                add_scheduled_game(
                    game_date=today,
                    away=game['away_team'],
                    home=game['home_team'],
                    vegas_total=game.get('vegas_total')
                )
                
                # Save prediction
                save_prediction(
                    game_date=today,
                    away=game['away_team'],
                    home=game['home_team'],
                    predicted_total=result['predicted_total'],
                    vegas_total=game.get('vegas_total'),
                    model_version='v1.0'
                )
            except Exception as e:
                print(f"  (Note: Could not save to DB: {e})")
    
    return predictions


def run_interactive():
    """
    Run the interactive prediction workflow.
    """
    print("\n" + "=" * 60)
    print("nba-game-predictor - Today's Game Predictions")
    print("=" * 60)
    
    # Fetch today's games
    matchups = fetch_todays_matchups()
    
    if not matchups:
        print("\nNo games found for today. Try again later or check if games are scheduled.")
        return
    
    # Display games
    display_todays_games(matchups)
    
    # Ask if user wants to add Vegas totals
    print("\nWould you like to enter Vegas O/U totals for better predictions?")
    choice = input("Enter 'y' for yes, or press Enter to predict without Vegas lines: ").strip().lower()
    
    if choice == 'y':
        matchups = add_vegas_totals(matchups)
    
    # Make predictions
    predictions = predict_todays_games(matchups)
    
    # Summary
    if predictions:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total games predicted: {len(predictions)}")
        
        # Count recommendations
        overs = sum(1 for p in predictions if p.get('recommendation') == 'OVER')
        unders = sum(1 for p in predictions if p.get('recommendation') == 'UNDER')
        
        if overs or unders:
            print(f"OVER recommendations:  {overs}")
            print(f"UNDER recommendations: {unders}")
            
            # High confidence picks
            high_conf = [p for p in predictions if abs(p.get('edge', 0)) >= 5]
            if high_conf:
                print(f"\nHIGH CONFIDENCE PICKS (5+ point edge):")
                for p in high_conf:
                    print(f"   {p['away_team'].upper()} @ {p['home_team'].upper()}: "
                          f"{p['recommendation']} (edge: {p['edge']:+.1f})")


def quick_predict(away: str, home: str, vegas_total: float = None) -> Dict[str, Any]:
    """
    Quick prediction for a single game.
    
    Parameters:
        away: Away team abbreviation
        home: Home team abbreviation  
        vegas_total: Optional Vegas O/U line
        
    Returns:
        Prediction result
    """
    result = predict_game(away_team=away, home_team=home, vegas_total=vegas_total)
    
    print(f"\n{away.upper()} @ {home.upper()}")
    if 'error' in result:
        print(f"  Error: {result['error']}")
    else:
        print(f"  Predicted Total: {result['predicted_total']:.1f}")
        if vegas_total:
            print(f"  Vegas Total:     {vegas_total}")
            print(f"  Edge:            {result.get('edge', 0):+.1f}")
            print(f"  Recommendation:  {result.get('recommendation', 'N/A')}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='nba-game-predictor Today\'s Predictions')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Quick mode: just fetch and predict without interaction')
    parser.add_argument('--away', type=str, help='Away team for single game prediction')
    parser.add_argument('--home', type=str, help='Home team for single game prediction')
    parser.add_argument('--vegas', type=float, help='Vegas total for single game prediction')
    
    args = parser.parse_args()
    
    if args.away and args.home:
        # Single game prediction
        quick_predict(args.away, args.home, args.vegas)
    elif args.quick:
        # Quick mode - fetch and predict without interaction
        matchups = fetch_todays_matchups()
        if matchups:
            display_todays_games(matchups)
            predict_todays_games(matchups, save_to_db=False)
    else:
        # Interactive mode
        run_interactive()
