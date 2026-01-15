"""
Daily update script for nba-game-predictor
Fetches new game results from the API and updates the database.
Can be run daily via cron job or Windows Task Scheduler.
"""
import sys
from datetime import datetime, timedelta
from typing import Dict
import pandas as pd

from .database import (
    init_database, 
    insert_game, 
    get_all_games,
    get_games_by_date_range,
    update_team_stats,
    update_prediction_result,
    get_database_stats
)
from .api_client import (
    check_api_available,
    fetch_completed_game_scores,
)
from .features import calculate_team_stats, calculate_rolling_features
from .config import CURRENT_SEASON


def determine_season(game_date: str) -> int:
    """
    Determine the NBA season for a given date.
    NBA season starts in October, so games from Oct-Dec are season+1.
    
    Parameters:
        game_date: Date in 'YYYY-MM-DD' format
        
    Returns:
        Season year (e.g., 2025 for 2024-25 season)
    """
    dt = datetime.strptime(game_date, '%Y-%m-%d')
    year = dt.year
    month = dt.month
    
   
    if month >= 10:  
        return year + 1
    else:  
        return year


def update_games_from_api(target_date: str = None) -> int:
    """
    Fetch games from API for a specific date and insert into database.
    
    Parameters:
        target_date: Date in 'YYYY-MM-DD' format, defaults to yesterday
        
    Returns:
        Number of games added/updated
    """
    if not check_api_available():
        print("Error: nba_api is not installed. Run: pip install nba_api")
        return 0
    
    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"Fetching game results for {target_date}...")
    
    game_results = fetch_completed_game_scores(target_date)
    
    if not game_results:
        print(f"No completed games found for {target_date}")
        return 0
    
    games_added = 0
    
    for result in game_results:
        season = determine_season(result['date'])
        
        game_data = {
            'season': season,
            'date': result['date'],
            'regular': True,  # API fetches regular season by default
            'playoffs': False,
            'away': result['away'],
            'home': result['home'],
            'score_away': result['score_away'],
            'score_home': result['score_home'],
            # Quarter scores would need additional API calls
            'q1_away': 0, 'q2_away': 0, 'q3_away': 0, 'q4_away': 0, 'ot_away': 0,
            'q1_home': 0, 'q2_home': 0, 'q3_home': 0, 'q4_home': 0, 'ot_home': 0,
        }
        
        try:
            insert_game(game_data)
            games_added += 1
            print(f"  Added: {result['away']} @ {result['home']}: "
                  f"{result['score_away']}-{result['score_home']}")
        except Exception as e:
            print(f"  Error adding game: {e}")
    
    print(f"Added {games_added} games to database")
    return games_added


def update_team_rolling_stats() -> None:
    """
    Recalculate rolling statistics for all teams based on current database.
    This should be run after adding new games.
    """
    print("Updating team rolling statistics...")
    
    # Get all games from database
    all_games = get_all_games()
    
    if all_games.empty:
        print("No games in database to calculate stats from")
        return
    
    # Calculate team stats using existing feature engineering
    team_games = calculate_team_stats(all_games)
    team_games = calculate_rolling_features(team_games)
    
    # Get unique teams
    teams = team_games['team'].unique()
    
    for team in teams:
        team_data = team_games[team_games['team'] == team].copy()
        
        if team_data.empty:
            continue
        
        # Get the most recent row for this team
        latest = team_data.sort_values('date').iloc[-1]
        
        # Prepare stats dict for database
        stats = {
            'date': str(latest['date']),
            'season': int(latest['season']),
            'games_played': len(team_data),
        }
        
        # Add rolling averages if they exist
        for window in [3, 5, 10, 15]:
            col = f'avg_points_scored_{window}'
            if col in latest.index and pd.notna(latest[col]):
                stats[f'avg_points_scored_{window}'] = float(latest[col])
            
            col = f'avg_points_allowed_{window}'
            if col in latest.index and pd.notna(latest[col]):
                stats[f'avg_points_allowed_{window}'] = float(latest[col])
            
            col = f'avg_total_points_{window}'
            if col in latest.index and pd.notna(latest[col]):
                stats[f'avg_total_points_{window}'] = float(latest[col])
            
            col = f'win_rate_{window}'
            if col in latest.index and pd.notna(latest[col]):
                stats[f'win_rate_{window}'] = float(latest[col])
        
        # Calculate season stats
        current_season_games = team_data[team_data['season'] == CURRENT_SEASON]
        if not current_season_games.empty:
            stats['season_wins'] = int(current_season_games['win'].sum())
            stats['season_losses'] = len(current_season_games) - stats['season_wins']
            stats['season_avg_scored'] = float(current_season_games['points_scored'].mean())
            stats['season_avg_allowed'] = float(current_season_games['points_allowed'].mean())
        
        update_team_stats(team, stats)
    
    print(f"Updated stats for {len(teams)} teams")


def update_prediction_results(target_date: str = None) -> int:
    """
    Update prediction records with actual game results.
    
    Parameters:
        target_date: Date to check, defaults to yesterday
        
    Returns:
        Number of predictions updated
    """
    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"Updating prediction results for {target_date}...")
    
    # Get games for that date from database
    games = get_games_by_date_range(target_date, target_date)
    
    updated = 0
    for _, game in games.iterrows():
        if pd.notna(game['score_away']) and pd.notna(game['score_home']):
            actual_total = int(game['score_away']) + int(game['score_home'])
            
            try:
                update_prediction_result(
                    game_date=target_date,
                    away=game['away'],
                    home=game['home'],
                    actual_total=actual_total
                )
                updated += 1
            except Exception as e:
                pass  # Prediction might not exist for this game
    
    print(f"Updated {updated} prediction results")
    return updated


def run_daily_update(days_back: int = 1) -> Dict[str, int]:
    """
    Run the complete daily update process.
    
    Parameters:
        days_back: Number of days to look back for games (default 1 = yesterday)
        
    Returns:
        Dictionary with update statistics
    """
    print("=" * 60)
    print(f"nba-game-predictor Daily Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Initialize database if needed
    init_database()
    
    stats = {
        'games_added': 0,
        'predictions_updated': 0
    }
    
    # Update games for each day
    for i in range(days_back):
        target_date = (datetime.now() - timedelta(days=i+1)).strftime('%Y-%m-%d')
        stats['games_added'] += update_games_from_api(target_date)
        stats['predictions_updated'] += update_prediction_results(target_date)
    
    # Update rolling stats
    update_team_rolling_stats()
    
    # Show database stats
    print("\n" + "=" * 60)
    print("Database Summary:")
    db_stats = get_database_stats()
    print(f"  Total games: {db_stats['total_games']}")
    print(f"  Date range: {db_stats['earliest_game']} to {db_stats['latest_game']}")
    print(f"  Seasons: {db_stats['seasons']}")
    print(f"  Total predictions: {db_stats['total_predictions']}")
    print("=" * 60)
    
    return stats


def backfill_games(start_date: str, end_date: str) -> int:
    """
    Backfill games for a date range.
    Useful for catching up on missed days.
    
    Parameters:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        Total number of games added
    """
    print(f"Backfilling games from {start_date} to {end_date}...")
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    total_added = 0
    current = start
    
    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        added = update_games_from_api(date_str)
        total_added += added
        current += timedelta(days=1)
    
    # Update rolling stats after backfill
    update_team_rolling_stats()
    
    print(f"Backfill complete. Total games added: {total_added}")
    return total_added


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='nba-game-predictor Daily Update')
    parser.add_argument('--days-back', type=int, default=1,
                        help='Number of days to look back for games')
    parser.add_argument('--backfill', action='store_true',
                        help='Run backfill mode')
    parser.add_argument('--start-date', type=str,
                        help='Start date for backfill (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                        help='End date for backfill (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    if args.backfill:
        if not args.start_date or not args.end_date:
            print("Error: --start-date and --end-date required for backfill")
            sys.exit(1)
        backfill_games(args.start_date, args.end_date)
    else:
        run_daily_update(days_back=args.days_back)
