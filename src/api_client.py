"""
NBA API Client for fetching game data.
Uses the free nba_api library which scrapes from stats.nba.com
"""
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import sys
from pathlib import Path

# Allow running as a script (python src/api_client.py)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    __package__ = "src"

try:
    from nba_api.stats.endpoints import ScoreboardV2, LeagueGameFinder, BoxScoreTraditionalV3
    from nba_api.stats.static import teams
    NBA_API_AVAILABLE = True
except ImportError as e:
    NBA_API_AVAILABLE = False
    print(f"Warning: nba_api not installed or failed to import. Error: {e}")
    print("Run: pip install nba_api")

from .config import TEAM_ABBREV_MAP, TEAM_ABBREV_REVERSE


def check_api_available() -> bool:
    """Check if nba_api is available."""
    return NBA_API_AVAILABLE


def get_all_nba_teams() -> List[Dict[str, Any]]:
    """
    Get list of all NBA teams from the API.
    
    Returns:
        List of team dictionaries with id, name, abbreviation, etc.
    """
    if not NBA_API_AVAILABLE:
        raise ImportError("nba_api is not installed")
    
    return teams.get_teams()


def normalize_team_abbrev(nba_api_abbrev: str) -> str:
    """
    Convert NBA API team abbreviation to our format.
    
    Parameters:
        nba_api_abbrev: Abbreviation from NBA API (e.g., 'GSW')
        
    Returns:
        Our abbreviated format (e.g., 'gs')
    """
    abbrev_upper = nba_api_abbrev.upper()
    if abbrev_upper in TEAM_ABBREV_REVERSE:
        return TEAM_ABBREV_REVERSE[abbrev_upper]
    return nba_api_abbrev.lower()


def get_games_by_date(game_date: str) -> pd.DataFrame:
    """
    Fetch all NBA games for a specific date.
    
    Parameters:
        game_date: Date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with game data
    """
    if not NBA_API_AVAILABLE:
        raise ImportError("nba_api is not installed")
    
    dt = datetime.strptime(game_date, '%Y-%m-%d')
    
    time.sleep(0.6)
    
    try:
        sb = ScoreboardV2(
            game_date=dt.strftime('%Y-%m-%d'),
            league_id='00'
        )
        
        games_data = sb.get_normalized_dict()
        
        if 'GameHeader' not in games_data or not games_data['GameHeader']:
            return pd.DataFrame()
        
        games = []
        for game in games_data['GameHeader']:
            game_info = {
                'game_id': game.get('GAME_ID'),
                'date': game_date,
                'home_team_id': game.get('HOME_TEAM_ID'),
                'away_team_id': game.get('VISITOR_TEAM_ID'),
                'game_status': game.get('GAME_STATUS_ID'),  # 1=scheduled, 2=in progress, 3=final
                'game_time': game.get('GAME_STATUS_TEXT')
            }
            games.append(game_info)
        
        return pd.DataFrame(games)
        
    except Exception as e:
        print(f"Error fetching games for {game_date}: {e}")
        return pd.DataFrame()


def get_game_details(game_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch detailed box score for a specific game.
    
    Parameters:
        game_id: The NBA game ID
        
    Returns:
        Dictionary with game details including scores
    """
    if not NBA_API_AVAILABLE:
        raise ImportError("nba_api is not installed")
    
    time.sleep(0.6)  # Rate limiting
    
    try:
        boxscore = BoxScoreTraditionalV3(game_id=game_id)
        data = boxscore.get_normalized_dict()
        
        team_stats = data.get('TeamStats', [])
        
        if len(team_stats) < 2:
            return None
        
        # Team stats are ordered [away, home] or [visitor, home]
        away_stats = team_stats[0]
        home_stats = team_stats[1]
        
        game_details = {
            'game_id': game_id,
            'away_team_id': away_stats.get('TEAM_ID'),
            'away_abbrev': away_stats.get('TEAM_ABBREVIATION'),
            'home_team_id': home_stats.get('TEAM_ID'),
            'home_abbrev': home_stats.get('TEAM_ABBREVIATION'),
            'score_away': away_stats.get('PTS'),
            'score_home': home_stats.get('PTS'),
        }
        
        return game_details
        
    except Exception as e:
        print(f"Error fetching game details for {game_id}: {e}")
        return None


def get_season_games(season: str, season_type: str = 'Regular Season') -> pd.DataFrame:
    """
    Fetch all games for a specific season.
    
    Parameters:
        season: Season in format 'YYYY-YY' (e.g., '2024-25')
        season_type: 'Regular Season', 'Playoffs', or 'All Star'
        
    Returns:
        DataFrame with all games for the season
    """
    if not NBA_API_AVAILABLE:
        raise ImportError("nba_api is not installed")
    
    time.sleep(0.6)
    
    try:
        
        nba_teams = teams.get_teams()
        
        all_games = []
        processed_game_ids = set()
        
        for team in nba_teams:
            time.sleep(0.6)  #
            
            try:
                game_finder = LeagueGameFinder.LeagueGameFinder(
                    team_id_nullable=team['id'],
                    season_nullable=season,
                    season_type_nullable=season_type
                )
                
                games = game_finder.get_data_frames()[0]
                
                for _, game in games.iterrows():
                    game_id = game['GAME_ID']
                    
                    # Avoid duplicate games
                    if game_id in processed_game_ids:
                        continue
                    processed_game_ids.add(game_id)
                    
                    # Parse matchup to determine home/away
                    matchup = game.get('MATCHUP', '')
                    is_home = '@' not in matchup
                    
                    game_data = {
                        'game_id': game_id,
                        'date': game.get('GAME_DATE'),
                        'team_id': game.get('TEAM_ID'),
                        'team_abbrev': game.get('TEAM_ABBREVIATION'),
                        'matchup': matchup,
                        'is_home': is_home,
                        'win_loss': game.get('WL'),
                        'points': game.get('PTS'),
                        'plus_minus': game.get('PLUS_MINUS'),
                    }
                    all_games.append(game_data)
                    
            except Exception as e:
                print(f"Error fetching games for {team['abbreviation']}: {e}")
                continue
        
        return pd.DataFrame(all_games)
        
    except Exception as e:
        print(f"Error fetching season games: {e}")
        return pd.DataFrame()


def get_todays_games() -> pd.DataFrame:
    """
    Get today's scheduled NBA games.
    
    Returns:
        DataFrame with today's games
    """
    today = datetime.now().strftime('%Y-%m-%d')
    return get_games_by_date(today)


def get_yesterdays_games() -> pd.DataFrame:
    """
    Get yesterday's NBA games (for updating results).
    
    Returns:
        DataFrame with yesterday's games
    """
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    return get_games_by_date(yesterday)


def fetch_completed_game_scores(game_date: str) -> List[Dict[str, Any]]:
    """
    Fetch completed game scores for a specific date.
    Used for updating the database with actual results.
    
    Parameters:
        game_date: Date in 'YYYY-MM-DD' format
        
    Returns:
        List of dictionaries with game results
    """
    if not NBA_API_AVAILABLE:
        raise ImportError("nba_api is not installed")
    
    games = get_games_by_date(game_date)
    
    if games.empty:
        return []
    
    results = []
    
    for _, game in games.iterrows():
        # Only process completed games (status 3)
        if game.get('game_status') != 3:
            continue
        
        game_id = game.get('game_id')
        details = get_game_details(game_id)
        
        if details:
            # Convert team abbreviations to our format
            away = normalize_team_abbrev(details.get('away_abbrev', ''))
            home = normalize_team_abbrev(details.get('home_abbrev', ''))
            
            result = {
                'date': game_date,
                'away': away,
                'home': home,
                'score_away': details.get('score_away'),
                'score_home': details.get('score_home'),
                'game_id': game_id
            }
            results.append(result)
    
    return results


def get_team_id(team_abbrev: str) -> Optional[int]:
    """
    Get NBA API team ID from abbreviation.
    
    Parameters:
        team_abbrev: Our team abbreviation (e.g., 'gs')
        
    Returns:
        NBA API team ID or None
    """
    if not NBA_API_AVAILABLE:
        return None
    
    # Convert to NBA API format
    nba_abbrev = TEAM_ABBREV_MAP.get(team_abbrev, team_abbrev.upper())
    
    nba_teams = teams.get_teams()
    for team in nba_teams:
        if team['abbreviation'] == nba_abbrev:
            return team['id']
    
    return None


if __name__ == "__main__":
    if not NBA_API_AVAILABLE:
        print("Please install nba_api: pip install nba_api")
    else:
        print("NBA API is available!")
        
       
        print("\nNBA Teams:")
        nba_teams = get_all_nba_teams()
        for team in nba_teams[:5]:
            print(f"  {team['abbreviation']}: {team['full_name']}")
        print(f"  ... and {len(nba_teams) - 5} more")
        
        print("\nToday's games:")
        today_games = get_todays_games()
        if not today_games.empty:
            print(today_games)
        else:
            print("  No games today or unable to fetch")
