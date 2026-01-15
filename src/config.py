"""
Configuration settings for nba-game-predictor.
"""
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


DATABASE_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATABASE_DIR / "nba.db"

# Ensure data directory exists
DATABASE_DIR.mkdir(parents=True, exist_ok=True)



TEAM_ABBREV_MAP = {
    # Standard abbreviations used in our database
    'atl': 'ATL', 'bos': 'BOS', 'bkn': 'BKN', 'cha': 'CHA',
    'chi': 'CHI', 'cle': 'CLE', 'dal': 'DAL', 'den': 'DEN',
    'det': 'DET', 'gs': 'GSW', 'hou': 'HOU', 'ind': 'IND',
    'lac': 'LAC', 'lal': 'LAL', 'mem': 'MEM', 'mia': 'MIA',
    'mil': 'MIL', 'min': 'MIN', 'no': 'NOP', 'ny': 'NYK',
    'okc': 'OKC', 'orl': 'ORL', 'phi': 'PHI', 'phx': 'PHX',
    'por': 'POR', 'sac': 'SAC', 'sa': 'SAS', 'tor': 'TOR',
    'utah': 'UTA', 'wsh': 'WAS',
    # Alternate abbreviations (aliases)
    'uta': 'UTA', 'was': 'WAS',
    'nj': 'NJN', 'sea': 'SEA', 'van': 'VAN', 'cha_old': 'CHH',
    'nok': 'NOK', 'noh': 'NOH'
}

# Reverse mapping (NBA API -> our format)
TEAM_ABBREV_REVERSE = {v: k for k, v in TEAM_ABBREV_MAP.items()}

TEAM_FULL_NAMES = {
    'atl': 'Atlanta Hawks', 'bos': 'Boston Celtics', 'bkn': 'Brooklyn Nets',
    'cha': 'Charlotte Hornets', 'chi': 'Chicago Bulls', 'cle': 'Cleveland Cavaliers',
    'dal': 'Dallas Mavericks', 'den': 'Denver Nuggets', 'det': 'Detroit Pistons',
    'gs': 'Golden State Warriors', 'hou': 'Houston Rockets', 'ind': 'Indiana Pacers',
    'lac': 'LA Clippers', 'lal': 'Los Angeles Lakers', 'mem': 'Memphis Grizzlies',
    'mia': 'Miami Heat', 'mil': 'Milwaukee Bucks', 'min': 'Minnesota Timberwolves',
    'no': 'New Orleans Pelicans', 'ny': 'New York Knicks', 'okc': 'Oklahoma City Thunder',
    'orl': 'Orlando Magic', 'phi': 'Philadelphia 76ers', 'phx': 'Phoenix Suns',
    'por': 'Portland Trail Blazers', 'sac': 'Sacramento Kings', 'sa': 'San Antonio Spurs',
    'tor': 'Toronto Raptors', 'utah': 'Utah Jazz', 'wsh': 'Washington Wizards',
    # Aliases
    'uta': 'Utah Jazz', 'was': 'Washington Wizards'
}


CURRENT_SEASON = 2025  
TRAINING_SEASONS = 5   


ROLLING_WINDOWS = [3, 5, 10, 15]


EXCLUDE_FROM_FEATURES = [
    'season', 'date', 'regular', 'playoffs', 'away', 'home',
    'score_away', 'score_home', 'q1_away', 'q2_away', 'q3_away', 'q4_away', 'ot_away',
    'q1_home', 'q2_home', 'q3_home', 'q4_home', 'ot_home',
    'whos_favored', 'spread', 'total', 'moneyline_away', 'moneyline_home',
    'h2_spread', 'h2_total', 'id_spread', 'id_total',
    'actual_total', 'went_over', 'season_phase', 'game_id',
    'created_at', 'updated_at', 'home_season_phase', 'away_season_phase'
]


MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_GAME_COLUMNS = frozenset([
    'game_id', 'season', 'date', 'regular', 'playoffs', 'away', 'home',
    'score_away', 'score_home', 'q1_away', 'q2_away', 'q3_away', 'q4_away', 'ot_away',
    'q1_home', 'q2_home', 'q3_home', 'q4_home', 'ot_home',
    'whos_favored', 'spread', 'total', 'moneyline_away', 'moneyline_home',
    'h2_spread', 'h2_total', 'id_spread', 'id_total',
    'created_at', 'updated_at'
])

ALLOWED_TEAM_STATS_COLUMNS = frozenset([
    'id', 'team', 'date', 'season', 'games_played',
    'avg_points_scored_3', 'avg_points_scored_5', 'avg_points_scored_10', 'avg_points_scored_15',
    'avg_points_allowed_3', 'avg_points_allowed_5', 'avg_points_allowed_10', 'avg_points_allowed_15',
    'avg_total_points_3', 'avg_total_points_5', 'avg_total_points_10', 'avg_total_points_15',
    'win_rate_3', 'win_rate_5', 'win_rate_10', 'win_rate_15',
    'avg_pace_3', 'avg_pace_5', 'avg_pace_10', 'avg_pace_15',
    'season_wins', 'season_losses', 'season_avg_scored', 'season_avg_allowed',
    'updated_at'
])

ALLOWED_TEAMS = frozenset(TEAM_ABBREV_MAP.keys())
