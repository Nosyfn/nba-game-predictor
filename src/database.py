"""
Database module for nba-game-predictor.
Handles SQLite database operations for storing and retrieving NBA game data.
"""
import sqlite3
import re
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
from contextlib import contextmanager
import sys
from pathlib import Path

# Allow running as a script (python src/database.py)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    __package__ = "src"

from .config import (
    DATABASE_PATH,
    ALLOWED_GAME_COLUMNS,
    ALLOWED_TEAM_STATS_COLUMNS,
    ALLOWED_TEAMS
)


def _validate_column_name(column: str, allowed_columns: frozenset) -> bool:
    return column in allowed_columns


def _sanitize_identifier(name: str) -> str:
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
        raise ValueError(f"Invalid SQL identifier: {name}")
    return name


def _build_insert_query(table: str, columns: List[str]) -> str:
    """
    Safely build an INSERT query with validated column names.
    
    Parameters:
        table: Table name (must be a known table)
        columns: List of column names (must be pre-validated)
        
    Returns:
        Safe SQL INSERT query string
    """
    # Validate table name
    allowed_tables = {'games', 'team_stats', 'scheduled_games', 'predictions'}
    if table not in allowed_tables:
        raise ValueError(f"Invalid table name: {table}")
    
    # Double-check each column name is safe (alphanumeric + underscore only)
    safe_columns = []
    for col in columns:
        sanitized = _sanitize_identifier(col)
        safe_columns.append(sanitized)
    
    columns_str = ', '.join(safe_columns)
    placeholders = ', '.join(['?']*len(safe_columns))
    
    return f"INSERT OR REPLACE INTO {table} ({columns_str}) VALUES ({placeholders})"


def _validate_columns(columns: List[str], allowed_columns: frozenset) -> List[str]:
    """
    Filter columns to only include allowed ones.
    
    Parameters:
        columns: List of column names to validate
        allowed_columns: Set of allowed column names
        
    Returns:
        List of valid column names only
    """
    return [col for col in columns if _validate_column_name(col, allowed_columns)]


def _validate_team(team: str) -> bool:
    """
    Validate team abbreviation.
    
    Parameters:
        team: Team abbreviation
        
    Returns:
        True if valid team
    """
    return team.lower() in ALLOWED_TEAMS


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    """
    Initialize the SQLite database with required tables.
    Creates tables if they don't exist.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Games table - stores raw game data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                game_id INTEGER PRIMARY KEY AUTOINCREMENT,
                season INTEGER NOT NULL,
                date TEXT NOT NULL,
                regular INTEGER DEFAULT 1,
                playoffs INTEGER DEFAULT 0,
                away TEXT NOT NULL,
                home TEXT NOT NULL,
                score_away INTEGER,
                score_home INTEGER,
                q1_away INTEGER DEFAULT 0,
                q2_away INTEGER DEFAULT 0,
                q3_away INTEGER DEFAULT 0,
                q4_away INTEGER DEFAULT 0,
                ot_away INTEGER DEFAULT 0,
                q1_home INTEGER DEFAULT 0,
                q2_home INTEGER DEFAULT 0,
                q3_home INTEGER DEFAULT 0,
                q4_home INTEGER DEFAULT 0,
                ot_home INTEGER DEFAULT 0,
                whos_favored TEXT,
                spread REAL,
                total REAL,
                moneyline_away INTEGER,
                moneyline_home INTEGER,
                h2_spread REAL,
                h2_total REAL,
                id_spread INTEGER DEFAULT 0,
                id_total INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, away, home)
            )
        ''')
        
        # Team stats table - stores computed rolling features for each team
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS team_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team TEXT NOT NULL,
                date TEXT NOT NULL,
                season INTEGER NOT NULL,
                games_played INTEGER DEFAULT 0,
                -- Rolling averages for different windows (stored as JSON or individual columns)
                -- Points
                avg_points_scored_3 REAL,
                avg_points_scored_5 REAL,
                avg_points_scored_10 REAL,
                avg_points_scored_15 REAL,
                avg_points_allowed_3 REAL,
                avg_points_allowed_5 REAL,
                avg_points_allowed_10 REAL,
                avg_points_allowed_15 REAL,
                -- Total points
                avg_total_points_3 REAL,
                avg_total_points_5 REAL,
                avg_total_points_10 REAL,
                avg_total_points_15 REAL,
                -- Win rate
                win_rate_3 REAL,
                win_rate_5 REAL,
                win_rate_10 REAL,
                win_rate_15 REAL,
                -- Pace (estimated possessions)
                avg_pace_3 REAL,
                avg_pace_5 REAL,
                avg_pace_10 REAL,
                avg_pace_15 REAL,
                -- Season totals
                season_wins INTEGER DEFAULT 0,
                season_losses INTEGER DEFAULT 0,
                season_avg_scored REAL,
                season_avg_allowed REAL,
                -- Last update
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(team, date)
            )
        ''')
        
        # Today's games table - stores scheduled games for prediction
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scheduled_games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_date TEXT NOT NULL,
                away TEXT NOT NULL,
                home TEXT NOT NULL,
                scheduled_time TEXT,
                predicted_total REAL,
                prediction_confidence REAL,
                vegas_total REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(game_date, away, home)
            )
        ''')
        
        # Predictions history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_date TEXT NOT NULL,
                away TEXT NOT NULL,
                home TEXT NOT NULL,
                predicted_total REAL NOT NULL,
                vegas_total REAL,
                actual_total REAL,
                prediction_error REAL,
                model_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_date ON games(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_season ON games(season)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_teams ON games(away, home)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_team_stats_team ON team_stats(team)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_team_stats_date ON team_stats(date)')
        
        conn.commit()
        print(f"Database initialized at {DATABASE_PATH}")


def insert_game(game_data: Dict[str, Any]) -> int:
    """
    Insert a single game into the database.
    
    Security: Column names are validated against whitelist before use.
    
    Parameters:
        game_data: Dictionary containing game information
        
    Returns:
        The game_id of the inserted record
        
    Raises:
        ValueError: If invalid columns are provided
    """
    # Security: Validate all column names
    valid_columns = _validate_columns(list(game_data.keys()), ALLOWED_GAME_COLUMNS)
    
    if not valid_columns:
        raise ValueError("No valid columns provided for insert")
    
    # Filter to only valid columns
    filtered_data = {k: v for k, v in game_data.items() if k in valid_columns}
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Build query using safe query builder (no f-string with user data)
        query = _build_insert_query('games', list(filtered_data.keys()))
        cursor.execute(query, list(filtered_data.values()))
        
        conn.commit()
        return cursor.lastrowid


def insert_games_batch(games: List[Dict[str, Any]]) -> int:
    """
    Insert multiple games into the database efficiently.
    
    Security: Column names are validated against whitelist before use.
    
    Parameters:
        games: List of dictionaries containing game information
        
    Returns:
        Number of games inserted
    """
    if not games:
        return 0
    
    # Security: Validate all column names from first game
    valid_columns = _validate_columns(list(games[0].keys()), ALLOWED_GAME_COLUMNS)
    
    if not valid_columns:
        raise ValueError("No valid columns provided for insert")
        
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Build query using safe query builder (no f-string with user data)
        query = _build_insert_query('games', valid_columns)
        
        # Only use valid columns
        values = [tuple(game.get(col) for col in valid_columns) for game in games]
        
        cursor.executemany(query, values)
        
        conn.commit()
        return len(games)


def get_all_games() -> pd.DataFrame:
    """
    Retrieve all games from the database as a DataFrame.
    
    Returns:
        DataFrame containing all game data
    """
    with get_db_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM games ORDER BY date", conn)
        
    # Convert boolean columns
    if 'regular' in df.columns:
        df['regular'] = df['regular'].astype(bool)
    if 'playoffs' in df.columns:
        df['playoffs'] = df['playoffs'].astype(bool)
        
    return df


def get_games_by_season(season: int) -> pd.DataFrame:
    """
    Retrieve games for a specific season.
    
    Parameters:
        season: The season year (e.g., 2024 for 2023-24 season)
        
    Returns:
        DataFrame containing games for that season
    """
    with get_db_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM games WHERE season = ? ORDER BY date",
            conn, params=(season,)
        )
    return df


def get_games_by_date_range(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieve games within a date range.
    
    Parameters:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame containing games in the date range
    """
    with get_db_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM games WHERE date BETWEEN ? AND ? ORDER BY date",
            conn, params=(start_date, end_date)
        )
    return df


def get_team_recent_games(team: str, num_games: int = 15) -> pd.DataFrame:
    """
    Get recent games for a specific team.
    
    Security: Team abbreviation is validated before use.
    
    Parameters:
        team: Team abbreviation
        num_games: Number of recent games to retrieve
        
    Returns:
        DataFrame with team's recent games
    """
    # Security: Validate team abbreviation
    if not _validate_team(team):
        print(f"Invalid team abbreviation: {team}")
        return pd.DataFrame()
    
    # Security: Validate num_games is a reasonable integer
    num_games = min(max(1, int(num_games)), 100)
    
    with get_db_connection() as conn:
        df = pd.read_sql_query('''
            SELECT * FROM games 
            WHERE away = ? OR home = ?
            ORDER BY date DESC
            LIMIT ?
        ''', conn, params=(team, team, num_games))
    return df


def get_latest_team_stats(team: str) -> Optional[Dict[str, Any]]:
    """
    Get the most recent computed stats for a team.
    
    Security: Team abbreviation is validated before use.
    
    Parameters:
        team: Team abbreviation
        
    Returns:
        Dictionary with team stats or None if not found
    """
    # Security: Validate team abbreviation
    if not _validate_team(team):
        print(f"Invalid team abbreviation: {team}")
        return None
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM team_stats
            WHERE team = ?
            ORDER BY date DESC
            LIMIT 1
        ''', (team,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None


def update_team_stats(team: str, stats: Dict[str, Any]) -> None:
    """
    Update or insert team statistics.
    
    Security: Team abbreviation and column names are validated.
    
    Parameters:
        team: Team abbreviation
        stats: Dictionary containing stat values
    """
    # Security: Validate team
    if not _validate_team(team):
        raise ValueError(f"Invalid team abbreviation: {team}")
    
    stats['team'] = team
    stats['updated_at'] = datetime.now().isoformat()
    
    # Security: Validate column names
    valid_columns = _validate_columns(list(stats.keys()), ALLOWED_TEAM_STATS_COLUMNS)
    
    if not valid_columns:
        raise ValueError("No valid columns provided for update")
    
    filtered_stats = {k: v for k, v in stats.items() if k in valid_columns}
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Build query using safe query builder (no f-string with user data)
        query = _build_insert_query('team_stats', list(filtered_stats.keys()))
        cursor.execute(query, list(filtered_stats.values()))
        
        conn.commit()


def _validate_date_format(date_str: str) -> bool:
    """
    Validate that a date string is in YYYY-MM-DD format.
    
    Parameters:
        date_str: Date string to validate
        
    Returns:
        True if valid format
    """
    import re
    if not isinstance(date_str, str):
        return False
    # Match YYYY-MM-DD format
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    return bool(re.match(pattern, date_str))


def get_scheduled_games(game_date: str) -> pd.DataFrame:
    """
    Get scheduled games for a specific date.
    
    Security: Date format is validated.
    
    Parameters:
        game_date: Date in YYYY-MM-DD format
        
    Returns:
        DataFrame with scheduled games
    """
    # Security: Validate date format
    if not _validate_date_format(game_date):
        print(f"Invalid date format: {game_date}. Use YYYY-MM-DD.")
        return pd.DataFrame()
    
    with get_db_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM scheduled_games WHERE game_date = ?",
            conn, params=(game_date,)
        )
    return df


def add_scheduled_game(game_date: str, away: str, home: str, 
                       scheduled_time: str = None, vegas_total: float = None) -> int:
    """
    Add a scheduled game for prediction.
    
    Security: Date and team inputs are validated.
    
    Parameters:
        game_date: Date in YYYY-MM-DD format
        away: Away team abbreviation
        home: Home team abbreviation
        scheduled_time: Game time
        vegas_total: Vegas total line if available
        
    Returns:
        ID of inserted record
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Security: Validate date
    if not _validate_date_format(game_date):
        raise ValueError(f"Invalid date format: {game_date}. Use YYYY-MM-DD.")
    
    # Security: Validate teams
    if not _validate_team(away):
        raise ValueError(f"Invalid away team: {away}")
    if not _validate_team(home):
        raise ValueError(f"Invalid home team: {home}")
    
    # Security: Validate vegas_total is a number if provided
    if vegas_total is not None:
        try:
            vegas_total = float(vegas_total)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid vegas_total: {vegas_total}")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO scheduled_games 
            (game_date, away, home, scheduled_time, vegas_total)
            VALUES (?, ?, ?, ?, ?)
        ''', (game_date, away, home, scheduled_time, vegas_total))
        conn.commit()
        return cursor.lastrowid


def save_prediction(game_date: str, away: str, home: str, 
                    predicted_total: float, vegas_total: float = None,
                    model_version: str = None) -> int:
    """
    Save a prediction to the database.
    
    Security: All inputs are validated.
    
    Parameters:
        game_date: Date of the game
        away: Away team
        home: Home team
        predicted_total: Model's predicted total
        vegas_total: Vegas line if available
        model_version: Version/name of model used
        
    Returns:
        ID of inserted prediction
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Security: Validate inputs
    if not _validate_date_format(game_date):
        raise ValueError(f"Invalid date format: {game_date}")
    if not _validate_team(away):
        raise ValueError(f"Invalid away team: {away}")
    if not _validate_team(home):
        raise ValueError(f"Invalid home team: {home}")
    
    try:
        predicted_total = float(predicted_total)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid predicted_total: {predicted_total}")
    
    if vegas_total is not None:
        try:
            vegas_total = float(vegas_total)
        except (TypeError, ValueError):
            vegas_total = None
    
    # Sanitize model_version (alphanumeric and dots only)
    if model_version is not None:
        import re
        model_version = re.sub(r'[^a-zA-Z0-9._-]', '', str(model_version))[:50]
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions 
            (game_date, away, home, predicted_total, vegas_total, model_version)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (game_date, away, home, predicted_total, vegas_total, model_version))
        conn.commit()
        return cursor.lastrowid


def update_prediction_result(game_date: str, away: str, home: str, 
                             actual_total: int) -> None:
    """
    Update a prediction with the actual game result.
    
    Parameters:
        game_date: Date of the game
        away: Away team
        home: Home team
        actual_total: Actual total points scored
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE predictions
            SET actual_total = ?,
                prediction_error = predicted_total - ?
            WHERE game_date = ? AND away = ? AND home = ?
        ''', (actual_total, actual_total, game_date, away, home))
        conn.commit()


def get_database_stats() -> Dict[str, Any]:
    """
    Get summary statistics about the database.
    
    Returns:
        Dictionary with database statistics
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        stats = {}
        
        # Total games
        cursor.execute("SELECT COUNT(*) FROM games")
        stats['total_games'] = cursor.fetchone()[0]
        
        # Date range
        cursor.execute("SELECT MIN(date), MAX(date) FROM games")
        row = cursor.fetchone()
        stats['earliest_game'] = row[0]
        stats['latest_game'] = row[1]
        
        # Seasons
        cursor.execute("SELECT DISTINCT season FROM games ORDER BY season")
        stats['seasons'] = [row[0] for row in cursor.fetchall()]
        
        # Teams
        cursor.execute("SELECT DISTINCT home FROM games ORDER BY home")
        stats['teams'] = [row[0] for row in cursor.fetchall()]
        
        # Predictions count
        cursor.execute("SELECT COUNT(*) FROM predictions")
        stats['total_predictions'] = cursor.fetchone()[0]
        
        return stats


if __name__ == "__main__":
    # Initialize database when run directly
    init_database()
    print("\nDatabase tables created successfully!")
    
    # Show stats
    stats = get_database_stats()
    print(f"Total games: {stats['total_games']}")
    print(f"Seasons: {stats['seasons']}")
