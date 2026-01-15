"""
Seed the SQLite database with data from the existing CSV file.
Run this once to initialize the database with historical data.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.database import init_database, insert_games_batch, get_database_stats
from src.config import PROJECT_ROOT


def seed_database_from_csv(csv_path: str = None) -> int:
    """
    Import all games from the CSV file into the SQLite database.
    
    Parameters:
        csv_path: Path to CSV file. Defaults to data/nba_2008-2025.csv
        
    Returns:
        Number of games imported
    """
    if csv_path is None:
        csv_path = PROJECT_ROOT / "data" / "nba_2008-2025.csv"
    
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return 0
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Initialize database
    init_database()
    
    games = []
    
    for _, row in df.iterrows():
        game = {
            'season': int(row['season']),
            'date': str(row['date']),
            'regular': int(row['regular']) if pd.notna(row.get('regular')) else 1,
            'playoffs': int(row['playoffs']) if pd.notna(row.get('playoffs')) else 0,
            'away': str(row['away']),
            'home': str(row['home']),
            'score_away': int(row['score_away']) if pd.notna(row.get('score_away')) else None,
            'score_home': int(row['score_home']) if pd.notna(row.get('score_home')) else None,
            'q1_away': int(row['q1_away']) if pd.notna(row.get('q1_away')) else 0,
            'q2_away': int(row['q2_away']) if pd.notna(row.get('q2_away')) else 0,
            'q3_away': int(row['q3_away']) if pd.notna(row.get('q3_away')) else 0,
            'q4_away': int(row['q4_away']) if pd.notna(row.get('q4_away')) else 0,
            'ot_away': int(row['ot_away']) if pd.notna(row.get('ot_away')) else 0,
            'q1_home': int(row['q1_home']) if pd.notna(row.get('q1_home')) else 0,
            'q2_home': int(row['q2_home']) if pd.notna(row.get('q2_home')) else 0,
            'q3_home': int(row['q3_home']) if pd.notna(row.get('q3_home')) else 0,
            'q4_home': int(row['q4_home']) if pd.notna(row.get('q4_home')) else 0,
            'ot_home': int(row['ot_home']) if pd.notna(row.get('ot_home')) else 0,
            'whos_favored': str(row['whos_favored']) if pd.notna(row.get('whos_favored')) else None,
            'spread': float(row['spread']) if pd.notna(row.get('spread')) else None,
            'total': float(row['total']) if pd.notna(row.get('total')) else None,
            'moneyline_away': int(row['moneyline_away']) if pd.notna(row.get('moneyline_away')) else None,
            'moneyline_home': int(row['moneyline_home']) if pd.notna(row.get('moneyline_home')) else None,
            'h2_spread': float(row['h2_spread']) if pd.notna(row.get('h2_spread')) else None,
            'h2_total': float(row['h2_total']) if pd.notna(row.get('h2_total')) else None,
            'id_spread': int(row['id_spread']) if pd.notna(row.get('id_spread')) else 0,
            'id_total': int(row['id_total']) if pd.notna(row.get('id_total')) else 0,
        }
        games.append(game)
    
    # Insert in batches
    batch_size = 1000
    total_inserted = 0
    
    for i in range(0, len(games), batch_size):
        batch = games[i:i+batch_size]
        inserted = insert_games_batch(batch)
        total_inserted += inserted
    
    # Show database stats
    print("\n" + "=" * 60)
    print("Database Summary:")
    stats = get_database_stats()
    print(f"  Total games: {stats['total_games']}")
    print(f"  Date range: {stats['earliest_game']} to {stats['latest_game']}")
    print(f"  Seasons: {min(stats['seasons'])} to {max(stats['seasons'])}")
    print(f"  Teams: {len(stats['teams'])}")
    print("=" * 60)
    
    return total_inserted


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Seed database from CSV')
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    
    args = parser.parse_args()
    
    seed_database_from_csv(args.csv)
