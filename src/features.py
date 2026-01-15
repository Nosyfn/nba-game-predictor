import pandas as pd
from typing import List

# Rolling window sizes for feature engineering
ROLLING_WINDOWS = [3, 5, 10, 15]

def calculate_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate per-team statistics needed for feature engineering.
    Creates a long-form dataframe with each team's game-by-game stats.
    
    Parameters:
    df (pd.DataFrame): Raw NBA game data.
    
    Returns:
    pd.DataFrame: DataFrame with team-centric stats for each game.
    """
    # Create separate records for home and away teams
    home_games = df.copy()
    home_games['team'] = home_games['home']
    home_games['opponent'] = home_games['away']
    home_games['is_home'] = 1
    home_games['points_scored'] = home_games['score_home']
    home_games['points_allowed'] = home_games['score_away']
    home_games['q1_scored'] = home_games['q1_home']
    home_games['q2_scored'] = home_games['q2_home']
    home_games['q3_scored'] = home_games['q3_home']
    home_games['q4_scored'] = home_games['q4_home']
    home_games['ot_scored'] = home_games['ot_home']
    home_games['q1_allowed'] = home_games['q1_away']
    home_games['q2_allowed'] = home_games['q2_away']
    home_games['q3_allowed'] = home_games['q3_away']
    home_games['q4_allowed'] = home_games['q4_away']
    home_games['ot_allowed'] = home_games['ot_away']
    
    away_games = df.copy()
    away_games['team'] = away_games['away']
    away_games['opponent'] = away_games['home']
    away_games['is_home'] = 0
    away_games['points_scored'] = away_games['score_away']
    away_games['points_allowed'] = away_games['score_home']
    away_games['q1_scored'] = away_games['q1_away']
    away_games['q2_scored'] = away_games['q2_away']
    away_games['q3_scored'] = away_games['q3_away']
    away_games['q4_scored'] = away_games['q4_away']
    away_games['ot_scored'] = away_games['ot_away']
    away_games['q1_allowed'] = away_games['q1_home']
    away_games['q2_allowed'] = away_games['q2_home']
    away_games['q3_allowed'] = away_games['q3_home']
    away_games['q4_allowed'] = away_games['q4_home']
    away_games['ot_allowed'] = away_games['ot_home']
    
    # Combine home and away records
    team_games = pd.concat([home_games, away_games], ignore_index=True)
    
    # Calculate derived stats
    team_games['total_points'] = team_games['points_scored'] + team_games['points_allowed']
    team_games['point_diff'] = team_games['points_scored'] - team_games['points_allowed']
    team_games['win'] = (team_games['point_diff'] > 0).astype(int)
    team_games['went_to_ot'] = (team_games['ot_scored'] > 0).astype(int)
    
    # Regulation points (excluding OT)
    team_games['reg_points_scored'] = (team_games['q1_scored'] + team_games['q2_scored'] + 
                                        team_games['q3_scored'] + team_games['q4_scored'])
    team_games['reg_points_allowed'] = (team_games['q1_allowed'] + team_games['q2_allowed'] + 
                                         team_games['q3_allowed'] + team_games['q4_allowed'])
    team_games['reg_total'] = team_games['reg_points_scored'] + team_games['reg_points_allowed']
    
    # First half and second half stats
    team_games['first_half_scored'] = team_games['q1_scored'] + team_games['q2_scored']
    team_games['second_half_scored'] = team_games['q3_scored'] + team_games['q4_scored']
    team_games['first_half_allowed'] = team_games['q1_allowed'] + team_games['q2_allowed']
    team_games['second_half_allowed'] = team_games['q3_allowed'] + team_games['q4_allowed']
    
    # Remove redundant columns that were mapped to team/opponent specific columns
    cols_to_drop = [
        'home', 'away', 'score_home', 'score_away',
        'q1_home', 'q2_home', 'q3_home', 'q4_home', 'ot_home',
        'q1_away', 'q2_away', 'q3_away', 'q4_away', 'ot_away'
    ]
    # Filter to only columns that actually exist
    cols_to_drop = [c for c in cols_to_drop if c in team_games.columns]
    team_games = team_games.drop(columns=cols_to_drop)

    # Sort by team and date for rolling calculations
    team_games = team_games.sort_values(['team', 'date']).reset_index(drop=True)
    
    return team_games


def calculate_rolling_features(team_games: pd.DataFrame, windows: List[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    """
    Calculate rolling window features for each team.
    
    Parameters:
    team_games (pd.DataFrame): Team-centric game data.
    windows (List[int]): List of rolling window sizes.
    
    Returns:
    pd.DataFrame: DataFrame with rolling features added.
    """
    # Stats to calculate rolling averages for
    rolling_stats = [
        'points_scored', 'points_allowed', 'total_points', 'point_diff',
        'reg_points_scored', 'reg_points_allowed', 'reg_total',
        'first_half_scored', 'second_half_scored', 
        'first_half_allowed', 'second_half_allowed',
        'q1_scored', 'q2_scored', 'q3_scored', 'q4_scored',
        'q1_allowed', 'q2_allowed', 'q3_allowed', 'q4_allowed',
        'win', 'went_to_ot'
    ]
    
    # Calculate all rolling features and collect them
    new_columns = {}
    
    for window in windows:
        for stat in rolling_stats:
            col_name = f'{stat}_roll_{window}'
            # Use shift(1) to avoid data leakage (don't include current game)
            new_columns[col_name] = (
                team_games.groupby('team')[stat]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            )
    
    # Calculate rolling standard deviations for volatility
    volatility_stats = ['points_scored', 'points_allowed', 'total_points']
    for window in windows:
        for stat in volatility_stats:
            col_name = f'{stat}_std_{window}'
            new_columns[col_name] = (
                team_games.groupby('team')[stat]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=2).std())
            )
    
    # Add all columns at once using pd.concat to avoid fragmentation
    new_df = pd.concat([team_games, pd.DataFrame(new_columns)], axis=1)
    
    return new_df


def calculate_pace_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate pace-related features (proxy for possessions).
    Since we don't have possession data, we estimate pace from scoring patterns.
    
    Parameters:
    team_games (pd.DataFrame): Team-centric game data.
    
    Returns:
    pd.DataFrame: DataFrame with pace features added.
    """
    # Pace proxy: total points is a rough indicator of game pace
    # Higher scoring games generally have more possessions
    
    new_columns = {}
    for window in ROLLING_WINDOWS:
        # Offensive pace (points scored per game)
        new_columns[f'off_pace_{window}'] = team_games[f'points_scored_roll_{window}']
        
        # Defensive pace (points allowed per game)
        new_columns[f'def_pace_{window}'] = team_games[f'points_allowed_roll_{window}']
        
        # Combined pace
        new_columns[f'pace_{window}'] = team_games[f'total_points_roll_{window}']
        
        # Pace differential (teams that score more than allow)
        new_columns[f'pace_diff_{window}'] = (
            team_games[f'points_scored_roll_{window}'] - team_games[f'points_allowed_roll_{window}']
        )
    
    # Add all columns at once
    result = pd.concat([team_games, pd.DataFrame(new_columns)], axis=1)
    
    return result


def calculate_trend_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate trend features to capture momentum.
    
    Parameters:
    team_games (pd.DataFrame): Team-centric game data.
    
    Returns:
    pd.DataFrame: DataFrame with trend features added.
    """
    new_columns = {}
    
    # Short-term vs long-term trends
    if 3 in ROLLING_WINDOWS and 10 in ROLLING_WINDOWS:
        new_columns['scoring_trend'] = (
            team_games['points_scored_roll_3'] - team_games['points_scored_roll_10']
        )
        new_columns['defense_trend'] = (
            team_games['points_allowed_roll_3'] - team_games['points_allowed_roll_10']
        )
        new_columns['total_trend'] = (
            team_games['total_points_roll_3'] - team_games['total_points_roll_10']
        )
        new_columns['win_trend'] = (
            team_games['win_roll_3'] - team_games['win_roll_10']
        )
    
    # Days rest calculation
    team_games['date'] = pd.to_datetime(team_games['date'])
    new_columns['days_rest'] = (
        team_games.groupby('team')['date']
        .diff()
        .dt.days
        .fillna(3)  # Assume 3 days rest for first game
    )
    
    # Add columns first, then compute dependent columns
    result = pd.concat([team_games, pd.DataFrame(new_columns)], axis=1)
    
    # Back-to-back indicator
    result['is_b2b'] = (result['days_rest'] == 1).astype(int)
    
    # Long rest indicator (4+ days)
    result['long_rest'] = (result['days_rest'] >= 4).astype(int)
    
    return result


def calculate_season_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate season-level features.
    
    Parameters:
    team_games (pd.DataFrame): Team-centric game data.
    
    Returns:
    pd.DataFrame: DataFrame with season features added.
    """
    new_columns = {}
    
    # Game number in season
    new_columns['game_num'] = team_games.groupby(['team', 'season']).cumcount() + 1
    
    # Season totals to date
    new_columns['season_ppg'] = (
        team_games.groupby(['team', 'season'])['points_scored']
        .transform(lambda x: x.expanding().mean().shift(1))
    )
    new_columns['season_papg'] = (
        team_games.groupby(['team', 'season'])['points_allowed']
        .transform(lambda x: x.expanding().mean().shift(1))
    )
    new_columns['season_total_avg'] = (
        team_games.groupby(['team', 'season'])['total_points']
        .transform(lambda x: x.expanding().mean().shift(1))
    )
    new_columns['season_win_pct'] = (
        team_games.groupby(['team', 'season'])['win']
        .transform(lambda x: x.expanding().mean().shift(1))
    )
    
    # Add all columns at once
    result = pd.concat([team_games, pd.DataFrame(new_columns)], axis=1)
    
    # Season phase (early, mid, late)
    result['season_phase'] = pd.cut(
        result['game_num'],
        bins=[0, 25, 55, 100],
        labels=['early', 'mid', 'late']
    )
    
    return result


def calculate_matchup_features(df: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate head-to-head matchup features between teams.
    Uses efficient merge-based approach instead of row-by-row apply.
    
    Parameters:
    df (pd.DataFrame): Original game data.
    team_games (pd.DataFrame): Team-centric game data with rolling features.
    
    Returns:
    pd.DataFrame: Original df with matchup features added.
    """
    # Get feature columns to merge
    feature_cols = [col for col in team_games.columns if any(
        x in col for x in ['roll_', 'std_', 'pace_', 'trend', 'season_', 
                           'days_rest', 'is_b2b', 'long_rest', 'game_num']
    )]
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    team_games['date'] = pd.to_datetime(team_games['date'])
    
    # Prepare team features for merging
    home_features = team_games[['team', 'date'] + feature_cols].copy()
    home_features.columns = ['home', 'date'] + [f'home_{col}' for col in feature_cols]
    
    away_features = team_games[['team', 'date'] + feature_cols].copy()
    away_features.columns = ['away', 'date'] + [f'away_{col}' for col in feature_cols]
    
    # Merge home team features
    df = df.merge(home_features, on=['home', 'date'], how='left')
    
    # Merge away team features
    df = df.merge(away_features, on=['away', 'date'], how='left')
    
    return df


def create_over_under_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features specifically designed for Over/Under prediction.
    
    Parameters:
    df (pd.DataFrame): Game data with team features.
    
    Returns:
    pd.DataFrame: DataFrame with over/under specific features.
    """
    df = df.copy()
    
    # Target variable: actual total points
    df['actual_total'] = df['score_home'] + df['score_away']
    
    # Binary target: did game go over?
    df['went_over'] = (df['actual_total'] > df['total']).astype(int)
    
    # Combined pace features
    for window in ROLLING_WINDOWS:
        # Combined team scoring averages
        df[f'combined_scoring_{window}'] = (
            df[f'home_points_scored_roll_{window}'] + df[f'away_points_scored_roll_{window}']
        )
        
        # Combined team defense averages (points allowed)
        df[f'combined_defense_{window}'] = (
            df[f'home_points_allowed_roll_{window}'] + df[f'away_points_allowed_roll_{window}']
        )
        
        # Expected total from combined stats
        df[f'expected_total_{window}'] = (
            (df[f'home_points_scored_roll_{window}'] + df[f'away_points_allowed_roll_{window}']) / 2 +
            (df[f'away_points_scored_roll_{window}'] + df[f'home_points_allowed_roll_{window}']) / 2
        )
        
        # Pace matchup
        df[f'pace_matchup_{window}'] = (
            df[f'home_pace_{window}'] + df[f'away_pace_{window}']
        ) / 2
        
        # Combined volatility
        if f'home_total_points_std_{window}' in df.columns:
            df[f'combined_volatility_{window}'] = (
                df[f'home_total_points_std_{window}'] + df[f'away_total_points_std_{window}']
            ) / 2
    
    # Line value features
    for window in ROLLING_WINDOWS:
        # Difference between Vegas line and expected total
        df[f'line_vs_expected_{window}'] = df['total'] - df[f'expected_total_{window}']
        
        # Historical over/under tendency (if teams tend to go over/under)
        df[f'home_over_tendency_{window}'] = (
            df[f'home_total_points_roll_{window}'] - df['total']
        )
        df[f'away_over_tendency_{window}'] = (
            df[f'away_total_points_roll_{window}'] - df['total']
        )
    
    # Rest impact on totals
    df['total_rest'] = df['home_days_rest'] + df['away_days_rest']
    df['rest_diff'] = df['home_days_rest'] - df['away_days_rest']
    df['both_rested'] = ((df['home_days_rest'] >= 2) & (df['away_days_rest'] >= 2)).astype(int)
    df['both_b2b'] = (df['home_is_b2b'] + df['away_is_b2b'] == 2).astype(int)
    
    # Trend combinations
    if 'home_scoring_trend' in df.columns:
        df['combined_scoring_trend'] = df['home_scoring_trend'] + df['away_scoring_trend']
        df['combined_total_trend'] = df['home_total_trend'] + df['away_total_trend']
    
    # Season phase dummies
    df['home_early_season'] = (df['home_game_num'] <= 25).astype(int)
    df['away_early_season'] = (df['away_game_num'] <= 25).astype(int)
    
    return df
