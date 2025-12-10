import pandas as pd

def make_churn_labels(df_cleaned, window_days=60, reference_date=None):
    if reference_date is None:
        reference_date = df_cleaned['started_at_dt'].max()
    if hasattr(reference_date, 'tzinfo') and reference_date.tzinfo is not None:
        reference_date = reference_date.tz_localize(None)
    if pd.api.types.is_datetime64tz_dtype(df_cleaned['started_at_dt']):
        df_cleaned['started_at_dt'] = df_cleaned['started_at_dt'].dt.tz_localize(None)

    tmp = (
        df_cleaned
        .melt(id_vars=['started_at_dt'],
              value_vars=['users.0._id', 'users.1._id'],
              var_name='side',
              value_name='player_id')
        .dropna(subset=['player_id'])
    )

    invalid_ids = ['unknown', 'nan', 'none']
    tmp = tmp[~tmp['player_id'].astype(str).str.lower().str.strip().isin(invalid_ids)]

    last_games = (
        tmp
        .groupby('player_id', as_index=False)['started_at_dt']
        .max()
        .rename(columns={'started_at_dt': 'last_game_date'})
    )

    if pd.api.types.is_datetime64tz_dtype(last_games['last_game_date']):
        last_games['last_game_date'] = last_games['last_game_date'].dt.tz_localize(None)

    last_games['days_since_last_game'] = (reference_date - last_games['last_game_date']).dt.days
    last_games['churn'] = (last_games['days_since_last_game'] > window_days).astype(int)
    last_games['player_id'] = last_games['player_id'].astype(str)

    return last_games[['player_id', 'churn']]