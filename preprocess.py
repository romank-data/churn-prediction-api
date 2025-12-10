import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def to_datetime_safe(df, col, unit='s'):
    return pd.to_datetime(df[col], unit=unit, errors='coerce')

class GamesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.end_stats_cols = [
            "end_stats.rating_points.0", "end_stats.rating_points.1",
            "end_stats.highest_break.0", "end_stats.highest_break.1",
            "end_stats.balls_potted.0", "end_stats.balls_potted.1",
            "end_stats.total_points.0", "end_stats.total_points.1",
            "end_stats.table_time.0", "end_stats.table_time.1",
            "end_stats.pot_success.0", "end_stats.pot_success.1",
            "end_stats.shot_time.0", "end_stats.shot_time.1"
        ]
        self.exclude_dt = {"end_stats.game_id", "end_stats.updated_at", "end_stats.created_at"}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # ID к строке
        df["users.0._id"] = df["users.0._id"].astype(str)
        df["users.1._id"] = df["users.1._id"].astype(str)

        # флаги и пропуски
        df["isRematch"] = df["isRematch"].fillna(0).astype(int)
        for col in self.end_stats_cols:
            df[col + "_missing"] = df[col].isna().astype(int)
            df[col] = df[col].fillna(0)

        df.drop(columns=self.exclude_dt, inplace=True, errors='ignore')
        df["game_finished"] = df["ended_at"].notna().astype(int)
        df["has_winner"] = df["winner"].notna().astype(int)

        # users.1
        cat_cols_u1 = ["users.1.username", "users.1._id", "users.1.created_at"]
        for c in cat_cols_u1:
            df[c] = df[c].fillna("unknown")
        num_cols_u1 = [
            "users.1.seconds_in_game", "users.1.online.online_sessions",
            "users.1.online_game_rating.value", "users.1.energy.count"
        ]
        for c in num_cols_u1:
            df[c + "_missing"] = df[c].isna().astype(int)
            df[c] = df[c].fillna(0)

        # users.0
        df["users.0.username"] = df["users.0.username"].fillna("unknown")
        df["users.0.online_game_rating.value_missing"] = df["users.0.online_game_rating.value"].isna().astype(int)
        df["users.0.online_game_rating.value"] = df["users.0.online_game_rating.value"].fillna(0)

        # даты
        df["started_at_dt"] = pd.to_datetime(df["started_at"], unit='s', errors='coerce').dt.tz_localize(None)
        df["ended_at_dt"] = pd.to_datetime(df["ended_at"], unit='s', errors='coerce').dt.tz_localize(None)
        df["users.0.created_at"] = pd.to_datetime(df["users.0.created_at"], errors='coerce').dt.tz_localize(None)
        df["users.1.created_at"] = pd.to_datetime(df["users.1.created_at"], errors='coerce').dt.tz_localize(None)

        df["duration_sec"] = (df["ended_at_dt"] - df["started_at_dt"]).dt.total_seconds().clip(lower=0).fillna(0)
        df["start_hour"] = df["started_at_dt"].dt.hour.fillna(-1).astype(int)
        df["start_dow"] = df["started_at_dt"].dt.dayofweek.fillna(-1).astype(int)
        df["user0_account_age_days"] = (df["started_at_dt"] - df["users.0.created_at"]).dt.days.fillna(0)
        df["user1_account_age_days"] = (df["started_at_dt"] - df["users.1.created_at"]).dt.days.fillna(0)

        # long format
        p0_cols = [
            "users.0._id", "isRematch", "game_finished", "has_winner", "duration_sec",
            "end_stats.total_points.0", "end_stats.highest_break.0", "end_stats.pot_success.0",
            "end_stats.shot_time.0", "end_stats.table_time.0", "end_stats.rating_points.0",
            "start_hour", "start_dow", "user0_account_age_days"
        ]
        p0 = df[p0_cols].rename(columns={"users.0._id": "player_id"}).copy()
        p0 = p0[p0["player_id"].str.lower().str.strip().isin(["unknown", "nan", "none"]) == False]

        p1_cols = [
            "users.1._id", "isRematch", "game_finished", "has_winner", "duration_sec",
            "end_stats.total_points.1", "end_stats.highest_break.1", "end_stats.pot_success.1",
            "end_stats.shot_time.1", "end_stats.table_time.1", "end_stats.rating_points.1",
            "start_hour", "start_dow", "user1_account_age_days"
        ]
        p1 = df[p1_cols].rename(columns={"users.1._id": "player_id"}).copy()
        p1 = p1[p1["player_id"].str.lower().str.strip().isin(["unknown", "nan", "none"]) == False]

        players_long = pd.concat([p0, p1], ignore_index=True)
        players_long["is_win"] = (players_long["has_winner"] == 1).astype(int)

        # agg
        agg_feat = players_long.groupby("player_id", as_index=False).agg(
            games_played=("is_win", "count"),
            wins=("is_win", "sum"),
            rematch_rate=("isRematch", "mean"),
            avg_duration_sec=("duration_sec", "mean"),
            avg_points=("end_stats.total_points.0", "mean"),
            avg_highest_break=("end_stats.highest_break.0", "mean"),
            avg_pot_success=("end_stats.pot_success.0", "mean"),
            avg_shot_time=("end_stats.shot_time.0", "mean"),
            avg_table_time=("end_stats.table_time.0", "mean"),
            avg_rating_delta=("end_stats.rating_points.0", "mean"),
            user_account_age_days=("user0_account_age_days", "mean"),
            start_hour_mode=("start_hour", lambda x: x.mode().iloc[0] if not x.mode().empty else -1),
            start_dow_mode=("start_dow", lambda x: x.mode().iloc[0] if not x.mode().empty else -1),
        )

        # защита от деления на 0
        agg_feat["winrate"] = agg_feat["wins"] / np.clip(agg_feat["games_played"], 1, None)

        # клиппинг
        agg_feat["avg_duration_sec"] = np.clip(agg_feat["avg_duration_sec"], 0, 3600 * 24)
        agg_feat["avg_points"] = np.clip(agg_feat["avg_points"], 0, 1e6)
        agg_feat["avg_highest_break"] = np.clip(agg_feat["avg_highest_break"], 0, 1e5)
        agg_feat["avg_rating_delta"] = np.clip(agg_feat["avg_rating_delta"], -1e4, 1e4)
        agg_feat["user_account_age_days"] = np.clip(agg_feat["user_account_age_days"], 0, 365 * 10)

        # inf и NaN
        agg_feat.replace([np.inf, -np.inf], 0, inplace=True)
        agg_feat.fillna(0, inplace=True)

        agg_feat["player_id"] = agg_feat["player_id"].astype(str)
        agg_feat.set_index("player_id", inplace=True)
        return agg_feat

class ChestsPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df["user._id"] = df["user._id"].astype(str)
        df["opened_with"] = (
            df["opened_with"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"game store": "store", "gamestore": "store", "shop": "store"})
        )
        df["chest.type"] = df["chest.type"].astype(str).str.strip().str.lower()
        df["open_at"] = pd.to_datetime(df["open_at"], unit="s", errors="coerce")

        df = df[df["user._id"].str.lower().str.strip().isin(["unknown", "nan", "none"]) == False]

        agg_chests = df.groupby("user._id").agg(
            total_chests=("chest.type", "count"),
            unique_chests=("chest.type", "nunique"),
            last_open=("open_at", "max"),
            first_open=("open_at", "min")
        ).reset_index()

        agg_chests["last_open"] = pd.to_datetime(agg_chests["last_open"], errors='coerce')
        agg_chests["first_open"] = pd.to_datetime(agg_chests["first_open"], errors='coerce')

        agg_chests["last_open_dow"] = agg_chests["last_open"].dt.dayofweek.fillna(-1).astype(int)
        agg_chests["last_open_month"] = agg_chests["last_open"].dt.month.fillna(-1).astype(int)
        agg_chests["first_open_dow"] = agg_chests["first_open"].dt.dayofweek.fillna(-1).astype(int)
        agg_chests["first_open_month"] = agg_chests["first_open"].dt.month.fillna(-1).astype(int)

        chest_type_counts = pd.crosstab(df["user._id"], df["chest.type"]).add_prefix("chest_").reset_index()
        open_with_counts = pd.crosstab(df["user._id"], df["opened_with"]).add_prefix("open_with_").reset_index()

        df_feat = agg_chests.merge(chest_type_counts, on="user._id", how="left").merge(open_with_counts, on="user._id", how="left")

        df_feat["days_between_first_last"] = (agg_chests["last_open"] - agg_chests["first_open"]).dt.days.fillna(0)
        df_feat["days_since_last"] = (pd.Timestamp.now() - agg_chests["last_open"]).dt.days.fillna(0)

        # клиппинг
        df_feat["total_chests"] = np.clip(df_feat["total_chests"], 0, 1e5)
        df_feat["days_between_first_last"] = np.clip(df_feat["days_between_first_last"], 0, 365 * 5)
        df_feat["days_since_last"] = np.clip(df_feat["days_since_last"], 0, 365 * 5)

        # защита от деления на 0
        df_feat["avg_chests_per_day"] = df_feat["total_chests"] / np.clip(df_feat["days_between_first_last"], 1, None)
        df_feat["open_with_paid"] = df_feat.get("open_with_store", 0) + df_feat.get("open_with_gems", 0)
        df_feat["paid_ratio"] = df_feat["open_with_paid"] / np.clip(df_feat["total_chests"], 1, None)
        df_feat["daily_ratio"] = df_feat.get("chest_daily", 0) / np.clip(df_feat["total_chests"], 1, None)

        # inf и NaN
        df_feat.replace([np.inf, -np.inf], 0, inplace=True)
        df_feat.fillna(0, inplace=True)

        df_feat = df_feat.drop(columns=["last_open", "first_open"], errors="ignore")
        df_feat.rename(columns={"user._id": "player_id"}, inplace=True)
        df_feat["player_id"] = df_feat["player_id"].astype(str)
        df_feat.set_index("player_id", inplace=True)
        return df_feat