from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
import pandas as pd

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

pipeline = joblib.load('churn_pipeline.pkl')

games_columns = [
    "_id", "game_mode", "creator_id", "status", "started_at", "ended_at", "winner",
    "score.0", "score.1", "frames_count", "isRematch", "updated_at", "created_at",
    "end_stats.rating_points.0", "end_stats.rating_points.1",
    "end_stats.highest_break.0", "end_stats.highest_break.1",
    "end_stats.balls_potted.0", "end_stats.balls_potted.1",
    "end_stats.total_points.0", "end_stats.total_points.1",
    "end_stats.table_time.0", "end_stats.table_time.1",
    "end_stats.pot_success.0", "end_stats.pot_success.1",
    "end_stats.shot_time.0", "end_stats.shot_time.1",
    "end_stats.game_id", "end_stats.updated_at", "end_stats.created_at",
    "users.0._id", "users.0.username", "users.0.created_at",
    "users.0.seconds_in_game", "users.0.online.online_sessions", "users.0.online_game_rating.value", "users.0.energy.count",
    "users.1._id", "users.1.username", "users.1.created_at",
    "users.1.seconds_in_game", "users.1.online.online_sessions", "users.1.online_game_rating.value", "users.1.energy.count"
]

chests_columns = [
    "user._id", "user.username", "chest.type", "opened_with", "open_at"
]

class RequestData(BaseModel):
    games: List[Dict[str, Any]]
    chests: List[Dict[str, Any]]

def flatten_nested_arrays(obj: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    # поддерживает dict + list -> плоские ключи типа users.0._id, score.1, end_stats.rating_points.0
    items = []
    for k, v in obj.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_arrays(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                idx_key = f"{new_key}{sep}{i}"
                if isinstance(item, dict):
                    items.extend(flatten_nested_arrays(item, idx_key, sep=sep).items())
                else:
                    items.append((idx_key, item))
        else:
            items.append((new_key, v))
    return dict(items)

def json_to_games_df(json_list: List[Dict[str, Any]], columns: List[str]) -> pd.DataFrame:
    flattened_data = [flatten_nested_arrays(item) for item in json_list]
    df = pd.DataFrame(flattened_data)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df[columns]

def json_to_chests_df(json_list: List[Dict[str, Any]], columns: List[str]) -> pd.DataFrame:
    flattened_data = [flatten_nested_arrays(item) for item in json_list]
    df = pd.DataFrame(flattened_data)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df[columns]


@app.post("/predict")
async def predict(data: RequestData):
    try:
        games_df = json_to_games_df(data.games, games_columns)
        chests_df = json_to_chests_df(data.chests, chests_columns)

        if games_df.empty:
            raise ValueError("Empty games dataframe - at least 1 game required")

        logger.info(f"Games: {games_df.shape}, Chests: {chests_df.shape}")
        
        proba_series = pipeline.predict_proba(games_df, chests_df)

        result = proba_series.to_dict()
        
        logger.info(f"Predictions for {len(result)} players")
        
        return {"probabilities": result}

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=400, detail=f"Error: {e}")
