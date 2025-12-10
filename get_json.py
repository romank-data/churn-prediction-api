import pandas as pd
import json
import math

games_df = pd.read_csv('data/online-games.csv')
chests_df = pd.read_csv('data/chests.csv')

users_games_0 = games_df['users.0._id'].dropna().unique()
users_games_1 = games_df['users.1._id'].dropna().unique()
users_games = set(users_games_0) | set(users_games_1)

users_chests = set(chests_df['user._id'].dropna().unique())

common_users = users_games.intersection(users_chests)
print(f"Common users: {len(common_users)}")

def nan_to_none(x):
    return None if isinstance(x, float) and math.isnan(x) else x

def build_game_record(row):
    # users
    users = []
    if pd.notna(row.get('users.0._id')):
        users.append({
            "_id": row.get('users.0._id'),
            "username": row.get('users.0.username'),
            "created_at": row.get('users.0.created_at'),
            "seconds_in_game": row.get('users.0.seconds_in_game'),
            "online": {
                "online_sessions": row.get('users.0.online.online_sessions')
            },
            "online_game_rating": {
                "value": row.get('users.0.online_game_rating.value')
            },
            "energy": {
                "count": row.get('users.0.energy.count')
            }
        })
    if pd.notna(row.get('users.1._id')):
        users.append({
            "_id": row.get('users.1._id'),
            "username": row.get('users.1.username'),
            "created_at": row.get('users.1.created_at'),
            "seconds_in_game": row.get('users.1.seconds_in_game'),
            "online": {
                "online_sessions": row.get('users.1.online.online_sessions')
            },
            "online_game_rating": {
                "value": row.get('users.1.online_game_rating.value')
            },
            "energy": {
                "count": row.get('users.1.energy.count')
            }
        })

    # score как массив
    score = [
        nan_to_none(row.get('score.0')),
        nan_to_none(row.get('score.1'))
    ]

    end_stats = {
        "rating_points": [
            nan_to_none(row.get('end_stats.rating_points.0')),
            nan_to_none(row.get('end_stats.rating_points.1'))
        ],
        "highest_break": [
            nan_to_none(row.get('end_stats.highest_break.0')),
            nan_to_none(row.get('end_stats.highest_break.1'))
        ],
        "balls_potted": [
            nan_to_none(row.get('end_stats.balls_potted.0')),
            nan_to_none(row.get('end_stats.balls_potted.1'))
        ],
        "total_points": [
            nan_to_none(row.get('end_stats.total_points.0')),
            nan_to_none(row.get('end_stats.total_points.1'))
        ],
        "table_time": [
            nan_to_none(row.get('end_stats.table_time.0')),
            nan_to_none(row.get('end_stats.table_time.1'))
        ],
        "pot_success": [
            nan_to_none(row.get('end_stats.pot_success.0')),
            nan_to_none(row.get('end_stats.pot_success.1'))
        ],
        "shot_time": [
            nan_to_none(row.get('end_stats.shot_time.0')),
            nan_to_none(row.get('end_stats.shot_time.1'))
        ],
        "game_id": row.get('end_stats.game_id'),
        "updated_at": row.get('end_stats.updated_at'),
        "created_at": row.get('end_stats.created_at')
    }

    game = {
        "_id": row.get('_id'),
        "game_mode": row.get('game_mode'),
        "creator_id": row.get('creator_id'),
        "users": users,
        "status": row.get('status'),
        "started_at": row.get('started_at'),
        "ended_at": row.get('ended_at'),
        "winner": row.get('winner'),
        "score": score,
        "frames_count": row.get('frames_count'),
        "isRematch": nan_to_none(row.get('isRematch')),
        "updated_at": row.get('updated_at'),
        "created_at": row.get('created_at'),
        "end_stats": end_stats
    }
    return game

def build_chest_record(row):
    return {
        "chest": {
            "type": row.get('chest.type')
        },
        "user": {
            "_id": row.get('user._id'),
            "username": row.get('user.username')
        },
        "opened_with": row.get('opened_with'),
        "open_at": row.get('open_at')
    }

if len(common_users) == 0:
    print("No common users")
else:
    user_id = list(common_users)[0]
    print(f"Use user_id: {user_id}")

    user_games_all = games_df[
        (games_df['users.0._id'] == user_id) |
        (games_df['users.1._id'] == user_id)
    ]
    user_chests_all = chests_df[chests_df['user._id'] == user_id]

    user_games = user_games_all.sample(1)
    user_chests = user_chests_all.sample(1)

    games_payload = [build_game_record(row) for _, row in user_games.iterrows()]
    chests_payload = [build_chest_record(row) for _, row in user_chests.iterrows()]

    payload = {
        "games": games_payload,
        "chests": chests_payload
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))
