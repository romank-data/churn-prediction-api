import pandas as pd
import json

games_df = pd.read_csv('data/online-games.csv')
chests_df = pd.read_csv('data/chests.csv')

# все активные игроки
users_games_0 = games_df['users.0._id'].dropna().unique()
users_games_1 = games_df['users.1._id'].dropna().unique()
users_games = list(set(users_games_0) | set(users_games_1))

if len(users_games) == 0:
    print("No active players")
else:
    user_id = users_games[0]
    print(f"User: {user_id}")

    # ВСЕ игры пользователя
    user_games_all = games_df[
        (games_df['users.0._id'] == user_id) |
        (games_df['users.1._id'] == user_id)
        ]

    # Сундуки (может быть 0)
    user_chests_all = chests_df[chests_df['user._id'] == user_id]

    print(f"Games: {len(user_games_all)}, Chests: {len(user_chests_all)}")

    if len(user_games_all) == 0:
        print("No games for user")
    else:
        payload = {
            "games": user_games_all.to_dict(orient='records'),
            "chests": user_chests_all.to_dict(orient='records')
        }

        print(json.dumps(payload, indent=2, default=str))
