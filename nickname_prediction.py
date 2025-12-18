import pandas as pd
import joblib

# загружаем данные
games_df = pd.read_csv('data/online-games.csv')
chests_df = pd.read_csv('data/chests.csv')
pipeline = joblib.load('churn_pipeline.pkl')

username = "Najibelgo"

# ищем соответствующие user_id по нику среди всех пользователей
game_users_0 = games_df[games_df['users.0.username'] == username]['users.0._id'].unique()
game_users_1 = games_df[games_df['users.1.username'] == username]['users.1._id'].unique()
chest_users = chests_df[chests_df['user.username'] == username]['user._id'].unique()

# собираем все user_id этого ника
user_ids = set(game_users_0) | set(game_users_1) | set(chest_users)

if not user_ids:
    print(f"There's no '{username}' here")
else:
    print(f"Found user IDs for '{username}': {user_ids}\n")

    # все игры и сундуки по этим user_id
    games_sample = games_df[
        (games_df['users.0._id'].isin(user_ids)) |
        (games_df['users.1._id'].isin(user_ids))
        ]
    chests_sample = chests_df[chests_df['user._id'].isin(user_ids)]

    print(f"Games: {len(games_sample)}, Chests: {len(chests_sample)}\n")

    if len(games_sample) == 0:
        print("No games found for this user")
    else:
        # преобразуем в нужный формат
        games_json = games_sample.to_dict(orient="records")
        chests_json = chests_sample.to_dict(orient="records")

        # получаем имена колонок
        games_columns = games_df.columns.tolist()
        chests_columns = chests_df.columns.tolist()


        def json_to_df(json_list, columns):
            df = pd.DataFrame(json_list)
            for col in columns:
                if col not in df.columns:
                    df[col] = None
            return df[columns]


        games_df_ready = json_to_df(games_json, games_columns)
        chests_df_ready = json_to_df(chests_json, chests_columns)

        probs = pipeline.predict_proba(games_df_ready, chests_df_ready)

        X_features = pipeline.transform(games_df_ready, chests_df_ready)
        player_ids_ordered = X_features.index.tolist()

        predictions_map = dict(zip(player_ids_ordered, probs))

        print(f"Churn predictions for username '{username}':\n")

        found = False
        for uid in user_ids:
            if uid in predictions_map:
                score = predictions_map[uid]
                print(f"  User ID: {uid}")
                print(f"  Churn Probability: {score:.4f}")
                print(
                    f"  Status: {'HIGH RISK' if score > 0.6 else 'LOW RISK' if score < 0.4 else 'MEDIUM RISK'}\n")
                found = True

        if not found:
            print("Model returned predictions, but target user IDs were not found in results (unexpected).")
