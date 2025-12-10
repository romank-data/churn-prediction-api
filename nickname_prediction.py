import pandas as pd
import joblib

# загружаем данные
games_df = pd.read_csv('data/online-games.csv')
chests_df = pd.read_csv('data/chests.csv')
pipeline = joblib.load('churn_pipeline.pkl')

username = "straykov"

# ищем соответствующие user_id по нику среди всех пользователей
game_users_0 = games_df[games_df['users.0.username'] == username]['users.0._id'].unique()
game_users_1 = games_df[games_df['users.1.username'] == username]['users.1._id'].unique()
chest_users = chests_df[chests_df['user.username'] == username]['user._id'].unique()

# собираем все user_id этого ника
user_ids = set(game_users_0) | set(game_users_1) | set(chest_users)

if not user_ids:
    print(f"There's no {username} here")
else:
    # все игры и сундуки по этим user_id
    games_sample = games_df[(games_df['users.0._id'].isin(user_ids)) | (games_df['users.1._id'].isin(user_ids))]
    chests_sample = chests_df[chests_df['user._id'].isin(user_ids)]

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

    proba = pipeline.predict_proba(games_df_ready, chests_df_ready)
    print(f"Churn probability for player '{username}': {proba.mean():.4f}")