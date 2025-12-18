import pandas as pd
import joblib
import random

from pipeline import ChurnPipeline

def json_to_games_df(json_list, columns):
    df = pd.DataFrame(json_list)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df[columns]

def json_to_chests_df(json_list, columns):
    df = pd.DataFrame(json_list)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df[columns]

if __name__ == "__main__":
    # загрузка полных таблиц
    games_full = pd.read_csv('data/online-games.csv')
    chests_full = pd.read_csv('data/chests.csv')

    # список всех user_id (player_id) из игр и сундуков
    ids_from_games = pd.concat([games_full["users.0._id"], games_full["users.1._id"]], ignore_index=True).dropna().unique()
    ids_from_chests = chests_full["user._id"].dropna().unique()
    common_ids = list(set(ids_from_games) & set(ids_from_chests))

    # выбираем случайного игрока
    random_id = random.choice(common_ids)
    print(f"Random player_id: {random_id}\n")

    # выбираем все игры и сундуки этого игрока
    games_sample = games_full[(games_full["users.0._id"] == random_id) | (games_full["users.1._id"] == random_id)]
    chests_sample = chests_full[chests_full["user._id"] == random_id]

    print(f"Games: {len(games_sample)}, Chests: {len(chests_sample)}\n")

    if len(games_sample) == 0:
        print("No games found for this player")
    else:
        # грязный json
        games_json = games_sample.to_dict(orient="records")
        chests_json = chests_sample.to_dict(orient="records")

        # шаблоны/имена колонок
        games_columns = games_full.columns.tolist()
        chests_columns = chests_full.columns.tolist()

        # конвертируем обратно в DataFrame, как будто json пришёл снаружи
        games_df = json_to_games_df(games_json, games_columns)
        chests_df = json_to_chests_df(chests_json, chests_columns)

        # загружаем пайплайн и делаем предсказание
        pipeline = joblib.load('churn_pipeline.pkl')
        probs = pipeline.predict_proba(games_df, chests_df)

        X_features = pipeline.transform(games_df, chests_df)
        player_ids_ordered = X_features.index.tolist()

        predictions_map = dict(zip(player_ids_ordered, probs))

        target_prob = predictions_map.get(random_id)

        if target_prob is not None:
            print(f"Churn probability for player {random_id}: {target_prob:.4f}")
            
            # Опционально: красивый риск-анализ
            if target_prob > 0.6:
                print("HIGH RISK")
            elif target_prob < 0.4:
                print("LOW RISK")
            else:
                print("MEDIUM RISK")
        else:
            print(f"Error: Target player {random_id} not found in predictions.")
