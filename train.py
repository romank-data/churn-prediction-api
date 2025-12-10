import pandas as pd
import numpy as np
import joblib
from pipeline import ChurnPipeline
from utils import make_churn_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

games_df = pd.read_csv('data/online-games.csv')
chests_df = pd.read_csv('data/chests.csv')
games_df['started_at_dt'] = pd.to_datetime(games_df['started_at'], unit='s', errors='coerce').dt.tz_localize(None)

# метки
churn_labels = make_churn_labels(games_df, window_days=60)
churn_labels['player_id'] = churn_labels['player_id'].astype(str)
churn_labels.set_index('player_id', inplace=True)

# сплит
X = ChurnPipeline.aggregate_features_for_split(games_df, chests_df)
X.index = X.index.astype(str)

y = churn_labels['churn'].reindex(X.index)
mask = y.notna()
X = X.loc[mask]
y = y.loc[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train.index = X_train.index.astype(str)
X_test.index = X_test.index.astype(str)
y_train.index = y_train.index.astype(str)
y_test.index = y_test.index.astype(str)

# веса
pos = y_train.sum()
neg = len(y_train) - pos
scale_pos_weight = neg / pos if pos > 0 else 1.0

# raw
train_ids = X_train.index.tolist()
test_ids = X_test.index.tolist()

def filter_games(df, ids):
    return df[
        df['users.0._id'].astype(str).isin(ids) |
        df['users.1._id'].astype(str).isin(ids)
    ]

games_train = filter_games(games_df, train_ids)
games_test = filter_games(games_df, test_ids)
chests_train = chests_df[chests_df['user._id'].astype(str).isin(train_ids)]
chests_test = chests_df[chests_df['user._id'].astype(str).isin(test_ids)]

# train
pipeline = ChurnPipeline(scale_pos_weight=scale_pos_weight)
pipeline.fit(games_train, chests_train, y_train)

# pred
X_test_final = pipeline.transform(games_test, chests_test)
X_test_final.index = X_test_final.index.astype(str)
y_test.index = y_test.index.astype(str)

common_ids = X_test_final.index.intersection(y_test.index)
X_test_final = X_test_final.loc[common_ids]
y_test_final = y_test.loc[common_ids]

# results
y_pred = pipeline.model.predict(X_test_final)
y_proba = pipeline.model.predict_proba(X_test_final)[:, 1]

print("Classification Report:")
print(classification_report(y_test_final, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test_final, y_proba):.4f}")

# сейвим обученный пайплайн
joblib.dump(pipeline, 'churn_pipeline.pkl')
print('Model is learned, tested, and saved')