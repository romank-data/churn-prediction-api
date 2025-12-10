import joblib
import pandas as pd

loaded_pipeline = joblib.load('churn_pipeline.pkl')

# возьмем условно новые df

new_games_df = pd.read_csv('data/online-games.csv')
new_chests_df = pd.read_csv('data/chests.csv')

probs = loaded_pipeline.predict_proba(new_games_df, new_chests_df)
print(probs)

#probs.to_csv('alldata.csv')
