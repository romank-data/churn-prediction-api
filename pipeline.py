import warnings
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from preprocess import GamesPreprocessor, ChestsPreprocessor
from sklearn.preprocessing import StandardScaler

class ChurnPipeline:
    def __init__(self, model=None, scale_pos_weight=1.0):
        self.games_processor = GamesPreprocessor()
        self.chests_processor = ChestsPreprocessor()
        self.scaler = StandardScaler()
        self.model = model if model else LGBMClassifier(
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            learning_rate=0.01
        )
        self.feature_names_ = None

    def fit(self, games_df, chests_df, churn_labels):
        games_features = self.games_processor.fit_transform(games_df)
        chests_features = self.chests_processor.fit_transform(chests_df)

        games_features.index = games_features.index.astype(str)
        chests_features.index = chests_features.index.astype(str)

        features = games_features.join(chests_features, how="left")

        # inf и NaN
        features.replace([np.inf, -np.inf], 0, inplace=True)
        features.fillna(0, inplace=True)

        # метки
        if isinstance(churn_labels, pd.DataFrame):
            y = churn_labels['churn'] if 'churn' in churn_labels.columns else churn_labels.iloc[:, 0]
        else:
            y = churn_labels
        y.index = y.index.astype(str)

        common_index = features.index.intersection(y.index)
        features = features.loc[common_index]
        y = y.loc[common_index]

        # масштабирование
        features_scaled = self.scaler.fit_transform(features)
        features_scaled = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)

        self.feature_names_ = features.columns.tolist()
        self.model.fit(features_scaled, y)
        return self

    def transform(self, games_df, chests_df):
        if self.feature_names_ is None:
            raise AttributeError("Pipeline not fitted")

        games_features = self.games_processor.transform(games_df)
        chests_features = self.chests_processor.transform(chests_df)

        games_features.index = games_features.index.astype(str)
        chests_features.index = chests_features.index.astype(str)

        features = games_features.join(chests_features, how="left")

        features.replace([np.inf, -np.inf], 0, inplace=True)
        features.fillna(0, inplace=True)

        for col in self.feature_names_:
            if col not in features.columns:
                features[col] = 0
        features = features[self.feature_names_]

        features_scaled = self.scaler.transform(features)
        return pd.DataFrame(features_scaled, columns=self.feature_names_, index=features.index)

    def predict_proba(self, games_df, chests_df):
        features = self.transform(games_df, chests_df)
        proba = self.model.predict_proba(features)[:, 1]
        return pd.Series(proba, index=features.index, name='prob_churn')

    @staticmethod
    def aggregate_features_for_split(games_df, chests_df):
        gproc = GamesPreprocessor()
        cproc = ChestsPreprocessor()
        try:
            g_feats = gproc.transform(games_df)
        except:
            warnings.warn("GamesPreprocessor failed, using fit_transform")
            g_feats = gproc.fit_transform(games_df.copy())
        try:
            c_feats = cproc.transform(chests_df)
        except:
            warnings.warn("ChestsPreprocessor failed, using fit_transform")
            c_feats = cproc.fit_transform(chests_df.copy())

        g_feats.index = g_feats.index.astype(str)
        c_feats.index = c_feats.index.astype(str)

        features = g_feats.join(c_feats, how="left")
        features.replace([np.inf, -np.inf], 0, inplace=True)
        features.fillna(0, inplace=True)
        return features