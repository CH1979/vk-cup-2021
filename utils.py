from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import settings


def get_score(y_true, y_pred):
    account_score = np.sqrt(mean_squared_error(y_true, y_pred))
    leaderboard_score = 1 / account_score
    return account_score, leaderboard_score


def get_data(inference=False):

    if inference:
        DATA_PATH = Path('/tmp/data')
        users = pd.read_csv(DATA_PATH / 'test.csv')
        education = pd.read_csv(DATA_PATH / 'testEducationFeatures.csv')
        groups = pd.read_csv(DATA_PATH / 'testGroups.csv')

    else:
        DATA_PATH = Path('./data')
        users = pd.read_csv(DATA_PATH / 'train.csv')
        education = pd.read_csv(DATA_PATH / 'trainEducationFeatures.csv')
        groups = pd.read_csv(DATA_PATH / 'trainGroups.csv')

    friends = pd.read_csv(DATA_PATH / 'friends.csv')

    return users, education, groups, friends


def create_features(users, education, groups, friends):
    df = friends.groupby('uid')['fuid'].count().reset_index()
    df = users.merge(df, how='left', on='uid')
    df = df.fillna(0)
    df['fuid'] = df['fuid'].astype(int)
    df = df.rename(columns={'fuid': 'n_friends'})

    df = df.merge(education, how='left', on='uid')

    groups = groups.groupby('uid')['gid'].count().reset_index()
    groups = groups.rename(columns={'gid': 'n_groups'})

    df = df.merge(groups, how='left', on='uid')

    return df


def train_model(df):
    folds = KFold(n_splits=settings.N_FOLDS)
    models = dict()
    oof = np.zeros(df.shape[0])
    
    for n, (trn_idx, val_idx) in enumerate(folds.split(df)):
        print(f'Fold #{n}')
        models[n] = LGBMRegressor(
            metric='rmse',
            n_estimators=500,
            learning_rate=0.01
        )
        models[n].fit(
            df.loc[trn_idx, settings.FEATURES],
            df.loc[trn_idx, settings.TARGET],
            eval_set=[(
                df.loc[val_idx, settings.FEATURES],
                df.loc[val_idx, settings.TARGET],
            )],
            verbose=20
        )
        oof[val_idx] = models[n].predict(
            df.loc[val_idx, settings.FEATURES]
        )
    return oof, models
