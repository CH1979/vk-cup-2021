from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from lightgbm.plotting import plot_importance
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import settings


def get_score(y_true, y_pred, show_plot=False):
    account_score = np.sqrt(mean_squared_error(y_true, y_pred))
    leaderboard_score = 1 / account_score

    if show_plot:
        bins = len(set(y_true))
        fig, (ax_1, ax_2) = plt.subplots(ncols=2, nrows=1)
        ax_1.hist(y_true, bins=bins)
        ax_2.hist(y_pred, bins=bins)
        plt.show()

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

    right_df = education[['uid', 'school_education']]
    right_df = right_df.rename(columns={'uid': 'fuid'})

    temp = friends.merge(
        right_df,
        how='left',
        on='fuid'
    )
    temp = temp.groupby('uid')['school_education'] \
        .agg(['min', 'max', 'mean', 'median']) \
        .reset_index()
    temp = temp.rename(
        columns={
            'min': 'friends_min_school_education',
            'max': 'friends_max_school_education',
            'mean': 'friends_mean_school_education',
            'median': 'friends_median_school_education'
        }
    )

    df = df.merge(temp, how='left', on='uid')

    return df


def get_friends_embeddings(users, friends):
    user_encoder = LabelEncoder()
    friend_encoder = LabelEncoder()

    data = pd.concat([
        friends,
        friends.rename(columns={'uid': 'fuid', 'fuid': 'uid'})
    ], sort=False)
    full_data = pd.concat([
        data['uid'],
        users['uid']
    ])
    friend_encoder.fit(full_data)

    data = data.drop_duplicates()
    data = users.merge(data, how='left', on='uid')
    data = data.dropna()

    user_encoder.fit(users['uid'])
    t_uid = user_encoder.transform(data['uid'])
    t_fuid = friend_encoder.transform(data['fuid'])

    matrix = csr_matrix(
        (np.ones(len(data)), (t_uid, t_fuid)),
        shape=(max(t_uid) + 1, full_data.max() + 1)
    )
    order = user_encoder.transform(users['uid'])

    svd = TruncatedSVD(n_components=settings.N_COMPONENTS)
    result = svd.fit_transform(matrix)

    return result[order]


def train_model(df, target):
    if target == 'age':
        folds = KFold(n_splits=settings.N_FOLDS)
        splits = folds.split(df)
        model = LGBMRegressor(
            metric='rmse',
            n_estimators=800,
            learning_rate=0.01
        )
        verbose = 100
    else:
        folds = StratifiedKFold(n_splits=settings.N_FOLDS)
        splits = folds.split(df, df['outlier_21'])
        model = LGBMClassifier(
            metric = 'auc',
            n_estimators=100,
            learning_rate=0.01
        )
        verbose = 20

    models = dict()
    oof = np.zeros(df.shape[0])
    
    for n, (trn_idx, val_idx) in enumerate(splits):
        print(f'Fold #{n}')
        models[n] = model
        models[n].fit(
            df.loc[trn_idx, settings.FEATURES],
            df.loc[trn_idx, target],
            eval_set=[(
                df.loc[val_idx, settings.FEATURES],
                df.loc[val_idx, target],
            )],
            verbose=verbose
        )
        features = df.loc[val_idx, settings.FEATURES]
        if target == 'age':
            oof[val_idx] = models[n].predict(
                features
            )
        else:
            oof[val_idx] = models[n].predict_proba(
                features
            )[:, 1]
    # for model in models.values():
    #     fig = plot_importance(model)
    #     plt.show()
    return oof, models
