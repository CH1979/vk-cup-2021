import joblib
from argparse import ArgumentParser

import numpy as np
from sklearn.metrics import roc_auc_score

import settings, utils

parser = ArgumentParser()
parser.add_argument('-m', dest='model', required=False)
args = parser.parse_args()

if args.model:
    models = joblib.load(args.model)
    users, education, groups, friends = utils.get_data(inference=True)
    df = utils.create_features(
        users,
        education,
        groups,
        friends
    )
    users['age'] = 0

    df['linear_pred'] = utils.get_linear_predict(users, friends)
    
    for model in models.values():
        pred = model.predict(df[settings.FEATURES])
        users['age'] += pred / settings.N_FOLDS
    
    users[['uid', 'age']].to_csv(
        settings.OUTPUT_REMOTE,
        index=False
    )
else:
    users, education, groups, friends = utils.get_data(inference=False)
    df = utils.create_features(
        users,
        education,
        groups,
        friends
    )
    
    df[settings.FRIENDS_EMBEDDINGS] = utils.get_friends_embeddings(users, friends)
    df['outlier_21'] = 0
    df.loc[df['age']==21, 'outlier_21'] = 1

    oof_21, models = utils.train_model(df, 'outlier_21')
    score_21 = roc_auc_score(
        df['outlier_21'],
        oof_21
    )
    print(f'oof mean: {np.mean(oof_21)}')
    target_mean = np.mean(df['outlier_21'])
    print(f'target mean: {target_mean}')
    print(f'Roc-auc score for outlier: {score_21}')

    oof, models = utils.train_model(df, 'age')
    ac_score, lb_score = utils.get_score(
        df['age'],
        oof,
        show_plot=True
    )

    print('Single model result:')    
    print(f'Account score: {ac_score}')
    print(f'Leaderboard score: {lb_score}')

    oof = oof * (1 - oof_21) + 21 * oof_21

    ac_score, lb_score = utils.get_score(
        df['age'],
        oof,
        show_plot=True
    )

    print('Blend result:')
    print(f'Account score: {ac_score}')
    print(f'Leaderboard score: {lb_score}')


    joblib.dump(models, 'model.pkl')
