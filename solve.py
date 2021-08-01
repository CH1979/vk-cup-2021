import joblib
from argparse import ArgumentParser

import pandas as pd

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

    fe = utils.get_friends_embeddings(users, friends, inference=True)
    df = pd.concat([
        df,
        pd.DataFrame(fe, columns=settings.FRIENDS_EMBEDDINGS)
    ], axis=1)
    
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
    
    fe = utils.get_friends_embeddings(users, friends)
    df = pd.concat([
        df,
        pd.DataFrame(fe, columns=settings.FRIENDS_EMBEDDINGS)
    ], axis=1)

    oof, models = utils.train_model(df, 'age')
    ac_score, lb_score = utils.get_score(
        df['age'],
        oof,
        show_plot=True
    )

    print('Model result:')    
    print(f'Account score: {ac_score}')
    print(f'Leaderboard score: {lb_score}')

    joblib.dump(models, 'model.pkl')
