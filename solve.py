import joblib
from argparse import ArgumentParser

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
    oof, models = utils.train_model(df)
    ac_score, lb_score = utils.get_score(
        df[settings.TARGET],
        oof
    )
    
    print(f'Account score: {ac_score}')
    print(f'Leaderboard score: {lb_score}')

    joblib.dump(models, 'model.pkl')
