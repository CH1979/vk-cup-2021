FEATURES = [
    'registered_year',
    'n_friends',
    'n_groups',
    'school_education',
    'graduation_1',
    'graduation_2',
    'graduation_3',
    'graduation_4',
    'graduation_5',
    'graduation_6',
    'graduation_7',
    'friends_min_school_education',
    'friends_max_school_education',
    'friends_mean_school_education',
    'friends_median_school_education',
]

N_COMPONENTS = 30
FRIENDS_EMBEDDINGS = [f'emb_{x}' for x in range(N_COMPONENTS)]
FEATURES.extend(FRIENDS_EMBEDDINGS)

N_FOLDS = 5
OUTPUT_LOCAL = './subs/result'
OUTPUT_REMOTE = '/var/log/result'
