"""
This is a boilerplate pipeline 'PreparacaoDados'
generated using Kedro 0.18.7
"""
import mlflow
from sklearn.model_selection import train_test_split


def prepare_data(df_raw, shot_type):

    # delete rows with NA values
    df_prep = df_raw.dropna()

    # filter shot_type => 2PT Field Goal
    df_prep = df_prep[df_prep['shot_type'] == shot_type]

    # select column
    cols = ['lat', 'lon', 'minutes_remaining', 'period',
            'playoffs', 'shot_distance', 'shot_made_flag']
    df_prep = df_prep[cols]

    # transform shot_made_flag to int
    df_prep['shot_made_flag'] = df_prep['shot_made_flag'].astype('int64')

    return df_prep


""" def train_test_split(data, test_size, test_split_random_state, target):

    # stratified and randomized split
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=test_split_random_state)

    for train_index, test_index in split.split(data, data[target]):
        df_train = data.iloc[train_index].copy()
        df_test = data.iloc[test_index].copy()

    return df_train, df_test """


def split_train_test(data, test_size, test_split_random_state, target):

    df_train, df_test = train_test_split(
        data, test_size=test_size, stratify=data[target], random_state=test_split_random_state
    )

    return df_train, df_test


def data_metrics(data_train, data_test):

    metrics = {
        'train_rows': data_train.shape[0],
        'test_rows': data_test.shape[0],
        'num_features': data_train.shape[1],
    }

    return {
        key: {'value': int(value), 'step': 1}
        for key, value in metrics.items()
    }
