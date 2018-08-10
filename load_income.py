from influence.dataset import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
import pandas as pd
import numpy as np


def load_income(ex_to_leave_out=None, num_examples=None):
    np.random.seed(0)
    train_fraction = 0.8
    valid_fraction = 0.0

    df = pd.read_csv('data/Adult/adult.data.csv', na_values='?')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                        'native-country']
    numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    feature_names = numerical_cols

    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], drop_first=True)
        feature_names += list(dummies.columns.values)
        df = df.join(dummies)

    # If num_examples is specified, split provided training data into train, valid, and test sets
    # Else, use the pre-split training and test sets
    if num_examples is not None:
        num_examples = min(num_examples * 2, df.shape[0])
        df = df[:num_examples]

        num_train_examples = int(train_fraction * num_examples)
        num_valid_examples = int(valid_fraction * num_examples)
        num_test_examples = num_examples - num_train_examples - num_valid_examples

        train_df = df[:num_train_examples]
        valid_df = df[num_train_examples: num_train_examples + num_valid_examples]
        test_df = df[-num_test_examples:]
    else:
        # FLAG: This doesn't work yet! For now, always pass a value num_examples
        train_df = df
        valid_df = df[0:0]
        test_df = pd.read_csv('data/adult/adult.test.csv', na_values='?')
        test_df.dropna(inplace=True)
        test_df.reset_index(drop=True, inplace=True)

    if ex_to_leave_out is not None:
        train_df.drop(ex_to_leave_out, inplace=True)
        train_df.reset_index(drop=True, inplace=True)
        number_of_elements_excluded = 1
    else:
        number_of_elements_excluded = 0

    print('Based on provided data and CLI args, there are {} train examples and {} test examples'.format(
        len(train_df.index), len(test_df.index)
    ))

    assert (len(train_df.index) + len(valid_df.index) + len(test_df.index) == num_examples - number_of_elements_excluded)

    # categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
    #                     'native-country']
    # numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    #
    # feature_names = numerical_cols

    # for col in categorical_cols:
    #     dummies_train = pd.get_dummies(train_df[col], drop_first=True)
    #     dummies_test = pd.get_dummies(test_df[col], drop_first=True)
    #     if not valid_df.empty:
    #         dummies_valid = pd.get_dummies(valid_df[col], drop_first=True)
    #     feature_names += list(dummies_train.columns.values)
    #     train_df = train_df.join(dummies_train)
    #     test_df = test_df.join(dummies_test)
    #     if not valid_df.empty:
    #         valid_df = valid_df.join(dummies_valid)

    X_train = train_df[feature_names].values
    Y_train = train_df.label.values

    if not valid_df.empty:
        X_valid = valid_df[feature_names].values
        Y_valid = valid_df.label.values

    X_test = test_df[feature_names].values
    Y_test = test_df.label.values

    train = DataSet(X_train, Y_train)
    test = DataSet(X_test, Y_test)
    if not valid_df.empty:
        validation = DataSet(X_valid, Y_valid)
    else:
        validation = None

    return base.Datasets(train=train, validation=validation, test=test)