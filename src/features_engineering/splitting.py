import argparse

import pandas as pd
from os.path import join

from sklearn.model_selection import train_test_split


from settings import *


def split_data(data_engineered, output_train, output_test):

    # ['name', 'category', 'main_category', 'currency', 'deadline', 'goal',
    #  'launched', 'backers', 'country', 'usd pledged', 'usd_pledged_real',
    #  'usd_goal_real', 'success', 'contain_special_symbols', 'name_length',
    #  'num_vowels', 'num_cap', 'num_whitespace', 'contain_bad_word',
    #  'polarity', 'subjectivity', 'launch_month', 'duration',
    #  'pledged_per_backer']
    dataset = pd.read_csv(data_engineered, encoding='latin1', low_memory=False)
    # Remove duplicate field
    dataset = dataset.drop(['name', 'usd pledged', 'backers', 'goal', 'usd_pledged_real', 'deadline', 'launched'], axis=1)

    # Group feature
    pre_launched_features = ['category', 'main_category', 'currency', 'usd_goal_real', 'country', 'contain_special_symbols',
                             'name_length',
                             'num_vowels', 'num_cap', 'num_whitespace', 'contain_bad_word',
                             'polarity', 'subjectivity', 'launch_month', 'duration']
    post_launched_features = ['backers', 'pledged_per_backer']

    # Label encoding
    dataset = pd.get_dummies(dataset, dummy_na=True, drop_first=True)


    # Splitting the dataset
    train, test = train_test_split(dataset, shuffle=True, test_size=0.3, random_state=0)

    print('Train size', train.shape)
    print('Test size', test.shape)

    train.to_csv(path_or_buf=output_train, encoding='latin1', index=False)

    test.to_csv(path_or_buf=output_test, encoding='latin1', index=False)

def main():
    dataset = join(DATA_ENGINEER_ROOT, 'ks-projects-201801_engineered.csv')
    train_set = join(DATA_SPLIT_ROOT, 'train.csv')
    test_set = join(DATA_SPLIT_ROOT, 'test.csv')
    split_data(dataset, train_set, test_set)

main()
