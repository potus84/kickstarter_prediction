import argparse

import pandas as pd
from os.path import join

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance


from settings import *


def main():
    dataset = pd.read_csv(join(DATA_ENGINEER_ROOT, 'ks-projects-201801_engineered.csv'))

    # ['name', 'category', 'main_category', 'currency', 'deadline', 'goal',
    #  'launched', 'backers', 'country', 'usd pledged', 'usd_pledged_real',
    #  'usd_goal_real', 'success', 'contain_special_symbols', 'name_length',
    #  'num_vowels', 'num_cap', 'num_whitespace', 'contain_bad_word',
    #  'polarity', 'subjectivity', 'launch_month', 'duration',
    #  'pledged_per_backer']

    # Remove duplicate field
    dataset = dataset.drop(['name', 'usd pledged', 'backers', 'goal', 'usd_pledged_real', 'deadline', 'launched'], axis=1)

    # Group feature
    pre_launched_features = ['category', 'main_category', 'currency', 'usd_goal_real', 'country', 'contain_special_symbols', 'name_length',
     'num_vowels', 'num_cap', 'num_whitespace', 'contain_bad_word',
     'polarity', 'subjectivity', 'launch_month', 'duration']
    post_launched_features = ['backers', 'pledged_per_backer']

    # Label encoding
    gle = LabelEncoder()
    category_labels = gle.fit_transform(dataset['main_category'])
    dataset['main_category'] = category_labels

    sub_category_labels = gle.fit_transform(dataset['category'])
    dataset['category'] = sub_category_labels

    country = gle.fit_transform(dataset['country'])
    dataset['country'] = country

    currency = gle.fit_transform(dataset['currency'])
    dataset['currency'] = currency

    # One hot encoding
    # pd.get_dummies(dataset, prefix=['currency', 'country', 'main_category', 'category'],
    #                columns=['currency', 'country', 'main_category', 'category'],
    #                dummy_na=True, drop_first=True)


    X = dataset.drop(['success'], axis=1)
    print('Predictors ', X.columns)
    y = dataset['success']

    # One hot

    # Splitting the dataset
    train_features, test_features, train_targets, test_targets = \
        train_test_split(X, y, test_size=0.3, random_state=0)
    print('Train size', train_features.shape)
    print('Test size', test_features.shape)

    parser = argparse.ArgumentParser()
    # For whole folder processing
    parser.add_argument('--alg', help='The training algorithm')

    args = parser.parse_args()

    if args.alg == 'CART':
        carl = DecisionTreeClassifier()
        tree = carl.fit(train_features, train_targets)
        print("The CART accuracy is: ", tree.score(test_features, test_targets) * 100, "%")
    elif args.alg == 'xgboost':
        xgb = XGBClassifier().fit(train_features, train_targets)
        print("The XGBoost accuracy is: ", xgb.score(test_features, test_targets) * 100, "%")
        plot_importance(xgb)
        plt.show()
    elif args.alg == 'rf':
        rf = RandomForestClassifier().fit(train_features, train_targets)
        print("The Random Forest accuracy is: ", rf.score(test_features, test_targets) * 100, "%")
    elif args.alg == 'catboost':
        cb = CatBoostClassifier().fit(train_features, train_targets)
        print("The Cat Boost accuracy is: ", cb.score(test_features, test_targets) * 100, "%")




main()