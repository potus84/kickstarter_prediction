import argparse

import pandas as pd
from os.path import join

from catboost import CatBoostClassifier
from pandas import DataFrame
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

    train_set = join(DATA_SPLIT_ROOT, 'train.csv')
    test_set = join(DATA_SPLIT_ROOT, 'test.csv')

    train = pd.read_csv(train_set, encoding='latin1', low_memory=True)
    test = pd.read_csv(test_set, encoding='latin1', low_memory=True)

    train_features = train.drop(['success'], axis=1)
    train_targets = train['success']

    test_features = test.drop(['success'], axis=1)
    test_targets = test['success']

    parser = argparse.ArgumentParser()
    # For whole folder processing
    parser.add_argument('--alg', help='The training algorithm')

    args = parser.parse_args()

    if args.alg == 'CART':
        carl = DecisionTreeClassifier()
        tree = carl.fit(train_features, train_targets)
        print("The CART accuracy is: ", tree.score(test_features, test_targets) * 100, "%")
    elif args.alg == 'xgboost':
        xgb = XGBClassifier()
        forest = xgb.fit(train_features, train_targets)
        print("The XGBoost accuracy is: ", forest.score(test_features, test_targets) * 100, "%")
        plot_importance(xgb)
        plt.show()
    elif args.alg == 'rf':
        rf = RandomForestClassifier()
        forest = rf.fit(train_features, train_targets)
        print("The Random Forest accuracy is: ", forest.score(test_features, test_targets) * 100, "%")
    elif args.alg == 'catboost':
        cb = CatBoostClassifier().fit(train_features, train_targets)
        print("The Cat Boost accuracy is: ", cb.score(test_features, test_targets) * 100, "%")




main()