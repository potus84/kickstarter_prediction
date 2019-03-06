
import argparse

import pandas as pd
from os.path import join

from catboost import CatBoostClassifier
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance

import os
os.chdir('/home/tai/Projects/kickstarter_prediction')
PROJECT_ROOT = '/home/tai/Projects/kickstarter_prediction'
DATA_RAW_ROOT = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PREPROCESSED_ROOT = os.path.join(PROJECT_ROOT, 'data', 'pre_processed')
DATA_ENGINEER_ROOT = os.path.join(PROJECT_ROOT, 'data', 'engineered')
DATA_EXTERNAL_ROOT = os.path.join(PROJECT_ROOT, 'data', 'external')
DATA_SPLIT_ROOT = os.path.join(PROJECT_ROOT, 'data', 'data_spliting')

MODELS_ROOT = os.path.join(PROJECT_ROOT, 'models')

#export PYTHONPATH=$PYTHONPATH:'pwd'


def main():    
    
    train_set = join(DATA_SPLIT_ROOT, 'train.csv')
    test_set = join(DATA_SPLIT_ROOT, 'test.csv')

    train = pd.read_csv(train_set, encoding='latin1', low_memory=True)
    test = pd.read_csv(test_set, encoding='latin1', low_memory=True)
    
  
    dataset = pd.concat([train, test])
    post_features = ['pledged_per_backer', 'required_backers', 'required_backers_per_day']
    dataset = dataset.drop(columns=post_features)
    
    features = dataset.drop(['success'], axis=1)
    targets = dataset['success']



    parser = argparse.ArgumentParser()
    # For whole folder processing
    parser.add_argument('--alg', help='The training algorithm')

    args = parser.parse_args()

    if args.alg == 'CART':
        model = DecisionTreeClassifier()        
    elif args.alg == 'xgboost':
        model = XGBClassifier()
    elif args.alg == 'rf':
        model = RandomForestClassifier()
    elif args.alg == 'catboost':
        model = CatBoostClassifier()

    scores = cross_val_score(model, features, targets, cv=5)
    print('{} 5-fold scores: '.format(args.alg), scores)
    print('{} accuracy: %.4f%% (+/- %.4f%%)'.format(args.alg) % (scores.mean() * 100.0, scores.std() * 2 * 100.0))


main()