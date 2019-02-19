"""
Import the DecisionTreeClassifier model.
"""
#Import the DecisionTreeClassifier
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from os.path import join
import numpy as np
from settings import *
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# from sklearn import cubist.Cubist
from sklearn.decomposition import PCA, IncrementalPCA
###########################################################################################################
##########################################################################################################
"""
Import the KickStater Dataset
"""
#Import the dataset
# dataset = pd.read_csv(join(DATA_PREPROCESSED_ROOT, 'ks-projects-201801.csv'))
dataset = pd.read_csv(join(DATA_ENGINEER_ROOT, 'ks-projects-201801.csv'))
# drop the features because it is useless now
dataset = dataset.drop(['usd_pledged', 'country'], axis=1)
# print the features that we used
print(dataset.columns)
##########################################################################################################
##########################################################################################################
"""
Transform nominal dataset into numeric
"""
gle = LabelEncoder()
main_category = np.unique(dataset['main_category'])
category_labels = gle.fit_transform(dataset['main_category'])
dataset['main_category'] = category_labels

category = np.unique(dataset['category'])
sub_category_labels = gle.fit_transform(dataset['category'])
dataset['category'] = sub_category_labels

main_state = np.unique(dataset['state'])
state_labels = gle.fit_transform(dataset['state'])
dataset['state'] = state_labels


"""
Split the data into a training and a testing set
"""
X = dataset.drop(['state'], axis=1)
y = dataset['state']
# train_features, test_features, train_targets, test_targets = \
#     train_test_split(X, y, test_size=0.33, random_state=0)


###########################################################################################################
##########################################################################################################
"""
Train different model
"""
# CART tree
# carl = DecisionTreeClassifier(criterion = 'entropy').fit(train_features, train_targets)
# XGBoost
xgb = XGBClassifier()
# Set the parameters by cross-validation
# tuned_parameters = {
#               'objective':['binary:logistic'],
#               'learning_rate': [0.05, 0.1, 0.15, 0.2], #so called `eta` value
#               'max_depth': [3, 4, 5, 6],
#               'min_child_weight': [1, 2, 3, 4, 5, 6],
#               'colsample_bytree': [0.7],
#                'reg_alpha': [1e-5, 1e-2,  0.75],
#                'reg_lambda': [1e-5, 1e-2, 0.45],
#                'subsample': [0.6, 0.8, 0.95],
#               'n_estimators': [100, 500, 1000, 1500], #number of trees, change it to 1000 for better results
#             }
param_test1 = {
'reg_alpha':[1e-5, 5e-5, 7e-5]
}
xgb = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=10,
 min_child_weight=5, gamma=0.1, subsample=0.9, colsample_bytree=0.7,
 objective= 'binary:logistic', scale_pos_weight=1, seed=27)
gsearch1 = GridSearchCV(xgb, 
 param_grid = param_test1, scoring='accuracy',n_jobs=-1, iid=False, cv=5)
gsearch1.fit(X, y)
# print(gsearch1.grid_scores_)
print(gsearch1.best_params_)
print(gsearch1.best_score_)

# scores = ['accuracy']

# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()

#     clf = GridSearchCV(xgb, tuned_parameters, cv=5,
#                        scoring=score)
#     clf.fit(train_features, train_targets)

#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()

#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = test_targets, clf.predict(test_features)
#     print(classification_report(y_true, y_pred))
#     print()
