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
import graphviz
from settings import *
from xgboost import XGBClassifier
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
train_features, test_features, train_targets, test_targets = \
    train_test_split(X, y, test_size=0.33, random_state=0)


###########################################################################################################
##########################################################################################################
"""
Train different model
"""
# CART tree
# carl = DecisionTreeClassifier(criterion = 'entropy').fit(train_features, train_targets)
# XGBoost
xgb = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1500,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)\
 # .fit(train_features, train_targets)
# # Random forest
# rf = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(train_features, train_targets)

###########################################################################################################
##########################################################################################################
"""
Predict the classes of new, unseen data
"""
# new_data = pca.transform(test_features)
# prediction = model.predict(new_data)
# prediction = model.predict(test_features)

# make predictions for test data
# y_pred = model.predict(test_features)
# predictions = [round(value) for value in y_pred]

"""
Visualize the tree
"""
# dot_data = tree.export_graphviz(model, out_file="kickstater.dot", feature_names= ['main_category', 'launched_month', 'campaign_length', 'goal', 'pledge_per_packer'],
# class_names=['failed', 'successful'], filled=True, rounded=True, special_characters=True)
# graph = graphviz.Source(dot_data)
###########################################################################################################
##########################################################################################################
"""
Check the accuracy
"""
# print("The CART accuracy is: ", carl.score(test_features, test_targets)*100, "%")
# print("The XGBoost training accuracy is: ", xgb.score(train_features, train_targets)*100, "%")
# print("The XGBoost accuracy is: ", xgb.score(test_features, test_targets)*100, "%")
scores = cross_val_score(xgb, X, y, cv=5)
print(scores)
# print("The Random Forest accuracy is: ", rf.score(test_features, test_targets)*100, "%")

# print("The prediction accuracy is: ", model.score(new_data, test_targets)*100, "%")
# evaluate predictions
# accuracy = accuracy_score(test_targets, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))