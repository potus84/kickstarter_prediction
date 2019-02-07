import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join
from settings import *
from src.data_process.constants import *
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, IncrementalPCA

dataset = pd.read_csv(join(DATA_PREPROCESSED_ROOT, 'andreas-ksp-201801.csv'))

dataset = dataset.drop([
#name_chars,
#daily_goal
#launched,
    #goal, main_category, campaign_length, currency, country,
    #name_questionmark, name_exclamation, name_punctuation, name_symbols, name_whitespace, name_wovels, name_caps,
    #pledge_per_backer
                        ], axis=1)

X = dataset.drop([state], axis=1)
y = dataset[state]
train_features, test_features, train_targets, test_targets = \
    train_test_split(X, y, test_size=0.33, random_state=0)


pca = PCA()
train_features = pca.fit_transform(train_features)
test_features = pca.transform(test_features)



#result = XGBClassifier().fit(train_features, train_targets)
#result = DecisionTreeClassifier(criterion = 'entropy').fit(train_features, train_targets)
result = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(train_features, train_targets)
print("The training accuracy is: ", result.score(train_features, train_targets)*100, "%")
print("The accuracy is: ", result.score(test_features, test_targets)*100, "%")