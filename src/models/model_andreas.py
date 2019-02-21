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
        #name_whitespace, name_symbols, name_wovels, name_caps, <-
        #name_avr_length,
        ##name_exclamation, name_questionmark, name_punctuation, name_whitespace, name_symbols, name_wovels, name_chars, name_caps, name_badmouth,
    #name_sentiment_subjectivity,
    #name_sentiment_polarity,
    #name_avr_length,
        #main_category, category, currency, goal, launched, campaign_length, <-
        country,
        ##main_category, category, launched, campaign_length, goal,  currency, country,

        pledge_per_backer, required_backers, required_daily_backers
                        ], axis=1)

X = dataset.drop([state], axis=1)
y = dataset[state]
train_features, test_features, train_targets, test_targets = \
    train_test_split(X, y, test_size=0.20, random_state=0)


#pca = PCA()
#train_features = pca.fit_transform(train_features)
#test_features = pca.transform(test_features)

result = XGBClassifier( learning_rate=0.1, n_estimators=500, max_depth=7, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27).fit(train_features, train_targets)
#result = DecisionTreeClassifier(criterion = 'entropy', max_depth = 7).fit(train_features, train_targets)
#result = RandomForestClassifier(n_jobs=-1, n_estimators=1500, max_depth=7).fit(train_features, train_targets)
print("The training accuracy is: ", result.score(train_features, train_targets)*100, "%")
print("The accuracy is: ", result.score(test_features, test_targets)*100, "%")