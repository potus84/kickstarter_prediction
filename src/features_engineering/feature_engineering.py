import pandas as pd
import pickle

from os.path import join

from settings import *
from src.features_engineering.TextPreprocess import TextPreprocess


def devide_by_zero(a, b):
    if b == 0 or a == 0:
        return 0
    else:
        return a / b


def data_engineer(input, output):

    google_bad_words_path = join(DATA_EXTERNAL_ROOT, 'google_bad_words_list.txt')
    no_swearing_words_path = join(DATA_EXTERNAL_ROOT, 'noswearing_bad_words_list.txt')
    google_bad_words_list = pickle.load(open(google_bad_words_path, 'rb'))
    no_swearing_words_list = pickle.load(open(no_swearing_words_path, 'rb'))
    df = pd.read_csv(input, encoding='latin1', low_memory=False)

    # Convert the numeric features to be correctly in order usd pledged, goal, usd_pledged_real, usd_goal_real, backer
    df.loc[:, 'usd pledged'] = pd.to_numeric(df['usd pledged'], downcast='float', errors='coerce')
    df.loc[:, 'goal'] = pd.to_numeric(df['goal'], downcast='float', errors='coerce')
    df.loc[:, 'usd_pledged_real'] = pd.to_numeric(df['usd_pledged_real'], downcast='float', errors='coerce')
    df.loc[:, 'usd_goal_real'] = pd.to_numeric(df['usd_goal_real'], downcast='float', errors='coerce')
    df.loc[:, 'backers'] = pd.to_numeric(df.backers, errors='raise', downcast='integer')

    # Convert launched, and deadline to datetime objects
    for col in ['launched', 'deadline']:
        df.loc[:, col] = pd.to_datetime(df[col], errors='coerce')

    features = df.copy()

    # Filter out the null name
    features = features[features.name.notna()]

    # Post features
    # Engineer for name
    features['contain_special_symbols'] = pd.get_dummies(df.name.str.contains(r'[.,:!?#*]'), drop_first=True)
    features['name_length'] = df.name.str.len()
    features['num_vowels'] = df.name.str.count(r'[aeiouywAEIOUYW]')
    features['num_cap'] = df.name.str.count(r'[A-Z]')
    features['num_whitespace'] = df.name.str.count(r'\s')
    features['contain_bad_word'] = df.name.apply(lambda row: TextPreprocess(row).contain_bad_words(google_bad_words_list, no_swearing_words_list))

    # Sentiment from TextBlob
    features[['subjectivity', 'polarity']] = df.apply(lambda row: pd.Series(TextPreprocess(row['name']).get_sentiment_value()), axis=1)
    features['subjectivity'] = features['subjectivity'].round(3)
    features['polarity'] = features['polarity'].round(3)

    # Datetime features
    features['launch_month'] = df.launched.dt.month
    features['duration'] = (df.deadline - df.launched).dt.days + 1


    # Deduction features
    features['pledged_per_backer'] = features.apply(lambda row: devide_by_zero(row.usd_pledged_real, row.backers), axis=1)
    features['pledged_per_backer'] = features['pledged_per_backer'].round(2)

    features['required_backers'] = features.apply(lambda row: devide_by_zero(row.usd_goal_real, row['pledged_per_backer']), axis=1)
    features['required_backers'] = features['required_backers'].round()

    features['required_backers_per_day'] = features.apply(lambda row: devide_by_zero(row['required_backers'], row['duration']), axis=1)
    features['required_backers_per_day'] = features['required_backers_per_day'].round()



    features.to_csv(path_or_buf=output, encoding='latin1', index=False)

    print(features.columns)





def main():
    kickstarter_pre_process_file = join(DATA_PREPROCESSED_ROOT, 'ks-projects-201801_pre_process.csv')
    kickstarter_engineer_file = join(DATA_ENGINEER_ROOT, 'ks-projects-201801_engineered.csv')
    data_engineer(kickstarter_pre_process_file, kickstarter_engineer_file)

main()