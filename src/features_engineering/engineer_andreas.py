import csv
from os.path import join

from settings import *
import datetime
from src.data_process.constants import *

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import re
import random

from textblob import TextBlob

raw_file = 'ks-projects-201801.csv'
processed_file = 'andreas-ksp-201801.csv'


header = [random_mod,
            name_whitespace, name_symbols, name_wovels, name_caps, name_avr_length, name_sentiment_polarity, name_sentiment_subjectivity,
            main_category, category, launched, campaign_length, goal, currency, country,
            pledge_per_backer, required_backers, required_daily_backers,
            state]



def preprocess_data(row):
    if row[state] == 'failed' or row[state] == 'successful':

        random_mod_value = random.randint(0, 10)

        #num_exclaimation = len(re.findall(r'!', row[name]))
        #num_questionmark = len(re.findall(r'\?', row[name]))
        #num_punctuation = len(re.findall(r'\.', row[name]))
        num_whitespace = len(re.findall(r'\s', row[name]))
        num_symbols = len(re.findall(r'[.,:!?#*]', row[name]))
        num_wovels = len(re.findall(r'[aeiouywAEIOUYW]', row[name]))
        if num_wovels > 0:
            num_wovels = round(num_wovels / len(row[name]), 3)
        #num_chars = len(row[name])
        num_caps = len(re.findall(r'[A-Z]', row[name]))
        if num_caps > 0:
            num_caps = round(num_caps/len(row[name]), 3)

        words = row[name].split()
        num_avr_length = len(row[name])
        if len(words) > 0:
            num_avr_length = round(sum(len(word) for word in words) / len(words))

        name_sentiment = TextBlob(row[name]).sentiment
        num_sentiment_polarity = round(name_sentiment.polarity, 3)
        num_sentiment_subjectivity = round(name_sentiment.subjectivity, 3)
        #num_badmouth = len(re.findall(r'[A-Za-z]\*[A-Za-z]', row[name]))



        try:
            launched_date = datetime.datetime.strptime(row[launched], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            launched_date = datetime.datetime.strptime(row[launched], "%Y-%m-%d")
        try:
            deadline_date = datetime.datetime.strptime(row[deadline], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            deadline_date = datetime.datetime.strptime(row[deadline], "%Y-%m-%d")

        campaign_length_value = (deadline_date - launched_date).days

        pledge_per_packer = 0.0
        if row[usd_pledged_real] != '':
            usd_pledged_value = round(float(row[usd_pledged_real]), 3)
        elif row[usd_pledged] != '' and row[currency] == 'USD':
            usd_pledged_value = round(float(row[usd_pledged]), 3)
        else:
            print('No value in USD for the pledge', row)
            return None

        if row[usd_goal_real] != '':
            goal_value = round(float(row[usd_goal_real]))
        elif row[goal] != '' and row[currency] == 'USD':
            goal_value = round(float(row[goal]))
        else:
            print('No value in USD for the goal', row)
            return None

        if row[backers] != '0':
            pledge_per_packer = round(usd_pledged_value / int(row[backers]), 3)


        required_backers_number = 0

        if goal_value > 0 and pledge_per_packer > 0.0:
            required_backers_number = round(goal_value / pledge_per_packer)

        required_daily_backers_number = 0
        if required_backers_number > 0 and campaign_length_value > 0:
            required_daily_backers_number = round(required_backers_number / campaign_length_value)

        #daily_goal_value = 0.0
        #if goal_value > 0 and campaign_length_value > 0:
        #   daily_goal_value = round(goal_value / campaign_length_value, 2)

        return [random_mod_value,
                num_whitespace, num_symbols, num_wovels, num_caps, num_avr_length, num_sentiment_polarity, num_sentiment_subjectivity,
                row[main_category], row[category], launched_date.month, campaign_length_value, goal_value, row[currency], row[country],
                pledge_per_packer, required_backers_number, required_daily_backers_number,
                row[state]]

    else:
        return None


if __name__ == '__main__':
    num_failed = 0
    num_successful = 0
    with open(join(DATA_RAW_ROOT, raw_file), encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        data = [row for row in csv_reader]

        with open(join(DATA_PREPROCESSED_ROOT, processed_file), 'w') as pre_processed_file:
            writer = csv.writer(pre_processed_file)
            writer.writerow(header)
            total_rows = len(data)

            for row in data:
                new_row = preprocess_data(row)
                if new_row is not None:

                    if new_row[len(header)-1] == 'failed' and num_failed < num_successful + 10:
                        writer.writerow(new_row)
                        num_failed += 1
                    elif new_row[len(header)-1] == 'successful' and num_successful < num_failed + 10:
                        writer.writerow(new_row)
                        num_successful += 1

    dataset = pd.read_csv(join(DATA_PREPROCESSED_ROOT, 'andreas-ksp-201801.csv'))

    gle = LabelEncoder()

    dataset[main_category] = gle.fit_transform(dataset[main_category])
    dataset[category] = gle.fit_transform(dataset[category])
    dataset[currency] = gle.fit_transform(dataset[currency])
    dataset[country] = gle.fit_transform(dataset[country])
    dataset[state] = gle.fit_transform(dataset[state])

    dataset.to_csv(join(DATA_PREPROCESSED_ROOT, 'andreas-ksp-201801.csv'))

    print('Number of failed projects: ', num_failed)
    print('Number of successful projects: ', num_successful)