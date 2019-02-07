import csv
from os.path import join

from settings import *
import datetime
from src.data_process.constants import *

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import re

raw_file = 'ks-projects-201801.csv'
processed_file = 'andreas-ksp-201801.csv'


header = [name_exclamation, name_questionmark, name_punctuation, name_whitespace, name_symbols, name_wovels, name_chars, name_caps, name_badmouth,
          main_category, category, launched, campaign_length, goal, daily_goal, pledge_per_backer, currency, country, state]

def preprocess_data(row):
    if row[state] == 'failed' or row[state] == 'successful':

        num_exclaimation = len(re.findall(r'!', row[name]))
        num_questionmark = len(re.findall(r'\?', row[name]))
        num_punctuation = len(re.findall(r'\.', row[name]))
        num_whitespace = len(re.findall(r'\s', row[name]))
        num_symbols = len(re.findall(r'[.,:!?#*]', row[name]))
        num_wovels = len(re.findall(r'[aeiouywAEIOUYW]', row[name]))
        if num_wovels > 0:
            num_wovels = round(num_wovels / len(row[name]), 2)
        num_chars = len(row[name])
        num_caps = len(re.findall(r'[A-Z]', row[name]))
        if num_caps > 0:
            num_caps = round(num_caps/len(row[name]), 2)

        num_badmouth = len(re.findall(r'[A-Za-z]\*[A-Za-z]', row[name]))

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
            usd_pledged_value = round(float(row[usd_pledged_real]), 2)
        elif row[usd_pledged] != '' and row[currency] == 'USD':
            usd_pledged_value = round(float(row[usd_pledged]), 2)
        else:
            print('No value in USD for the pledge', row)
            return None

        if row[usd_goal_real] != '':
            goal_value = round(float(row[usd_goal_real]), 2)
        elif row[goal] != '' and row[currency] == 'USD':
            goal_value = round(float(row[goal]), 2)
        else:
            print('No value in USD for the goal', row)
            return None

        if row[backers] != '0':
            pledge_per_packer = round(usd_pledged_value / int(row[backers]), 2)

        daily_goal_value = 0.0
        if goal_value > 0 and campaign_length_value > 0:
            daily_goal_value = round(goal_value / campaign_length_value, 2)

        return [num_exclaimation, num_questionmark, num_punctuation, num_whitespace, num_symbols, num_wovels, num_chars, num_caps, num_badmouth,
                row[main_category], row[category], launched_date.month, campaign_length_value,
                goal_value, daily_goal_value, pledge_per_packer, row[currency], row[country], row[state]]

    else:
        return None


if __name__ == '__main__':
    with open(join(DATA_RAW_ROOT, raw_file), encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        data = [row for row in csv_reader]

        with open(join(DATA_PREPROCESSED_ROOT, processed_file), 'w') as pre_processed_file:
            writer = csv.writer(pre_processed_file)
            writer.writerow(header)
            total_rows = len(data)
            print('...')
            for row in data:
                new_row = preprocess_data(row)
                if new_row is not None:
                    writer.writerow(new_row)

    dataset = pd.read_csv(join(DATA_PREPROCESSED_ROOT, 'andreas-ksp-201801.csv'))

    gle = LabelEncoder()

    dataset[main_category] = gle.fit_transform(dataset[main_category])
    dataset[category] = gle.fit_transform(dataset[category])
    dataset[currency] = gle.fit_transform(dataset[currency])
    dataset[country] = gle.fit_transform(dataset[country])
    dataset[state] = gle.fit_transform(dataset[state])

    dataset.to_csv(join(DATA_PREPROCESSED_ROOT, 'andreas-ksp-201801.csv'))

    print('done.')