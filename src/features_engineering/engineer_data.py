import csv
import math

import numpy as np
from os.path import join

from settings import *
from src.data_process.constants import *
from src.features_engineering.TextPreprocess import TextPreprocess

kickstater_file = 'ks-projects-201801.csv'

def read_preprocess_file():
    # with open(join(DATA_RAW_ROOT, kickstater_file), encoding='cp1252') as csv_file:
    with open(join(DATA_PREPROCESSED_ROOT, kickstater_file), encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        # csv_file = csv.reader(csv_file, delimiter=',')
        data = [row for row in csv_reader]
        return data

def engineer_data(row):
    if(row[name] != ''):
        processor = TextPreprocess(row[name])

        num_exclaimation = processor.num_occurrences(r'!')
        num_question_mark = processor.num_occurrences(r'\?')
        try:
            sentiment = processor.get_sentiment_value()
        except IndexError:
            print(row)
            return None
        if(row['goal'] != 0):
            log_goal = math.log10(float(row[goal]))
        else:
            return None
        # Order ['contain_exclamation', 'contain_question_mark', 'main_category',
        # st'launched_month', 'country', 'campaign_length', 'goal', 'usd_pledged', 'pledge_per_packer', 'state']
        new_row = [num_exclaimation, num_question_mark, sentiment, row['category'], row['main_category'],
                   row['launched_month'], row['country'], row['campaign_length'], row[goal],
                   row['usd_pledged'], row['pledge_per_packer'], row['state']]
        return new_row
    else:
        return None


if __name__ == '__main__':
    data = read_preprocess_file()
    with open(join(DATA_ENGINEER_ROOT, 'ks-projects-201801.csv'), 'w') as engineer_file:
        writer = csv.writer(engineer_file)
        writer.writerow(engineer_header)
        count_row = 0
        total_rows = len(data)
        for row in data:
            new_row = engineer_data(row)
            if new_row is not None:
                # print(new_row)
                writer.writerow(new_row)
            count_row += 1
            print('Processed {}/{}'.format(count_row, total_rows))
        # preprocess_data(data[0])
