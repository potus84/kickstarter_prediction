import csv
from os.path import join
from settings import *
import datetime
from src.data_process.constants import *

kickstater_file = 'ks-projects-201612.csv'

def read_kickstarter_file():
    # with open(join(DATA_RAW_ROOT, kickstater_file), encoding='cp1252') as csv_file:
    with open(join(DATA_RAW_ROOT, kickstater_file), encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        # csv_file = csv.reader(csv_file, delimiter=',')
        data = [row for row in csv_reader]
        return data

def preprocess_data(row):
    new_row = row
    # text_processor = TextPreprocess(row['name '])
    # print(row['name '])
    # print(text_processor.preprocess_and_tokenize_tweet())
    if row[state] == 'failed' or row[state] == 'successful':
        try:
            launched_date = datetime.datetime.strptime(row[launched], "%Y-%m-%d %H:%M:%S") #%H:%M:%S
        except ValueError:
            launched_date = datetime.datetime.strptime(row[launched], "%Y-%m-%d")
        try:
            deadline_date = datetime.datetime.strptime(row[deadline], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            deadline_date = datetime.datetime.strptime(row[deadline], "%Y-%m-%d")

        campaign_length = (deadline_date - launched_date).days
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
        elif (row[goal] != '' and row[currency] == 'USD'):
            goal_value = round(float(row[goal]), 2)
        else:
            print('No value in USD for the goal', row)
            return None
        if row[backers] != '0':
            pledge_per_packer = round(usd_pledged_value / int(row[backers]), 2)
        # Order ['name', 'main_category', 'launched_month', 'country', 'campaign_length', 'goal', 'usd_pledge', 'pledge_per_packer', 'state']
        new_row = [row[name], row[main_category], launched_date.month, row[country], campaign_length,
                   goal_value,
                   usd_pledged_value, pledge_per_packer, row[state]]
        return new_row

    else:
        return None


if __name__ == '__main__':
    data = read_kickstarter_file()
    with open(join(DATA_PREPROCESSED_ROOT, kickstater_file), 'w') as pre_processed_file:
        writer = csv.writer(pre_processed_file)
        writer.writerow(new_header)
        count_row = 0
        total_rows = len(data)
        for row in data:
            new_row = preprocess_data(row)
            if new_row is not None:
                # print(new_row)
                writer.writerow(new_row)
            count_row += 1
            print('Processed {}/{}'.format(count_row, total_rows))
        # preprocess_data(data[0])
