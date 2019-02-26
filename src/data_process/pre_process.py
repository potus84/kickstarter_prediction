import pandas as pd
import numpy as np
from os.path import join
from settings import *

def pre_process_data(file, output):
    """

    :param file: The raw csv file
    :param output: The preprocessed csv file with only 'successful' and 'failed'
    :return:
    """
    df = pd.read_csv(file, encoding='latin1', low_memory=False)

    # Standardize the columns names
    # df.columns = ['ID', 'name', 'category', 'main_category', 'currency', 'deadline',
    #        'goal', 'launched', 'pledged', 'state', 'backers', 'country',
    #        'usd pledged', 'usd_pledged_real', 'usd_goal_real']

    # Filter out only project with tag successful or failed
    df = df[(df.state == 'successful') | (df.state == 'failed')]

    # # Convert the numeric features to be correctly in order usd pledged, goal, usd_pledged_real, usd_goal_real, backer
    # df.loc[:, 'usd pledged'] = pd.to_numeric(df['usd pledged'], downcast='float', errors='coerce')
    # df.loc[:, 'goal'] = pd.to_numeric(df['goal'], downcast='float', errors='coerce')
    # df.loc[:, 'usd_pledged_real'] = pd.to_numeric(df['usd_pledged_real'], downcast='float', errors='coerce')
    # df.loc[:, 'usd_goal_real'] = pd.to_numeric(df['usd_pledged_real'], downcast='float', errors='coerce')
    # df.loc[:, 'backers'] = pd.to_numeric(df.backers, errors='raise', downcast='integer')
    #
    # # Convert launched, and deadline to datetime objects
    # for col in ['launched', 'deadline']:
    #     df.loc[:, col] = pd.to_datetime(df[col], errors='coerce')

    # Make binary output for model
    df['success'] = np.where(df.state == 'successful', 1, 0)

    # Remove neccessary features: ID, pledged, and state (replaced by success already)
    df = df.drop(['ID', 'pledged', 'state'], axis=1)

    # TEST Print the columns to test
    # print(df.columns)
    # print(((df.drop(['category', 'main_category', 'launched', 'deadline'], axis=1))[df.success == 0]).head(100))

    df.to_csv(path_or_buf=output, encoding='latin1', index=False)


def main():
    kickstarter_file = join(DATA_RAW_ROOT, 'ks-projects-201801.csv')
    kickstarter_pre_process_file = join(DATA_PREPROCESSED_ROOT, 'ks-projects-201801_pre_process.csv')
    pre_process_data(kickstarter_file, kickstarter_pre_process_file)

main()