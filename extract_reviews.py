import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import argparse
import os

#custom libraries
from load_dataset import split

if __name__ == '__main__':

    #directories
    DIR = 'Dataset/'

    #datasets
    DATASET = os.listdir(DIR)

    #columns to maintain
    c = ['reviewerID', 'asin', 'helpful', 'reviewText', 'overall',
       'unixReviewTime', 'reviewTime']

    for d in DATASET:
        #read the csv
        in_path = f'{DIR}/{d}/df_featured.csv'
        ds = pd.read_csv(in_path, index_col = 'Unnamed: 0')
        print(f"{d}\t{len(ds)}")

        ds.to_csv(f"{DIR}/{d}/{d}_reviews.csv")
