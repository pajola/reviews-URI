import argparse
import pandas as pd
import gzip
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from tqdm import tqdm
import pickle
import os.path

#custom library
# from utils.tree import *

""" Implement a set of function to load Amazon reviews + metadata zip files """
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

"""
    Define the data loader function.
"""
def dataloader(rev_path, meta_path, out_path):
    #check the output path
    if os.path.isfile(out_path):
        print("The file already exists")
    else:
        #check the input paths
        if not os.path.isfile(rev_path):
            raise Exception(f"review file not found. Check the path={rev_path}")

        if not os.path.isfile(meta_path):
            raise Exception(f"meta file not found. Check the path={meta_path}")

        #load the datasets
        print("Load the dataset")
        # df_rev = pd.read_json(rev_path,lines=True)
        # df_meta = pd.read_json(meta_path,lines=True)
        df_rev = getDF(rev_path)
        df_meta = getDF(meta_path)
        print("\t\t...done!")

        #print some statistics
        print(f"Statistics.\tThe dataset contains {len(df_rev)} reviews and {len(df_meta)} products")

        #we first need to merge the two dataframes based on the asin
        print("\n\nMerge the dataset")
        df_final = pd.merge(df_rev, df_meta, on = ['asin'])
        df_final = df_final.drop_duplicates(subset=['reviewerID', 'asin', 'unixReviewTime', 'reviewText'])
        df_final = df_final.reset_index()
        print("\t\t...done!")

        #print some statistics
        print(f"Statistics.\tThe dataset contains {len(df_final)} reviews")

        #save the file
        df_final.to_csv(out_path)

""" Preprocessing """
def str2lst(x):
    """ Voters should be a list of two numbers.
    The loading returns a string.
    Here we fix it.
    """
    #split with the comma
    x_list = x.split(',')

    #the object should contain two items
    if len(x_list) != 2:
        raise Exception(f"Error in the voters object={x}")

    #extract the two numbers
    n1 = int(x_list[0][1:])
    n2 = int(x_list[1][:-1])

    return [n1, n2]


def preprocessing(in_path, out_path, out_path2, dataset):
    #check the existency of the paths
    if not os.path.isfile(in_path):
        print("The input path does not exist.")

    if os.path.isfile(out_path):
        print("The file already exists")
    else: #preprocess
        #laod the dataset
        print("Load the raw dataset")
        df = pd.read_csv(in_path, index_col= 'Unnamed: 0',
            low_memory = False)
        print("\t\t...done!")

        if "category" in df.columns.tolist():
            category = "category"
        elif "categories" in df.columns.tolist():
            category = "categories"
        else:
            raise Exception("Error in category variable.")

        #drop those samples with na value in helpful review
        df = df[df["reviewText"].notna()]

        #extract the sentence length (num chars)
        df["nchars"] = df['reviewText'].apply(lambda x: len(x))
        df = df[df["nchars"].notna()]

        #maintain only those features with more than 1 char
        df = df[df["nchars"] >=1]

        #set the list of features
        features = ['reviewerID', 'asin', 'helpful', 'reviewText', 'overall',
            'unixReviewTime', 'reviewTime', "price", "salesRank", category]

        #coerce the price into floating numbers. if None, we discard the items
        df['price'] = pd.to_numeric(df['price'], errors = 'coerce')
        df = df[df["price"].notna()]

        #coerce the salesRank info
        df = df[df["salesRank"].notna()]
        df['salesRank'] = [np.mean(list(eval(x).values())) if len(eval(x)) > 0 else np.nan for x in df['salesRank']]
        df = df[df["salesRank"].notna()]

        #maintain only the useful features
        df_filtered = df[features]

        #cast the helpful column
        #from string to list
        df_filtered['helpful'] = [str2lst(x) for x in df_filtered['helpful']]

        #extract the helpful votes
        df_filtered['voters'] = [x[1] for x in tqdm(df_filtered['helpful'])]

        #filter by number of voters
        df_filtered = df_filtered[df_filtered['voters'] >= 3]

        #calculate the helpful score
        df_filtered['help_score'] = [x[0] / x[1] if x[1] > 0 else 0
            for x in tqdm(df_filtered['helpful'])]

        #calculate the helpful score
        df_filtered['help_label'] = [1 if x >= 0.75 else 0
            for x in tqdm(df_filtered['help_score'])]

        #extract time information
        tmp_l = len(df_filtered)
        df_filtered['reviewTime'] = pd.to_datetime(df_filtered.reviewTime,
            errors= 'coerce') #cast the time in a proper object
        df_filtered = df_filtered[~df_filtered.reviewTime.isna()]
        print(f"we discard {len(df_filtered) - tmp_l} samples that do not contain time info")

        #order the dataset with ascending time and reset indexes
        df_filtered = df_filtered.sort_values(by = 'reviewTime')
        df_filtered = df_filtered.reset_index(drop=True)

        #save
        print(f"The dataset contains = {len(df_filtered)} reviews")
        df_filtered.to_csv(out_path)
        # df_filtered[["reviewText", "asin"]].to_csv(out_path2)

"""Train / Validation / Test Splitting"""
def temporal_splitting(in_path, out_path, tr_ratio, val_ratio, te_ratio,
    save_splits = False, voters = 3):
    #check the existency of the path
    if not os.path.isfile(in_path):
        print("The input path does not exist.")

    if (tr_ratio + val_ratio + te_ratio) != 1:
        raise Exception("The sum of the ratios must be equal to 1")

    #laod the dataset
    print("Load the raw dataset")
    df = pd.read_csv(in_path, index_col= 'Unnamed: 0',
        low_memory = False)
    print("\t\t...done!")

    #filter by the number of voters
    df= df[df['voters'] >= voters]

    print(f"The dataset is defined between {df['reviewTime'].min()} and {df['reviewTime'].max()}")

    #count the dates frequencies
    dates_cnt = df['reviewTime'].value_counts(normalize = True, sort = False)

    #extract unique dates
    dates_unique = df['reviewTime'].unique()

    #search for the two splits
    split1 = None #split between train and val
    split2 = None #split between val and test

    freq = 0
    for xx in dates_unique:
        #update the freq
        freq += dates_cnt[xx]

        #check if we can update the splits
        if split1 is None and freq >= tr_ratio:
            split1 = xx

        #check if we can update the splits
        if split2 is None and freq >= (tr_ratio + val_ratio):
            split2 = xx

    print(f"Split1={split1}\tSplit2={split2}")

    #save the splits
    splits = {
        'split1': split1,
        'split2': split2
    }
    with open(out_path + 'split.pkl', 'wb') as file:
        pickle.dump(splits, file)

    # # save the dataset
    # df.to_csv(out_path + 'dataset.csv')

    #save splits
    if save_splits:
        #define the three splits
        train = df[df['reviewTime'] < split1]
        val = df[(df['reviewTime'] < split2) & (df['reviewTime'] >= split1)]
        test = df[df['reviewTime'] >= split2]

        print(f"Train len={len(train)}\tVal len={len(val)}\tTest len={len(test)}")

        #save the data
        train.to_csv(out_path + 'train.csv')
        val.to_csv(out_path + 'val.csv')
        test.to_csv(out_path + 'test.csv')

def split(df, path, debug = False):
    #load the splits info
    with open(path + 'split.pkl', 'rb') as file:
        splits = pickle.load(file)

    split1 = splits['split1']
    split2 = splits['split2']

    if debug:
        print(f"split1 = {split1}\tsplit2 = {split2}")

    #define the three splits
    train = df[df['reviewTime'] < split1]
    val = df[(df['reviewTime'] < split2) & (df['reviewTime'] >= split1)]
    test = df[df['reviewTime'] >= split2]

    return train, val, test

""" Main function """

if __name__ == '__main__':
    # Define the parsing
    parser = argparse.ArgumentParser(description='prepare data')
    parser.add_argument('--dataset', '-d', type= str, help='name of the dataset')
    parser.add_argument('--min_voters', type = int, default = 3, help = 'minimum number of voters per review')
    parser.add_argument('--train_split', type = float, default = 0.7, help = 'training ratio')
    parser.add_argument('--valid_split', type = float, default = 0.1, help = 'validation ratio')
    parser.add_argument('--test_split', type = float, default = 0.2, help = 'test ratio')

    args = parser.parse_args()

    #extract the arguments
    dataset = args.dataset
    voters = args.min_voters
    tr_ratio = args.train_split
    val_ratio = args.valid_split
    te_ratio = args.test_split

    #define the execution paths - custom
    # rev_path = f'./Dataset/{dataset}/reviews_Toys_and_Games_5.json.gz'
    # meta_path = f'./Dataset/{dataset}/meta_Toys_and_Games.json.gz'
    rev_path = f'./Dataset/{dataset}/reviews_{dataset}_5.json.gz'
    meta_path = f'./Dataset/{dataset}/meta_{dataset}.json.gz'


    #define the execution paths -- standards
    raw_path = f'./Dataset/{dataset}/raw.csv'
    data_path = f'./Dataset/{dataset}/'
    pre_path = f'./Dataset/{dataset}/df_preprocessed.csv'
    rev_only_path = f'./Dataset/{dataset}/df_rev_only.csv'

    ## splitting and data loading
    dataloader(rev_path, meta_path, raw_path)
    preprocessing(raw_path, pre_path, rev_only_path, dataset)
    temporal_splitting(pre_path, data_path, 0.7, 0.1, 0.2,
        voters = voters, save_splits = False)

    #prepare the product tree
    tree_template = f"./Dataset/{dataset}/DG.pkl"
    train, _, _= split(pd.read_csv(pre_path,
                index_col = 'Unnamed: 0'), data_path) #get the training partition

    if "category" in train.columns.tolist():
        category = "category"
    elif "categories" in train.columns.tolist():
        category = "categories"
    else:
        raise Exception("Error in category variable.")
