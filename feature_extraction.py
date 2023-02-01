import time
import os
import argparse
import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import  Pool

from utils.metadata_extractor import has_metadata, metadata_extractor
from utils.structural_extractor import has_structural, structural_extractor
from utils.product_extractor import has_prod_metadata, has_prod_history, product_metadata_extractor,  prod_history_extractor
from utils.readability_extractor import has_readability, readability_extractor, load_requirements
from utils.emotions_extractor import has_emotions, emotions_extractor, sentiwordnet
from utils.reviewer_extractor import has_reviewer_history, reviewer_history_extractor
from utils.embedding_extractor import has_embedding, embedding_extractor

""" The function allows to parallelize the feature extraction process """
def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

#
# #get english stopwords
# sw = set(stopwords.words('english'))
#
# #define the lemmatizer
# lemmatizer = WordNetLemmatizer()

#
#
#
# def custom_tokenizer(x):
#     #tokenize the sentene
#     x_tok = word_tokenize(x)
#
#     #lemmatize
#     x_lemm = [lemmatizer.lemmatize(x) for x in x_tok]
#
#     return x_lemm
#
# def get_unigram(corpus, max_features = None):
#     #define the engine
#     engine = TfidfVectorizer(lowercase = True,
#         tokenizer = custom_tokenizer,
#         analyzer = 'word',
#         stop_words = sw,
#         ngram_range = (1, 1),
#         max_features = max_features
#     )
#
#     #fit
#     engine.fit(corpus)
#
#     return engine


""" Main function """
# Define the parsing
parser = argparse.ArgumentParser(description='prepare data')
parser.add_argument('--dataset', '-d', type= str, help='name of the dataset')

args = parser.parse_args()

if __name__ == '__main__':
    #extract the arguments
    dataset = args.dataset

    #define input and output path
    in_path = f'./Dataset/{dataset}/df_preprocessed.csv'
    raw_path = f'./Dataset/{dataset}/raw.csv'
    out_path = f'./Dataset/{dataset}/df_featured.csv'

    if os.path.isfile(out_path): #load already existing file if it exists
        in_path = out_path

    #read the csv

    dataset = pd.read_csv(in_path, index_col = 'Unnamed: 0')
    dataset_raw = pd.read_csv(raw_path, index_col = 'Unnamed: 0')

    #extract the features -- metadata
    if not has_metadata(dataset):
        print(f"\tFound an incomplete list of metadata features. Extraction")
        start = time.time()
        dataset = parallelize_dataframe(dataset, metadata_extractor, n_cores = 8)
        print(f"\t\tExecution Time: {time.time() - start:.2f} [s]\n")
    else:
        print(f"\tSkip metadata features extraction\n")


    # #extract the features -- structural_features
    if not has_structural(dataset):
        print(f"\tFound an incomplete list of structural features. Extraction")
        start = time.time()
        dataset = parallelize_dataframe(dataset, structural_extractor, n_cores = 8)
        print(f"\t\tExecution Time: {time.time() - start:.2f} [s]\n")
    else:
        print(f"\tSkip structural features extraction\n")
    #


    # extract the features -- product info
    if not has_prod_metadata(dataset):
        print(f"\tFound an incomplete list of product metadata features. Extraction")
        start = time.time()
        dataset = parallelize_dataframe(dataset, product_metadata_extractor, n_cores = 8)
        print(f"\t\tExecution Time: {time.time() - start:.2f} [s]\n")
    else:
        print(f"\tSkip product metadata features extraction\n")
    #

    # extract the features -- readability_features
    if not has_readability(dataset):
        print(f"\tFound an incomplete list of readability features. Extraction")
        start = time.time()
        load_requirements(dataset)
        dataset = parallelize_dataframe(dataset, readability_extractor, n_cores = 8)
        print(f"\t\tExecution Time: {time.time() - start:.2f} [s]\n")
    else:
        print(f"\tSkip readability features extraction\n")

    # extract the features -- emotions_features
    if not has_emotions(dataset):
        print(f"\tFound an incomplete list of emotions features. Extraction")
        start = time.time()
        dataset = parallelize_dataframe(dataset, emotions_extractor, n_cores = 8)
        print(f"\t\tExecution Time: {time.time() - start:.2f} [s]\n")
    else:
        print(f"\tSkip emotions features extraction\n")


    #extract the features -- product history
    if not has_prod_history(dataset):
        print(f"\tFound an incomplete list of product history features. Extraction")
        start = time.time()
        dataset = prod_history_extractor(dataset_raw, dataset)
        print(f"\t\tExecution Time: {time.time() - start:.2f} [s]\n")
    else:
        print(f"\tSkip product history features extraction\n")

    # extract the features -- reviewer history
    if not has_reviewer_history(dataset):
        print(f"\tFound an incomplete list of reviewer history features. Extraction")
        start = time.time()
        dataset = reviewer_history_extractor(dataset_raw, dataset)
        print(f"\t\tExecution Time: {time.time() - start:.2f} [s]\n")
    else:
        print(f"\tSkip reviewer history features extraction\n")

    #extract embedding features
    if not has_embedding(dataset):
        print(f"\tFound an incomplete list of embedding. Extraction")
        start = time.time()
        dataset = embedding_extractor(dataset)
        print(f"\t\tExecution Time: {time.time() - start:.2f} [s]\n")
    else:
        print(f"\tSkip embedding extraction\n")

    #save the csv
    dataset.to_csv(out_path)
