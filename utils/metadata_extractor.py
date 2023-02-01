import pandas as pd
import datetime

#define the scraping date
SCRAPE_DATE = datetime.date(2014, 8, 1)

metadata_features = [
    "MetaData.StarRating", "MetaData.ReviewElapsedTime", "MetaData.StarRating_isModerate"
]

def has_metadata(df):
    return len([c for c in df.columns if c in metadata_features]) == len(metadata_features)

def is_moderate(x):
    #convert to int
    x = int(x)
    return x == 3

def metadata_extractor(df):
    #star rating
    if "MetaData.StarRating" not in df.columns:
        df["MetaData.StarRating"] = df['overall']

    if "MetaData.ReviewElapsedTime" not in df.columns:
        df["reviewTime"] = pd.to_datetime(df.reviewTime, errors= 'coerce')
        df["MetaData.ReviewElapsedTime"] = df["reviewTime"].apply(lambda x: (SCRAPE_DATE - x.date()).days)

    if "MetaData.StarRating_isModerate" not in df.columns:
        df["MetaData.StarRating_isModerate"] = df['overall'].apply(lambda x: is_moderate(x))



    return df
