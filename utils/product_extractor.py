import pandas as pd
import numpy as np

prod_metadata_features = [
    "ProductInfo.ProductPrice", "ProductInfo.ProductRank"
]

def has_prod_metadata(df):
    return len([c for c in df.columns if c in prod_metadata_features]) == len(prod_metadata_features)

def product_metadata_extractor(df):
    #price
    if "ProductInfo.ProductPrice" not in df.columns:
        df["ProductInfo.ProductPrice"] = df["price"]

    #product rank
    if "ProductInfo.ProductRank" not in df.columns:
        df["ProductInfo.ProductRank"] = df["salesRank"]

    return df

prod_history_features = [
    "ProductHistory.NReviews", "ProductHistory.AVGRatingOverTime", "ProductHistory.StarRatingDivergence"
]

def has_prod_history(df):
    return len([c for c in df.columns if c in prod_history_features]) == len(prod_history_features)

def prod_history_extractor(df_raw, df_feat):
    if "ProductHistory.NReviews" not in df_feat.columns:
        #sort both dataframes
        df_raw["reviewTime"] = pd.to_datetime(df_raw.reviewTime,
            errors= 'coerce')
        df_feat["reviewTime"] = pd.to_datetime(df_feat.reviewTime,
            errors= 'coerce')
        df_raw = df_raw.sort_values(by = 'reviewTime')
        df_feat = df_feat.sort_values(by = 'reviewTime')
        df_feat["ProductHistory.NReviews"] = None
        df_feat["ProductHistory.AVGRatingOverTime"] = None
        df_feat["ProductHistory.StarRatingDivergence"] = None

        #get the list of products
        prod_ids = df_feat["asin"].unique()

        #iterate over the prod_ids
        for pid in prod_ids:
            #get the indexes of those reviews belongin in the product id
            df_raw_ids = df_raw[df_raw["asin"] == pid].index.values
            df_feat_ids = df_feat[df_feat["asin"] == pid].index.values

            #select the subset of the raw_dataset
            curr_df_raw = df_raw.loc[df_raw_ids][["reviewTime", "reviewerID", 'overall']]
            curr_df_feat = df_feat.loc[df_feat_ids][["reviewTime", "reviewerID"]]

            #add a counter column
            curr_df_raw["NREV"] = np.arange(len(curr_df_raw))

            #compute the cumulative sum.
            #note that we do not consider the current review in the counter
            curr_df_raw["CUM_RATING"] = np.cumsum(curr_df_raw["overall"].values)
            curr_df_raw["CUM_RATING"] = curr_df_raw["CUM_RATING"].values - curr_df_raw["overall"].values

            #compute the average rate. Note that when NREV = 0, the operation produces NaN.
            #we thus cast it to 0.
            curr_df_raw = curr_df_raw.assign(CUM_AVG_RATE=lambda x: (x["CUM_RATING"] / (x["NREV"])))
            curr_df_raw["CUM_AVG_RATE"] = curr_df_raw["CUM_AVG_RATE"].fillna(0)

            #compute the absolute divergence between the product avg star rating and the current review score
            curr_df_raw = curr_df_raw.assign(SRD=lambda x: np.abs(x["CUM_AVG_RATE"] - x["overall"]))

            #we now need to maintain only those samples that are contained in the filtered
            #version as well
            merged_df = pd.merge(curr_df_raw, curr_df_feat, how = "inner", on = ["reviewTime", "reviewerID"])

            df_feat.loc[df_feat_ids, 'ProductHistory.NReviews'] = merged_df["NREV"].tolist()
            df_feat.loc[df_feat_ids, 'ProductHistory.AVGRatingOverTime'] = merged_df["CUM_AVG_RATE"].tolist()
            df_feat.loc[df_feat_ids, 'ProductHistory.StarRatingDivergence'] = merged_df["SRD"].tolist()


    return df_feat
