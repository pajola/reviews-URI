import pandas as pd
import numpy as np

prod_history_features = [
    "ReviewerHistory.NReviews", "ReviewerHistory.RevHistMacro", 'ReviewerHistory.CumHelpVotes',
    'ReviewerHistory.CumVotes'
]

def has_reviewer_history(df):
    return len([c for c in df.columns if c in prod_history_features]) == len(prod_history_features)

def reviewer_history_extractor(df_raw, df_feat):
    #sort both dataframes
    df_raw["reviewTime"] = pd.to_datetime(df_raw.reviewTime,
        errors= 'coerce')
    df_feat["reviewTime"] = pd.to_datetime(df_feat.reviewTime,
        errors= 'coerce')
    df_raw = df_raw.sort_values(by = 'reviewTime')
    df_feat = df_feat.sort_values(by = 'reviewTime')

    df_feat["ReviewerHistory.NReviews"] = None
    df_feat["ReviewerHistory.RevHistMacro"] = None

    #get the list of userids
    rev_ids = df_feat["reviewerID"].unique()

    #iterate over the prod_ids
    for rid in rev_ids:
        #get the indexes of those reviews belongin in the product id
        df_raw_ids = df_raw[df_raw["reviewerID"] == rid].index.values
        df_feat_ids = df_feat[df_feat["reviewerID"] == rid].index.values

        #select the subset of the raw_dataset
        curr_df_raw = df_raw.loc[df_raw_ids][["reviewTime", "reviewerID", "asin", 'helpful']]
        curr_df_feat = df_feat.loc[df_feat_ids][["reviewTime", "reviewerID", "asin"]]

        #add a counter column
        curr_df_raw["NREV"] = np.arange(len(curr_df_raw))

        #get both positve votes and total votes info
        curr_df_raw["UP_VOTERS"] = curr_df_raw["helpful"].apply(lambda x: eval(x)[0])
        curr_df_raw["VOTERS"] = curr_df_raw["helpful"].apply(lambda x: eval(x)[1])

        #compute the cumulative vote info. We subtract current reviews from the computation
        curr_df_raw["CUM_UP_VOTERS"] = np.cumsum(curr_df_raw["UP_VOTERS"].values)
        curr_df_raw["CUM_UP_VOTERS"] = curr_df_raw["CUM_UP_VOTERS"].values - curr_df_raw["UP_VOTERS"].values
        curr_df_raw["CUM_VOTERS"] = np.cumsum(curr_df_raw["VOTERS"].values)
        curr_df_raw["CUM_VOTERS"] = curr_df_raw["CUM_VOTERS"].values - curr_df_raw["VOTERS"].values

        #compute the ratio between the cumulative voting info
        # Note that when CUM_VOTERS = 0, the operation produces NaN.
        #we thus cast it to 0.
        curr_df_raw = curr_df_raw.assign(CUM_AVG_RATE=lambda x: (x["CUM_UP_VOTERS"] / x["CUM_VOTERS"]))
        curr_df_raw["CUM_AVG_RATE"] = curr_df_raw["CUM_AVG_RATE"].fillna(0)

        #we now need to maintain only those samples that are contained in the filtered
        #version as well
        merged_df = pd.merge(curr_df_raw, curr_df_feat, how = "inner", on = ["reviewTime", "reviewerID", "asin"])

        df_feat.loc[df_feat_ids, 'ReviewerHistory.NReviews'] = merged_df["NREV"].tolist()
        df_feat.loc[df_feat_ids, 'ReviewerHistory.RevHistMacro'] = merged_df["CUM_AVG_RATE"].tolist()
        df_feat.loc[df_feat_ids, 'ReviewerHistory.CumHelpVotes'] = merged_df["CUM_UP_VOTERS"].tolist()
        df_feat.loc[df_feat_ids, 'ReviewerHistory.CumVotes'] = merged_df["CUM_VOTERS"].tolist()

    return df_feat
