import pandas as pd
import numpy as np
import argparse
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, make_scorer, precision_score, recall_score, plot_confusion_matrix,roc_auc_score
from sklearn.pipeline import Pipeline

#custom libraries
from load_dataset import split
from joblib import dump, load


families = ["Emotions", "MetaData", "ProductInfo", "Readability", "ReviewerHistory", "StructFeat", "Embedding", "ProductHistory"]


def get_dataset(ds, data_path):
    #get the columns with features
    cols_overall = list(ds.columns)
    extracted = [x for x in cols_overall if (families[0] in x or
        families[1] in x or families[2] in x or families[3] in x or
        families[4] in x or families[5] in x or families[6] in x or families[7] in x)
    ]

    #train / val / test split
    train_feat_ds, val_feat_ds, test_feat_ds = split(ds, data_path, debug = True)

    #find columns with nan
    na_names = ds.isnull().any()
    na_names = list(na_names.where(na_names == True).dropna().index)
    if len(na_names) > 0:
        raise Exception(f"Found NaN in the following columns = {na_names}")

    # #check for infinity
    # inf_name = ds.columns.to_series()[np.isinf(ds).any()]
    # if len(inf_name) > 0:
    #     raise Exception(f"Found Inf in the following columns = {inf_name}")

    #split in X and Y features
    X_train = train_feat_ds[extracted]#.to_numpy(dtype = np.float32)
    X_val = val_feat_ds[extracted]#.to_numpy(dtype = np.float32)
    X_test = test_feat_ds[extracted]#.to_numpy(dtype = np.float32)

    Y_train = train_feat_ds['help_label'].tolist()
    Y_val= val_feat_ds['help_label'].tolist()
    Y_test = test_feat_ds['help_label'].tolist()

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare data')
    parser.add_argument('--dataset', '-d', type= str, help='name of the dataset')
    parser.add_argument('--jobs', '-j', type = int, default = -1, help = 'model')
    args = parser.parse_args()

    #extract the arguments
    dataset = args.dataset
    # model = args.model
    njobs = args.jobs

    #define input and output path
    LIWC_dir = './LIWC/'
    data_path = f'./Dataset/{dataset}/'
    in_path = f'./Dataset/{dataset}/df_featured.csv'
    # out_path = f'./Models/{dataset}__{model}.joblib'
    log_path = f'./Results/test__{dataset}.csv'

    #read the log file
    if not os.path.exists(log_path):
        log_df = pd.DataFrame(None, columns = ['Dataset', 'Feature', 'Model', 'Metric', 'Score'])
        log_df.to_csv(log_path)
    else:
        log_df = pd.read_csv(log_path, index_col = 'Unnamed: 0')


    #memory optimizer -- we read only one column at a time
    ds_feat = pd.read_csv(f'Dataset/Toys_and_Games/df_featured.csv', index_col = 'Unnamed: 0')
    cols_overall = list(ds_feat.columns)
    cols_overall_nofeat = [x for x in cols_overall if (families[0] not in x and
        families[1] not in x and families[2] not in x and families[3] not in x and
        families[4] not in x and families[5] not in x and families[6] not in x and families[7] not in x)
    ]
    cols_overall_feat = [x for x in cols_overall if (families[0] in x or
        families[1] in x or families[2] in x or families[3] in x or
        families[4] in x or families[5] in x or families[6] in x or families[7] in x)
    ]

    #iterate over the families
    for f in families:
        print(f"\t\t{f}\n\n")
        #open the ds
        c = [x for x in cols_overall_feat if f in x]
        df_overall = pd.read_csv(in_path, usecols = cols_overall_nofeat + c)
        if f == 'Emotions': #add LIWC info
            ds_liwc = pd.read_csv(f'{LIWC_dir}LIWC__{dataset}.csv')
            #rename liwc columns
            ds_liwc.columns = [f"Emotions.LIWC.{x}" for x in ds_liwc.columns]
            liwc_features = list(ds_liwc.columns)[-93:]
            ds_liwc= ds_liwc[liwc_features + ['Emotions.LIWC.B', 'Emotions.LIWC.C' , 'Emotions.LIWC.G']]

            #merge
            # df_overall = df_overall.merge(ds_liwc, left_on = 'unixReviewTime', right_on = 'Emotions.LIWC.G').drop(columns = ['Emotions.LIWC.G'])
            df_overall = df_overall.merge(ds_liwc, left_on = ['reviewerID', 'asin' ,'unixReviewTime'],
                right_on = ['Emotions.LIWC.B', 'Emotions.LIWC.C', 'Emotions.LIWC.G']).drop(columns = ['Emotions.LIWC.B', 'Emotions.LIWC.C', 'Emotions.LIWC.G'])

        ##############################################################3
        # load training settings
        #get the data
        X_train, X_val, X_test, Y_train, Y_val, Y_test = get_dataset(df_overall, data_path)

        #PCA analysis
        if "Embedding" in set([x.split('.')[0] for x in X_train.columns]):
            #get the target columns
            target_cols = [x for x in X_train.columns if "Embedding." in x]

            #extract the info from the training set
            X_train_emb = X_train[target_cols]
            X_val_emb = X_val[target_cols]
            X_test_emb = X_test[target_cols]

            #PCA
            pca_scaler = PCA(n_components = 100, random_state = 123)
            pca_scaler.fit(X_train_emb)
            new_c = [f"Embedding.PCA{i}"for i in range(100)]
            X_train_emb_pca100 = pd.DataFrame(pca_scaler.transform(X_train_emb), 
                columns = new_c, index = X_train.index)
            X_val_emb_pca100 = pd.DataFrame(pca_scaler.transform(X_val_emb), 
                columns = new_c, index = X_val.index)
            X_test_emb_pca100 = pd.DataFrame(pca_scaler.transform(X_test_emb), 
                columns = new_c, index = X_test.index)

            #drop old colums
            X_train = X_train.drop(target_cols, axis = 1)
            X_val = X_val.drop(target_cols, axis = 1)
            X_test = X_test.drop(target_cols, axis = 1)

            #concatenate the current dataset with the previous one
            X_train = pd.concat([X_train, X_train_emb_pca100], axis = 1)
            X_val = pd.concat([X_val, X_val_emb_pca100], axis = 1)
            X_test = pd.concat([X_test, X_test_emb_pca100], axis = 1)

        X_train = X_train.to_numpy(dtype = np.float32)
        X_val = X_val.to_numpy(dtype = np.float32)
        X_test = X_test.to_numpy(dtype = np.float32)



        #scale the data
        scl = StandardScaler().fit(X_train)
        X_train = scl.transform(X_train)
        X_val = scl.transform(X_val)
        X_test = scl.transform(X_test)

        ### prepare the data
        #since we have temporal data, we cannot use CV.
        #to use the sckit CV module, we thus need to create something ad hoc
        X_train_val = np.concatenate([X_train, X_val], axis = 0)
        Y_train_val = Y_train + Y_val


        split_index = [-1] * len(X_train) + [0] * len(X_val)
        ps = PredefinedSplit(test_fold= split_index) #this avoids random splits

        #define the evaluation metric
        f1_weighted = make_scorer(f1_score, average = 'weighted')

        ############################################################
        # logistic regression
        if len(log_df[(log_df['Model'] == 'lr') & (log_df['Dataset'] == dataset) & (log_df['Feature'] == f)]) == 0:
            #
            model = "lr"

            #load the model
            clf = load(f"./Models/{dataset}__{model}__{f}.joblib")

            y_pred = clf.predict(X_test)

            f1_ = f1_score(Y_test, y_pred,average = 'weighted')
            prec_ = precision_score(Y_test, y_pred)
            rec_ = recall_score(Y_test, y_pred)
            roc_ = roc_auc_score(Y_test, y_pred)

            log_df.loc[len(log_df)] = [dataset, f, 'lr', 'f1', f1_]
            log_df.loc[len(log_df)] = [dataset, f, 'lr', 'prec', prec_]
            log_df.loc[len(log_df)] = [dataset, f, 'lr', 'recall', rec_]
            log_df.loc[len(log_df)] = [dataset, f, 'lr', 'roc', roc_]


        #############################################################
        # decision tree
        if len(log_df[(log_df['Model'] == 'dt') & (log_df['Dataset'] == dataset) & (log_df['Feature'] == f)]) == 0:
            model = "dt"

            #load the model
            clf = load(f"./Models/{dataset}__{model}__{f}.joblib")
            y_pred = clf.predict(X_test)

            f1_ = f1_score(Y_test, y_pred,average = 'weighted')
            prec_ = precision_score(Y_test, y_pred)
            rec_ = recall_score(Y_test, y_pred)
            roc_ = roc_auc_score(Y_test, y_pred)

            log_df.loc[len(log_df)] = [dataset, f, 'dt', 'f1', f1_]
            log_df.loc[len(log_df)] = [dataset, f, 'dt', 'prec', prec_]
            log_df.loc[len(log_df)] = [dataset, f, 'dt', 'recall', rec_]
            log_df.loc[len(log_df)] = [dataset, f, 'dt', 'roc', roc_]


        #############################################################
        # random forest
        if len(log_df[(log_df['Model'] == 'rf') & (log_df['Dataset'] == dataset) & (log_df['Feature'] == f)]) == 0:
            model = "rf"
            #load the model
            clf = load(f"./Models/{dataset}__{model}__{f}.joblib")
            y_pred = clf.predict(X_test)

            f1_ = f1_score(Y_test, y_pred,average = 'weighted')
            prec_ = precision_score(Y_test, y_pred)
            rec_ = recall_score(Y_test, y_pred)
            roc_ = roc_auc_score(Y_test, y_pred)

            log_df.loc[len(log_df)] = [dataset, f, 'rf', 'f1', f1_]
            log_df.loc[len(log_df)] = [dataset, f, 'rf', 'prec', prec_]
            log_df.loc[len(log_df)] = [dataset, f, 'rf', 'recall', rec_]
            log_df.loc[len(log_df)] = [dataset, f, 'rf', 'roc', roc_]


        #############################################################
        # ada boost
        if len(log_df[(log_df['Model'] == 'ab') & (log_df['Dataset'] == dataset) & (log_df['Feature'] == f)]) == 0:
            model = "ab"
            #load the model
            clf = load(f"./Models/{dataset}__{model}__{f}.joblib")
            y_pred = clf.predict(X_test)

            f1_ = f1_score(Y_test, y_pred,average = 'weighted')
            prec_ = precision_score(Y_test, y_pred)
            rec_ = recall_score(Y_test, y_pred)
            roc_ = roc_auc_score(Y_test, y_pred)

            log_df.loc[len(log_df)] = [dataset, f, 'ab', 'f1', f1_]
            log_df.loc[len(log_df)] = [dataset, f, 'ab', 'prec', prec_]
            log_df.loc[len(log_df)] = [dataset, f, 'ab', 'recall', rec_]
            log_df.loc[len(log_df)] = [dataset, f, 'ab', 'roc', roc_]


        #############################################################
        # naive bayes
        if len(log_df[(log_df['Model'] == 'nb') & (log_df['Dataset'] == dataset) & (log_df['Feature'] == f)]) == 0:
            model = "nb"
            #get the classifier
            #load the model
            clf = load(f"./Models/{dataset}__{model}__{f}.joblib")
            y_pred = clf.predict(X_test)

            f1_ = f1_score(Y_test, y_pred,average = 'weighted')
            prec_ = precision_score(Y_test, y_pred)
            rec_ = recall_score(Y_test, y_pred)
            roc_ = roc_auc_score(Y_test, y_pred)

            log_df.loc[len(log_df)] = [dataset, f, 'nb', 'f1', f1_]
            log_df.loc[len(log_df)] = [dataset, f, 'nb', 'prec', prec_]
            log_df.loc[len(log_df)] = [dataset, f, 'nb', 'recall', rec_]
            log_df.loc[len(log_df)] = [dataset, f, 'nb', 'roc', roc_]


    log_df.to_csv(log_path)

    ##
    ## ===============================================================================

    ### train over all features
    print(f"\t\tALL\n\n")
    #open the ds
    c = cols_overall_feat
    f = "ALL"
    df_overall = pd.read_csv(in_path, usecols = cols_overall_nofeat + c)
    ds_liwc = pd.read_csv(f'{LIWC_dir}LIWC__{dataset}.csv')
    #rename liwc columns
    ds_liwc.columns = [f"Emotions.LIWC.{x}" for x in ds_liwc.columns]
    liwc_features = list(ds_liwc.columns)[-93:]
    ds_liwc= ds_liwc[liwc_features + ['Emotions.LIWC.B', 'Emotions.LIWC.C' , 'Emotions.LIWC.G']]

    #merge
    # df_overall = df_overall.merge(ds_liwc, left_on = 'unixReviewTime', right_on = 'Emotions.LIWC.G').drop(columns = ['Emotions.LIWC.G'])
    df_overall = df_overall.merge(ds_liwc, left_on = ['reviewerID', 'asin' ,'unixReviewTime'],
        right_on = ['Emotions.LIWC.B', 'Emotions.LIWC.C', 'Emotions.LIWC.G']).drop(columns = ['Emotions.LIWC.B', 'Emotions.LIWC.C', 'Emotions.LIWC.G'])
    ##############################################################3
    # load training settings
    #get the data
    X_train, X_val, X_test, Y_train, Y_val, Y_test = get_dataset(df_overall, data_path)
    print(f"\nf={f}\tshape:{X_train.shape}")

    #PCA analysis
    if "Embedding" in set([x.split('.')[0] for x in X_train.columns]):
        #get the target columns
        target_cols = [x for x in X_train.columns if "Embedding." in x]

        #extract the info from the training set
        X_train_emb = X_train[target_cols]
        X_val_emb = X_val[target_cols]
        X_test_emb = X_test[target_cols]

        #PCA
        pca_scaler = PCA(n_components = 100, random_state = 123)
        pca_scaler.fit(X_train_emb)
        new_c = [f"Embedding.PCA{i}"for i in range(100)]
        X_train_emb_pca100 = pd.DataFrame(pca_scaler.transform(X_train_emb), 
            columns = new_c, index = X_train.index)
        X_val_emb_pca100 = pd.DataFrame(pca_scaler.transform(X_val_emb), 
            columns = new_c, index = X_val.index)
        X_test_emb_pca100 = pd.DataFrame(pca_scaler.transform(X_test_emb), 
            columns = new_c, index = X_test.index)

        #drop old colums
        X_train = X_train.drop(target_cols, axis = 1)
        X_val = X_val.drop(target_cols, axis = 1)
        X_test = X_test.drop(target_cols, axis = 1)

        #concatenate the current dataset with the previous one
        X_train = pd.concat([X_train, X_train_emb_pca100], axis = 1)
        X_val = pd.concat([X_val, X_val_emb_pca100], axis = 1)
        X_test = pd.concat([X_test, X_test_emb_pca100], axis = 1)

    X_train = X_train.to_numpy(dtype = np.float32)
    X_val = X_val.to_numpy(dtype = np.float32)
    X_test = X_test.to_numpy(dtype = np.float32)



    #scale the data
    scl = StandardScaler().fit(X_train)
    X_train = scl.transform(X_train)
    X_val = scl.transform(X_val)
    X_test = scl.transform(X_test)

    ### prepare the data
    #since we have temporal data, we cannot use CV.
    #to use the sckit CV module, we thus need to create something ad hoc
    X_train_val = np.concatenate([X_train, X_val], axis = 0)
    Y_train_val = Y_train + Y_val


    split_index = [-1] * len(X_train) + [0] * len(X_val)
    ps = PredefinedSplit(test_fold= split_index) #this avoids random splits

    #define the evaluation metric
    f1_weighted = make_scorer(f1_score, average = 'weighted')

    ############################################################
    # logistic regression
    if len(log_df[(log_df['Model'] == 'lr') & (log_df['Dataset'] == dataset) & (log_df['Feature'] == f)]) == 0:
        #
        print("LR")
        model = "lr"
        #load the model
        clf = load(f"./Models/{dataset}__{model}__{f}.joblib")
        y_pred = clf.predict(X_test)

        f1_ = f1_score(Y_test, y_pred,average = 'weighted')
        prec_ = precision_score(Y_test, y_pred)
        rec_ = recall_score(Y_test, y_pred)
        roc_ = roc_auc_score(Y_test, y_pred)

        log_df.loc[len(log_df)] = [dataset, f, 'lr', 'f1', f1_]
        log_df.loc[len(log_df)] = [dataset, f, 'lr', 'prec', prec_]
        log_df.loc[len(log_df)] = [dataset, f, 'lr', 'recall', rec_]
        log_df.loc[len(log_df)] = [dataset, f, 'lr', 'roc', roc_]


    #############################################################
    # decision tree
    if len(log_df[(log_df['Model'] == 'dt') & (log_df['Dataset'] == dataset) & (log_df['Feature'] == f)]) == 0:
        print("DT")
        model = "dt"
        #get the classifier
        #load the model
        clf = load(f"./Models/{dataset}__{model}__{f}.joblib")
        y_pred = clf.predict(X_test)

        f1_ = f1_score(Y_test, y_pred,average = 'weighted')
        prec_ = precision_score(Y_test, y_pred)
        rec_ = recall_score(Y_test, y_pred)
        roc_ = roc_auc_score(Y_test, y_pred)

        log_df.loc[len(log_df)] = [dataset, f, 'dt', 'f1', f1_]
        log_df.loc[len(log_df)] = [dataset, f, 'dt', 'prec', prec_]
        log_df.loc[len(log_df)] = [dataset, f, 'dt', 'recall', rec_]
        log_df.loc[len(log_df)] = [dataset, f, 'dt', 'roc', roc_]


    #############################################################
    # random forest
    if len(log_df[(log_df['Model'] == 'rf') & (log_df['Dataset'] == dataset) & (log_df['Feature'] == f)]) == 0:
        print("RF")
        model = "rf"
        #get the classifier
        #load the model
        clf = load(f"./Models/{dataset}__{model}__{f}.joblib")
        y_pred = clf.predict(X_test)

        f1_ = f1_score(Y_test, y_pred,average = 'weighted')
        prec_ = precision_score(Y_test, y_pred)
        rec_ = recall_score(Y_test, y_pred)
        roc_ = roc_auc_score(Y_test, y_pred)

        log_df.loc[len(log_df)] = [dataset, f, 'rf', 'f1', f1_]
        log_df.loc[len(log_df)] = [dataset, f, 'rf', 'prec', prec_]
        log_df.loc[len(log_df)] = [dataset, f, 'rf', 'recall', rec_]
        log_df.loc[len(log_df)] = [dataset, f, 'rf', 'roc', roc_]


    #############################################################
    # ada boost
    if len(log_df[(log_df['Model'] == 'ab') & (log_df['Dataset'] == dataset) & (log_df['Feature'] == f)]) == 0:
        print("AB")
        model = "ab"
        #load the model
        clf = load(f"./Models/{dataset}__{model}__{f}.joblib")
        y_pred = clf.predict(X_test)

        f1_ = f1_score(Y_test, y_pred,average = 'weighted')
        prec_ = precision_score(Y_test, y_pred)
        rec_ = recall_score(Y_test, y_pred)
        roc_ = roc_auc_score(Y_test, y_pred)

        log_df.loc[len(log_df)] = [dataset, f, 'ab', 'f1', f1_]
        log_df.loc[len(log_df)] = [dataset, f, 'ab', 'prec', prec_]
        log_df.loc[len(log_df)] = [dataset, f, 'ab', 'recall', rec_]
        log_df.loc[len(log_df)] = [dataset, f, 'ab', 'roc', roc_]


    #############################################################
    # naive bayes
    if len(log_df[(log_df['Model'] == 'nb') & (log_df['Dataset'] == dataset) & (log_df['Feature'] == f)]) == 0:
        print("NB")
        model = "nb"
        #load the model
        clf = load(f"./Models/{dataset}__{model}__{f}.joblib")
        y_pred = clf.predict(X_test)

        f1_ = f1_score(Y_test, y_pred,average = 'weighted')
        prec_ = precision_score(Y_test, y_pred)
        rec_ = recall_score(Y_test, y_pred)
        roc_ = roc_auc_score(Y_test, y_pred)

        log_df.loc[len(log_df)] = [dataset, f, 'nb', 'f1', f1_]
        log_df.loc[len(log_df)] = [dataset, f, 'nb', 'prec', prec_]
        log_df.loc[len(log_df)] = [dataset, f, 'nb', 'recall', rec_]
        log_df.loc[len(log_df)] = [dataset, f, 'nb', 'roc', roc_]


    log_df.to_csv(log_path)
