import pandas as pd
import numpy as np
import argparse
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, make_scorer, precision_score, recall_score, plot_confusion_matrix,roc_auc_score
from sklearn.pipeline import Pipeline

#custom libraries
from load_dataset import split
from joblib import dump, load

class CustomScaler(BaseEstimator,TransformerMixin): 
    #following what said in: https://stackoverflow.com/questions/37685412/avoid-scaling-binary-columns-in-sci-kit-learn-standsardscaler
    # note: returns the feature matrix with the binary columns ordered first  
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X[:, :-1])
        return self

    def transform(self, X):
        X_transf = self.scaler.transform(X[:, :-1])
        return np.concatenate((X_transf,X[:, -1:]), axis=1)


class CustomPCA(BaseEstimator,TransformerMixin): 
    #following what said in: https://stackoverflow.com/questions/37685412/avoid-scaling-binary-columns-in-sci-kit-learn-standsardscaler
    # note: returns the feature matrix with the binary columns ordered first  
    def __init__(self, n_components = None):
        self.n_components = n_components
        self.pca = PCA(n_components = n_components)

    def fit(self, X, y=None):
        self.pca.fit(X[:, :-1], y)
        return self

    def transform(self, X, y=None, copy=None):
        X_transf = self.pca.transform(X[:, :-1])
        return np.concatenate((X_transf,X[:, -1:]), axis=1)

families = ["Emotions", "MetaData", "ProductInfo", "Readability", "ReviewerHistory", "StructFeat", "Embedding", "ProductHistory"]

def get_model(model, ps, eval_f, feature_size, use_custom_pca = True, njobs = -1, hidden_size = None):
    #create the array for the PCA parametersearch 
    # -- no considering values bigger than the current dataset
    k = [None]
    if feature_size >= 64:
        k.append(64)
    if feature_size >= 128:
        k.append(128)

    #define the pca to use
    estimator_pca = CustomPCA()

    if model == 'mlp':
        #mlp classifier
        estimator_mlp = MLPClassifier(max_iter=100, 
            verbose = False, 
            random_state = 123)

        if hidden_size is None:
            raise Exception("Invalid Parameter")

        #define the pipeline
        pipe_mlp = Pipeline( steps = [
            ('pca', estimator_pca),
            ('clf', estimator_mlp)
        ])

        param_grid_mlp = {
            'pca__n_components': k,
            'clf__hidden_layer_sizes': [ (hidden_size,), (hidden_size, hidden_size // 2), (hidden_size // 2,)],
            'clf__activation': ['tanh', 'relu'],
            'clf__solver': ['adam'],
            'clf__alpha': [0.0001, 0.05],
            'clf__learning_rate': ['constant','adaptive'],
        }

        clf_mlp = GridSearchCV(estimator= pipe_mlp,
                             cv = ps,
                             param_grid= param_grid_mlp,
                             scoring= eval_f,
                             n_jobs=njobs,
                             refit = True)
        return clf_mlp

    else:
        raise Exception("Invalid model choice.")

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
    X_train = train_feat_ds[extracted] #.to_numpy(dtype = np.float32)
    X_val = val_feat_ds[extracted] #.to_numpy(dtype = np.float32)
    X_test = test_feat_ds[extracted] #.to_numpy(dtype = np.float32)


    #reordering is necessary for those model using such a boolean variable. Preprocessing stages must not consider 
    #such a column. placing it in the last position ease the operation
    if "MetaData.StarRating_isModerate" in X_train.columns:
        X_train["MetaData.StarRating_isModerate_"] = X_train.loc[:, "MetaData.StarRating_isModerate"]
        X_val["MetaData.StarRating_isModerate_"] = X_val.loc[:, "MetaData.StarRating_isModerate"]
        X_test["MetaData.StarRating_isModerate_"] = X_test.loc[:, "MetaData.StarRating_isModerate"]
        X_train.drop(['MetaData.StarRating_isModerate'], axis = 1, inplace = True)
        X_val.drop(['MetaData.StarRating_isModerate'], axis = 1, inplace = True)
        X_test.drop(['MetaData.StarRating_isModerate'], axis = 1, inplace = True)

    #cast to numpy
    X_train = X_train.to_numpy(dtype = np.float32)
    X_val = X_val.to_numpy(dtype = np.float32)
    X_test = X_test.to_numpy(dtype = np.float32)

    Y_train = train_feat_ds['help_label'].tolist()
    Y_val= val_feat_ds['help_label'].tolist()
    Y_test = test_feat_ds['help_label'].tolist()

    #define the credibility masks
    test_mask = []
    for x in test_feat_ds.voters.tolist():
        if x <= 5:
            test_mask.append('low')
        elif x <= 10:
            test_mask.append('medium')
        elif x > 10:
            test_mask.append('high')
        else:
            raise Exception("Error")

    test_mask = np.array(test_mask)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, test_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare data')
    parser.add_argument('--dataset', '-d', type= str, help='name of the dataset')
    parser.add_argument('--model', '-m', type = str, default = 'lr', help = 'model')
    parser.add_argument('--jobs', '-j', type = int, default = -1, help = 'model')
    args = parser.parse_args()

    #extract the arguments
    dataset = args.dataset
    model = 'mlp'
    njobs = args.jobs

    #define input and output path
    LIWC_dir = './LIWC/'
    data_path = f'./Dataset/{dataset}/'
    in_path = f'./Dataset/{dataset}/df_featured.csv'
    filename =  f"./Models/{dataset}__{model}__all.joblib"


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
    X_train, X_val, X_test, Y_train, Y_val, Y_test, test_mask = get_dataset(df_overall, data_path)
    print(f"\nf={f}\tshape:{X_train.shape}")


    #scale the data
    scl = CustomScaler().fit(X_train)
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
    ##train
    hidden_size = X_train.shape[1]
    clf = get_model('mlp', ps, f1_weighted, njobs = njobs, feature_size = X_train_val.shape[1], hidden_size = hidden_size)

    #train -- if required
    if not os.path.exists(filename):
        clf.fit(X_train_val, Y_train_val) #fit
        dump(clf, filename)
    else:
        clf = load(filename)

    print('Best parameters found:\n', clf.best_params_)

    #split the test set in low, medium, and high credibility
    Y_test = np.array(Y_test, dtype = int)
    X_test_low = X_test[test_mask == 'low']
    Y_test_low = Y_test[test_mask == 'low']
    X_test_medium = X_test[test_mask == 'medium']
    Y_test_medium = Y_test[test_mask == 'medium']
    X_test_high = X_test[test_mask == 'high']
    Y_test_high = Y_test[test_mask == 'high']

    y_pred_low = clf.predict(X_test_low)
    f1_low = f1_score(Y_test_low, y_pred_low, average = 'weighted')

    y_pred_medium = clf.predict(X_test_medium)
    f1_medium = f1_score(Y_test_medium, y_pred_medium, average = 'weighted')

    y_pred_high = clf.predict(X_test_high)
    f1_high = f1_score(Y_test_high, y_pred_high, average = 'weighted')

    y_pred_over = clf.predict(X_test)
    f1_overall = f1_score(Y_test, y_pred_over, average = 'weighted')
    print("PERFORMANCE")
    print(f"\t--->Low: {f1_low * 100:.2f}")
    print(f"\t--->Medium: {f1_medium * 100:.2f}")
    print(f"\t--->High: {f1_high * 100:.2f}")
    print(f"\t--->Overall: {f1_overall * 100:.2f}")
