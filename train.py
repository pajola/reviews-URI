import pandas as pd
import numpy as np
import argparse
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
from sklearn.base import BaseEstimator, TransformerMixin


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

def get_model(model, ps, eval_f, feature_size, use_custom_pca = False, njobs = -1):
    #create the array for the PCA parametersearch 
    # -- no considering values bigger than the current dataset
    k = [None]
    if feature_size >= 64:
        k.append(64)
    if feature_size >= 128:
        k.append(128)

    #define the pca to use
    if use_custom_pca:
        estimator_pca = CustomPCA()
    else:
        estimator_pca = PCA()



    if model == "lr":
        #define an instance of the model
        estimator_lr = LogisticRegression(random_state= 123, max_iter = 100000)

        #define the pipeline
        pipe_lr = Pipeline( steps = [
            ('pca', estimator_pca),
            ('clf', estimator_lr)
        ])

        #define GS hyperparameters
        param_grid_lr ={
        'pca__n_components': k,
        'clf__C': [0.01, 0.1, 1, 10, 10],
        'clf__fit_intercept': (True, False),
        'clf__class_weight': [None, "balanced"]
        }

        #create the grid search instance
        clf_lr = GridSearchCV(estimator= pipe_lr,
                             cv = ps,
                             param_grid= param_grid_lr,
                             scoring= eval_f,
                             n_jobs=njobs,
                             refit = True)
        return clf_lr
    elif model == 'dt':
        #define the components of the model's pipeline
        estimator_dt = DecisionTreeClassifier(random_state= 123)

        #define pipeline
        pipe_dt = Pipeline( steps = [
            ('pca', estimator_pca),
            ('clf', estimator_dt)
        ])

        #define GS hyperparameters
        param_grid_dt ={
        'pca__n_components': k,
        'clf__criterion': ["gini", "entropy"],
        'clf__max_depth': [None, 2, 3, 4, 5, 10],
        'clf__class_weight': [None, "balanced"],
        'clf__min_samples_split': [2, 3, 5],
        'clf__min_samples_leaf': [2, 3, 4, 5]
        }


        clf_dt = GridSearchCV(estimator= pipe_dt,
                             cv = ps,
                             param_grid= param_grid_dt,
                             scoring= eval_f,
                             n_jobs=njobs,
                             refit = True)
        return clf_dt
    elif model == 'rf':
        #define the components of the model's pipeline
        estimator_rf = RandomForestClassifier(random_state= 123)

        #define pipeline
        pipe_rf = Pipeline( steps = [
            ('pca', estimator_pca),
            ('clf', estimator_rf)
        ])

        #define GS hyperparameters
        param_grid_rf ={
        'pca__n_components': k,
        'clf__criterion': ["gini", "entropy"],
        'clf__max_depth': [None, 2, 3, 4, 5, 10],
        'clf__class_weight': [None, "balanced"],
        'clf__min_samples_split': [2, 3, 5],
        'clf__min_samples_leaf': [2, 3, 4, 5],
        'clf__n_estimators': [4, 16, 32, 64, 128, 256]
        }

        clf_rf = GridSearchCV(estimator= pipe_rf,
                             cv = ps,
                             param_grid= param_grid_rf,
                             scoring= eval_f,
                             n_jobs=njobs,
                             refit = True)

        return clf_rf
    elif model == 'ab':
        #define the components of the model's pipeline
        estimator_ab = AdaBoostClassifier(random_state= 123)

        #define pipeline
        pipe_ab = Pipeline( steps = [
            ('pca', estimator_pca),
            ('clf', estimator_ab)
        ])

        #define GS hyperparameters
        param_grid_ab ={
        'pca__n_components': k,
        'clf__n_estimators': [2, 8, 16, 32, 64],
        'clf__learning_rate': [0.1, 1., 10],
        }

        clf_ab = GridSearchCV(estimator= pipe_ab,
                             cv = ps,
                             param_grid= param_grid_ab,
                             scoring= eval_f,
                             n_jobs=njobs,
                             refit = True)
        return clf_ab

    elif model == 'nb':
        #define the components of the model's pipeline
        estimator_nb = GaussianNB()

        #define pipeline
        pipe_nb= Pipeline( steps = [
            ('pca', estimator_pca),
            ('clf', estimator_nb)
        ])

        #define GS hyperparameters
        param_grid_nb ={
            'pca__n_components': k,
            'clf__var_smoothing' : [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
        }

        clf_nb = GridSearchCV(estimator= pipe_nb,
                             cv = ps,
                             param_grid= param_grid_nb,
                             scoring= eval_f,
                             n_jobs=njobs,
                             refit = True)
        return clf_nb
    else:
        raise Exception("Invalid model choice.")

def get_dataset(ds, data_path):
    ## the only boolean variable is ``MetaData.StarRating_isModerate''. 
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

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare data')
    parser.add_argument('--dataset', '-d', type= str, help='name of the dataset')
    parser.add_argument('--model', '-m', type = str, default = 'lr', help = 'model')
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
    log_path = f'./Results/log_models.csv'

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
        use_custom_pca = False
        if f == "MetaData":
            use_custom_pca = True

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


        #scale the data
        if f == "MetaData":
            scl = CustomScaler().fit(X_train)
        else:
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
            #get the classifier
            clf = get_model('lr', ps, f1_weighted, feature_size = X_train_val.shape[1], use_custom_pca = use_custom_pca, njobs = njobs)

            #train and evaluate
            clf.fit(X_train_val, Y_train_val) #fit
            y_pred = clf.predict(X_test)

            f1_ = f1_score(Y_test, y_pred,average = 'weighted')
            prec_ = precision_score(Y_test, y_pred)
            rec_ = recall_score(Y_test, y_pred)
            roc_ = roc_auc_score(Y_test, y_pred)

            log_df.loc[len(log_df)] = [dataset, f, 'lr', 'f1', f1_]
            log_df.loc[len(log_df)] = [dataset, f, 'lr', 'prec', prec_]
            log_df.loc[len(log_df)] = [dataset, f, 'lr', 'recall', rec_]
            log_df.loc[len(log_df)] = [dataset, f, 'lr', 'roc', roc_]

            #save the model
            dump(clf, f"./Models/{dataset}__{model}__{f}.joblib")

        #############################################################
        # decision tree
        if len(log_df[(log_df['Model'] == 'dt') & (log_df['Dataset'] == dataset) & (log_df['Feature'] == f)]) == 0:
            model = "dt"
            #get the classifier
            clf = get_model('dt', ps, f1_weighted, feature_size = X_train_val.shape[1], use_custom_pca = use_custom_pca, njobs = njobs)

            #train and evaluate
            clf.fit(X_train_val, Y_train_val) #fit
            y_pred = clf.predict(X_test)

            f1_ = f1_score(Y_test, y_pred,average = 'weighted')
            prec_ = precision_score(Y_test, y_pred)
            rec_ = recall_score(Y_test, y_pred)
            roc_ = roc_auc_score(Y_test, y_pred)

            log_df.loc[len(log_df)] = [dataset, f, 'dt', 'f1', f1_]
            log_df.loc[len(log_df)] = [dataset, f, 'dt', 'prec', prec_]
            log_df.loc[len(log_df)] = [dataset, f, 'dt', 'recall', rec_]
            log_df.loc[len(log_df)] = [dataset, f, 'dt', 'roc', roc_]

            #save the model
            dump(clf, f"./Models/{dataset}__{model}__{f}.joblib")

        #############################################################
        # random forest
        if len(log_df[(log_df['Model'] == 'rf') & (log_df['Dataset'] == dataset) & (log_df['Feature'] == f)]) == 0:
            model = "rf"
            #get the classifier
            clf = get_model('rf', ps, f1_weighted, feature_size = X_train_val.shape[1], use_custom_pca = use_custom_pca, njobs = njobs)

            #train and evaluate
            clf.fit(X_train_val, Y_train_val) #fit
            y_pred = clf.predict(X_test)

            f1_ = f1_score(Y_test, y_pred,average = 'weighted')
            prec_ = precision_score(Y_test, y_pred)
            rec_ = recall_score(Y_test, y_pred)
            roc_ = roc_auc_score(Y_test, y_pred)

            log_df.loc[len(log_df)] = [dataset, f, 'rf', 'f1', f1_]
            log_df.loc[len(log_df)] = [dataset, f, 'rf', 'prec', prec_]
            log_df.loc[len(log_df)] = [dataset, f, 'rf', 'recall', rec_]
            log_df.loc[len(log_df)] = [dataset, f, 'rf', 'roc', roc_]

            #save the model
            dump(clf, f"./Models/{dataset}__{model}__{f}.joblib")

        #############################################################
        # ada boost
        if len(log_df[(log_df['Model'] == 'ab') & (log_df['Dataset'] == dataset) & (log_df['Feature'] == f)]) == 0:
            model = "ab"
            #get the classifier
            clf = get_model('ab', ps, f1_weighted, feature_size = X_train_val.shape[1], use_custom_pca = use_custom_pca, njobs = njobs)

            #train and evaluate
            clf.fit(X_train_val, Y_train_val) #fit
            y_pred = clf.predict(X_test)

            f1_ = f1_score(Y_test, y_pred,average = 'weighted')
            prec_ = precision_score(Y_test, y_pred)
            rec_ = recall_score(Y_test, y_pred)
            roc_ = roc_auc_score(Y_test, y_pred)

            log_df.loc[len(log_df)] = [dataset, f, 'ab', 'f1', f1_]
            log_df.loc[len(log_df)] = [dataset, f, 'ab', 'prec', prec_]
            log_df.loc[len(log_df)] = [dataset, f, 'ab', 'recall', rec_]
            log_df.loc[len(log_df)] = [dataset, f, 'ab', 'roc', roc_]

            #save the model
            dump(clf, f"./Models/{dataset}__{model}__{f}.joblib")

        #############################################################
        # naive bayes
        if len(log_df[(log_df['Model'] == 'nb') & (log_df['Dataset'] == dataset) & (log_df['Feature'] == f)]) == 0:
            model = "nb"
            #get the classifier
            clf = get_model('nb', ps, f1_weighted, feature_size = X_train_val.shape[1], use_custom_pca = use_custom_pca, njobs = njobs)

            #train and evaluate
            clf.fit(X_train_val, Y_train_val) #fit
            y_pred = clf.predict(X_test)

            f1_ = f1_score(Y_test, y_pred,average = 'weighted')
            prec_ = precision_score(Y_test, y_pred)
            rec_ = recall_score(Y_test, y_pred)
            roc_ = roc_auc_score(Y_test, y_pred)

            log_df.loc[len(log_df)] = [dataset, f, 'nb', 'f1', f1_]
            log_df.loc[len(log_df)] = [dataset, f, 'nb', 'prec', prec_]
            log_df.loc[len(log_df)] = [dataset, f, 'nb', 'recall', rec_]
            log_df.loc[len(log_df)] = [dataset, f, 'nb', 'roc', roc_]

            #save the model
            dump(clf, f"./Models/{dataset}__{model}__{f}.joblib")

    log_df.to_csv(log_path)

    # ##
    # ## ===============================================================================

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


    #scale the data
    # scl = StandardScaler().fit(X_train)
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

    use_custom_pca = True

    ############################################################
    # logistic regression
    if len(log_df[(log_df['Model'] == 'lr') & (log_df['Dataset'] == dataset) & (log_df['Feature'] == f)]) == 0:
        #
        print("LR")
        model = "lr"
        #get the classifier
        clf = get_model('lr', ps, f1_weighted, feature_size = X_train_val.shape[1], use_custom_pca = use_custom_pca, njobs = njobs)

        #train and evaluate
        clf.fit(X_train_val, Y_train_val) #fit
        y_pred = clf.predict(X_test)

        f1_ = f1_score(Y_test, y_pred,average = 'weighted')
        prec_ = precision_score(Y_test, y_pred)
        rec_ = recall_score(Y_test, y_pred)
        roc_ = roc_auc_score(Y_test, y_pred)

        log_df.loc[len(log_df)] = [dataset, f, 'lr', 'f1', f1_]
        log_df.loc[len(log_df)] = [dataset, f, 'lr', 'prec', prec_]
        log_df.loc[len(log_df)] = [dataset, f, 'lr', 'recall', rec_]
        log_df.loc[len(log_df)] = [dataset, f, 'lr', 'roc', roc_]

        #save the model
        dump(clf, f"./Models/{dataset}__{model}__{f}.joblib")

    #############################################################
    # decision tree
    if len(log_df[(log_df['Model'] == 'dt') & (log_df['Dataset'] == dataset) & (log_df['Feature'] == f)]) == 0:
        print("DT")
        model = "dt"
        #get the classifier
        clf = get_model('dt', ps, f1_weighted, feature_size = X_train_val.shape[1], use_custom_pca = use_custom_pca, njobs = njobs)

        #train and evaluate
        clf.fit(X_train_val, Y_train_val) #fit
        y_pred = clf.predict(X_test)

        f1_ = f1_score(Y_test, y_pred,average = 'weighted')
        prec_ = precision_score(Y_test, y_pred)
        rec_ = recall_score(Y_test, y_pred)
        roc_ = roc_auc_score(Y_test, y_pred)

        log_df.loc[len(log_df)] = [dataset, f, 'dt', 'f1', f1_]
        log_df.loc[len(log_df)] = [dataset, f, 'dt', 'prec', prec_]
        log_df.loc[len(log_df)] = [dataset, f, 'dt', 'recall', rec_]
        log_df.loc[len(log_df)] = [dataset, f, 'dt', 'roc', roc_]

        #save the model
        dump(clf, f"./Models/{dataset}__{model}__{f}.joblib")

    #############################################################
    # random forest
    if len(log_df[(log_df['Model'] == 'rf') & (log_df['Dataset'] == dataset) & (log_df['Feature'] == f)]) == 0:
        print("RF")
        model = "rf"
        #get the classifier
        clf = get_model('rf', ps, f1_weighted, feature_size = X_train_val.shape[1], use_custom_pca = use_custom_pca, njobs = njobs)

        #train and evaluate
        clf.fit(X_train_val, Y_train_val) #fit
        y_pred = clf.predict(X_test)

        f1_ = f1_score(Y_test, y_pred,average = 'weighted')
        prec_ = precision_score(Y_test, y_pred)
        rec_ = recall_score(Y_test, y_pred)
        roc_ = roc_auc_score(Y_test, y_pred)

        log_df.loc[len(log_df)] = [dataset, f, 'rf', 'f1', f1_]
        log_df.loc[len(log_df)] = [dataset, f, 'rf', 'prec', prec_]
        log_df.loc[len(log_df)] = [dataset, f, 'rf', 'recall', rec_]
        log_df.loc[len(log_df)] = [dataset, f, 'rf', 'roc', roc_]

        #save the model
        dump(clf, f"./Models/{dataset}__{model}__{f}.joblib")

    #############################################################
    # ada boost
    if len(log_df[(log_df['Model'] == 'ab') & (log_df['Dataset'] == dataset) & (log_df['Feature'] == f)]) == 0:
        print("AB")
        model = "ab"
        #get the classifier
        clf = get_model('ab', ps, f1_weighted, feature_size = X_train_val.shape[1], use_custom_pca = use_custom_pca, njobs = njobs)

        #train and evaluate
        clf.fit(X_train_val, Y_train_val) #fit
        y_pred = clf.predict(X_test)

        f1_ = f1_score(Y_test, y_pred,average = 'weighted')
        prec_ = precision_score(Y_test, y_pred)
        rec_ = recall_score(Y_test, y_pred)
        roc_ = roc_auc_score(Y_test, y_pred)

        log_df.loc[len(log_df)] = [dataset, f, 'ab', 'f1', f1_]
        log_df.loc[len(log_df)] = [dataset, f, 'ab', 'prec', prec_]
        log_df.loc[len(log_df)] = [dataset, f, 'ab', 'recall', rec_]
        log_df.loc[len(log_df)] = [dataset, f, 'ab', 'roc', roc_]

        #save the model
        dump(clf, f"./Models/{dataset}__{model}__{f}.joblib")

    #############################################################
    # naive bayes
    if len(log_df[(log_df['Model'] == 'nb') & (log_df['Dataset'] == dataset) & (log_df['Feature'] == f)]) == 0:
        print("NB")
        model = "nb"
        #get the classifier
        clf = get_model('nb', ps, f1_weighted, feature_size = X_train_val.shape[1], use_custom_pca = use_custom_pca, njobs = njobs)

        #train and evaluate
        clf.fit(X_train_val, Y_train_val) #fit
        y_pred = clf.predict(X_test)

        f1_ = f1_score(Y_test, y_pred,average = 'weighted')
        prec_ = precision_score(Y_test, y_pred)
        rec_ = recall_score(Y_test, y_pred)
        roc_ = roc_auc_score(Y_test, y_pred)

        log_df.loc[len(log_df)] = [dataset, f, 'nb', 'f1', f1_]
        log_df.loc[len(log_df)] = [dataset, f, 'nb', 'prec', prec_]
        log_df.loc[len(log_df)] = [dataset, f, 'nb', 'recall', rec_]
        log_df.loc[len(log_df)] = [dataset, f, 'nb', 'roc', roc_]

        #save the model
        dump(clf, f"./Models/{dataset}__{model}__{f}.joblib")

    log_df.to_csv(log_path)
