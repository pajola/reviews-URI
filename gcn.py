"""
In this script, we are going to train and test simple GNN.
We follow Pytorch Geometric tutorial.

https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
"""
from typing import Callable, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import copy
import random

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

def load_amazon_dataset(in_path, LIWC_dir, dataset):
    # families = ["Emotions", "MetaData", "Readability", "StructFeat", "Embedding"]
    families = ["Emotions", "MetaData", "ProductInfo", "Readability", "ReviewerHistory", "StructFeat", "Embedding", "ProductHistory"]

    #laod the data
    #memory optimizer -- we read only one column at a time
    ds_feat = pd.read_csv(f'Dataset/Toys_and_Games/df_featured.csv', index_col = 'Unnamed: 0')
    cols_overall = list(ds_feat.columns)
    cols_overall_nofeat = [x for x in cols_overall if (families[0] not in x and
        families[1] not in x and families[2] not in x and families[3] not in x and families[4] not in x)
    ]
    cols_overall_feat = [x for x in cols_overall if (families[0] in x or
        families[1] in x or families[2] in x or families[3] in x or
        families[4] in x)
    ]

    c = cols_overall_feat
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

    return df_overall

def generate_graph(dataset, path, n_components = None):
    #generate the node idx field
    dataset['node-idx'] = list(range(len(dataset)))

    #define the training, validation, and testing splits
    #load the splits info
    with open(path + 'split.pkl', 'rb') as file:
        splits = pickle.load(file)

    split1 = splits['split1']
    split2 = splits['split2']
    # print(split1, split2)

    dataset['reviewTime'] = pd.to_datetime(dataset.reviewTime,
        errors= 'coerce') #cast the time in a proper object

    # print(dataset['reviewTime'].min(), dataset['reviewTime'].max())
    # print(np.sum(dataset['reviewTime'] < split1))
    # print(np.sum(dataset['reviewTime'] > split2))
    dataset['split'] = ''
    dataset.at[dataset['reviewTime'] < split1, 'split'] = 'train'
    dataset.at[(dataset['reviewTime'] < split2) & (dataset['reviewTime'] >= split1), 'split'] = 'val'
    dataset.at[dataset['reviewTime'] >= split2, 'split'] = 'test'
    mask = np.array(dataset['split'].values)

    #we first need to define the nodes with their embedding.
    #we only use readability based features for now
    families = ["Emotions", "MetaData", "Readability", "StructFeat", "Embedding"]
    target_col =[]
    target_col += [x for x in dataset.columns if families[0] in x]
    target_col += [x for x in dataset.columns if families[1] in x]
    target_col += [x for x in dataset.columns if families[2] in x]
    target_col += [x for x in dataset.columns if families[3] in x]
    target_col += [x for x in dataset.columns if families[4] in x]

    #obtain the nodes, represented as embeddings
    # nodes = dataset[target_col].to_numpy(dtype = np.float)
    nodes = dataset[target_col] 
    if "MetaData.StarRating_isModerate" in nodes.columns:
        nodes["MetaData.StarRating_isModerate_"] = nodes.loc[:, "MetaData.StarRating_isModerate"]
        nodes.drop(['MetaData.StarRating_isModerate'], axis = 1, inplace = True)       
    nodes = nodes.to_numpy(dtype = np.float)

    # print(nodes[:10])

    #we need to scale the input
    scl = CustomScaler().fit(nodes[mask == 'train', :])
    nodes = scl.transform(nodes)

    #and now we apply the pca
    pca = CustomPCA(n_components= n_components).fit(nodes[mask == 'train', :])
    nodes = pca.transform(nodes)

    #define the adjacency matrix
    print("Define the edges")
    adj = []
    for i, idx_pivot in tqdm(enumerate(dataset.index), total = len(dataset)):
        #get the current userid, asin, and timestamp
        ref_uid = dataset.at[idx_pivot, 'reviewerID']
        ref_pid = dataset.at[idx_pivot, 'asin']
        ref_ts = dataset.at[idx_pivot, 'unixReviewTime']
        ref_idx = dataset.at[idx_pivot, 'node-idx']

        #we filter the dataset
        curr_df = dataset[(dataset['reviewerID'] == ref_uid) | (dataset['asin'] == ref_pid)]

        for j, idx_ref in enumerate(curr_df.index):
            if idx_pivot != idx_ref: #we avoid node self-connections
                cand_idx = curr_df.at[idx_pivot, 'node-idx']
                adj.append([ref_idx, cand_idx])



    #convert to tensor objects
    x = torch.tensor(nodes, dtype = torch.float)
    edge_index = torch.tensor(adj, dtype = torch.long).t().contiguous()

    #define the ground truth variable
    y = torch.tensor(dataset['help_label'].tolist(), dtype = torch.long)

    #create the graph object
    data = Data(x=x, edge_index=edge_index, y = y)


    #define training - validation - testing masks
    data.train_mask = torch.tensor( mask == 'train', dtype = torch.bool)
    data.val_mask = torch.tensor( mask == 'val', dtype = torch.bool)
    data.test_mask = torch.tensor( mask == 'test', dtype = torch.bool)

    #define the testing mask with low, medium and high credibility
    test_mask = []
    for x in dataset.voters.tolist():
        if x <= 5:
            test_mask.append('low')
        elif x <= 10:
            test_mask.append('medium')
        elif x > 10:
            test_mask.append('high')
        else:
            raise Exception("Error")
    test_mask = np.array(test_mask)
    data.test_low_credibility_mask = (data.test_mask) & (torch.tensor(test_mask == 'low'))
    data.test_med_credibility_mask = (data.test_mask) & (torch.tensor(test_mask == 'medium'))
    data.test_hig_credibility_mask = (data.test_mask) & (torch.tensor(test_mask == 'high'))

    print(data.has_isolated_nodes(), data.has_self_loops(), data.is_directed())
    print(data.train_mask.sum().item(), data.val_mask.sum().item(), data.test_mask.sum().item())
    print(data.test_low_credibility_mask.sum().item(), data.test_med_credibility_mask.sum().item(), data.test_hig_credibility_mask.sum().item())
    return data

class GCN(torch.nn.Module):
    def __init__(self, embedding_dim, num_classes, n_linear = 1, n_conv = 1, hidden_dim = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_conv = n_conv
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.n_linear = n_linear

        if self.n_linear == 0:
            if self.n_conv == 1:
                self.conv1 = GCNConv(self.embedding_dim, self.num_classes)
            elif self.n_conv == 2:
                self.conv1 = GCNConv(self.embedding_dim, self.hidden_dim)
                self.conv2 = GCNConv(self.hidden_dim, self.num_classes)
            else:
                raise Except("DEB")

        else:
            if self.n_conv == 1:
                self.conv1 = GCNConv(self.embedding_dim, self.hidden_dim)
            elif self.n_conv == 2:
                self.conv1 = GCNConv(self.embedding_dim, self.hidden_dim)
                self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
            else:
                raise Except("DEB")

            if self.n_linear == 1:
                self.linear1 = Linear(self.hidden_dim, self.num_classes)
            elif self.n_linear == 2:
                self.linear1 = Linear(self.hidden_dim, int(self.hidden_dim / 2))
                self.linear2 = Linear(int(self.hidden_dim / 2), self.num_classes)
            else:
                raise Except("DEB")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if self.n_linear == 0:
            if self.n_conv == 1:
                x = self.conv1(x, edge_index)
            elif self.n_conv == 2:
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
                x = self.conv2(x, edge_index)
            else:
                raise Except("DEB")

        else:
            if self.n_conv == 1:
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
            elif self.n_conv == 2:
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
                x = self.conv2(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
            else:
                raise Except("DEB")

            if self.n_linear == 1:
                x = self.linear1(x)
            elif self.n_linear == 2:
                x = self.linear1(x)
                x = F.relu(x)
                x = self.linear2(x)
            else:
                raise Except("DEB")

        # x = self.conv1(x, edge_index)
        # if self.n_linear == 1:
        #     x = F.relu(x)
        #     x = F.dropout(x, training=self.training)
        #     x = self.linear1(x)
        #
        # if self.n_linear == 2:
        #     x = F.relu(x)
        #     x = F.dropout(x, training=self.training)
        #     x = self.linear1(x)
        #     x = F.relu(x)
        #     x = self.linear2(x)

        return F.log_softmax(x, dim=1)

def train(model, data):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    #
    trainer = True
    best_score = -1.
    early_stopping = 5
    early_stopping_cnt = 0
    max_it = 200
    it = 0
    best_model_wts = copy.deepcopy(model.state_dict())


    while trainer and it < max_it:
        it +=1

        #optimize the model
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        #performance
        model.eval()
        pred = model(data).argmax(dim=1)

        y_train_pred = pred[data.train_mask]
        y_train_true = data.y[data.train_mask]
        train_score = f1_score(y_train_true, y_train_pred, average = 'weighted')
        y_val_pred = pred[data.val_mask]
        y_val_true = data.y[data.val_mask]
        val_score = f1_score(y_val_true, y_val_pred, average = 'weighted')

        if val_score >= best_score:
            best_score = val_score
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stopping_cnt = 0
            # print(f"Improvement at it: {it}.\tVal:{val_score:.4f}\tTrain:{train_score:.4f}")
        else:
            early_stopping_cnt += 1
            # print(f"\t\t-->No Improvement at it: {it}.\tVal:{val_score:.4f}\tTrain:{train_score:.4f}")

            if early_stopping_cnt == early_stopping:
                trainer = False

    return best_model_wts, best_score

if __name__ == '__main__':
    #setting up the machine
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #configuration settings
    datasets = ['Cell_Phones_and_Accessories',  'Digital_Music', 'Electronics', 'Pet_Supplies', 'Toys_and_Games', 'Video_Games']

    #dataset
    for ds in datasets:
        print("\n\n\n\n\n\n\n",ds)
        for nc in [64, 128, None]:
            print(f"\tn_components = {nc}")
            #define input and output path
            LIWC_dir = './LIWC/'
            data_path = f'./Dataset/{ds}/'
            in_path = f'./Dataset/{ds}/df_featured.csv'

            #load the amazon dataset
            dataset = load_amazon_dataset(in_path, LIWC_dir, ds)

            #convert it into a torch geometric structure
            data = generate_graph(dataset, data_path, n_components=nc).to(device)

            print("\t 1 Conv - NO Linear - LAYERS")
            model = GCN(embedding_dim = data.num_node_features,
            num_classes = 2, n_linear = 0)

            best_model_wts, best_score = train(model, data)
            print(f"\tValidation score = {100* best_score:.2f}")

            #test
            # print(f"\n\n\nBEST MODEL dim= {gs_best_dim} n_layers = {gs_best_nl}")
            model = GCN(embedding_dim = data.num_node_features,
            num_classes = 2, n_linear = 0).to(device)
            model.load_state_dict(best_model_wts)

            #performance
            model.eval()
            pred = model(data).argmax(dim=1)

            y_test_pred = pred[data.test_low_credibility_mask]
            y_test_true = data.y[data.test_low_credibility_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test Low: {test_score * 100:.2f}")

            y_test_pred = pred[data.test_med_credibility_mask]
            y_test_true = data.y[data.test_med_credibility_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test Medium: {test_score* 100:.2f}")

            y_test_pred = pred[data.test_hig_credibility_mask]
            y_test_true = data.y[data.test_hig_credibility_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test High: {test_score * 100:.2f}")

            y_test_pred = pred[data.test_mask]
            y_test_true = data.y[data.test_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test Overall: {test_score * 100:.2f}")

            print("\n\n\n\t 2 Conv - NO Linear - LAYERS")
            best_hd = None
            best_val = -1
            best_wts = None
            for hd in [32, 64, 128, 256, 512]:
                model = GCN(embedding_dim = data.num_node_features,
                num_classes = 2, n_conv = 2, n_linear = 0, hidden_dim = hd)

                best_model_wts, best_score = train(model, data)
                print(f"\tHDIM = {hd}\tValidation score = {100* best_score:.2f}")

                if best_score > best_val:
                    best_val = best_score
                    best_wts = best_model_wts
                    best_hd = hd

            print(f"\tBest setting. HDIM={best_hd}")

            print(f"\tBest setting. HDIM={best_hd}")

            model = GCN(embedding_dim = data.num_node_features,
            num_classes = 2, n_conv = 2, n_linear = 0, hidden_dim = best_hd).to(device)
            model.load_state_dict(best_wts)

            #performance
            model.eval()
            pred = model(data).argmax(dim=1)

            y_test_pred = pred[data.test_low_credibility_mask]
            y_test_true = data.y[data.test_low_credibility_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test Low: {test_score * 100:.2f}")

            y_test_pred = pred[data.test_med_credibility_mask]
            y_test_true = data.y[data.test_med_credibility_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test Medium: {test_score* 100:.2f}")

            y_test_pred = pred[data.test_hig_credibility_mask]
            y_test_true = data.y[data.test_hig_credibility_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test High: {test_score * 100:.2f}")

            y_test_pred = pred[data.test_mask]
            y_test_true = data.y[data.test_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test Overall: {test_score * 100:.2f}")

            print("\n\n\t 1 Conv - 1 LINEAR LAYER ")
            best_hd = None
            best_val = -1
            best_wts = None
            for hd in [32, 64, 128, 256, 512]:
                model = GCN(embedding_dim = data.num_node_features,
                num_classes = 2, n_linear = 1, hidden_dim = hd)

                best_model_wts, best_score = train(model, data)
                print(f"\tHDIM = {hd}\tValidation score = {100* best_score:.2f}")

                if best_score > best_val:
                    best_val = best_score
                    best_wts = best_model_wts
                    best_hd = hd

            print(f"\tBest setting. HDIM={best_hd}")

            model = GCN(embedding_dim = data.num_node_features,
            num_classes = 2, n_linear = 1, hidden_dim = best_hd).to(device)
            model.load_state_dict(best_wts)

            #performance
            model.eval()
            pred = model(data).argmax(dim=1)

            y_test_pred = pred[data.test_low_credibility_mask]
            y_test_true = data.y[data.test_low_credibility_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test Low: {test_score * 100:.2f}")

            y_test_pred = pred[data.test_med_credibility_mask]
            y_test_true = data.y[data.test_med_credibility_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test Medium: {test_score* 100:.2f}")

            y_test_pred = pred[data.test_hig_credibility_mask]
            y_test_true = data.y[data.test_hig_credibility_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test High: {test_score * 100:.2f}")

            y_test_pred = pred[data.test_mask]
            y_test_true = data.y[data.test_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test Overall: {test_score * 100:.2f}")

            print("\n\n\t 1 Conv - 2 LINEAR LAYER")
            best_hd = None
            best_val = -1
            best_wts = None
            for hd in [32, 64, 128, 256, 512]:
                model = GCN(embedding_dim = data.num_node_features,
                num_classes = 2, n_linear = 2, hidden_dim = hd)

                best_model_wts, best_score = train(model, data)
                print(f"\tHDIM = {hd}\tValidation score = {100* best_score:.2f}")

                if best_score > best_val:
                    best_val = best_score
                    best_wts = best_model_wts
                    best_hd = hd

            print(f"\tBest setting. HDIM={best_hd}")

            model = GCN(embedding_dim = data.num_node_features,
            num_classes = 2, n_linear = 2, hidden_dim = best_hd).to(device)
            model.load_state_dict(best_wts)

            #performance
            model.eval()
            pred = model(data).argmax(dim=1)

            y_test_pred = pred[data.test_low_credibility_mask]
            y_test_true = data.y[data.test_low_credibility_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test Low: {test_score * 100:.2f}")

            y_test_pred = pred[data.test_med_credibility_mask]
            y_test_true = data.y[data.test_med_credibility_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test Medium: {test_score* 100:.2f}")

            y_test_pred = pred[data.test_hig_credibility_mask]
            y_test_true = data.y[data.test_hig_credibility_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test High: {test_score * 100:.2f}")

            y_test_pred = pred[data.test_mask]
            y_test_true = data.y[data.test_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test Overall: {test_score * 100:.2f}")


            print("\n\n\t 2 Conv - 1 LINEAR LAYER ")
            best_hd = None
            best_val = -1
            best_wts = None
            for hd in [32, 64, 128, 256, 512]:
                model = GCN(embedding_dim = data.num_node_features,
                num_classes = 2, n_conv = 2, n_linear = 1, hidden_dim = hd)

                best_model_wts, best_score = train(model, data)
                print(f"\tHDIM = {hd}\tValidation score = {100* best_score:.2f}")

                if best_score > best_val:
                    best_val = best_score
                    best_wts = best_model_wts
                    best_hd = hd

            print(f"\tBest setting. HDIM={best_hd}")

            model = GCN(embedding_dim = data.num_node_features,
            num_classes = 2, n_conv = 2, n_linear = 1, hidden_dim = best_hd).to(device)
            model.load_state_dict(best_wts)

            #performance
            model.eval()
            pred = model(data).argmax(dim=1)

            y_test_pred = pred[data.test_low_credibility_mask]
            y_test_true = data.y[data.test_low_credibility_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test Low: {test_score * 100:.2f}")

            y_test_pred = pred[data.test_med_credibility_mask]
            y_test_true = data.y[data.test_med_credibility_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test Medium: {test_score* 100:.2f}")

            y_test_pred = pred[data.test_hig_credibility_mask]
            y_test_true = data.y[data.test_hig_credibility_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test High: {test_score * 100:.2f}")

            y_test_pred = pred[data.test_mask]
            y_test_true = data.y[data.test_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test Overall: {test_score * 100:.2f}")

            print("\n\n\t 2 Conv - 2 LINEAR LAYER")
            best_hd = None
            best_val = -1
            best_wts = None
            for hd in [32, 64, 128, 256, 512]:
                model = GCN(embedding_dim = data.num_node_features,
                num_classes = 2, n_conv = 2, n_linear = 2, hidden_dim = hd)

                best_model_wts, best_score = train(model, data)
                print(f"\tHDIM = {hd}\tValidation score = {100* best_score:.2f}")

                if best_score > best_val:
                    best_val = best_score
                    best_wts = best_model_wts
                    best_hd = hd

            print(f"\tBest setting. HDIM={best_hd}")

            model = GCN(embedding_dim = data.num_node_features,
            num_classes = 2, n_conv = 2, n_linear = 2, hidden_dim = best_hd).to(device)
            model.load_state_dict(best_wts)

            #performance
            model.eval()
            pred = model(data).argmax(dim=1)

            y_test_pred = pred[data.test_low_credibility_mask]
            y_test_true = data.y[data.test_low_credibility_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test Low: {test_score * 100:.2f}")

            y_test_pred = pred[data.test_med_credibility_mask]
            y_test_true = data.y[data.test_med_credibility_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test Medium: {test_score* 100:.2f}")

            y_test_pred = pred[data.test_hig_credibility_mask]
            y_test_true = data.y[data.test_hig_credibility_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test High: {test_score * 100:.2f}")

            y_test_pred = pred[data.test_mask]
            y_test_true = data.y[data.test_mask]
            test_score = f1_score(y_test_true, y_test_pred, average = 'weighted')
            print(f"\t--->Test Overall: {test_score * 100:.2f}")
