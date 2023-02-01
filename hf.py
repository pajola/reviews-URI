import pandas as pd
import numpy as np
import argparse
import os

import torch
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments


#custom libraries
from load_dataset import split
from joblib import dump, load

families = ["Emotions", "MetaData", "ProductInfo", "Readability", "ReviewerHistory", "StructFeat", "Embedding", "ProductHistory"]

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    #configuration settings
    datasets = ['Cell_Phones_and_Accessories',  'Digital_Music', 'Electronics', 'Pet_Supplies', 'Toys_and_Games', 'Video_Games']

    #memory optimizer -- we read only one column at a time
    ds_feat = pd.read_csv(f'Dataset/Toys_and_Games/df_featured.csv', index_col = 'Unnamed: 0', nrows = 5)
    cols_overall = list(ds_feat.columns)
    cols_overall_nofeat = [x for x in cols_overall if (families[0] not in x and
        families[1] not in x and families[2] not in x and families[3] not in x and
        families[4] not in x and families[5] not in x and families[6] not in x and families[7] not in x)
    ]

    #load BERT tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    #iterate over the datasets
    for dataset in datasets[:1]:
        print(dataset)
        data_path = f'./Dataset/{dataset}/'
        in_path = f'./Dataset/{dataset}/df_featured.csv'

        #read the dataset
        df_overall = pd.read_csv(in_path, usecols = cols_overall_nofeat)

        #split into training, validation, and testing set
        df_train, df_val, df_test = split(df_overall, data_path)

        #extract text and ground truth
        X_train = df_train['reviewText'].tolist()
        X_val = df_val['reviewText'].tolist()
        X_test = df_test['reviewText'].tolist()
        y_train = df_train['help_label'].tolist()
        y_val = df_val['help_label'].tolist()
        y_test = df_test['help_label'].tolist()

        #tokenize
        train_encodings = tokenizer(X_train, truncation=True, padding=True)
        val_encodings = tokenizer(X_val, truncation=True, padding=True)
        test_encodings = tokenizer(X_test, truncation=True, padding=True)

        #move tokenized items into a torch datasect object
        train_dataset = CustomDataset(train_encodings, y_train)
        val_dataset = CustomDataset(val_encodings, y_val)
        test_dataset = CustomDataset(test_encodings, y_test)

        #define the trainer arguments
        training_args = TrainingArguments(
            output_dir='./hf_results',          # output directory
            num_train_epochs=3,              # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
        )

        #define the model
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

        #define the trainer object
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset             # evaluation dataset
        )

        #fine tune
        trainer.train()
