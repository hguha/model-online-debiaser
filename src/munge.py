import pandas as pd
import numpy as np
import random
from os.path import join
import torch
from torch.utils.data import Dataset, DataLoader
from config import path_configs, hyperparameters
from copy import deepcopy

raw_filename = path_configs['raw_data_path']

#if the use is true, then we will change some percent of Caucasians to African-Americans(or vice versa!)
use_change_sample = False
change_sample = {"use": use_change_sample, "prob": 0.2, "from": "Caucasian", "to": "African-American"}

# COLUMNS
cat_cols = ['sex', 'age_cat', 'race', 'c_charge_degree', 'score_text']
date_cols = ['c_jail_in', 'c_jail_out']
num_cols = ['age', 'decile_score', 'priors_count', 'days_b_screening_arrest', 'is_recid']  # leaving out two_year_recid

cols = cat_cols + num_cols
# this may be a config, it will change depending on the csv
filters = ['days_b_screening_arrest <= 30', 'days_b_screening_arrest >= -30', "is_recid != -1",
           "c_charge_degree != 'O'", "score_text != 'N/A'"]


def get_raw_df(filename, use_col_list=True):
    df = None
    if use_col_list:
        df = pd.read_csv(filename, usecols=cols)
    else:
        df = pd.read_csv(filename)
    df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)  # remove rows with missing data
    return df


def munge_data(df, filters=None, mutations=None):
    # filter out rows
    if filters:
        for f in filters:
            df = df.query(f)

    # ensure proper dtypes
    for c in cat_cols:
        df[c] = df[c].astype('category')
    # for c in date_cols:
    #     df[c] = pd.to_datetime(df[c]).astype(np.int64)  # we want to turn the dates into a single number

    # There are some mutations to add factors but I don't know what they are used for or how to do it in pandas
    # so I think I'll skip it for now and dive deeper later

    return df


def get_categories():
    with open(join(path_configs['root_directory'], 'data', 'categories_dict.txt'), 'r') as f:
        return eval(f.read())


class RecidivismDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.columns = list(self.data.columns)[1:]
        self.categories = get_categories()

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        covariate_cols = [x for x in self.columns if x != 'is_recid']
        covariates = row.loc[covariate_cols]
        score = row.loc['is_recid']
        return torch.cat([x.float() for x in covariates.values.tolist()]), score

    # change a specific tensor row
    def invert_tensor_to_row(self, t):
        new_df = {}
        new_t = t[0].tolist()
        # okay we need the size of each key in the categories dict
        cats = {}
        for col in self.columns:
            if (col in self.categories):
                cats[col] = len(self.categories[col])
            elif col == "is_recid":
                pass
            else:
                cats[col] = 1
        count = 0
        for col in cats:
            tensor_by_col = new_t[count:count + cats[col]]
            if cats[col] > 1:
                idx = tensor_by_col.index(1)
                new_df[col] = self.categories[col][idx]
            else:
                new_df[col] = tensor_by_col[0]
            count += cats[col]

        # returns a dict of the raw_data row
        return new_df

    def check_invert(self):
        for idx in range(len(self.data)):
            print(self.invert_tensor_to_row(self.__getitem__(idx)))

    # change the whole ass thing back for whatever reason
    def invert_tensor_to_df(self):
        new_df = {}
        with open(join(path_configs['root_directory'], 'data', 'categories_dict.txt'), 'r') as f:
            categories = eval(f.read())
        for col in self.data:
            x = self.data[col]
            new_data = []
            if (col in categories):  # If the col is a cat row
                for row in x:
                    f = [int(i) for i in row.tolist()]
                    idx = f.index(1)
                    new_data.append(categories[col][idx])
            else:  # for num rows
                for row in x: new_data.append(int(row))
            new_df[col] = new_data
        return pd.DataFrame(data=new_df)


def create_data_loader(df):
    categories_dict = {}

    if(change_sample["use"]):
        l = df["race"].tolist()
        for i in range(len(l)):
            if l[i]==change_sample["from"] and random.uniform(0,1) < change_sample["prob"]: l[i] = change_sample["to"]
        df["race"] = l

    def format_categorical_col(df, col):
        df[col] = df[col].astype('category')
        try:
            categories = df[col].cat.categories.tolist()
            categories_dict[col] = categories
        except Exception as e:
            print(col)
            print(df[col])
            breakpoint()
        num_categories = len(categories)
        df[col] = df[col].astype('object')

        def format_single_cell(x):
            t = torch.zeros(num_categories)
            idx = categories.index(x)
            t[idx] = 1
            return t

        df[col] = df[col].apply(lambda x: format_single_cell(x))

    def format_numerical_col(df, col):
        df[col] = df[col].apply(lambda x: torch.tensor([x]))

    for col in cat_cols:
        format_categorical_col(df, col)

    for col in num_cols:
        format_numerical_col(df, col)

    with open(join(path_configs['root_directory'], 'data', 'categories_dict.txt'), 'w') as f:
        print(categories_dict, file=f)

    dataset = RecidivismDataset(df)
    # dataset.check_invert()

    return DataLoader(dataset, batch_size=hyperparameters['batch_size'])


def divide_data_set(filename, size=1000, split=[60, 20, 20]):
    df = get_raw_df(filename)
    n = len(df)
    # shuffle
    df = df.sample(frac=1)

    # create dfs for each set(can technically be skipped, but good practice)
    amts = []
    for i in split:
        amts.append(int(n * i / 100))
    train_df = df[:amts[0]]
    validation_df = df[amts[0]:(amts[0] + amts[1])]
    test_df = df[(amts[0] + amts[1]):n]

    # create csv's
    train_df.to_csv(path_configs['train_data_path'])
    validation_df.to_csv(path_configs['validation_data_path'])
    test_df.to_csv(path_configs['test_data_path'])

    # return our dfs
    return train_df, validation_df, test_df


# Uncomment to get all the data
# divide_data_set(raw_filename)

def read_datasets():
    train = pd.read_csv(path_configs['train_data_path'])
    validation = pd.read_csv(path_configs['validation_data_path'])
    test = pd.read_csv(path_configs['test_data_path'])
    return train, validation, test

# train, validation, test = read_datasets()
# train_loader = create_data_loader(munge_data(train))
# validation_loader = create_data_loader(munge_data(validation))
# test_loader = create_data_loader(munge_data(test))


# test_small_invert = create_data_loader(munge_data(pd.read_csv(path_configs['small_data_path'])))
