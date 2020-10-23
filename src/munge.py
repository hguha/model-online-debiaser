import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from config import path_configs, hyperparameters

raw_filename = path_configs['raw_data_path']
n = 1000  # size of training set we want to consider
total_size = sum(1 for line in open(raw_filename))
use_full = True  # if True, gives the entire data set. If False, gives us multiple trials

# COLUMNS
cat_cols = ['sex', 'age_cat', 'race', 'score_text', 'c_charge_degree']
date_cols = ['c_jail_in', 'c_jail_out']
num_cols = ['age', 'decile_score', 'priors_count', 'days_b_screening_arrest', 'is_recid',
            'two_year_recid']

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


class RecidivismDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        covariate_cols = [x for x in cat_cols + num_cols if x != 'score_text']
        covariates = row.loc[covariate_cols]
        score = row.loc['score_text']

        return torch.cat([x.float() for x in covariates.values.tolist()]), score.argmax()


def create_data_loader(df):

    def format_categorical_col(df, col):
        df[col] = df[col].astype('category')
        try:
            categories = df[col].cat.categories.tolist()
        except Exception as e:
            print(col)
            print(df[col])
            breakpoint()
        num_categories = len(categories)

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

    dataset = RecidivismDataset(df)
    return DataLoader(dataset, batch_size=hyperparameters['batch_size'])


    # Cat Tensor
    # code_vals = []
    # for c in cat_cols:
    #     code_vals.append(df[c].cat.codes.values)
    # cat_data = np.stack(code_vals, 1)
    # cat_tensor = torch.tensor(cat_data, dtype=torch.int64)
    # cat_tensor_ds = torch.utils.data.TensorDataset(cat_tensor)
    #
    # # this is a vectorization as opposed to a linearization that might be useful, idk
    # cat_col_sizes = [len(df[c].cat.categories) for c in cat_cols]
    # cat_embedding_sizes = [(col_size, min(50, (col_size + 1) // 2)) for col_size in cat_col_sizes]
    #
    # # Num Tensor
    # num_data = np.stack([df[c].values for c in num_cols], 1)
    # num_tensor = torch.tensor(num_data, dtype=torch.float)
    # num_tensor_ds = torch.utils.data.TensorDataset(num_tensor)

    # Not sure if these can be combined, or whether that would make everything confusing?
    # return num_tensor_ds, cat_tensor_ds


# def create_dataloader(df):
#     num_tensor, cat_tensor = create_tensors(df)
#     num_loader = torch.utils.data.DataLoader(num_tensor, batch_size=2, pin_memory=True)
#     cat_loader = torch.utils.data.DataLoader(cat_tensor, batch_size=2, pin_memory=True)
#     return num_loader, cat_loader


def divide_data_set(filename, size=1000, split=[60, 20, 20]):
    df = get_raw_df(filename)
    n = len(df)
    # shuffle
    df = df.sample(frac=1)

    # small_data(for other analysis)
    small_df = df.head(size)
    small_df.to_csv(path_configs['small_data_path'])

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
divide_data_set(raw_filename)

def read_datasets():
    train = pd.read_csv(path_configs['train_data_path'])
    validation = pd.read_csv(path_configs['validation_data_path'])
    test = pd.read_csv(path_configs['test_data_path'])
    return train, validation, test

# create train, validation, test data loaders
train, validation, test = read_datasets()
train_loader = create_data_loader(munge_data(train))
validation_loader = create_data_loader(munge_data(validation))
test_loader = create_data_loader(munge_data(test))

# for x, y in train_loader:
#     print(x)
#     print(y)
#     break;