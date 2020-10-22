import pandas as pd
import numpy as np
import random
import torch
from config import configs

raw_filename = configs['raw_data_path']
n = 1000  # size of training set we want to consider
total_size = sum(1 for line in open(raw_filename))
use_full = True  # if True, gives the entire data set. If False, gives us multiple trials

# COLUMNS
cat_cols = ['sex', 'age_cat', 'race', 'score_text', 'c_charge_degree']
date_cols = ['c_jail_in', 'c_jail_out']
num_cols = ['age', 'decile_score', 'priors_count', 'days_b_screening_arrest', 'is_recid',
            'two_year_recid'] + date_cols

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
    for f in filters: df = df.query(f)

    # ensure proper dtypes
    for c in cat_cols:
        df[c] = df[c].astype('category')
    for c in date_cols:
        df[c] = pd.to_datetime(df[c]).astype(np.int64)  # we want to turn the dates into a single number

    # There are some mutations to add factors but I don't know what they are used for or how to do it in pandas
    # so I think I'll skip it for now and dive deeper later

    return df


def create_tensors(df):
    # Cat Tensor
    code_vals = []
    for c in cat_cols:
        code_vals.append(df[c].cat.codes.values)
    cat_data = np.stack(code_vals, 1)
    cat_tensor = torch.tensor(cat_data, dtype=torch.int64)
    cat_tensor_ds = torch.utils.data.TensorDataset(cat_tensor)

    # this is a vectorization as opposed to a linearization that might be useful, idk
    cat_col_sizes = [len(df[c].cat.categories) for c in cat_cols]
    cat_embedding_sizes = [(col_size, min(50, (col_size + 1) // 2)) for col_size in cat_col_sizes]

    # Num Tensor
    num_data = np.stack([df[c].values for c in num_cols], 1)
    num_tensor = torch.tensor(num_data, dtype=torch.float)
    num_tensor_ds = torch.utils.data.TensorDataset(num_tensor)
    
    # Not sure if these can be combined, or whether that would make everything confusing?
    return num_tensor_ds, cat_tensor_ds


def create_dataloader(df):
    num_tensor, cat_tensor = create_tensors(df)
    num_loader = torch.utils.data.DataLoader(num_tensor, batch_size=2, collate_fn=collate_wrapper, pin_memory=True)
    cat_loader = torch.utils.data.DataLoader(cat_tensor, batch_size=2, collate_fn=collate_wrapper, pin_memory=True)
    return num_loader, cat_loader


def divide_data_set(filename, size=1000, split=[60, 20, 20]):
    df = get_raw_df(filename)
    n = len(df)
    # shuffle
    df = df.sample(frac=1)

    # small_data(for other analysis)
    small_df = df.head(size)
    small_df.to_csv('data/small_data.csv')

    # create dfs for each set(can technically be skipped, but good practice)
    amts = []
    for i in split:
        amts.append(int(n * i / 100))
    print(amts)
    train_df = df[:amts[0]]
    validate_df = df[amts[0]:(amts[0] + amts[1])]
    test_df = df[(amts[0] + amts[1]):n]

    # create csv's
    train_df.to_csv('data/training_data.csv')
    validate_df.to_csv('data/validation_data.csv')
    test_df.to_csv('data/test_data.csv')

    # return our dfs
    return train_df, validate_df, test_df

# Uncomment to get all the data
divide_data_set(get_raw_data(raw_filename))
