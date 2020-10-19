import pandas as pd
import numpy as np
import random
import torch
from config import configs

raw_filename = configs['raw_data_path']
n = 1000  # size of training set we want to consider
total_size = sum(1 for line in open(raw_filename))
use_full = True #if True, gives the entire data set. If False, gives us multiple trials

#COLUMNS
cat_cols = ['sex', 'age_cat', 'race', 'score_text', 'c_charge_degree']
date_cols = ['c_jail_in', 'c_jail_out']
num_cols = ['age', 'decile_score', 'priors_count', 'days_b_screening_arrest','is_recid', 
            'two_year_recid'] + date_cols

cols = cat_cols+num_cols
#this may be a config, it will change depending on the csv
filters = ['days_b_screening_arrest <= 30', 'days_b_screening_arrest >= -30', "is_recid != -1", 
                "c_charge_degree != 'O'", "score_text != 'N/A'"]
#  SUBSET DATA(fast, gives us tons of testing options, but non-consistent)
#  there's (11700 choose 1000) possible datasets, or 3x10^1481 datasets

def get_raw_data(filename):
        df = None
        if(not use_full):
                keep = random.sample(range(1, total_size), n)
                keep.append(0)
                to_exclude = [i for i in range(total_size) if i not in keep]
                df = pd.read_csv(filename, skiprows=to_exclude, usecols=cols)
                # df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)  #remove rows with missing data

        else:
                df = pd.read_csv(filename, usecols = cols)
                df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)  #remove rows with missing data
        return df


def munge_data(filters, mutations = None):
        df = get_raw_data(raw_filename)
        
        #filter out rows
        for f in filters: df = df.query(f)

        #ensure proper dtypes
        for c in cat_cols: df[c] = df[c].astype('category')
        for c in date_cols: df[c] = pd.to_datetime(df[c]).astype(np.int64) #we want to turn the dates into a single number
        
        #There are some mutations to add factors but I don't know what they are used for or how to do it in pandas 
        #so I think I'll skip it for now and dive deeper later

        return df


def create_tensors():
        df = munge_data(filters)
        
        #Cat Tensor
        code_vals = []
        for c in cat_cols: code_vals.append(df[c].cat.codes.values)
        cat_data = np.stack(code_vals, 1)
        cat_tensor = torch.tensor(cat_data, dtype=torch.int64)

        #this is a vectorization as opposed to a linearization that might be useful, idk
        cat_col_sizes = [len(df[c].cat.categories) for c in cat_cols]
        cat_embedding_sizes = [(col_size, min(50, (col_size+1)//2)) for col_size in cat_col_sizes]
        print(num_tensor)
        
        #Num Tensor
        num_data = np.stack([df[c].values for c in num_cols], 1)
        num_tensor = torch.tensor(num_data, dtype=torch.float)
        
        return (num_tensor, cat_tensor)