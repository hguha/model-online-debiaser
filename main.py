#imports
import pandas as pd
import numpy as np
import random

filename = "compas-scores.csv"
n = 1000 #size of training set we want to consider
total_size = sum(1 for line in open(filename))
cols = ["name", "first", "last"] #pick whatever cols we find relevant

#SUBSET DATA(fast, gives us tons of testing options, but non-consistent)
#there's (11700 choose 1000) possible datasets, or 3x10^1481 datasets
keep = random.sample(range(1, total_size), n)
keep.append(0)
to_exclude = [i for i in range(total_size) if i not in keep]
df = pd.read_csv(filename, skiprows = to_exclude, usecols = cols)s
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True) #remove rows with missing data

#FULL DATA(slow, but consistent)
df_full = pd.read_csv(filename, usecols = cols)
df_full.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True) #remove rows with missing data

print(df)


#modeling pipeline?