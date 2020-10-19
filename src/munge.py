import pandas as pd
import numpy as np
import random
from config import configs

raw_filename = configs['raw_data_path']
n = 1000  # size of training set we want to consider
total_size = sum(1 for line in open(raw_filename))

cols = ['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count',
        'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number', 'c_offense_date',
        'c_arrest_date', 'c_days_from_compas', 'c_charge_degree', 'c_charge_desc', 'is_recid', 'num_r_cases',
        'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out',
        'is_violent_recid', 'num_vr_cases', 'vr_case_number', 'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc',
        'v_type_of_assessment', 'v_decile_score', 'v_score_text', 'v_screening_date', 'type_of_assessment',
        'decile_score', 'score_text', 'screening_date']

#  SUBSET DATA(fast, gives us tons of testing options, but non-consistent)
#  there's (11700 choose 1000) possible datasets, or 3x10^1481 datasets
keep = random.sample(range(1, total_size), n)
keep.append(0)
to_exclude = [i for i in range(total_size) if i not in keep]
df = pd.read_csv(raw_filename, skiprows=to_exclude, usecols=cols)
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)  #remove rows with missing data


#  FULL DATA(slow, but consistent)
df_full = pd.read_csv(raw_filename, usecols = cols)
df_full.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)  #remove rows with missing data


#  modeling pipeline?

