import os
from os.path import join, dirname

_root_directory = dirname(__file__)

configs = {'root_directory': _root_directory,
           'raw_data_path': join(_root_directory, 'data', 'raw_data.csv'),
           'small_data_path': join(_root_directory, 'data', 'small_data.csv'),
           'train_data_path': join(_root_directory, 'data', 'train_data.csv'),
           'validation_data_path': join(_root_directory, 'data', 'validation_data.csv'),
           'test_data_path': join(_root_directory, 'data', 'test_data.csv')
           }