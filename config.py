import os
from os.path import join, dirname

_root_directory: dirname(__file__)

configs = {'root_directory': _root_directory,
           'raw_data_path': join(_root_directory, 'raw_data.csv'),
           'small_data_path': join(_root_directory, 'small_data.csv'),
           'training_data_path': join(_root_directory, 'training_data.csv'),
           'validation_data_path': join(_root_directory, 'validation_data.csv'),
           'test_data_path': join(_root_directory, 'test_data.csv')
           }