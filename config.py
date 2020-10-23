import os
from os.path import join, dirname
import torch

_root_directory = dirname(__file__)

path_configs = {'root_directory': _root_directory,
                'raw_data_path': join(_root_directory, 'data', 'raw_data.csv'),
                'small_data_path': join(_root_directory, 'data', 'small_data.csv'),
                'train_data_path': join(_root_directory, 'data', 'train_data.csv'),
                'validation_data_path': join(_root_directory, 'data', 'validation_data.csv'),
                'test_data_path': join(_root_directory, 'data', 'test_data.csv')
                }

hyperparameters = {'batch_size': 20,
                   'input_dim': 19,
                   'hidden_dim': 100,  # default
                   'learning_rate': 10 ** (-5),  # default
                   'num_epochs': 1000,  # default
                   'output_dim': 3,
                   'base_loss_fn': torch.nn.CrossEntropyLoss}

if torch.cuda.is_available():
    torch.cuda.device(0)
    _device = 0
else:
    _device = 'cpu'

machine_configs = {'device': _device}


def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


safe_mkdir(join(path_configs['root_directory'], 'data'))
safe_mkdir(join(path_configs['root_directory'], 'runs'))
safe_mkdir(join(path_configs['root_directory'], 'models'))
