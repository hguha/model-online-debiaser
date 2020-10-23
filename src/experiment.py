from src.model_pipeline import train
from config import hyperparameters
import time

hyperparameter_list = [{}]


def format_hyperparameters_to_name(hp):
    vals = '-'.join([str(hp[x]) for x in hp])
    return vals + str(time.time())  # TODO(jeffrey): make not stupid


def try_hp(hp, hyperparameters: dict):
    trial_hp = hyperparameters.copy()
    for key in hp:
        trial_hp = hp[key]
    return trial_hp


for hp in hyperparameter_list:
    name = format_hyperparameters_to_name(hp)
    trial_hp = try_hp(hp, hyperparameters)
    # train(trial_hp, experimental=True)
    train(trial_hp, experimental=False, model_name=name)
