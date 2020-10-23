from src.model_pipeline import train
from config import hyperparameters

hyperparameter_list = [{}]


def try_hp(hp, hyperparameters: dict):
    trial_hp = hyperparameters.copy()
    for key in hp:
        trial_hp = hp[key]
    return trial_hp


for hp in hyperparameter_list:
    trial_hp = try_hp(hp, hyperparameters)
    # train(trial_hp, experimental=True)
    train(trial_hp, experimental=False)
