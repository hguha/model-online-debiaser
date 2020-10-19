from src.model_pipeline import train

hyperparameter_list = []

for hp in hyperparameter_list:
    train(hp, experimental=True)
    train(hp, experimental=False)
