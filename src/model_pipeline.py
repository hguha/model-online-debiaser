import torch
import time
import random
import copy
from os.path import join
from config import path_configs, machine_configs, hyperparameters
from torch.utils.tensorboard import SummaryWriter
from src.model_architectures import ShallowFFN
from src.munge import read_datasets, create_data_loader, RecidivismDataset, get_categories, change_sample

train_df, validation_df, test_df = read_datasets()
train_data_loader = create_data_loader(train_df)
validation_data_loader = create_data_loader(validation_df)
test_data_loader = create_data_loader(test_df)
categories = get_categories()


# def experimental_loss(classwise_errors, x, y_hat, y, loss_fn):
#     # get race for every example in the batch
#     races = []  # apply invert fn to x
#     losses = [classwise_errors[races[i]] * loss_fn(y_hat[i], y[i]) for i in range(x.size()[0])]
#     return sum(losses) / len(losses)


def is_biased(classwise_accuracies):
    return max(classwise_accuracies) - min(classwise_accuracies) < hyperparameters['epsilon']


def write_model(model, model_path):
    torch.save(model.state_dict(), model_path)


#
#
# def read_model(model_path):
#     model = torch.load(model_path)
#     return model.eval()


def train(hyperparameters, experimental: bool = False, test: bool = False,
          model_name=''):
    model_path = join(path_configs['root_directory'], 'models', model_name + '.model')
    writer_path = join(path_configs['root_directory'], 'runs', model_name + str(time.time()))
    writer = SummaryWriter(writer_path)
    running_model = ShallowFFN(hyperparameters['input_dim'],
                               hyperparameters['hidden_dim'],
                               hyperparameters['output_dim'])
    running_model = running_model.to(machine_configs['device'])
    optimizer = torch.optim.Adam(running_model.parameters(), lr=hyperparameters['learning_rate'])
    criterion = hyperparameters['base_loss_fn']()

    safety_model = copy.deepcopy(running_model)
    unique_models_validated = 0

    for epoch in range(hyperparameters['num_epochs']):
        num_batches = len(train_data_loader)

        running_model.train()
        for batch_idx, (x, y) in enumerate(train_data_loader):
            x = x.to(machine_configs['device']).float()
            y = y.to(machine_configs['device']).float()
            optimizer.zero_grad()
            y_hat = running_model(x)

            loss = criterion(y_hat, y)
            loss.backward()
            writer.add_scalar('train_loss', loss, batch_idx + epoch * num_batches)

            optimizer.step()

        def run_validation(model, write: bool):
            model.eval()

            if test:
                dl = test_data_loader
                name = 'Test'
            else:
                dl = validation_data_loader
                name = 'Validation'

            losses = []

            classwise_accuracies = {race: [] for race in categories['race']}
            score_text = {"High": 0, "Medium": 0, "Low": 0}
            for batch_idx, (x, y) in enumerate(dl):
                x = x.to(machine_configs['device']).float()
                y = y.to(machine_configs['device']).float()
                y_hat = model(x)
                loss = criterion(y_hat, y)
                losses.append(loss)

                separate_tensor_examples = torch.unbind(x, dim=0)
                for idx, example in enumerate(separate_tensor_examples):
                    assert isinstance(dl.dataset, RecidivismDataset)
                    reconstruction = dl.dataset.invert_tensor_to_row(example.unsqueeze(dim=0))
                    race = reconstruction['race']
                    # example_loss = criterion(y_hat[idx], y[idx])  # why tf was I using BCE on the validation set?
                    is_correct = int(torch.round(y_hat[idx]) == y[idx])
                    y_hat_val = float(y_hat[idx])

                    # using the same ranges as COMPAS to compare
                    if 0 <= y_hat_val <= 0.4:
                        score_text["Low"] += 1
                    elif 0.4 < y_hat_val <= 0.7:
                        score_text["Medium"] += 1
                    elif 0.7 < y_hat_val <= 1:
                        score_text["High"] += 1
                    else:
                        print(y_hat_val)

                    classwise_accuracies[race].append(is_correct)

            df_grouped = dl.dataset.invert_tensor_to_df().groupby("score_text").count()["sex"].tolist()
            compas_score_text = {"High": df_grouped[0], "Low": df_grouped[1], "Medium": df_grouped[2]}

            avg_classwise_accuracies = {race: sum(classwise_accuracies[race]) / len(classwise_accuracies[race]) for race
                                        in classwise_accuracies}
            overall_accuracy_num = sum([sum(classwise_accuracies[race]) for race in classwise_accuracies])
            overall_accuracy_denom = sum([len(classwise_accuracies[race]) for race in classwise_accuracies])
            overall_accuracy = overall_accuracy_num / overall_accuracy_denom

            if write:
                writer.add_scalar('Average ' + name + 'Accuracy', overall_accuracy, epoch)
                writer.add_scalar('Worst Classwise Accuracy', min(avg_classwise_accuracies.values()), epoch)
                writer.add_scalar('Worst Classwise Accuracy Difference',
                                  max(avg_classwise_accuracies.values()) - min(avg_classwise_accuracies.values()),
                                  epoch)
                writer.add_scalar('Caucasian Accuracy Minus African-American Accuracy',
                                  avg_classwise_accuracies['Caucasian'] - avg_classwise_accuracies['African-American'],
                                  epoch)
                writer.add_scalar('African-American Accuracy' + ("CHANGED" if change_sample["use"] else ""),
                                  avg_classwise_accuracies['African-American'], epoch)
                writer.add_scalar('Caucasian Accuracy' + ("CHANGED" if change_sample["use"] else ""),
                                  avg_classwise_accuracies['Caucasian'], epoch)
                # writer.add_scalar('avg classwise errors',
                #                   sum(avg_classwise_errors.values()) / len(avg_classwise_errors.values()), epoch)
                writer.add_scalar('compas_diff_high', abs(score_text["High"] - compas_score_text["High"]), epoch)
                writer.add_scalar('compas_diff_med', abs(score_text["Medium"] - compas_score_text["Medium"]), epoch)
                writer.add_scalar('compas_diff_low', abs(score_text["Low"] - compas_score_text["Low"]), epoch)
                writer.add_scalars('All Accuracies', avg_classwise_accuracies, epoch)
                write_model(model, model_path)

            return avg_classwise_accuracies

        if experimental:
            avg_running_classwise_errors = run_validation(running_model, False)
            _ = run_validation(safety_model)
            if is_biased(avg_running_classwise_errors):
                # do not update safety model
                # do not update # models validated
                pass
            else:
                # update safety model
                safety_model = copy.deepcopy(running_model)
                unique_models_validated += 1
            writer.add_scalar('Unique Models Validated', unique_models_validated, epoch)
        else:
            run_validation(running_model, True)
