import torch
import time
import random
from os.path import join
from config import path_configs, machine_configs
from torch.utils.tensorboard import SummaryWriter
from src.model_architectures import ShallowFFN
from src.munge import read_datasets, create_data_loader, RecidivismDataset, get_categories, change_sample

train_df, validation_df, test_df = read_datasets()
train_data_loader = create_data_loader(train_df)
validation_data_loader = create_data_loader(validation_df)
test_data_loader = create_data_loader(test_df)
categories = get_categories()


# TODO(hirsh): make sure this works
def experimental_loss(classwise_errors, x, y_hat, y, loss_fn):
    # get race for every example in the batch
    races = []  # apply invert fn to x
    losses = [classwise_errors[races[i]] * loss_fn(y_hat[i], y[i]) for i in range(x.size()[0])]
    return sum(losses) / len(losses)


def is_biased(classwise_errors, epoch):
    return max(classwise_errors) / min(classwise_errors) > 1 + 1 / epoch


def write_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def read_model(model_path):
    model = torch.load(model_path)
    return model.eval()


def train(hyperparameters, experimental: bool = False, test: bool = False,
          model_name=''):

    model_path = join(path_configs['root_directory'], 'models', model_name + '.model')
    writer_path = join(path_configs['root_directory'], 'runs', model_name + str(time.time()))
    writer = SummaryWriter(writer_path)
    model = ShallowFFN(hyperparameters['input_dim'],
                       hyperparameters['hidden_dim'],
                       hyperparameters['output_dim'])
    model = model.to(machine_configs['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
    criterion = hyperparameters['base_loss_fn']()

    classwise_errors = []

    for epoch in range(hyperparameters['num_epochs']):
        num_batches = len(train_data_loader)

        model.train()
        for batch_idx, (x, y) in enumerate(train_data_loader):
            x = x.to(machine_configs['device']).float()
            y = y.to(machine_configs['device']).float()
            optimizer.zero_grad()
            y_hat = model(x)

            if experimental:
                loss = experimental_loss(classwise_errors, y_hat, y, criterion)
            else:
                loss = criterion(y_hat, y)
            loss.backward()
            writer.add_scalar('train_loss', loss, batch_idx + epoch * num_batches)

            optimizer.step()

        model.eval()

        if test:
            dl = test_data_loader
            name = 'test'
        else:
            dl = validation_data_loader
            name = 'validation'

        losses = []

        classwise_errors = {race: [] for race in categories['race']}
        score_text = { "High": 0, "Medium": 0, "Low":0 }
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
                example_loss = criterion(y_hat[idx], y[idx])
                y_hat_val = float(y_hat[idx])
                #using the same ranges as COMPAS to compare
                if(0 <= y_hat_val <= 0.4): score_text["Low"]+=1
                elif(0.4 < y_hat_val <= 0.7): score_text["Medium"]+=1
                elif(0.7 < y_hat_val <= 1): score_text["High"]+=1
                else: print(y_hat_val)
                
                classwise_errors[race].append(example_loss)
        
        df_grouped = dl.dataset.invert_tensor_to_df().groupby("score_text").count()["sex"].tolist()
        compas_score_text = {"High": df_grouped[0], "Low": df_grouped[1], "Medium": df_grouped[2]}
        avg_loss = sum(losses) / len(losses)

        avg_classwise_errors = {race: sum(classwise_errors[race]) / len(classwise_errors[race]) for race in classwise_errors}

        if experimental:
            if is_biased(avg_classwise_errors.values(), epoch):
                # rollback model
                model = read_model(model_path)

        # breakpoint()
        writer.add_scalar('average_' + name + '_' + 'loss', avg_loss, epoch)
        writer.add_scalar('worst_classwise_error', max(avg_classwise_errors.values()), epoch)
        writer.add_scalar('worst_bias_ratio', max(avg_classwise_errors.values()) / min(avg_classwise_errors.values()), epoch)
        writer.add_scalar('black_to_write_bias_ratio', avg_classwise_errors['African-American'] / avg_classwise_errors['Caucasian'], epoch)
        writer.add_scalar('error on African-Americans - ' + ("CHANGED" if change_sample["use"] else "UNCHANGED"), avg_classwise_errors['African-American'], epoch)
        writer.add_scalar('error on Caucasians - ' + ("CHANGED" if change_sample["use"] else "UNCHANGED"), avg_classwise_errors['Caucasian'], epoch)
        writer.add_scalar('avg classwise errors', sum(avg_classwise_errors.values())/len(avg_classwise_errors.values()), epoch)
        writer.add_scalar('compas_diff_high', abs(score_text["High"] - compas_score_text["High"]), epoch)
        writer.add_scalar('compas_diff_med', abs(score_text["Medium"] - compas_score_text["Medium"]), epoch)
        writer.add_scalar('compas_diff_low', abs(score_text["Low"] - compas_score_text["Low"]), epoch)
        write_model(model, model_path)
        print(epoch)
