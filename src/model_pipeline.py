import torch
import time
from os.path import join
from config import path_configs, machine_configs
from torch.utils.tensorboard import SummaryWriter
from src.model_architectures import ShallowFFN
from src.munge import read_datasets, create_data_loader

train_df, validation_df, test_df = read_datasets()
train_data_loader = create_data_loader(train_df)
validation_data_loader = create_data_loader(validation_df)
test_data_loader = create_data_loader(test_df)


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
    pass  # TODO(Hirsh)


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
        for batch_idx, (x, y) in enumerate(dl):
            x = x.to(machine_configs['device']).float()
            y = y.to(machine_configs['device']).float()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            losses.append(loss)

        classwise_errors = None  # TODO(hirsh): implement.  Should be a list or vector of length |C|, the # of races in the dataset

        if experimental:
            if is_biased(classwise_errors, epoch):
                # rollback model
                model = read_model(model_path)  # TODO(hirsh): make sure this works

        writer.add_scalar('average_' + name + '_' + 'loss', sum(losses) / len(losses), epoch)
        # writer.add_scalar('worst_classwise_error', max(classwise_errors), epoch)  # TODO(hirsh): make sure this write makes sense
        # writer.add_scalar('bias ratio', max(classwise_errors) / min(classwise_errors), epoch)  # TODO(hirsh): make sure this write makes sense

        write_model(model, model_path)
