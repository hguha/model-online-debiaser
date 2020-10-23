import torch
import time
from os.path import join
from config import path_configs
from torch.utils.tensorboard import SummaryWriter
from src.model_architectures import ShallowFFN
from src.munge import read_datasets, create_data_loader

train_df, validation_df, test_df = read_datasets()
train_data_loader = create_data_loader(train_df)
validation_data_loader = create_data_loader(validation_df)
test_data_loader = create_data_loader(test_df)


def format_hyperparameters_to_name(hyperparameters):
    return str(hyperparameters)  # TODO(jeffrey): make not stupid


def experimental_loss(confusion_matrix, y_hat, y, loss_fn):
    breakpoint()
    return loss_fn(y_hat, y)  # TODO(jeffrey + hirsh): make not trivial


def is_biased(model):
    return False  # TODO(jeffrey + hirsh): make not trivial


def write_model(model):
    pass  # TODO(jeffrey): make not stupid


def read_model(model):
    pass  # TODO(jeffrey): make not stupid


def train(hyperparameters, experimental: bool = False, test: bool = False):

    model_name = format_hyperparameters_to_name(hyperparameters)
    model_path = join(path_configs['root_directory'], 'models', model_name + '.model')

    writer = SummaryWriter(model_name + time.time())  # TODO(jeffrey): add path
    model = ShallowFFN(hyperparameters['input_dim'],
                       hyperparameters['hidden_dim'],
                       hyperparameters['output_dim'])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
    criterion = hyperparameters['base_loss_fn']()

    running_confusion_matrix = None

    for epoch in range(hyperparameters['num_epochs']):
        num_batches = len(train_data_loader)

        model.train()
        for batch_idx, (x, y) in enumerate(train_data_loader):
            optimizer.zero_grad()

            y_hat = model(x)

            if experimental:
                loss = experimental_loss(running_confusion_matrix, y_hat, y, criterion)
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
        accuracies = []
        for batch_idx, (x, y) in enumerate(dl):
            y_hat = model(x)
            loss = criterion(y_hat, y)

            batch_size = y_hat.size()[0]
            best_guess = y_hat.argmax(dim=1)
            accuracy = float((best_guess.eq(y)).sum()) / batch_size

            losses.append(loss)
            accuracies.append(accuracy)

        running_confusion_matrix = None  # TODO(jeffrey): implement

        writer.add_scalar('average_' + name + '_' + 'loss', sum(losses) / len(losses), epoch)
        writer.add_scalar('average_' + name + '_' + 'accuracy', sum(accuracies) / len(accuracies), epoch)

        torch.save(model.state_dict(), model_path)

        # if experimental:  # TODO(jeffrey): implement
        #     if is_biased(model):
        #         model = read_model()  # rollback
