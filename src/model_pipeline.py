import torch
from torch.utils.tensorboard import SummaryWriter
from src.model_architectures import ShallowFFN
import src.munge

#get dataframes
train_df, validate_df, test_df = munge.divide_data_set("data/raw_data.csv")

#turn them into dataloaders maybe
train_data_loader = munge.create_dataloader(train_df)
validation_data_loader = munge.create_dataloader(validate_df)
test_data_loader = munge.create_dataloader(test_df)


def format_hyperparameters_to_name(hyperparameters):
    return str(hyperparameters)  # TODO(jeffrey): make not stupid


def experimental_loss(confusion_matrix, y, y_hat, loss_fn):
    return loss_fn(y, y_hat)  # TODO(jeffrey + hirsh): make not trivial


def is_biased(model):
    return False  # TODO(jeffrey + hirsh): make not trivial


def write_model(model):
    pass  # TODO(jeffrey): make not stupid


def read_model(model):
    pass  # TODO(jeffrey): make not stupid


def train(hyperparameters, experimental: bool = False, test: bool = False):
    writer = SummaryWriter()  # TODO(jeffrey): add path
    model = ShallowFFN(hyperparameters['input_dim'],
                       hyperparameters['hidden_dim'],
                       hyperparameters['output_dim'])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
    criterion = hyperparameters['loss_fn']

    running_confusion_matrix = None

    for epoch in range(hyperparameters['num_epochs']):
        num_batches = len(train_data_loader)

        model.train()
        for batch_idx, (x, y) in enumerate(train_data_loader):
            optimizer.zero_grad()

            y_hat = model(x)

            if experimental:
                loss = experimental_loss(running_confusion_matrix, y, y_hat)
            else:
                loss = criterion(y, y_hat)
            loss.backwards()
            writer.add_scalar('train_loss', batch_idx + epoch * num_batches)

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
            loss = criterion(y, y_hat)
            accuracy = 0  # TODO(jeffrey): implement
            losses.append(loss)

        running_confusion_matrix = None  # TODO(jeffrey): implement

        writer.add_scalar('average_' + name + '_' + 'loss', sum(losses) / len(losses))
        writer.add_scalar('average_' + name + '_' + 'accuracy', sum(accuracies) / len(accuracies))

        if experimental:  # TODO(jeffrey): implement
            if is_biased(model):
                model = read_model()  # rollback