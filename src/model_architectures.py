import torch.nn as nn


class ShallowFFN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ShallowFFN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))