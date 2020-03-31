import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
#         self.convnet = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=5, stride=2),
#             nn.BatchNorm2d(16),
#             nn.Conv2d(16, 32, kernel_size=5, stride=2),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 32, kernel_size=5, stride=2),
#             nn.BatchNorm2d(32),
#         )
        self.fcnet = nn.Sequential(
            nn.Linear(state_size , 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )
        # self.fc1 = nn.Linear(state_size, fc1_units)
        # self.fc2 = nn.Linear(fc1_units, fc2_units)
        # self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        return self.fcnet(state)

