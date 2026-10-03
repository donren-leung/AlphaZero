import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazero.games.GameBase import GameBase

class ResNet(nn.Module):
    def __init__(self,
                 game_type: GameBase,
                 num_resBlocks: int,
                 num_channels: int,
                 head_hidden_size: int,
                 device: torch.device,
                 state_dict: dict | None = None,
                 **kwargs
    ):
        super().__init__()

        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_channels) for _ in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_channels, head_hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(head_hidden_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(head_hidden_size * game_type.row_count * game_type.col_count, game_type.action_size)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_channels, head_hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(head_hidden_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(head_hidden_size * game_type.row_count * game_type.col_count, 1),
            nn.Tanh()
        )

        self.to(device)

        if state_dict is not None:
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x
