import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, num_hidden: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def residual(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(x))

    def forward(self, x):
        return F.relu(self.residual(x) + x)

class SE_ResBlock(ResBlock):
    def __init__(self, num_hidden: int, reduction: int=8):
        super().__init__(num_hidden)
        bottleneck = max(1, num_hidden // reduction)
        self.excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_hidden, bottleneck, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(bottleneck, num_hidden, kernel_size=1),
            nn.Sigmoid(),
        )

    def residual(self, x):
        features = super().residual(x)
        return features * self.excitation(features)

class ResNet(nn.Module):
    block_type = ResBlock

    def __init__(self,
                 game_input_channels: int,
                 game_spatial_size: int,
                 game_action_size: int,
                 num_resBlocks: int,
                 num_channels: int,
                 head_hidden_size: int,
                 state_dict: dict | None = None,
                 **kwargs
    ):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(game_input_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [self.block_type(num_channels) for _ in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_channels, head_hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(head_hidden_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(head_hidden_size * game_spatial_size, game_action_size)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_channels, head_hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(head_hidden_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(head_hidden_size * game_spatial_size, 1),
            nn.Tanh()
        )

        if state_dict is not None:
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value

class SE_ResNet(ResNet):
    block_type = SE_ResBlock

class FileRankPlaneFlatten(nn.Module):
    def forward(self, x):
        # (B, planes, rank, file)
        # -> (B, file, rank, planes)
        return x.permute(0, 3, 2, 1).reshape(x.shape[0], -1)

class SlimPolicy(ResNet):
    def __init__(self,
                 game_input_channels: int,
                 game_spatial_size: int,
                 game_action_size: int,
                 num_resBlocks: int,
                 num_channels: int,
                 head_hidden_size: int,
                 state_dict: dict | None = None,
                 **kwargs
    ):
        super().__init__(
            game_input_channels=game_input_channels,
            game_spatial_size=game_spatial_size,
            game_action_size=game_action_size,
            num_resBlocks=num_resBlocks,
            num_channels=num_channels,
            head_hidden_size=head_hidden_size,
            state_dict=None,
            **kwargs
        )

        game_type = kwargs["game_type_str"]
        if game_type in ('chess', 'tictactoe'):
            # Games with spatial action spaces can use 1x1 conv to reduce channels
            # before flattening, dramatically reducing params.
            assert game_action_size % game_spatial_size == 0

            flatten = FileRankPlaneFlatten() if game_type == 'chess' else nn.Flatten()

            self.policyHead = nn.Sequential(
                nn.Conv2d(num_channels, head_hidden_size, kernel_size=1),
                nn.BatchNorm2d(head_hidden_size),
                nn.ReLU(),
                nn.Conv2d(head_hidden_size, game_action_size // game_spatial_size, kernel_size=1),
                flatten
            )

        if state_dict is not None:
            self.load_state_dict(state_dict)

class SlimSE_ResNet(SlimPolicy, SE_ResNet):
    pass
