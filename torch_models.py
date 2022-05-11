import torch
from torch import nn


def get_1d_conv_net():
    return nn.Sequential(
        nn.BatchNorm1d(1), #TODO Keep this?
        # No Block
        nn.Dropout(p=0.1),
        nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=3),
        nn.ReLU(),
        nn.BatchNorm1d(128),

        # Block 1
        nn.Dropout(p=0.1),
        nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.MaxPool1d(kernel_size=3, stride=3),
        #
        nn.Dropout(p=0.1),
        nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.MaxPool1d(kernel_size=3, stride=3),

        # No Block
        nn.Dropout(p=0.1),
        nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.MaxPool1d(kernel_size=3, stride=3),

        # Block 2
        nn.Dropout(p=0.1),
        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.MaxPool1d(kernel_size=3, stride=3),

        # No Block
        nn.Dropout(p=0.1),
        nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.BatchNorm1d(512),
    )


# For production code, an efficient execution of the conv and maxpool layers should be used, where only the elements dependant on new data are calculated (with padding on the left from memory)

class Orig1d(nn.Module):
    def __init__(self, langs: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_net = get_1d_conv_net()
        self.dense_net = nn.Sequential(
            nn.AdaptiveMaxPool1d(output_size=1),
            nn.Flatten(start_dim=-2, end_dim=-1), # Transforms data according to Linear's input requirements (batch, 512, 1) -> (batch, 512)
            nn.Dropout(p=0.1), #TODO Is this correct?
            nn.Linear(in_features=512, out_features=langs),
        )

    def forward(self, input):
        return self.dense_net(self.conv_net(input))







# For switching the linear layer with the adaptive pooling, a custom transformation layer using torch.movedim(input, (-2, -1, ), (-1, -2, )) has to be used to move the 512-filter dim to the back.
#TODO Add dropout according to Linear layer in Orig1d

class Swapped1dMax(nn.Module):
    def __init__(self, langs: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_net = get_1d_conv_net()
        self.linear = nn.Linear(in_features=512, out_features=langs)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, input):
        conv_res = self.conv_net(input)
        lin_input = torch.movedim(conv_res, -2, -1) # Transforms data according to Linear's input requirements (batch, 512, len_input) -> (batch, len_input, 512)
        lin_res = self.linear(lin_input)
        pool_input = torch.movedim(lin_res, -2, -1) # Transforms data according to Pooling's input requirements (batch, len_input, n_classes) -> (batch, n_classes, len_input)
        return self.pool(pool_input)


class Swapped1dAvg(nn.Module):
    def __init__(self, langs: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_net = get_1d_conv_net()
        self.linear = nn.Linear(in_features=512, out_features=langs)
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, input):
        conv_res = self.conv_net(input)
        lin_input = torch.movedim(conv_res, -2, -1) # Transforms data according to Linear's input requirements (batch, 512, len_input) -> (batch, len_input, 512)
        lin_res = self.linear(lin_input)
        pool_input = torch.movedim(lin_res, -2, -1) # Transforms data according to Pooling's input requirements (batch, len_input, n_classes) -> (batch, n_classes, len_input)
        return self.pool(pool_input)