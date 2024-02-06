import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBaseNet(nn.Module):
    def __init__(self, input_dim, output_channels, num_time_steps, output_dim, train_on_gpu=True):
        super(CNNBaseNet, self).__init__()
        self.input_dim = input_dim
        self.num_time_steps = num_time_steps
        self.train_on_gpu = train_on_gpu

        self.cnn = nn.Sequential(
            nn.Conv2d(input_dim, output_channels, kernel_size=(5, 1)),  #reduce dimension by 4 -> 46
            nn.MaxPool2d((2, 1)),  # divide 2 -> 23
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=(5, 1)),  # -> 19
            nn.MaxPool2d((2, 1)),  # 9
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=(3, 1)),  # 7
            nn.ReLU(),
        )

        self.fc = nn.Linear(7 * output_channels, output_dim)

    def forward(self, x):
        x = x.reshape(-1, self.input_dim, self.num_time_steps)  # shape (batch_size, channel, win)
        x = x.unsqueeze(dim=3)  # input size: (batch_size, channel, win, 1)
        #basically image height = num_windows, image width = 1, channels = modalities
        # encoder
        x = self.cnn(x)  # shape (batch_size, channel_size, some_math_here, 1)
        x = x.reshape(x.size(0), -1)  # flatten
        x = self.fc(x)  # batch_size x embedding
        return x


class PopulationEncoder(nn.Module):
    def __init__(self, input_dim, hidden_1, output_dim, train_on_gpu=True):
        super(PopulationEncoder, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Sequential(nn.ReLU(),
                            nn.Linear(self.input_dim, hidden_1),
                              nn.ReLU(),
                                nn.Linear(hidden_1, hidden_1 // 2),
                                nn.ReLU(),
                                nn.Linear(hidden_1 // 2, output_dim))
        self.train_on_gpu = train_on_gpu

    def forward(self, x):
        return self.fc(x)

class PersonalizedEncoder(nn.Module):
    def __init__(self, input_dim, hidden_1, output_dim, train_on_gpu=True):
        super(PersonalizedEncoder, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Sequential(nn.ReLU(),
                                nn.Linear(self.input_dim, hidden_1),
                                nn.ReLU(),
                                nn.Linear(hidden_1, hidden_1 // 2),
                                nn.ReLU(),
                                nn.Linear(hidden_1 // 2, output_dim))
        self.train_on_gpu = train_on_gpu

    def forward(self, x):
        return self.fc(x)

class ReconstructNet(nn.Module):
    def __init__(self, input_dim, hidden_1, output_dim, train_on_gpu=True):
        super(ReconstructNet, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Sequential(nn.ReLU(),
                                nn.Linear(self.input_dim, hidden_1),
                              nn.ReLU(),
                              nn.Linear(hidden_1, hidden_1),
                                nn.ReLU(),
                                nn.Linear(hidden_1, output_dim))
        self.train_on_gpu = train_on_gpu

    def forward(self, x):
        return self.fc(x)