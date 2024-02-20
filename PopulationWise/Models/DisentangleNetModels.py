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
            nn.Conv2d(input_dim, output_channels, kernel_size=(3, 1)),  #reduce dimension by 2 -> 48
            nn.MaxPool2d((2, 1)),  # divide 2 -> 24
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=(3, 1)),  # -> 22
            nn.MaxPool2d((2, 1)),  # 9
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=(3, 1)),  # 9
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(9 * output_channels, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

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
        self.fc = nn.Sequential(
                            nn.Linear(input_dim, output_dim)
                                )
        self.train_on_gpu = train_on_gpu

    def forward(self, x):
        return self.fc(x)

class PersonalizedEncoder(nn.Module):
    def __init__(self, input_dim, hidden_1, output_dim, train_on_gpu=True):
        super(PersonalizedEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )
        self.train_on_gpu = train_on_gpu

    def forward(self, x):
        return self.fc(x)

class ReconstructNet(nn.Module):
    def __init__(self, input_dim, hidden_1, output_dim, train_on_gpu=True):
        super(ReconstructNet, self).__init__()
        self.fc = nn.Sequential(nn.ReLU(),
                                nn.Linear(input_dim, output_dim)
                                )
        self.train_on_gpu = train_on_gpu

    def forward(self, x):
        return self.fc(x)


class Mine(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim, train_on_gpu = True):
        super(Mine, self).__init__()
        self.fcx = nn.Sequential(nn.Linear(x_dim, hidden_dim // 4))
        self.fcz = nn.Sequential(nn.Linear(z_dim, hidden_dim // 4))
        self.fc = nn.Linear(hidden_dim // 4, 1)
        self.train_on_gpu = train_on_gpu

    def forward(self, x, z):
        h1 = F.leaky_relu(self.fcx(x) + self.fcz(z))
        h2 = self.fc(h1)
        return h2