import torch.nn as nn

class CNNBaseNet(nn.Module):
    def __init__(self, input_dim, output_channels, num_time_steps, embedding, train_on_gpu = True):
        super(CNNBaseNet, self).__init__()
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.num_time_steps = num_time_steps
        self.train_on_gpu = True
        self.embedding = embedding

        self.cnn = nn.Sequential(
            nn.Conv2d(input_dim, self.output_channels, kernel_size=(5, 1)),  #reduce dimension by 4 -> 46
            nn.MaxPool2d((2, 1)),  # divide 2 -> 23
            nn.ReLU(),
            nn.Conv2d(self.output_channels, self.output_channels, kernel_size=(5, 1)),  # -> 19
            nn.MaxPool2d((2, 1)),  # 9
            nn.ReLU(),
            nn.Conv2d(self.output_channels, self.output_channels, kernel_size=(3, 1)),  # 7
            nn.ReLU(),
        )

        self.fc = nn.Linear(7 * self.output_channels, self.embedding)

    def forward(self, x):
        x = x.view(-1, self.input_dim, self.num_time_steps)  # shape (batch_size, channel, win, 1)
        x = x.unsqueeze(dim=3)  # input size: (batch_size, channel, win, 1)
        #basically image height = num_windows, image width = 1, channels = modalities
        x = self.cnn(x)  # shape (batch_size, channel_size, some_math_here, 1)
        x = x.reshape(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x
        # return x