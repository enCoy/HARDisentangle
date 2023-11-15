import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNBaseNet(nn.Module):
    def __init__(self, input_dim, output_channels, num_time_steps, embedding, train_on_gpu=True):
        super(CNNBaseNet, self).__init__()
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.num_time_steps = num_time_steps
        self.train_on_gpu = train_on_gpu
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
        x = x.reshape(-1, self.input_dim, self.num_time_steps)  # shape (batch_size, channel, win)
        x = x.unsqueeze(dim=3)  # input size: (batch_size, channel, win, 1)
        #basically image height = num_windows, image width = 1, channels = modalities
        # encoder
        x = self.cnn(x)  # shape (batch_size, channel_size, some_math_here, 1)
        x = x.reshape(x.size(0), -1)  # flatten
        x = self.fc(x)  # batch_size x embedding
        return x


class CommonNet(nn.Module):
    def __init__(self, embedding_dim, hidden_1, hidden_2, train_on_gpu=True):
        super(CommonNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(nn.ReLU(),
                            nn.Linear(self.embedding_dim, hidden_1),
                              nn.ReLU(),
                                nn.Linear(hidden_1, hidden_1 // 2),
                                nn.ReLU(),
                                nn.Linear(hidden_1 // 2, hidden_2))
        self.train_on_gpu = train_on_gpu


    def forward(self, x):
        return self.fc(x)

class ReconstructNet(nn.Module):
    def __init__(self, embedding_dim, hidden_1, output, train_on_gpu=True):
        super(ReconstructNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(nn.ReLU(),
                                nn.Linear(self.embedding_dim, hidden_1),
                              nn.ReLU(),
                              nn.Linear(hidden_1, hidden_1),
                                nn.ReLU(),
                                nn.Linear(hidden_1, output))
        self.train_on_gpu = train_on_gpu


    def forward(self, x):
        return self.fc(x)

class UniqueNet(nn.Module):
    def __init__(self, embedding_dim, hidden_1, hidden_2, train_on_gpu=True):
        super(UniqueNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(nn.ReLU(),
                                nn.Linear(self.embedding_dim, hidden_1),
                                nn.ReLU(),
                                nn.Linear(hidden_1, hidden_1 // 2),
                                nn.ReLU(),
                                nn.Linear(hidden_1 // 2, hidden_2))
        self.train_on_gpu = train_on_gpu


    def forward(self, x):
        return self.fc(x)


class FuseNet(nn.Module):
    def __init__(self, base_net, common_net, unique_net, reconstruct_net, feature_dim, window_num=50,train_on_gpu=True):
        super(FuseNet, self).__init__()
        self.base_net = base_net
        self.common_net = common_net
        self.unique_net = unique_net
        self.reconstruct_net = reconstruct_net
        self.window_num = window_num
        self.feature_dim = feature_dim

    def forward(self, x):
        # input shape (N x NumWindows x feature_dim)
        base_representation = self.base_net(x) # should be shaped (N x embedding_dim)
        common_representation = self.common_net(base_representation)  # should be shaped (N x output_size)
        unique_representation = self.unique_net(base_representation)  # should be shaped (N x output_size)
        # concatenate these two
        concatted = torch.concat((common_representation, unique_representation), 1)  # shaped (N x (2output_size))
        reconstructed = self.reconstruct_net(concatted)
        # reshape reconstructed
        reconstructed = torch.reshape(reconstructed, (-1, self.window_num, self.feature_dim))
        return reconstructed, common_representation, unique_representation

class Mine(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim, train_on_gpu = True):
        super(Mine, self).__init__()
        self.fcx = nn.Sequential(nn.Linear(x_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim // 4),)
        self.fcz = nn.Sequential(nn.Linear(z_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim // 4))
        self.fc = nn.Linear(hidden_dim // 4, 1)
        self.train_on_gpu = train_on_gpu

    def forward(self, x, z):
        h1 = F.leaky_relu(self.fcx(x) + self.fcz(z))
        h2 = self.fc(h1)
        return h2



def train_one_epoch(loader, optimizer, fuse_net, positive_r_model, negative_r_model, mine,
                    mse_loss, contrastive_loss, mi_estimator, mode):
    batch_losses = {'total_loss': [],
        'reconstruction_loss': [],
        'c_loss_common_p': [],
        'c_loss_common_n': [],
        'c_loss_unique_p': [],
        'c_loss_unique_n': [],
        'MI_loss': []}
    ma_et = 1
    if mode == 'train':
        for batch_idx, (inputs, targets) in enumerate(loader):
            optimizer.zero_grad()
            total_loss, batch_losses, ma_et = forward_pass(inputs, targets, fuse_net, positive_r_model, negative_r_model, mine,
                 mse_loss, contrastive_loss, mi_estimator, batch_losses, ma_et)
            total_loss.backward()
            optimizer.step()
    else:  # val or test
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                total_loss, batch_losses, ma_et = forward_pass(inputs, targets, fuse_net, positive_r_model, negative_r_model,
                                                        mine,
                                                        mse_loss, contrastive_loss, mi_estimator, batch_losses, ma_et)
    return batch_losses

def forward_pass(inputs, targets, fuse_net, positive_r_model, negative_r_model, mine,
                 mse_loss, contrastive_loss, mi_estimator, batch_losses, ma_et):
    inputs, targets = inputs.cuda(), targets.cuda()
    reconstructed, common_representation, unique_representation = fuse_net(inputs)
    # input shape (N x NumWindows x feature_dim)
    positive_representation = positive_r_model(inputs)  # should be shaped (N x embedding_dim)
    negative_representation = negative_r_model(inputs)  # should be shaped (N x embedding_dim)
    negative_representation = -positive_representation + negative_representation
    # for common representation - positives are positives - negatives are negatives
    # for unique representation - vice versa
    # calculate losses
    # reconstruction
    reconstruction_loss = mse_loss(reconstructed, targets)
    # contrastive
    c_loss_common_p = contrastive_loss(common_representation, positive_representation, label=0)
    c_loss_common_n = contrastive_loss(common_representation, negative_representation, label=1)  # problematic
    c_loss_unique_p = contrastive_loss(unique_representation, positive_representation, label=1)  # problematic
    c_loss_unique_n = contrastive_loss(unique_representation, negative_representation, label=0)
    # mutual info
    # this is how several people sample marginal distribution
    unique_shuffled = torch.index_select(  # shuffles the noise
        unique_representation, 0, torch.randperm(unique_representation.shape[0]).cuda())
    # mi_score = mi_estimator(mine, common_representation, unique_representation, unique_shuffled)
    # source: https://github.com/sungyubkim/MINE-Mutual-Information-Neural-Estimation-/blob/master/MINE.ipynb
    ma_rate = 0.05
    mi_lb, joint, et = mi_estimator(mine, common_representation, unique_representation, unique_shuffled)
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)
    # unbiasing use moving average
    MI_loss = -(torch.mean(joint) - (1 / ma_et.mean()).detach() * torch.mean(et))

    lambda_reconst = 1.0
    lambda_contrast = 1.0
    lambda_MI = 0.5

    total_loss = lambda_reconst*reconstruction_loss + lambda_contrast*(c_loss_common_p + c_loss_common_n \
                 + c_loss_unique_p + c_loss_unique_n) + lambda_MI*MI_loss
    # add batch losses
    batch_losses['total_loss'].append(total_loss.item())
    batch_losses['reconstruction_loss'].append(reconstruction_loss.detach().item())
    batch_losses['c_loss_common_p'].append(c_loss_common_p.detach().item())
    batch_losses['c_loss_common_n'].append(c_loss_common_n.detach().item())
    batch_losses['c_loss_unique_p'].append(c_loss_unique_p.detach().item())
    batch_losses['c_loss_unique_n'].append(c_loss_unique_n.detach().item())
    batch_losses['MI_loss'].append(MI_loss.detach().item())
    return total_loss, batch_losses, ma_et


