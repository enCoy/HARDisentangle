"""

In this one we will not have pretrained positive and negative representation generators
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

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
        self.train_on_gpu = train_on_gpu

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

def train_one_epoch(loader, modalities, glob_optimizer, local_optimizers, glob_models, local_models,
                    mse_loss, contrastive_loss, mi_estimator, mode):
    # loaders has different modalities inside
    # version without positive and negative representations
    batch_losses = {'total_loss': [],
        'reconstruction_loss': [],
        'c_loss_common_p': [],
        'c_loss_common_n': [],
        'c_loss_unique_p': [],
        'c_loss_unique_n': [],
        'MI_loss': []}
    if mode == 'train':
        # take batches from each modality separately
        for batch_idx, (inputs, targets) in enumerate(loader):
            # load data and split into modalities
            inputs_list = np.split(inputs, len(modalities), axis=-1)
            targets_list = np.split(targets, len(modalities), axis=-1)
            for modality_idx in range(len(modalities)):
                modality = modalities[modality_idx]
                input_modality = torch.squeeze(inputs_list[modality_idx])
                target_modality = torch.squeeze(targets_list[modality_idx])
                # local pass first
                local_optimizers[modality].zero_grad()
                local_loss, negative_repr = local_net_forward_pass(input_modality, target_modality, local_models[modality]['fuse'], mse_loss)
                local_loss.backward(retain_graph=True)
                local_optimizers[modality].step()
                # global network pass
                glob_optimizer.zero_grad()
                total_loss, batch_losses = glob_net_forward_pass(input_modality, target_modality, negative_repr, glob_models['fuse'], glob_models['mine'],
                                      mse_loss, contrastive_loss, mi_estimator, batch_losses)
                total_loss.backward()
                glob_optimizer.step()
    else:  # val or test
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                # load data and split into modalities
                inputs_list = np.split(inputs, len(modalities), axis=-1)
                targets_list = np.split(targets, len(modalities), axis=-1)
                for modality_idx in range(len(modalities)):
                    modality = modalities[modality_idx]
                    input_modality = torch.squeeze(inputs_list[modality_idx])
                    target_modality = torch.squeeze(targets_list[modality_idx])
                    local_loss, negative_repr = local_net_forward_pass(input_modality, target_modality, local_models[modality]['fuse'], mse_loss)
                    total_loss, batch_losses = glob_net_forward_pass(input_modality, target_modality, negative_repr,
                                                                     glob_models['fuse'], glob_models['mine'],
                                                                     mse_loss, contrastive_loss, mi_estimator,
                                                                     batch_losses)
    return batch_losses


def local_net_forward_pass(inputs, targets, fuse_net, mse_loss):
    [_, _, negatives] = np.split(inputs, 3, axis=-1)
    [_, _, target_negatives] = np.split(targets, 3, axis=-1)
    negatives, target_negatives = negatives.cuda(), target_negatives.cuda()

    negative_reconstruct, _, negative_repr = fuse_net(negatives)
    # input shape (N x NumWindows x feature_dim)
    # reconstruction
    reconstruction_loss = mse_loss(negative_reconstruct, target_negatives)
    return reconstruction_loss, negative_repr


def glob_net_forward_pass(inputs, targets, neg_repr, fuse_net, mine,
                 mse_loss, contrastive_loss, mi_estimator, batch_losses):
    [inputs, positives, _] = np.split(inputs, 3, axis=-1)
    [outputs, pos_out, _] = np.split(targets, 3, axis=-1)
    inputs, positives, outputs, pos_out = inputs.cuda(), positives.cuda(), outputs.cuda(), pos_out.cuda()

    # positives are other modalities
    positive_reconstructed, positive_common, positive_unique = fuse_net(positives)
    # anchor signal
    reconstructed, common_representation, unique_representation = fuse_net(inputs)
    # input shape (N x NumWindows x feature_dim)

    # negative_common = neg_repr-positive_common
    # negative_unique = neg_repr-positive_unique

    # for common representation - positives are positives - negatives are negatives
    # for unique representation - vice versa
    # calculate losses
    # reconstruction
    reconstruction_loss = mse_loss(reconstructed, outputs) + mse_loss(positive_reconstructed, pos_out)
    # contrastive
    # we want common and positive_common to be similar
    c_loss_common_p = contrastive_loss(common_representation, positive_common, label=0)
    # we want common and negative_common to be dissimilar -i.e. common info of that window should be dissimilar to another window's common
    c_loss_common_n = contrastive_loss(common_representation, neg_repr, label=1)  # problematic
    # we want unique and positive unique to be dissimilar, i.e. unique info of that window
    c_loss_unique_p = contrastive_loss(unique_representation, positive_unique, label=1)  # problematic
    c_loss_unique_n = contrastive_loss(unique_representation, neg_repr, label=0)
    # mutual info
    # this is how several people sample marginal distribution
    unique_shuffled = torch.index_select(  # shuffles the noise
        unique_representation, 0, torch.randperm(unique_representation.shape[0]).cuda())
    # mi_score = mi_estimator(mine, common_representation, unique_representation, unique_shuffled)
    # source: https://github.com/sungyubkim/MINE-Mutual-Information-Neural-Estimation-/blob/master/MINE.ipynb
    mi_lb, joint, et = mi_estimator(mine, common_representation, unique_representation, unique_shuffled)
    # biased estimate
    MI_loss = -mi_lb

    lambda_reconst = 0.2
    lambda_contrast_c = 1.0
    lambda_contrast_u = 0.1
    lambda_MI = 0.1

    total_loss = lambda_reconst*reconstruction_loss + lambda_contrast_c*(c_loss_common_p + c_loss_common_n) \
                 + lambda_contrast_u*(c_loss_unique_p + c_loss_unique_n) + lambda_MI*MI_loss
    total_loss_to_report = reconstruction_loss + c_loss_common_p + c_loss_common_n + c_loss_unique_p + c_loss_unique_n + MI_loss
    # add batch losses
    batch_losses['total_loss'].append(total_loss_to_report.item())
    batch_losses['reconstruction_loss'].append(reconstruction_loss.detach().item())
    batch_losses['c_loss_common_p'].append(c_loss_common_p.detach().item())
    batch_losses['c_loss_common_n'].append(c_loss_common_n.detach().item())
    batch_losses['c_loss_unique_p'].append(c_loss_unique_p.detach().item())
    batch_losses['c_loss_unique_n'].append(c_loss_unique_n.detach().item())
    batch_losses['MI_loss'].append(MI_loss.detach().item())
    return total_loss, batch_losses