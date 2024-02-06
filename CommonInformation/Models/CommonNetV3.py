"""

In this one we will not have pretrained positive and negative representation generators
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


def l2_norm(input):  # unit norm conversion
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


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
        return x
        # x = self.fc(x)  # batch_size x embedding
        # return F.relu(x)


class CommonNet(nn.Module):
    def __init__(self, embedding_dim, hidden_1, hidden_2, train_on_gpu=True):
        super(CommonNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(
                            nn.Linear(self.embedding_dim, hidden_1),
                            nn.BatchNorm1d(hidden_1),
                            nn.ReLU(),
                            nn.Linear(hidden_1, hidden_2),
                            nn.BatchNorm1d(hidden_2),
                            nn.ReLU()

        )
        self.train_on_gpu = train_on_gpu


    def forward(self, x):
        return self.fc(x)

class ReconstructNet(nn.Module):
    def __init__(self, embedding_dim, hidden_1, output, train_on_gpu=True):
        super(ReconstructNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(
                                nn.Linear(self.embedding_dim, hidden_1),
                                nn.ReLU(),
                                nn.Linear(hidden_1, output))
        self.train_on_gpu = train_on_gpu


    def forward(self, x):
        return self.fc(x)

class UniqueNet(nn.Module):
    def __init__(self, embedding_dim, hidden_1, hidden_2, train_on_gpu=True):
        super(UniqueNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU()
        )
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
        self.fcx = nn.Sequential(nn.Linear(x_dim, hidden_dim // 4))
        self.fcz = nn.Sequential(nn.Linear(z_dim, hidden_dim // 4))
        self.fc = nn.Linear(hidden_dim // 4, 1)
        self.train_on_gpu = train_on_gpu

    def forward(self, x, z):
        h1 = F.leaky_relu(self.fcx(x) + self.fcz(z))
        h2 = self.fc(h1)
        return h2

def train_one_epoch(loader, modalities, glob_optimizer, local_optimizers_neg, local_optimizers_pos,
                    glob_models, local_models_neg, local_models_pos,
                    mse_loss, contrastive_loss, mi_estimator, mode):
    # loaders has different modalities inside
    # version without positive and negative representations
    batch_losses = {'total_loss': [],
        'reconstruction_loss': [],
        'common_and_positive': [],
        'unique_and_negative': [],
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
                local_optimizers_neg[modality].zero_grad()
                local_optimizers_pos[modality].zero_grad()
                local_loss_neg, negative_repr = local_net_forward_pass(input_modality, target_modality,
                                                                       local_models_neg[modality]['base'], local_models_neg[modality]['unique'],
                                                                       local_models_neg[modality]['reconst'],
                                                                       mse_loss)
                local_loss_pos, positive_repr = local_net_forward_pass(input_modality, target_modality,
                                                                       local_models_pos[modality]['base'], local_models_pos[modality]['common'],
                                                                       local_models_pos[modality]['reconst'],
                                                                       mse_loss)
                local_loss_neg.backward(retain_graph=True)
                local_loss_pos.backward(retain_graph=True)
                local_optimizers_neg[modality].step()
                local_optimizers_pos[modality].step()
                # global network pass
                glob_optimizer.zero_grad()
                total_loss, batch_losses = glob_net_forward_pass(input_modality, target_modality, negative_repr, positive_repr,
                                                                 glob_models['fuse'], glob_models['mine'],
                                                        mse_loss, contrastive_loss, mi_estimator, batch_losses)
                total_loss.backward()
                glob_optimizer.step()
        print("common net")
        for param in glob_models['common'].parameters():
            print(f"Number less threshold {torch.sum(torch.abs(param) < 0.00001)} out of {torch.numel(param)}")
        print("unique net")
        for param in glob_models['unique'].parameters():
            print(f"Number less threshold {torch.sum(torch.abs(param) < 0.00001)} out of {torch.numel(param)}")
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
                    local_loss_neg, negative_repr = local_net_forward_pass(input_modality, target_modality,
                                                                           local_models_neg[modality]['base'],
                                                                           local_models_neg[modality]['unique'],
                                                                           local_models_neg[modality]['reconst'], mse_loss)
                    local_loss_pos, positive_repr = local_net_forward_pass(input_modality, target_modality,
                                                                           local_models_pos[modality]['base'],
                                                                           local_models_pos[modality]['common'],
                                                                           local_models_pos[modality]['reconst'], mse_loss)
                    total_loss, batch_losses = glob_net_forward_pass(input_modality, target_modality, negative_repr, positive_repr,
                                                                     glob_models['fuse'], glob_models['mine'],
                                                                     mse_loss, contrastive_loss, mi_estimator,
                                                                     batch_losses)
    return batch_losses


def local_net_forward_pass(inputs, targets, base_net, encoding_net, reconst_net, mse_loss, net_type='pos'):
    # net_type = 'neg' or 'pos' --- positive or negative samples
    if net_type == 'pos':
        [_, samples, _] = np.split(inputs, 3, axis=-1)
        [_, target_samples, _] = np.split(targets, 3, axis=-1)
    else:  # neg
        [_, _, samples] = np.split(inputs, 3, axis=-1)
        [_, _, target_samples] = np.split(targets, 3, axis=-1)
    samples, target_samples = samples.cuda(), target_samples.cuda()

    base_representation = base_net(samples)  # should be shaped (N x embedding_dim)
    representation = encoding_net(base_representation)  # should be shaped (N x output_size)
    reconstructed = reconst_net(representation)
    # reshape reconstructed
    reconstructed = torch.reshape(reconstructed, (-1, base_net.num_time_steps, base_net.input_dim))
    # input shape (N x NumWindows x feature_dim)
    # reconstruction
    reconstruction_loss = mse_loss(reconstructed, target_samples)
    return reconstruction_loss, representation


def glob_net_forward_pass(inputs, targets, neg_repr, pos_repr, fuse_net, mine,
                 mse_loss, contrastive_loss, mi_estimator, batch_losses):
    [inputs, _, _] = np.split(inputs, 3, axis=-1)
    [outputs, _, _] = np.split(targets, 3, axis=-1)
    inputs, outputs = inputs.cuda(), outputs.cuda()
    # softmax_layer = nn.Softmax(dim=1)

    # anchor signal
    reconstructed, common_representation, unique_representation = fuse_net(inputs)
    # input shape (N x NumWindows x feature_dim)

    # negative_common = neg_repr-positive_common
    # negative_unique = neg_repr-positive_unique
    # neg_repr = neg_repr - positive_common
    reconstruction_loss = mse_loss(reconstructed, outputs)
    # contrastive
    # we want common and positive_common to be similar

    # common_representation_sm = softmax_layer(common_representation)
    # positive_common_sm = softmax_layer(positive_common)
    # neg_repr_sm = softmax_layer(neg_repr - positive_common)
    # unique_representation_sm = softmax_layer(unique_representation)
    neg_repr = neg_repr
    common_representation_sm = l2_norm(common_representation)
    positive_common_sm = l2_norm(pos_repr)
    neg_repr_sm = l2_norm(neg_repr)
    unique_representation_sm = l2_norm(unique_representation)


    commonality_triplet_loss = contrastive_loss(common_representation_sm, positive_common_sm, neg_repr_sm)
    uniqueness_triplet_loss = contrastive_loss(unique_representation_sm, neg_repr_sm, positive_common_sm)

    # commonality_triplet_loss = contrastive_loss(common_representation_sm, positive_common_sm, neg_repr_sm)
    # uniqueness_triplet_loss = contrastive_loss(unique_representation_sm, neg_repr_sm, positive_common_sm)

    # mutual info
    # this is how several people sample marginal distribution
    unique_shuffled = torch.index_select(  # shuffles the noise
        unique_representation, 0, torch.randperm(unique_representation.shape[0]).cuda())
    # mi_score = mi_estimator(mine, common_representation, unique_representation, unique_shuffled)
    # source: https://github.com/sungyubkim/MINE-Mutual-Information-Neural-Estimation-/blob/master/MINE.ipynb
    mi_lb, joint, et = mi_estimator(mine, common_representation, unique_representation, unique_shuffled)
    # biased estimate
    MI_loss = 0

    lambda_reconst = 0.2
    lambda_contrast_c = 2.0
    lambda_MI = 0.1

    # if torch.isnan(MI_loss):
    total_loss = lambda_reconst * reconstruction_loss + lambda_contrast_c * (
                    commonality_triplet_loss + uniqueness_triplet_loss)
    # else:
    #     total_loss = lambda_reconst*reconstruction_loss + lambda_contrast_c*(commonality_triplet_loss + uniqueness_triplet_loss) + lambda_MI*MI_loss
    total_loss_to_report = reconstruction_loss + commonality_triplet_loss + uniqueness_triplet_loss
    # add batch losses
    batch_losses['total_loss'].append(total_loss_to_report.item())
    batch_losses['reconstruction_loss'].append(reconstruction_loss.detach().item())
    batch_losses['common_and_positive'].append(commonality_triplet_loss.detach().item())
    batch_losses['unique_and_negative'].append(uniqueness_triplet_loss.detach().item())
    # batch_losses['MI_loss'].append(MI_loss.detach().item())
    batch_losses['MI_loss'].append(0)
    return total_loss, batch_losses
