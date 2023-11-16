"""

In this one we will not have pretrained positive and negative representation generators
Furthermore, for negative representations, we will have separate models
"""


import numpy as np
import os
from DataProcesses import DataProcessor
from Utils.HelperFunctions import  convert_to_torch_v2, plot_loss_curves
from CommonInformation.Models.CommonNetV3 import CNNBaseNet
from CommonInformation.Models.CommonNetV3 import train_one_epoch
from Models.CommonNetV3 import FuseNet, CommonNet, UniqueNet, ReconstructNet, Mine
import warnings
from time import localtime, strftime
import torch
import torch.utils.data as Data
import torch.nn as nn
from Utils.HelperFunctions import contrastive_loss_criterion, mi_estimator, mutual_information

warnings.filterwarnings('ignore')
machine = 'windows'
if machine == 'linux':
    BASE_DIR = r'/home/cmyldz/Dropbox (GaTech)/DisentangledHAR'
else:
    BASE_DIR = r'C:\Users\Cem Okan\Dropbox (GaTech)\DisentangledHAR'

def get_parameters(data_name):
    parameters_dict = {}
    parameters_dict['window_size'] = 50  # 50 hz = 1 sec
    parameters_dict['sampling_rate'] = 50 # hz
    parameters_dict['sliding_window_overlap_ratio'] = 0.5
    parameters_dict['num_epochs'] = 2
    parameters_dict['batch_size'] = 256
    parameters_dict['lr'] = 0.00003
    parameters_dict['embedding_dim'] = 128
    parameters_dict['weight_decay'] = 1e-6
    parameters_dict['use_bidirectional'] = False
    parameters_dict['num_lstm_layers'] = 2
    parameters_dict['generator_init_seed'] = 16
    parameters_dict['common_unique_shared_seed'] = 16
    if data_name == 'pamap2':
        parameters_dict['data_dir'] = os.path.join(BASE_DIR, r"PAMAP2_Dataset/PAMAP2_Dataset/Processed50Hz")
        parameters_dict['num_modalities'] = 52  # number of sensor channels
        parameters_dict['num_activities'] = 12
        parameters_dict['num_subjects'] = 8
    elif data_name == 'real':
        parameters_dict['data_dir'] = os.path.join(BASE_DIR, r"realworld2016_dataset/Processed")
        parameters_dict['num_modalities'] = 42  # number of sensor channels
        parameters_dict['num_activities'] = 8
        parameters_dict['num_subjects'] = 15
    else:
        print("Error! Data does not exist!!")
    return parameters_dict

def get_models(parameters_dict, disentangle_reconst=True):
    models = {}
    torch.manual_seed(parameters_dict['common_unique_shared_seed'])
    base_net = CNNBaseNet(input_dim=feature_dim, output_channels=128, embedding=1024,
                          num_time_steps=parameters_dict['window_size'])
    # initialize with the same seed to ensure that they start from the same place
    torch.manual_seed(parameters_dict['common_unique_shared_seed'])
    unique_net = UniqueNet(1024, 256, parameters_dict['embedding_dim'])
    torch.manual_seed(parameters_dict['common_unique_shared_seed'])
    common_net = CommonNet(1024, 256, parameters_dict['embedding_dim'])
    torch.manual_seed(parameters_dict['common_unique_shared_seed'])
    if disentangle_reconst:
        reconstruct_net = ReconstructNet(parameters_dict['embedding_dim'] * 2, 256,
                                         parameters_dict['window_size'] * feature_dim)
    else:
        reconstruct_net = ReconstructNet(parameters_dict['embedding_dim'], 256,
                                         parameters_dict['window_size'] * feature_dim)
    fuse_net = FuseNet(base_net, common_net, unique_net,
                       reconstruct_net, feature_dim,
                       window_num=parameters_dict['window_size'])
    mine = Mine(x_dim=parameters_dict['embedding_dim'], z_dim=parameters_dict['embedding_dim'],
                hidden_dim=parameters_dict['embedding_dim'] // 2)
    models['base'] = base_net
    models['common'] = common_net
    models['unique'] = unique_net
    models['reconst'] = reconstruct_net
    models['fuse'] = fuse_net
    models['mine'] = mine
    return models

def enable_mode(model_dict, mode='train'):
    keys = list(model_dict.keys())
    if mode == 'train':
        for key in keys:
            model_dict[key].train()
    else:
        for key in keys:
            model_dict[key].eval()
    return model_dict

def save_models(glob_models, local_models_neg, local_models_pos, save_dir):
    # save glob model
    glob_model_keys = list(glob_models.keys())
    for key in glob_model_keys:
        model = glob_models[key]
        # save state dict
        torch.save(model.state_dict(), os.path.join(save_dir, f"glob_{key}_stateDict.pth"))
        # save model
        torch.save(model, os.path.join(save_dir, f"glob_{key}.pth"))
    # save local models - first negative
    for modality in modalities:
        local_model_keys = list(local_models_neg[modality].keys())
        for key in local_model_keys:
            model = local_models_neg[modality][key]
            # save state dict
            torch.save(model.state_dict(), os.path.join(save_dir, f"{modality}_{key}_neg_stateDict.pth"))
            # save model
            torch.save(model, os.path.join(save_dir, f"{modality}_{key}.pth"))
    for modality in modalities:
        local_model_keys = list(local_models_pos[modality].keys())
        for key in local_model_keys:
            model = local_models_pos[modality][key]
            # save state dict
            torch.save(model.state_dict(), os.path.join(save_dir, f"{modality}_{key}_pos_stateDict.pth"))
            # save model
            torch.save(model, os.path.join(save_dir, f"{modality}_{key}.pth"))


if __name__ == "__main__":

    data_name = 'pamap2'  # 'pamap2' or 'real'
    target_subject = 3
    if data_name == 'pamap2':
        feature_dim = 9
    else:
        feature_dim = 6
    parameters_dict = get_parameters(data_name)

    data_processor = DataProcessor(data_dir=parameters_dict['data_dir'],
                                   data_name=data_name,
                                   target_subject_num=target_subject,
                                   num_subjects=parameters_dict['num_subjects'],
                    num_activities=parameters_dict['num_activities'],
                        window_size=parameters_dict['window_size'],
                                   num_modalities=parameters_dict['num_modalities'],
                 sampling_rate=parameters_dict['sampling_rate'],
                    sliding_window_overlap_ratio=parameters_dict['sliding_window_overlap_ratio'])
    modalities = data_processor.modalities
    train_X, train_y, test_X, test_y = data_processor.get_modality_separated_train_test_and_pn_v2(data_processor.data_dict,
                                                                                        data_processor.modalities)
    train_X, train_y, test_X, test_y = convert_to_torch_v2(train_X, train_y, test_X, test_y)

    trainset = Data.TensorDataset(train_X, train_y)
    trainloader = Data.DataLoader(dataset=trainset, batch_size=parameters_dict['batch_size'],
                                  shuffle=True, num_workers=0, drop_last=True)

    testset = Data.TensorDataset(test_X, test_y)
    testloader = Data.DataLoader(dataset=testset, batch_size=parameters_dict['batch_size'],
                                 shuffle=False, num_workers=0, drop_last=True)

    #start with global model that takes multiple modalities
    glob_models = get_models(parameters_dict)
    # now for each modality we are going to have separate base_net, unique_net and reconstruct net
    local_models_neg = {}
    local_models_pos = {}
    for modality in modalities:
        local_neg = get_models(parameters_dict, disentangle_reconst=False)
        local_pos = get_models(parameters_dict, disentangle_reconst=False)
        local_models_neg[modality] = local_neg
        local_models_pos[modality] = local_pos

    # push global models to gpu
    glob_models_to_push = ['base', 'common', 'unique', 'reconst', 'fuse', 'mine']
    for model_name in glob_models_to_push:
        if glob_models[model_name].train_on_gpu:
            glob_models[model_name].cuda()
    local_models_to_push_neg = ['base', 'unique', 'reconst', 'fuse']
    local_models_to_push_pos = ['base', 'common', 'reconst', 'fuse']

    for modality in modalities:
        for model_name in local_models_to_push_neg:
            if local_models_neg[modality][model_name].train_on_gpu:
                local_models_neg[modality][model_name].cuda()
        for model_name in local_models_to_push_pos:
            if local_models_pos[modality][model_name].train_on_gpu:
                local_models_pos[modality][model_name].cuda()


    # losses
    mse_loss = nn.MSELoss(reduction='mean')
    # initialize optimizers

    glob_optimizer = torch.optim.Adam(list(glob_models['base'].parameters()) + list(glob_models['unique'].parameters())
                                 + list(glob_models['common'].parameters()) + list(glob_models['reconst'].parameters()) + list(glob_models['mine'].parameters()),
                                 lr=parameters_dict['lr'], weight_decay=parameters_dict['weight_decay'])
    local_optimizers_neg = {}
    local_optimizers_pos = {}
    for modality in modalities:
        local_optimizers_neg[modality] = torch.optim.Adam(list(local_models_neg[modality]['base'].parameters())
                                                          + list(local_models_neg[modality]['unique'].parameters())
                                  + list(local_models_neg[modality]['reconst'].parameters()),
                                 lr=parameters_dict['lr'], weight_decay=parameters_dict['weight_decay'])
        local_optimizers_pos[modality] = torch.optim.Adam(list(local_models_pos[modality]['base'].parameters())
                                                          + list(local_models_pos[modality]['common'].parameters())
                                                          + list(local_models_pos[modality]['reconst'].parameters()),
                                                          lr=parameters_dict['lr'],
                                                          weight_decay=parameters_dict['weight_decay'])

    losses_neg = {
        'reconstruction_loss': []
    }
    losses_pos = {
        'reconstruction_loss': []
    }

    losses_train = {
        'total_loss': [],
        'reconstruction_loss': [],
        'common_and_positive': [],
        'unique_and_negative': [],
        'MI_loss': []
    }

    losses_test = {
        'total_loss': [],
        'reconstruction_loss': [],
        'common_and_positive': [],
        'unique_and_negative': [],
        'MI_loss': []
    }
    triplet_loss = nn.TripletMarginLoss(margin=0.5, p=2, eps=1e-7)
    for epoch in range(parameters_dict['num_epochs']):
        #enable train mode
        glob_models = enable_mode(glob_models, mode='train')
        for modality in modalities:
            local_models_neg[modality] = enable_mode(local_models_neg[modality], mode='train')
            local_models_pos[modality] = enable_mode(local_models_pos[modality], mode='train')

        all_batch_losses_train = train_one_epoch(trainloader, modalities,
                                                 glob_optimizer, local_optimizers_neg, local_optimizers_pos,
                                                 glob_models, local_models_neg, local_models_pos,
                    mse_loss, triplet_loss, mutual_information, mode='train')

        # add losses
        for loss_key in list(all_batch_losses_train.keys()):
            losses_train[loss_key].append(np.mean(np.array(all_batch_losses_train[loss_key])))

        # enable test mode
        glob_models = enable_mode(glob_models, mode='test')
        for modality in modalities:
            local_models_neg[modality] = enable_mode(local_models_neg[modality], mode='test')
            local_models_pos[modality] = enable_mode(local_models_pos[modality], mode='test')

        all_batch_losses_test = train_one_epoch(testloader, modalities,
                                                glob_optimizer, local_optimizers_neg, local_optimizers_pos,
                                                 glob_models, local_models_neg, local_models_pos,
                                                 mse_loss, triplet_loss, mutual_information, mode='test')

        # add losses
        for loss_key in list(all_batch_losses_test.keys()):
            losses_test[loss_key].append(np.mean(np.array(all_batch_losses_test[loss_key])))

        epoch_train_loss = np.mean(np.array(all_batch_losses_train['total_loss']))
        epoch_test_loss = np.mean(np.array(all_batch_losses_test['total_loss']))

        epoch_reconst_loss_train = np.mean(np.array(all_batch_losses_train['reconstruction_loss']))
        epoch_reconst_loss_test = np.mean(np.array(all_batch_losses_test['reconstruction_loss']))
        epoch_c_common_n_train = np.mean(np.array(all_batch_losses_train['c_loss_common_n']))
        epoch_c_common_n_test = np.mean(np.array(all_batch_losses_test['c_loss_common_n']))
        epoch_c_common_p_train = np.mean(np.array(all_batch_losses_train['c_loss_common_p']))
        epoch_c_common_p_test = np.mean(np.array(all_batch_losses_test['c_loss_common_p']))
        epoch_c_unique_n_train = np.mean(np.array(all_batch_losses_train['c_loss_unique_n']))
        epoch_c_unique_n_test = np.mean(np.array(all_batch_losses_test['c_loss_unique_n']))
        epoch_c_unique_p_train = np.mean(np.array(all_batch_losses_train['c_loss_unique_p']))
        epoch_c_unique_p_test = np.mean(np.array(all_batch_losses_test['c_loss_unique_p']))
        mutual_info_loss_train = np.mean(np.array(all_batch_losses_train['MI_loss']))
        mutual_info_loss_test = np.mean(np.array(all_batch_losses_test['MI_loss']))

        print(f'Epoch {epoch}: Train loss: {epoch_train_loss} Test: {epoch_test_loss}')
        print(f'Epoch {epoch}: Train reconst loss: {epoch_reconst_loss_train} Test: {epoch_reconst_loss_test}')
        print(f'Epoch {epoch}: Train C-common-p: {epoch_c_common_p_train} Test: {epoch_c_common_p_test}')
        print(f'Epoch {epoch}: Train C-common-n: {epoch_c_common_n_train} Test: {epoch_c_common_n_test}')
        print(f'Epoch {epoch}: Train C-unique-p: {epoch_c_unique_p_train} Test: {epoch_c_unique_p_test}')
        print(f'Epoch {epoch}: Train C-unique-n: {epoch_c_unique_n_train} Test: {epoch_c_unique_n_test}')
        print(f'Epoch {epoch}: Train MI Loss: {mutual_info_loss_train} Test: {mutual_info_loss_test}')
        print()

    timestring = strftime("%Y-%m-%d_%H-%M-%S", localtime()) + "_Subject%s" % str(
        target_subject)
    save_dir = os.path.join(BASE_DIR, 'CommonInformation', 'FuseNet', data_name, timestring)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_models(glob_models, local_models_neg, local_models_pos, save_dir)

    for m in range(len(list(losses_train.keys()))):
        key = list(losses_train.keys())[m]
        loss_train = losses_train[key]
        loss_test = losses_test[key]
        # save training result
        plot_loss_curves(loss_train, loss_test, save_loc=save_dir, show_fig=False,
                         title=key)

    with open(os.path.join(save_dir, 'parameters.txt'), 'w') as f:
        print(parameters_dict, file=f)
    print("END!")






