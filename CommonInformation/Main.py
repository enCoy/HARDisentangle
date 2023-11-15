import numpy as np
import os
from DataProcesses import DataProcessor
from Utils.HelperFunctions import  convert_to_torch, plot_loss_curves
from CommonInformation.Models.CommonNetModels import CNNBaseNet
from CommonInformation.Models.LSTMAE import LSTM_AE
from CommonInformation.Models.CommonNetModels import train_one_epoch
from Models.CommonNetModels import FuseNet, CommonNet, UniqueNet, ReconstructNet, Mine
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
    parameters_dict['num_epochs'] = 300
    parameters_dict['batch_size'] = 64
    parameters_dict['lr'] = 0.01
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

if __name__ == "__main__":

    data_name = 'pamap2'  # 'pamap2' or 'real'
    target_subject = 3
    if data_name == 'pamap2':
        feature_dim = 9
    else:
        feature_dim = 6
    parameters_dict = get_parameters(data_name)


    positive_representation_model_path = os.path.join(BASE_DIR, r"CommonInformation\PositiveSampleGenerator\pamap2\2023-11-02_07-47-04_Subject3")
    negative_representation_model_path = os.path.join(BASE_DIR, r"CommonInformation\NegativeSampleGenerator\pamap2\2023-11-02_05-50-32_Subject3")


    data_processor = DataProcessor(data_dir=parameters_dict['data_dir'],
                                   data_name=data_name,
                                   target_subject_num=target_subject,
                                   num_subjects=parameters_dict['num_subjects'],
                    num_activities=parameters_dict['num_activities'],
                        window_size=parameters_dict['window_size'],
                                   num_modalities=parameters_dict['num_modalities'],
                 sampling_rate=parameters_dict['sampling_rate'],
                    sliding_window_overlap_ratio=parameters_dict['sliding_window_overlap_ratio'])

    train_X, train_y, test_X, test_y = data_processor.get_modality_separated_train_test(data_processor.data_dict,
                                                                                        data_processor.modalities)
    train_X, train_y, test_X, test_y = convert_to_torch(train_X, train_y, test_X, test_y)

    trainset = Data.TensorDataset(train_X, train_y)
    trainloader = Data.DataLoader(dataset=trainset, batch_size=parameters_dict['batch_size'],
                                  shuffle=True, num_workers=0, drop_last=True)
    testset = Data.TensorDataset(test_X, test_y)
    testloader = Data.DataLoader(dataset=testset, batch_size=parameters_dict['batch_size'],
                                 shuffle=False, num_workers=0, drop_last=True)


    base_net = CNNBaseNet(input_dim=feature_dim, output_channels=128, embedding=1024,
                       num_time_steps=parameters_dict['window_size'])

    positive_r_model = LSTM_AE(feature_dim, parameters_dict['embedding_dim'],
                    use_bidirectional=parameters_dict['use_bidirectional'],
                    num_layers=parameters_dict['num_lstm_layers'])
    positive_r_model.load_state_dict(torch.load(os.path.join(positive_representation_model_path, 'positiveGeneratorStateDict.pth')))
    negative_r_model = LSTM_AE(feature_dim, parameters_dict['embedding_dim'],
                               use_bidirectional=parameters_dict['use_bidirectional'],
                               num_layers=parameters_dict['num_lstm_layers'])
    negative_r_model.load_state_dict(
        torch.load(os.path.join(negative_representation_model_path, 'negativeGeneratorStateDict.pth')))

    # freeze these two
    for param in positive_r_model.parameters():
        param.requires_grad = False
    for param in negative_r_model.parameters():
        param.requires_grad = False
    # initialize with the same seed to ensure that they start from the same place
    # torch.manual_seed(parameters_dict['common_unique_shared_seed'])
    unique_net = UniqueNet(1024, 256, parameters_dict['embedding_dim'])
    # torch.manual_seed(parameters_dict['common_unique_shared_seed'])
    common_net = CommonNet(1024, 256, parameters_dict['embedding_dim'])
    reconstruct_net = ReconstructNet(parameters_dict['embedding_dim'] * 2, 256,
                                     parameters_dict['window_size'] * feature_dim)
    fuse_net = FuseNet(base_net, common_net, unique_net,
                       reconstruct_net, feature_dim,
                       window_num=parameters_dict['window_size'])
    mine = Mine(x_dim=parameters_dict['embedding_dim'], z_dim=parameters_dict['embedding_dim'],
                hidden_dim=parameters_dict['embedding_dim'] // 2)

    all_models = [base_net, unique_net, common_net, reconstruct_net, mine, positive_r_model, negative_r_model]
    all_models_names = ['base_net', 'unique_net', 'common_net', 'reconstruct_net', 'mine', 'positive_r_model', 'negative_r_model']
    for m in range(len(all_models)):
        model = all_models[m]
        model_name = all_models_names[m]
        if (model.train_on_gpu):
            model.cuda()

    positive_r_model = positive_r_model.encoder
    negative_r_model = negative_r_model.encoder

    # losses
    mse_loss = nn.MSELoss(reduction='mean')
    # optimizer
    optimizer = torch.optim.Adam(list(base_net.parameters()) + list(unique_net.parameters())
                                 + list(common_net.parameters()) + list(reconstruct_net.parameters()) + list(mine.parameters()),
                                 lr=parameters_dict['lr'], weight_decay=parameters_dict['weight_decay'])

    losses_train = {
        'total_loss': [],
        'reconstruction_loss': [],
        'c_loss_common_p': [],
        'c_loss_common_n': [],
        'c_loss_unique_p': [],
        'c_loss_unique_n': [],
        'MI_loss': []
    }

    losses_test = {
        'total_loss': [],
        'reconstruction_loss': [],
        'c_loss_common_p': [],
        'c_loss_common_n': [],
        'c_loss_unique_p': [],
        'c_loss_unique_n': [],
        'MI_loss': []
    }
    for epoch in range(parameters_dict['num_epochs']):
        #enable train mode
        for model in all_models:
            model.train()
        all_batch_losses_train = train_one_epoch(trainloader, optimizer, fuse_net, positive_r_model, negative_r_model, mine,
                        mse_loss, contrastive_loss_criterion, mutual_information, mode='train')
        # add losses
        for loss_key in list(all_batch_losses_train.keys()):
            losses_train[loss_key].append(np.mean(np.array(all_batch_losses_train[loss_key])))

        # enable test mode
        for model in all_models:
            model.eval()
        all_batch_losses_test = train_one_epoch(testloader, optimizer, fuse_net, positive_r_model, negative_r_model, mine,
                                           mse_loss, contrastive_loss_criterion, mutual_information, mode='test')
        # add losses
        for loss_key in list(all_batch_losses_test.keys()):
            losses_test[loss_key].append(np.mean(np.array(all_batch_losses_test[loss_key])))

        epoch_train_loss = np.mean(np.array(all_batch_losses_train['total_loss']))
        epoch_test_loss = np.mean(np.array(all_batch_losses_test['total_loss']))

        epoch_c_common_n_train =  np.mean(np.array(all_batch_losses_train['c_loss_common_n']))
        epoch_c_unique_p_train =  np.mean(np.array(all_batch_losses_train['c_loss_unique_p']))
        epoch_c_common_n_test =  np.mean(np.array(all_batch_losses_test['c_loss_common_n']))
        epoch_c_unique_p_test =  np.mean(np.array(all_batch_losses_test['c_loss_unique_p']))
        print(f'Epoch {epoch}: Train loss: {epoch_train_loss} Test: {epoch_test_loss}')
        print(f'Epoch {epoch}: Train C-common-n: {epoch_c_common_n_train} Test: {epoch_c_common_n_test}')
        print(f'Epoch {epoch}: Train C-unique-p: {epoch_c_unique_p_train} Test: {epoch_c_unique_p_test}')
        print()

    timestring = strftime("%Y-%m-%d_%H-%M-%S", localtime()) + "_Subject%s" % str(
        target_subject)
    save_dir = os.path.join(BASE_DIR, 'CommonInformation', 'FuseNet', data_name, timestring)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for m in range(len(all_models)):
        model = all_models[m]
        model_name = all_models_names[m]
        # save state dict
        torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_stateDict.pth"))
        # save model
        torch.save(model, os.path.join(save_dir, f"{model_name}.pth"))
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






