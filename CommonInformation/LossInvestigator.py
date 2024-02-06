# compare commonality loss
import os
from CommonInformation.Models.CommonNetV3 import CNNBaseNet, CommonNet, UniqueNet, FuseNet
from CommonInformation.MainV3 import get_parameters, enable_mode
import numpy as np
from CommonInformation.DataProcesses import DataProcessor
from CommonInformation.Main import BASE_DIR
import torch
from Utils.HelperFunctions import convert_to_torch_v2
import torch.utils.data as Data
import itertools
import pandas as pd
import seaborn as sns
import  matplotlib.pyplot as plt


if __name__ == "__main__":

    machine = 'windows'
    if machine == 'linux':
        base_dir = r'/home/cmyldz/Dropbox (GaTech)/DisentangledHAR/'
    else:
        base_dir = r'C:\Users\Cem Okan\Dropbox (GaTech)\DisentangledHAR/'

    data_name = 'pamap2'  # 'pamap2' or 'real'
    target_subject = 1
    parameters_dict = get_parameters(data_name)
    if data_name == 'pamap2':
        input_size = 9  # number of sensor channels
    elif data_name == 'real':
        input_size = 6  # number of sensor channels

    # get the data loader
    data_processor = DataProcessor(data_dir=parameters_dict['data_dir'],
                                   data_name=data_name,
                                   target_subject_num=target_subject,
                                   num_subjects=parameters_dict['num_subjects'],
                                   num_activities=parameters_dict['num_activities'],
                                   window_size=parameters_dict['window_size'],
                                   num_modalities=parameters_dict['num_modalities'],
                                   sampling_rate=parameters_dict['sampling_rate'],
                                   sliding_window_overlap_ratio=parameters_dict['sliding_window_overlap_ratio'])
    # initial idea - look at the score of each modality alone % freeze earlier layers
    train_X, train_y, test_X, test_y = data_processor.get_modality_separated_train_test_classification_data_concatted_behind(
        data_processor.data_dict,
        data_processor.modalities)

    train_X, train_y, test_X, test_y = convert_to_torch_v2(train_X, train_y, test_X, test_y)

    dataset = Data.TensorDataset(train_X, train_y)
    loader = Data.DataLoader(dataset=dataset, batch_size=1,
                                  shuffle=False, num_workers=0, drop_last=True)

    num_modalities = parameters_dict['num_modalities']
    modalities = data_processor.modalities
    model_dir = os.path.join(BASE_DIR, r"CommonInformation\FuseNet\pamap2\2023-11-26_01-06-39_Subject3")
    models = {}

    models['base'] = CNNBaseNet(input_dim=input_size, output_channels=128, embedding=1024,
                                          num_time_steps=parameters_dict['window_size'])
    # models['common'] =  CommonNet(1024, 256, parameters_dict['embedding_dim'])
    # models['unique'] =  UniqueNet(1024, 256, parameters_dict['embedding_dim'])

    models['common'] = CommonNet(7 * 128, 256, parameters_dict['embedding_dim'])
    models['unique'] = UniqueNet(7 * 128, 256, parameters_dict['embedding_dim'])

    # load trained models
    models['base'].load_state_dict(torch.load(os.path.join(model_dir, 'glob_base_stateDict.pth')))
    models['common'].load_state_dict(torch.load(os.path.join(model_dir, 'glob_common_stateDict.pth')))
    models['unique'].load_state_dict(torch.load(os.path.join(model_dir, 'glob_unique_stateDict.pth')))

    models_to_push = ['base', 'common', 'unique']
    for modality in modalities:
        for model_name in models_to_push:
            if models[model_name].train_on_gpu:
                models[model_name].cuda()
                models[model_name].eval()

    common_encoder_out = {}  # key is (modality, batch_idx), value is common encoder out
    unique_encoder_out = {}  # key is (modality, batch_idx), value is unique encoder out
    mse_loss = torch.nn.MSELoss(reduction='mean')

    batch_indexes = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs_list = np.split(inputs, len(modalities), axis=-1)
            batch_indexes.append(batch_idx)
            for modality_idx in range(len(modalities)):
                modality = modalities[modality_idx]
                input_modality = torch.squeeze(inputs_list[modality_idx])
                base_repr = models['base'](input_modality)
                common_representation = models['common'](base_repr)
                unique_representation = models['unique'](base_repr)
                common_encoder_out[(modality, batch_idx)] = common_representation
                unique_encoder_out[(modality, batch_idx)] = unique_representation

    commonality_losses = {}  # key is (modality_i, modality_j) value is l2 norm difference
    uniqueness_losses = {}

    cross_modalities = list(itertools.combinations(modalities, 2))
    for modality_pair in cross_modalities:
        commonality_losses[modality_pair] = []
        uniqueness_losses[modality_pair] = []

        modality_1 = modality_pair[0]
        modality_2 = modality_pair[1]
        for batch_idx in batch_indexes:
            mod_1_common = common_encoder_out[(modality_1, batch_idx)]
            mod_2_common = common_encoder_out[(modality_2, batch_idx)]
            mod_1_unique = unique_encoder_out[(modality_1, batch_idx)]
            mod_2_unique = unique_encoder_out[(modality_2, batch_idx)]
            commonality_losses[modality_pair].append(np.linalg.norm(mod_1_common.cpu() - mod_2_common.cpu()))
            uniqueness_losses[modality_pair].append(np.linalg.norm(mod_1_unique.cpu() - mod_2_unique.cpu()))

            # print("mod_1_common  size: ", mod_1_common.size())
            # print("norm size: ", np.linalg.norm(mod_1_common.cpu() - mod_2_common.cpu()))
    # now common unique diff on the same modaity
    common_unique_loss = {}
    for modality in modalities:
        common_unique_loss[modality] = []
        for batch_idx in batch_indexes:
            common = common_encoder_out[(modality, batch_idx)]
            unique = unique_encoder_out[(modality, batch_idx)]
            common_unique_loss[modality].append(np.linalg.norm(common.cpu() - unique.cpu()))


    # create a dataframe y:loss, x=modality/modality pair , hue=common/unique
    all_xs = []
    all_ys = []
    all_hues = []
    df1 = pd.DataFrame()
    for modality_pair in cross_modalities:
        x_var_1 = [(modality_pair[0]+'-'+modality_pair[1]) for i in range(len(commonality_losses[modality_pair]))]
        y_var_1 = commonality_losses[modality_pair]
        hue_1 = ['common' for i in range(len(commonality_losses[modality_pair]))]

        x_var_2 = [(modality_pair[0] + '-' + modality_pair[1]) for i in range(len(uniqueness_losses[modality_pair]))]
        y_var_2 = uniqueness_losses[modality_pair]
        hue_2 = ['unique' for i in range(len(uniqueness_losses[modality_pair]))]

        all_xs = all_xs + x_var_1 + x_var_2
        all_ys = all_ys + y_var_1 + y_var_2
        all_hues = all_hues + hue_1 + hue_2

    df1['difference'] = all_ys
    df1['locations'] = all_xs
    df1['information'] = all_hues
    plt.figure()
    sns.boxplot(data=df1, x="locations", y="difference", hue="information", showfliers=False)


    df2 = pd.DataFrame()
    all_xs = []
    all_ys = []
    all_hues = []
    df1 = pd.DataFrame()
    for modality in modalities:
        x_var = [modality for i in range(len(common_unique_loss[modality]))]
        y_var = common_unique_loss[modality]
        hue = ['common' for i in range(len(common_unique_loss[modality]))]
        all_xs = all_xs + x_var
        all_ys = all_ys + y_var
        all_hues = all_hues + hue

    df2['difference'] = all_ys
    df2['locations'] = all_xs
    plt.figure()
    sns.boxplot(data=df2, x="locations", y="difference", showfliers=False)

    plt.show()


    #
    #
    # for batch_idx in batch_indexes:







