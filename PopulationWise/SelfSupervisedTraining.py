import warnings
from Utils.HelperFunctions import plot_loss_curves, save_models, save_best_models
from Utils.HelperFunctions import  mutual_information as mi_loss
from torch.utils.data import DataLoader
from DataProcesses import SelfSupervisedDataProcessor, SelfSupervisedDataProcessorForTesting
import time
from DataProcesses import get_pamap2_dataframe
import torch
from Models.DisentangleNetModels import CNNBaseNet, PopulationEncoder, PersonalizedEncoder, ReconstructNet, Mine
from time import localtime, strftime
import os
import torch.nn as nn
from info_nce import InfoNCE
import numpy as np
import json

def print_losses(train_losses, test_losses):
    # loss is dict with multiple losses
    loss_keys = list(train_losses.keys())
    for key in loss_keys:
        print(f"{key} --- Train: {train_losses[key][-1]}      Test:{test_losses[key][-1]}")
    print()


def get_loader(parameters_dict, target_subject, dataset_type='train'):
    if dataset_type == 'train':
        # handle data loader
        dataset = SelfSupervisedDataProcessor(dataframe_dir=parameters_dict['dataframe_dir'],
                                                    data_name=parameters_dict['data_name'],
                                                    target_subject_num=target_subject,
                                                    num_subjects=parameters_dict['num_subjects'],
                                                    window_size=parameters_dict['window_size'],
                                                    num_modalities=parameters_dict['num_modalities'], # for pamap2, 3 locations 9 axis per location
                                                    sampling_rate=parameters_dict['sampling_rate'],
                                                    sliding_window_overlap_ratio=parameters_dict['sliding_window_overlap_ratio'],
                                                    num_neg_samples=parameters_dict['num_neg_samples'])
    else:  # test
        dataset = SelfSupervisedDataProcessorForTesting(dataframe_dir=parameters_dict['dataframe_dir'],
                                              data_name=parameters_dict['data_name'],
                                              target_subject_num=target_subject,
                                              num_subjects=parameters_dict['num_subjects'],
                                              window_size=parameters_dict['window_size'],
                                              num_modalities=parameters_dict['num_modalities'],
                                              # for pamap2, 3 locations 9 axis per location
                                              sampling_rate=parameters_dict['sampling_rate'],
                                              sliding_window_overlap_ratio=parameters_dict[
                                                  'sliding_window_overlap_ratio'],
                                              num_neg_samples=parameters_dict['num_neg_samples'])
    loader = DataLoader(dataset, batch_size=parameters['batch_size'], shuffle=True, num_workers=0)
    return loader


def train_model_one_epoch(loader, base_net, person_enc, pop_enc, reconst_net, reconst_loss_fn, contrast_loss_fn, mine_loss, cost_params,
                          epoch_losses, parameters, ma_et, include_mine=False):
    batch_losses = {
        'total_loss': [],
        'reconstruction_loss': [],
        'info_pop_loss': [],
        'info_per_loss': [],
    }
    if include_mine:
        batch_losses['mine'] = []
    for batch in loader:
        anchor = batch['anchor'].view(-1, parameters['window_size'],
                                      parameters['num_modalities']).cuda()  # N x WindowSize x FeatureDim
        pop_enc_pos = batch['pop_enc_pos'].view(-1, parameters['window_size'],
                                                parameters['num_modalities']).cuda()  # N x WindowSize x FeatureDim
        pop_enc_neg = batch['pop_enc_neg'].cuda()  # N x NumNegSamples x WindowSize x FeatureDim
        person_enc_pos = batch['person_enc_pos'].view(-1, parameters['window_size'],
                                                      parameters['num_modalities']).cuda()  # N x WindowSize x FeatureDim
        person_enc_neg = batch['person_enc_neg'].cuda()  # N x NumNegSamples x WindowSize x FeatureDim

        optimizer.zero_grad()
        anc_per, anc_pop = get_encodings(anchor, base_net, person_enc, pop_enc)
        _, pos_pop = get_encodings(pop_enc_pos, base_net, person_enc, pop_enc)
        _, neg_pop = get_encodings(pop_enc_neg, base_net, person_enc, pop_enc)
        pos_per, _ = get_encodings(person_enc_pos, base_net, person_enc, pop_enc)
        neg_per, _ = get_encodings(person_enc_neg, base_net, person_enc, pop_enc)

        # concatenate two anchor representations
        concatted = torch.concat((anc_per, anc_pop), 1)  # shaped (N x (2output_size))
        reconstructed = reconst_net(concatted)
        # reshape reconstructed
        reconstructed = torch.reshape(reconstructed, (-1, parameters['window_size'], parameters['num_modalities']))

        # calculate losses
        reconstruction_loss = reconst_loss_fn(reconstructed, anchor)
        infoNCE_pop = contrast_loss_fn(anc_pop, pos_pop, neg_pop)
        infoNCE_person = contrast_loss_fn(anc_per, pos_per, neg_per)

        total_loss_w_mine = cost_params['lambda_reconst'] * reconstruction_loss + \
                     cost_params['lambda_info_per'] * infoNCE_person + cost_params['lambda_info_pop'] * infoNCE_pop
        if include_mine:
            # # this is how several people sample marginal distribution
            personalized_shuffled = torch.index_select(  # shuffles the noise
                anc_per, 0, torch.randperm(anc_per.shape[0]).cuda())
            # mi_score = -mine_loss(mine, anc_pop, anc_per, personalized_shuffled)
            # # source: https://github.com/sungyubkim/MINE-Mutual-Information-Neural-Estimation-/blob/master/MINE.ipynb
            ma_rate = 0.01
            mi_lb, joint, et = mine_loss(mine, anc_pop, anc_per, personalized_shuffled)
            ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)
            # unbiasing use moving average
            MI_loss = -(torch.mean(joint) - (1 / ma_et.mean()).detach() * torch.mean(et))
            total_loss = total_loss_w_mine + cost_params['lambda_mine']*MI_loss
            batch_losses['mine'].append(MI_loss.detach().item())
        else:
            total_loss = total_loss_w_mine
        total_loss.backward()
        optimizer.step()
        batch_losses['total_loss'].append(total_loss_w_mine.item())
        batch_losses['reconstruction_loss'].append(reconstruction_loss.detach().item())
        batch_losses['info_per_loss'].append(infoNCE_person.detach().item())
        batch_losses['info_pop_loss'].append(infoNCE_pop.detach().item())

    # add the losses to epoch loss
    epoch_losses['total_loss'].append(np.mean(batch_losses['total_loss']))
    epoch_losses['reconstruction_loss'].append(np.mean(batch_losses['reconstruction_loss']))
    epoch_losses['info_per_loss'].append(np.mean(batch_losses['info_per_loss']))
    epoch_losses['info_pop_loss'].append(np.mean(batch_losses['info_pop_loss']))
    if include_mine:
        epoch_losses['mine'].append(np.mean(batch_losses['mine']))
    return epoch_losses

def test_model(loader, base_net, person_enc, pop_enc, reconst_net, reconst_loss_fn, contrast_loss_fn, mine_loss, cost_params,
                          epoch_losses, parameters, ma_et, include_mine=False):
    batch_losses = {
        'total_loss': [],
        'reconstruction_loss': [],
        'info_pop_loss': [],
        'info_per_loss': [],
    }
    if include_mine:
        batch_losses['mine'] = []
    with torch.no_grad():
        for batch in loader:
            anchor = batch['anchor'].view(-1, parameters['window_size'],
                                          parameters['num_modalities']).cuda()  # N x WindowSize x FeatureDim
            pop_enc_pos = batch['pop_enc_pos'].view(-1, parameters['window_size'],
                                                    parameters['num_modalities']).cuda()  # N x WindowSize x FeatureDim
            pop_enc_neg = batch['pop_enc_neg'].cuda()  # N x NumNegSamples x WindowSize x FeatureDim
            person_enc_pos = batch['person_enc_pos'].view(-1, parameters['window_size'],
                                                          parameters['num_modalities']).cuda()  # N x WindowSize x FeatureDim
            person_enc_neg = batch['person_enc_neg'].cuda()  # N x NumNegSamples x WindowSize x FeatureDim

            anc_per, anc_pop = get_encodings(anchor, base_net, person_enc, pop_enc)
            _, pos_pop = get_encodings(pop_enc_pos, base_net, person_enc, pop_enc)
            _, neg_pop = get_encodings(pop_enc_neg, base_net, person_enc, pop_enc)
            pos_per, _ = get_encodings(person_enc_pos, base_net, person_enc, pop_enc)
            neg_per, _ = get_encodings(person_enc_neg, base_net, person_enc, pop_enc)

            # concatenate these two
            concatted = torch.concat((anc_per, anc_pop), 1)  # shaped (N x (2output_size))
            reconstructed = reconst_net(concatted)
            # reshape reconstructed
            reconstructed = torch.reshape(reconstructed, (-1, parameters['window_size'],
                                                          parameters['num_modalities']))

            # calculate losses
            reconstruction_loss = reconst_loss_fn(reconstructed, anchor)
            infoNCE_pop = contrast_loss_fn(anc_pop, pos_pop, neg_pop)
            infoNCE_person = contrast_loss_fn(anc_per, pos_per, neg_per)
            total_loss_w_mine = cost_params['lambda_reconst'] * reconstruction_loss + \
                                cost_params['lambda_info_per'] * infoNCE_person + cost_params[
                                    'lambda_info_pop'] * infoNCE_pop
            if include_mine:
                personalized_shuffled = torch.index_select(  # shuffles the noise
                    anc_per, 0, torch.randperm(anc_per.shape[0]).cuda())
                ma_rate = 0.01
                mi_lb, joint, et = mine_loss(mine, anc_pop, anc_per, personalized_shuffled)
                ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)
                # unbiasing use moving average
                MI_loss = -(torch.mean(joint) - (1 / ma_et.mean()).detach() * torch.mean(et))
                total_loss = total_loss_w_mine + cost_params['lambda_mine'] * MI_loss
                batch_losses['mine'].append(MI_loss.detach().item())
            else:
                total_loss = total_loss_w_mine

            batch_losses['total_loss'].append(total_loss_w_mine.item())
            batch_losses['reconstruction_loss'].append(reconstruction_loss.detach().item())
            batch_losses['info_per_loss'].append(infoNCE_person.detach().item())
            batch_losses['info_pop_loss'].append(infoNCE_pop.detach().item())

        # add the losses to epoch loss
        epoch_losses['total_loss'].append(np.mean(batch_losses['total_loss']))
        epoch_losses['reconstruction_loss'].append(np.mean(batch_losses['reconstruction_loss']))
        epoch_losses['info_per_loss'].append(np.mean(batch_losses['info_per_loss']))
        epoch_losses['info_pop_loss'].append(np.mean(batch_losses['info_pop_loss']))
        if include_mine:
            epoch_losses['mine'].append(np.mean(batch_losses['mine']))
    return epoch_losses


def get_encodings(x, base_net, person_enc, pop_enc):
    # input shape is either (N x NumWindows x feature_dim) or (N x NumNegSamples x NumWindows, feature_dim)
    initial_x_size = x.shape
    if x.dim() == 3:  # either positive or anchor sample
        is_negative = False
    else:  # negative
        is_negative = True

    if is_negative:
        x = x.view(-1, x.shape[-2], x.shape[-1])
    base_representation = base_net(x)  # should be shaped (N x embedding_dim)
    personalized_encoding = person_enc(base_representation)  # should be shaped (N x output_size)
    population_encoding = pop_enc(base_representation)  # should be shaped (N x output_size)
    # retransform if negative samples
    if is_negative:
        personalized_encoding = personalized_encoding.view(initial_x_size[0], initial_x_size[1], -1)
        population_encoding = population_encoding.view(initial_x_size[0], initial_x_size[1], -1)
    return personalized_encoding, population_encoding

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    machine = 'windows'
    include_mine = False
    if machine == 'linux':
        BASE_DIR = r'/home/cmyldz/Dropbox (GaTech)/DisentangledHAR'
    else:
        BASE_DIR = r'C:\Users\Cem Okan\Dropbox (GaTech)\DisentangledHAR'
    print("CUDA device available: ", torch.cuda.is_available())
    print("CUDA device name: ", torch.cuda.get_device_name(0))
    num_devices = torch.cuda.device_count()
    print("num_devices: ", num_devices)
    # Print information for each CUDA device
    for device_idx in range(num_devices):
        device = torch.cuda.get_device_properties(device_idx)
        print(f"CUDA Device {device_idx} - Name: {device.name}, Capability: {device.major}.{device.minor}")

    timestring = strftime("%Y-%m-%d_%H-%M-%S", localtime())
    output_dir = os.path.join(BASE_DIR, "Logging", timestring)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cost_params = {
        'lambda_info_pop': 1.0,
        'lambda_info_per': 1.0,
        'lambda_reconst': 1.0,
    }
    if include_mine:
        cost_params['lambda_mine'] = 1.0
    parameters = {
        'base_net_out_channel': 64,
        'base_net_output_dim': 256,
        'personalized_output_dim': 128,
        'population_output_dim': 128,
        'batch_size': 128,
        'model_shared_seed': 23,
        'learning_rate': 0.0003,
        'num_epochs': 50,
        'weight_decay': 1e-4,
        'cost_params': cost_params,
        'num_neg_samples': 128,
        'data_name': 'pamap2',
        'window_size': 50,
        'sampling_rate': 50,
        'sliding_window_overlap_ratio': 0.5,
        'best_epoch': 0
    }
    if parameters['data_name'] == 'pamap2':
        parameters['num_subjects'] = 8
        parameters['num_activities'] = 12
        parameters['num_modalities'] = 27  # two acc 1 gyro, 3 axis per sensor and 3 different locations = 3x3x3=27
        parameters['dataframe_dir'] = os.path.join(BASE_DIR, r"PAMAP2_Dataset/PAMAP2_Dataset/Processed50Hz")
    else:  # real
        parameters['num_subjects'] = 15
        parameters['num_activities'] = 8
        parameters['num_modalities'] = 42  # 1 acc 1 gyro per location, 3 axis per sensor and 7 different locations = 7x2x3
        parameters['dataframe_dir'] = os.path.join(BASE_DIR, r"realworld2016_dataset/Processed")

    # Save parameter dictionary as a text file
    with open(os.path.join(output_dir, 'parameters.txt'), 'w') as file:
        for key, value in parameters.items():
            file.write(f"{key}: {value}\n")

    if parameters['dataframe_dir'] == 'pamap2':
        dataframe = get_pamap2_dataframe(data_dir=parameters['dataframe_dir'],
                                         num_subjects=parameters['num_subjects'],
                                         window_size=parameters['window_size'],
                                         sliding_overlap=parameters['sliding_window_overlap_ratio'],
                                         num_activities=parameters['num_activities'])

    for test_subject in range(1, parameters['num_subjects']+1):
        subject_output_dir = os.path.join(output_dir, f'S{test_subject}')
        if not os.path.exists(subject_output_dir):
            os.makedirs(subject_output_dir)
        train_loader = get_loader(parameters, test_subject, dataset_type='train')
        test_loader = get_loader(parameters, test_subject, dataset_type='test')

        # load model etc.
        base_net = CNNBaseNet(input_dim=parameters['num_modalities'],
                              output_channels=parameters['base_net_out_channel'],
                              output_dim=parameters['base_net_output_dim'],
                              num_time_steps=parameters['window_size'])
        torch.manual_seed(parameters['model_shared_seed'])
        population_encoder = PopulationEncoder(input_dim=parameters['base_net_output_dim'],
                                               hidden_1=128, output_dim=parameters['population_output_dim'],
                                               train_on_gpu=True)
        torch.manual_seed(parameters['model_shared_seed'])
        personalized_encoder = PersonalizedEncoder(input_dim=parameters['base_net_output_dim'],
                                                   hidden_1=128, output_dim=parameters['personalized_output_dim'],
                                                   train_on_gpu=True)
        reconstruct_net = ReconstructNet(
            input_dim=parameters['personalized_output_dim'] + parameters['population_output_dim'],
            hidden_1=128, output_dim=parameters['window_size'] * parameters['num_modalities'], train_on_gpu=True)

        if include_mine:
            mine = Mine(x_dim=parameters['population_output_dim'], z_dim=parameters['personalized_output_dim'],
                    hidden_dim=parameters['population_output_dim'] // 2)

        all_models = {
            'base_net': base_net,
            'population_encoder': population_encoder,
            'personalized_encoder': personalized_encoder,
            'reconstruct_net': reconstruct_net,
        }
        if include_mine:
            all_models['mine'] = mine
        all_models_names = list(all_models.keys())
        # put the models in CUDA
        for m in range(len(all_models_names)):
            model = all_models[all_models_names[m]]
            if (model.train_on_gpu):
                model.cuda()

        # losses
        mse_loss = nn.MSELoss(reduction='mean')
        info_nce_loss = InfoNCE(negative_mode='paired')
        # optimizer
        if include_mine:
            optimizer = torch.optim.Adam(list(base_net.parameters()) + list(population_encoder.parameters())
                                     + list(personalized_encoder.parameters()) + list(reconstruct_net.parameters()) +
                                     list(mine.parameters()),
                                     lr=parameters['learning_rate'], weight_decay=parameters['weight_decay'])
        else:
            optimizer = torch.optim.Adam(list(base_net.parameters()) + list(population_encoder.parameters())
                                         + list(personalized_encoder.parameters()) + list(reconstruct_net.parameters()),
                                         lr=parameters['learning_rate'], weight_decay=parameters['weight_decay'])

        # training
        train_epoch_losses = {
            'total_loss': [],
            'reconstruction_loss': [],
            'info_pop_loss': [],
            'info_per_loss': [],
        }
        test_epoch_losses = {
            'total_loss': [],
            'reconstruction_loss': [],
            'info_pop_loss': [],
            'info_per_loss': [],
        }
        if include_mine:
            train_epoch_losses['mine'] = []
            test_epoch_losses['mine'] = []

        best_val_loss = np.inf
        ma_et = 1.0
        for epoch in range(parameters['num_epochs']):
            epoch_starting_time = time.time()
            print(f"Epoch {epoch + 1} is started...")
            # enable train mode
            for model in list(all_models.values()):
                model.train()
            train_epoch_losses = train_model_one_epoch(train_loader, base_net,
                                                       personalized_encoder, population_encoder, reconstruct_net,
                                                       reconst_loss_fn=mse_loss,
                                                       contrast_loss_fn=info_nce_loss,
                                                       mine_loss = mi_loss,
                                                       cost_params=cost_params,
                                                       epoch_losses=train_epoch_losses,
                                                       parameters=parameters,
                                                       ma_et=ma_et,
                                                       include_mine=include_mine)

            # testing
            test_epoch_losses = test_model(test_loader, base_net,
                                           personalized_encoder, population_encoder, reconstruct_net,
                                           reconst_loss_fn=mse_loss,
                                           contrast_loss_fn=info_nce_loss,
                                           mine_loss=mi_loss,
                                           cost_params=cost_params,
                                           epoch_losses=test_epoch_losses,
                                           parameters=parameters,
                                           ma_et=ma_et,
                                           include_mine=include_mine)
            print("Elapsed time: ", time.time() - epoch_starting_time)
            print_losses(train_epoch_losses, test_epoch_losses)

            # save models
            save_models(all_models, subject_output_dir)
            for loss_name in list(train_epoch_losses.keys()):
                plot_loss_curves(train_epoch_losses[loss_name], test_epoch_losses[loss_name],
                                 save_loc=subject_output_dir, show_fig=False, title=loss_name)

            # if it has best validation loss, save that model
            if test_epoch_losses['total_loss'][-1] < best_val_loss:
                best_val_loss = test_epoch_losses['total_loss'][-1]
                save_best_models(all_models, save_dir=subject_output_dir)
                parameters['best_epoch'] = epoch + 1
                # Save parameter dictionary as a text file
                with open(os.path.join(output_dir, 'parameters.txt'), 'w') as file:
                    for key, value in parameters.items():
                        file.write(f"{key}: {value}\n")

            # todo: look at domain discrimination loss
