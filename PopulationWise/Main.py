import warnings
import torch.utils.data as TorchData
from Utils.HelperFunctions import convert_to_torch
from DataProcesses import DownstreamDataProcessor
import torch
from Models.DisentangleNetModels import CNNBaseNet, PopulationEncoder, PersonalizedEncoder, ReconstructNet
import os
import torch.nn as nn

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



if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    machine = 'windows'
    if machine == 'linux':
        BASE_DIR = r'/home/cmyldz/Dropbox (GaTech)/DisentangledHAR'
    else:
        BASE_DIR = r'C:\Users\Cem Okan\Dropbox (GaTech)\DisentangledHAR'

    data_name = 'real'  # 'pamap2' or 'real'
    learning_rate = 0.001
    num_epochs = 1
    weight_decay = 1e-6
    num_time_steps = 50  # in a single window
    sampling_rate = 50
    sliding_overlap = 0.5
    model_shared_seed = 23
    if data_name == 'pamap2':
        num_subjects = 8
        num_activities = 12
        num_modalities = 27  # two acc 1 gyro, 3 axis per sensor and 3 different locations = 3x3x3=27
        data_dir = os.path.join(BASE_DIR, r"PAMAP2_Dataset/PAMAP2_Dataset/Processed50Hz")
    else:  # real
        num_subjects = 15
        num_activities = 8
        num_modalities =  42 # 1 acc 1 gyro per location, 3 axis per sensor and 7 different locations = 7x2x3
        data_dir = os.path.join(BASE_DIR, r"realworld2016_dataset/Processed")
    batch_size = 128

    parameters = {
        'base_net_out_channel': 256,
        'base_net_output_dim': 32,
        'personalized_output_dim': 128,
        'population_output_dim': 128
    }

    for test_subject in range(1, num_subjects+1):
        data_processor = DataProcessor(data_dir=data_dir,
                                       data_name=data_name,
                                       target_subject_num=test_subject,
                                       num_subjects=num_subjects,
                                       num_activities=num_activities,
                                       window_size=num_time_steps,
                                       num_modalities=num_modalities,  # for pamap2, 3 locations 9 axis per location
                                       sampling_rate=sampling_rate,
                                       sliding_window_overlap_ratio=sliding_overlap)

        if data_name == 'pamap2':
            train_X, train_y, test_X, test_y = data_processor.get_pamap2_train_test_data()
        else:  # if real
            train_X, train_y, test_X, test_y = data_processor.get_realworld_train_test_data()
        train_X, train_y, test_X, test_y = convert_to_torch(train_X, train_y, test_X, test_y)

        trainset = TorchData.TensorDataset(train_X, train_y)
        trainloader = TorchData.DataLoader(dataset=trainset, batch_size=batch_size,
                                      shuffle=True, num_workers=0, drop_last=True)
        testset = TorchData.TensorDataset(test_X, test_y)
        testloader = TorchData.DataLoader(dataset=testset, batch_size=batch_size,
                                     shuffle=False, num_workers=0, drop_last=True)


        base_net = CNNBaseNet(input_dim=num_modalities,
                              output_channels=parameters['base_net_out_channel'],
                              output_dim=parameters['base_net_output_dim'],
                              num_time_steps=num_time_steps)

        torch.manual_seed(model_shared_seed)
        population_encoder = PopulationEncoder(input_dim=parameters['base_net_output_dim'],
                                               hidden_1=128, output_dim=parameters['population_output_dim'], train_on_gpu=True)
        torch.manual_seed(model_shared_seed)
        personalized_encoder = PersonalizedEncoder(input_dim=parameters['base_net_output_dim'],
                                               hidden_1=128, output_dim=parameters['personalized_output_dim'], train_on_gpu=True)

        reconstruct_net = ReconstructNet(input_dim=parameters['personalized_output_dim'] + parameters['population_output_dim'],
                                         hidden_1=128, output_dim=num_time_steps*num_modalities, train_on_gpu=True)

        # todo: add MINE, add negative and positive representations
        # mine = Mine(x_dim=parameters_dict['embedding_dim'], z_dim=parameters_dict['embedding_dim'],
        #             hidden_dim=parameters_dict['embedding_dim'] // 2)

        all_models = [base_net, population_encoder, personalized_encoder, reconstruct_net]
        all_models_names = ['base_net', 'population_encoder', 'personalized_encoder', 'reconstruct_net']

        # put the models in CUDA
        for m in range(len(all_models_names)):
            model = all_models[m]
            if (model.train_on_gpu):
                model.cuda()

        # losses
        mse_loss = nn.MSELoss(reduction='mean')
        # optimizer
        optimizer = torch.optim.Adam(list(base_net.parameters()) + list(population_encoder.parameters())
                                     + list(personalized_encoder.parameters()) + list(reconstruct_net.parameters()),
                                     lr=learning_rate, weight_decay=weight_decay)


        # for epoch in range(num_epochs):
        #     # enable train mode
        #     for model in all_models:
        #         model.train()
        #     all_batch_losses_train = train_one_epoch(trainloader, optimizer, fuse_net, positive_r_model,
        #                                              negative_r_model, mine,
        #                                              mse_loss, contrastive_loss_criterion, mutual_information,
        #                                              mode='train')
        #     # add losses
        #     for loss_key in list(all_batch_losses_train.keys()):
        #         losses_train[loss_key].append(np.mean(np.array(all_batch_losses_train[loss_key])))
        #
        #     # enable test mode
        #     for model in all_models:
        #         model.eval()
        #     all_batch_losses_test = train_one_epoch(testloader, optimizer, fuse_net, positive_r_model, negative_r_model,
        #                                             mine,
        #                                             mse_loss, contrastive_loss_criterion, mutual_information,
        #                                             mode='test')
        #     # add losses
        #     for loss_key in list(all_batch_losses_test.keys()):
        #         losses_test[loss_key].append(np.mean(np.array(all_batch_losses_test[loss_key])))
        #
        #     epoch_train_loss = np.mean(np.array(all_batch_losses_train['total_loss']))
        #     epoch_test_loss = np.mean(np.array(all_batch_losses_test['total_loss']))
        #
        #     epoch_c_common_n_train = np.mean(np.array(all_batch_losses_train['c_loss_common_n']))
        #     epoch_c_unique_p_train = np.mean(np.array(all_batch_losses_train['c_loss_unique_p']))
        #     epoch_c_common_n_test = np.mean(np.array(all_batch_losses_test['c_loss_common_n']))
        #     epoch_c_unique_p_test = np.mean(np.array(all_batch_losses_test['c_loss_unique_p']))
        #     print(f'Epoch {epoch}: Train loss: {epoch_train_loss} Test: {epoch_test_loss}')
        #     print(f'Epoch {epoch}: Train C-common-n: {epoch_c_common_n_train} Test: {epoch_c_common_n_test}')
        #     print(f'Epoch {epoch}: Train C-unique-p: {epoch_c_unique_p_train} Test: {epoch_c_unique_p_test}')
        #     print()
        #
        # timestring = strftime("%Y-%m-%d_%H-%M-%S", localtime()) + "_Subject%s" % str(
        #     target_subject)
        # save_dir = os.path.join(BASE_DIR, 'CommonInformation', 'FuseNet', data_name, timestring)
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # for m in range(len(all_models)):
        #     model = all_models[m]
        #     model_name = all_models_names[m]
        #     # save state dict
        #     torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_stateDict.pth"))
        #     # save model
        #     torch.save(model, os.path.join(save_dir, f"{model_name}.pth"))
        # for m in range(len(list(losses_train.keys()))):
        #     key = list(losses_train.keys())[m]
        #     loss_train = losses_train[key]
        #     loss_test = losses_test[key]
        #     # save training result
        #     plot_loss_curves(loss_train, loss_test, save_loc=save_dir, show_fig=False,
        #                      title=key)
        #
        # with open(os.path.join(save_dir, 'parameters.txt'), 'w') as f:
        #     print(parameters_dict, file=f)
        # print("END!")