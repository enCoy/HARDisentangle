from CommonInformation.Models.Classifier import ClassifierNet, train_one_epoch
from CommonInformation.Models.CommonNetModels import CNNBaseNet, CommonNet
from CommonInformation.Main import get_parameters
from CommonInformation.DataProcesses import DataProcessor
from Utils.HelperFunctions import convert_to_torch
import torch.utils.data as Data
import torch
import numpy as np

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
    train_X, train_y, test_X, test_y = data_processor.get_modality_separated_train_test_classification_data(data_processor.data_dict,
                                                                                        data_processor.modalities)
    print("train y: ", train_y)
    train_X, train_y, test_X, test_y = convert_to_torch(train_X, train_y, test_X, test_y)

    trainset = Data.TensorDataset(train_X, train_y)
    trainloader = Data.DataLoader(dataset=trainset, batch_size=parameters_dict['batch_size'],
                                  shuffle=True, num_workers=0, drop_last=True)
    testset = Data.TensorDataset(test_X, test_y)
    testloader = Data.DataLoader(dataset=testset, batch_size=parameters_dict['batch_size'],
                                 shuffle=False, num_workers=0, drop_last=True)


    model_dir = r"C:\Users\Cem Okan\Dropbox (GaTech)\DisentangledHAR\CommonInformation\FuseNet\pamap2\2023-11-01_11-59-54_Subject1"
    base_net = CNNBaseNet(input_dim=input_size, output_channels=64, embedding=128,
                       num_time_steps=parameters_dict['window_size'])
    common_net = CommonNet(128, 256, parameters_dict['embedding_dim'])
    classification_net = ClassifierNet(common_rep_dim=parameters_dict['embedding_dim'], hidden_1=256, output_dim=parameters_dict['num_activities'])

    # freeze these two
    for param in base_net.parameters():
        param.requires_grad = False
    for param in common_net.parameters():
        param.requires_grad = False

    all_models = [base_net, common_net, classification_net]
    all_models_names = ['base_net', 'common_net', 'classification_net']
    for m in range(len(all_models)):
        model = all_models[m]
        model_name = all_models_names[m]
        if (model.train_on_gpu):
            model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(classification_net.parameters()),
                                 lr=parameters_dict['lr'], weight_decay=parameters_dict['weight_decay'])

    epoch_losses_train = []
    epoch_losses_test = []

    for epoch in range(parameters_dict['num_epochs']):
        #enable train mode
        for model in all_models:
            model.train()
        batch_losses = train_one_epoch(trainloader, optimizer, base_net, common_net, classification_net, criterion, mode='train')
        # add losses
        epoch_losses_train.append(np.mean(batch_losses))

        # enable test mode
        for model in all_models:
            model.eval()
        batch_losses = train_one_epoch(trainloader, optimizer, base_net, common_net, classification_net, criterion, mode='test')
        # add losses
        epoch_losses_test.append(np.mean(batch_losses))

        print(f'Epoch {epoch}: Train loss: {epoch_losses_train[-1]} Test: {epoch_losses_test[-1]}')












