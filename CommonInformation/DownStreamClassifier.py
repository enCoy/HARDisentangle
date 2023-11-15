from CommonInformation.Models.Classifier import ClassifierNet, train_one_epoch
from CommonInformation.Models.CommonNetModels import CNNBaseNet, CommonNet
from CommonInformation.Main import get_parameters
from CommonInformation.DataProcesses import DataProcessor
from Utils.HelperFunctions import convert_to_torch
from time import localtime, strftime
import torch.utils.data as Data
import torch
import sklearn.metrics as metrics
import numpy as np
import os
from CommonInformation.Main import BASE_DIR

if __name__ == "__main__":

    machine = 'windows'
    if machine == 'linux':
        base_dir = r'/home/cmyldz/Dropbox (GaTech)/DisentangledHAR/'
    else:
        base_dir = r'C:\Users\Cem Okan\Dropbox (GaTech)\DisentangledHAR/'

    data_name = 'pamap2'  # 'pamap2' or 'real'
    target_subject = 3
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
    train_X, train_y, test_X, test_y = convert_to_torch(train_X, train_y, test_X, test_y)

    trainset = Data.TensorDataset(train_X, train_y)
    trainloader = Data.DataLoader(dataset=trainset, batch_size=parameters_dict['batch_size'],
                                  shuffle=True, num_workers=0, drop_last=True)
    testset = Data.TensorDataset(test_X, test_y)
    testloader = Data.DataLoader(dataset=testset, batch_size=parameters_dict['batch_size'],
                                 shuffle=False, num_workers=0, drop_last=True)

    model_dir = os.path.join(BASE_DIR, r"CommonInformation\FuseNet\pamap2\2023-11-05_23-11-28_Subject3")
    base_net = CNNBaseNet(input_dim=input_size, output_channels=128, embedding=1024,
                       num_time_steps=parameters_dict['window_size'])
    common_net = CommonNet(1024, 1024, parameters_dict['embedding_dim'])
    classification_net = ClassifierNet(common_rep_dim=parameters_dict['embedding_dim'], hidden_1=1024, output_dim=parameters_dict['num_activities'])

    base_net.load_state_dict(torch.load(os.path.join(model_dir, 'base_net_stateDict.pth')))
    common_net.load_state_dict(torch.load(os.path.join(model_dir, 'common_net_stateDict.pth')))


    # freeze these two
    for param in base_net.parameters():
        print(f"Number less threshold {torch.sum(torch.abs(param) < 0.00001)} out of {torch.numel(param)}")
        param.requires_grad = False
    for param in common_net.parameters():
        print(f"Number less threshold {torch.sum(torch.abs(param) < 0.00001)} out of {torch.numel(param)}")
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
    test_results = []
    train_results = []
    # save configuration
    timestring = strftime("%Y-%m-%d_%H-%M-%S", localtime()) + "_Subject%s" % str(
        target_subject)
    save_dir = os.path.join(BASE_DIR, 'CommonInformation', 'ClassifierNet', data_name, timestring)
    train_result_csv = os.path.join(save_dir, 'train_classification_results.csv')
    test_result_csv = os.path.join(save_dir, 'test_classification_results.csv')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    for epoch in range(parameters_dict['num_epochs']):
        #enable train mode
        for model in all_models:
            model.train()
        batch_losses, all_ground_truth_train, all_predictions_train = train_one_epoch(trainloader, optimizer,
                                                                          base_net, common_net, classification_net, criterion, mode='train')
        # add losses
        epoch_losses_train.append(np.mean(batch_losses))
        # performance metrics
        f1_train = metrics.f1_score(all_ground_truth_train, all_predictions_train, average='macro')
        acc_train = metrics.accuracy_score(all_ground_truth_train, all_predictions_train)
        train_results.append([acc_train, f1_train, epoch+1])
        result_np = np.array(train_results, dtype=float)
        np.savetxt(train_result_csv, result_np, fmt='%.4f', delimiter=',')

        # enable test mode
        for model in all_models:
            model.eval()
        batch_losses, all_ground_truth_test, all_predictions_test = train_one_epoch(trainloader, optimizer,
                                                                          base_net, common_net, classification_net, criterion, mode='test')
        # add losses
        epoch_losses_test.append(np.mean(batch_losses))
        # performance metrics
        f1_test = metrics.f1_score(all_ground_truth_test, all_predictions_test, average='macro')
        acc_test = metrics.accuracy_score(all_ground_truth_test, all_predictions_test)
        test_results.append([acc_test, f1_test, epoch + 1])
        result_np = np.array(test_results, dtype=float)
        np.savetxt(test_result_csv, result_np, fmt='%.4f', delimiter=',')

        print(f'Epoch {epoch}: Train loss: {epoch_losses_train[-1]} Test: {epoch_losses_test[-1]}')
        print(f"Train f1: {f1_train},  Train acc: {acc_train}")
        print(f"Test f1: {f1_test}, test_acc:{acc_test}")
        print()












