from CommonInformation.Models.Classifier import ClassifierNet, train_one_epoch
from CommonInformation.Models.CommonNetV3 import CNNBaseNet, CommonNet
from CommonInformation.MainV3 import get_parameters, enable_mode
from CommonInformation.DataProcesses import DataProcessor
from Utils.HelperFunctions import convert_to_torch_v2
from time import localtime, strftime
import torch.utils.data as Data
import torch
import sklearn.metrics as metrics
import numpy as np
import os
from CommonInformation.Main import BASE_DIR

def fuse_predictions_from_multiple_modalities(predictions, modalities, num_classes):
    # fuse predictions - take the majority
    fused_predictions = []
    for j in range(len(predictions[modalities[0]])):  # over all examples
        classes_predicted = np.zeros(num_classes)
        modality_predictions = np.array([predictions[m][j] for m in modalities])
        # print("modality predictions: ", modality_predictions)
        for kk in range(len(modality_predictions)):
            classes_predicted[modality_predictions[kk]] += 1
        # print("class predicted: ", classes_predicted)
        fused_predictions.append(np.argmax(classes_predicted))
    return np.array(fused_predictions)

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
    train_X, train_y, test_X, test_y = data_processor.get_modality_separated_train_test_classification_data_concatted_behind(data_processor.data_dict,
                                                                                        data_processor.modalities)

    train_X, train_y, test_X, test_y = convert_to_torch_v2(train_X, train_y, test_X, test_y)



    trainset = Data.TensorDataset(train_X, train_y)
    trainloader = Data.DataLoader(dataset=trainset, batch_size=parameters_dict['batch_size'],
                                  shuffle=True, num_workers=0, drop_last=True)
    testset = Data.TensorDataset(test_X, test_y)
    testloader = Data.DataLoader(dataset=testset, batch_size=parameters_dict['batch_size'],
                                 shuffle=False, num_workers=0, drop_last=True)

    num_modalities = parameters_dict['num_modalities']
    modalities = data_processor.modalities
    # create as many models as modalities
    model_dir = os.path.join(BASE_DIR, r"CommonInformation\FuseNet\pamap2\2023-11-13_23-43-16_Subject3")
    models = {}
    for modality in modalities:
        models[modality] = {}
    for modality in modalities:
        models[modality]['base'] = CNNBaseNet(input_dim=input_size, output_channels=128, embedding=512,
                       num_time_steps=parameters_dict['window_size'])
        models[modality]['common'] =  CommonNet(512, 256, parameters_dict['embedding_dim'])
        models[modality]['classification'] = ClassifierNet(common_rep_dim=parameters_dict['embedding_dim'],
                                                           hidden_1=1024, output_dim=parameters_dict['num_activities'])

        # load trained models
        # models[modality]['base'].load_state_dict(torch.load(os.path.join(model_dir, 'glob_base_stateDict.pth')))
        # models[modality]['common'].load_state_dict(torch.load(os.path.join(model_dir, 'glob_common_stateDict.pth')))


        # # freeze these two if you wish
        # for param in models[modality]['base'].parameters():
        #     print(f"Number less threshold {torch.sum(torch.abs(param) < 0.00001)} out of {torch.numel(param)}")
        #     print(f"Number above threshold {torch.sum(torch.abs(param) > 10)} out of {torch.numel(param)}")
        #     param.requires_grad = False
        # for param in models[modality]['common'].parameters():
        #     print(f"Number less threshold {torch.sum(torch.abs(param) < 0.00001)} out of {torch.numel(param)}")
        #     print(f"Number above threshold {torch.sum(torch.abs(param) > 10)} out of {torch.numel(param)}")
        #     param.requires_grad = False

    models_to_push = ['base', 'common', 'classification']
    for modality in modalities:
        for model_name in models_to_push:
            if models[modality][model_name].train_on_gpu:
                models[modality][model_name].cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizers = {}
    for modality in modalities:
        optimizers[modality] = torch.optim.Adam(list(models[modality]['classification'].parameters()) +
                                 list(models[modality]['common'].parameters()) +
                                 list(models[modality]['base'].parameters()),
                                 lr=parameters_dict['lr'], weight_decay=parameters_dict['weight_decay'])


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
        for modality in modalities:
            models[modality] = enable_mode(models[modality], mode='train')
        batch_losses, all_ground_truth_train, all_predictions_train = train_one_epoch(trainloader, optimizers,
                                                                          models, criterion, modalities, mode='train')
        fused_predictions = fuse_predictions_from_multiple_modalities(all_predictions_train, modalities, parameters_dict['num_activities'])
        f1_train = metrics.f1_score(all_ground_truth_train[modalities[0]], fused_predictions, average='macro')
        acc_train = metrics.accuracy_score(all_ground_truth_train[modalities[0]], fused_predictions)
        train_results.append([acc_train, f1_train, epoch+1])
        result_np = np.array(train_results, dtype=float)
        np.savetxt(train_result_csv, result_np, fmt='%.4f', delimiter=',')

        # enable test mode
        for modality in modalities:
            models[modality] = enable_mode(models[modality], mode='test')
        batch_losses, all_ground_truth_test, all_predictions_test = train_one_epoch(testloader, optimizers,
                                                                          models, criterion, modalities, mode='test')
        fused_predictions = fuse_predictions_from_multiple_modalities(all_predictions_test, modalities, parameters_dict['num_activities'])

        # performance metrics
        print("fused predictions unique: ", np.unique(fused_predictions))
        print("gt unique: ", np.unique(all_ground_truth_test[modalities[0]]))
        f1_test = metrics.f1_score(all_ground_truth_test[modalities[0]], fused_predictions, average='macro')
        acc_test = metrics.accuracy_score(all_ground_truth_test[modalities[0]], fused_predictions)
        test_results.append([acc_test, f1_test, epoch + 1])
        result_np = np.array(test_results, dtype=float)
        np.savetxt(test_result_csv, result_np, fmt='%.4f', delimiter=',')
        print("Epoch ", epoch )
        print(f"Train f1: {f1_train},  Train acc: {acc_train}")
        print(f"Test f1: {f1_test}, test_acc:{acc_test}")
        print()












