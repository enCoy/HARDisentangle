from Utils.HelperFunctions import convert_to_torch
from Utils.Visualizers import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from time import localtime, strftime
from DataProcesses import DownstreamDataProcessor
from Models.DisentangleNetModels import CNNBaseNet, PopulationEncoder, PersonalizedEncoder
from Models.ClassifierNet import ClassifierNet
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch import optim
import sklearn.metrics as metrics
import numpy as np
import os
import re
import json
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

def read_params_dict(analysis_dir):
    parameters = {}
    with open(os.path.join(analysis_dir, 'parameters.txt'), 'r') as file:
        for line in file:
            match = re.match(r'^([^:]+):\s*(.*)$', line)
            if match:
                key = match.group(1)
                value = match.group(2)
                # If the value is a dictionary, evaluate it as a dictionary
                if value.startswith('{') and value.endswith('}'):
                    value = eval(value)
                else:
                    # Convert numeric values to int or float if possible
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # Value remains as string if not convertible to int or float
                parameters[key] = value
    # Print the loaded parameter dictionary
    return parameters


if __name__ == "__main__":
    machine = 'windows'
    if machine == 'linux':
        base_dir = r'/home/cmyldz/Dropbox (GaTech)/DisentangledHAR/'
    else:
        base_dir = r'C:\Users\Cem Okan\Dropbox (GaTech)\DisentangledHAR/'
    analysis_dir = os.path.join(base_dir, r"Logging", r"Deneme")
    target_subject = 1

    # Load parameter dictionary from the text file
    parameters = read_params_dict(analysis_dir)
    print("parameters: ", parameters)
    analysis_encoder = 'population'

    if parameters['data_name'] == 'pamap2':
        data_dir = os.path.join(base_dir, r"PAMAP2_Dataset\PAMAP2_Dataset\Processed50Hz")
    else:
        data_dir = os.path.join(base_dir, r"realworld2016_dataset\Processed")

    # get the data loader
    data_processor = DownstreamDataProcessor(data_dir=data_dir,
                                   data_name=parameters['data_name'],
                                   target_subject_num=target_subject,
                                   num_subjects=parameters['num_subjects'],
                                   num_activities=parameters['num_activities'],
                                   window_size=parameters['window_size'],
                                   num_modalities=parameters['num_modalities'],
                                   sampling_rate=parameters['sampling_rate'],
                                   sliding_window_overlap_ratio=parameters['sliding_window_overlap_ratio'])
    # initial idea - look at the score of each modality alone % freeze earlier layers
    train_X, train_y, test_X, test_y = data_processor.get_train_test_subjects_data()
    train_X, train_y, test_X, test_y = convert_to_torch(train_X, train_y, test_X, test_y)

    trainset = Data.TensorDataset(train_X, train_y)
    trainloader = Data.DataLoader(dataset=trainset, batch_size=36,
                                  shuffle=True, num_workers=0, drop_last=True)
    testset = Data.TensorDataset(test_X, test_y)
    testloader = Data.DataLoader(dataset=testset, batch_size=36,
                                 shuffle=False, num_workers=0, drop_last=True)

    model_dir = os.path.join(analysis_dir, f"S{target_subject}")
    # load models etc.
    base_net = CNNBaseNet(input_dim=parameters['num_modalities'],
                          output_channels=parameters['base_net_out_channel'],
                          output_dim=parameters['base_net_output_dim'],
                          num_time_steps=parameters['window_size'])
    population_encoder = PopulationEncoder(input_dim=parameters['base_net_output_dim'],
                                           hidden_1=256, output_dim=parameters['population_output_dim'],
                                           train_on_gpu=True)
    personalized_encoder = PersonalizedEncoder(input_dim=parameters['base_net_output_dim'],
                                       hidden_1=256, output_dim=parameters['personalized_output_dim'],
                                       train_on_gpu=True)
    # freeze these two
    for param in base_net.parameters():
        # print(f"Number less threshold {torch.sum(torch.abs(param) < 0.00001)} out of {torch.numel(param)}")
        param.requires_grad = False
    for param in population_encoder.parameters():
        # print(f"Number less threshold {torch.sum(torch.abs(param) < 0.00001)} out of {torch.numel(param)}")
        param.requires_grad = False
    for param in personalized_encoder.parameters():
        # print(f"Number less threshold {torch.sum(torch.abs(param) < 0.00001)} out of {torch.numel(param)}")
        param.requires_grad = False


    base_net.load_state_dict(torch.load(os.path.join(model_dir, 'best_base_net_stateDict.pth')))
    population_encoder.load_state_dict(torch.load(os.path.join(model_dir, 'best_population_encoder_stateDict.pth')))
    personalized_encoder.load_state_dict(torch.load(os.path.join(model_dir, 'best_personalized_encoder_stateDict.pth')))

    all_models = {
        'base_net': base_net,
        'population_encoder': population_encoder,
        'personalized_encoder': personalized_encoder
    }
    all_models_names = list(all_models.keys())
    # put the models in CUDA
    for m in range(len(all_models_names)):
        model = all_models[all_models_names[m]]
        if (model.train_on_gpu):
            model.cuda()
    X_embedding_train = np.empty((0, parameters['population_output_dim']))
    activities_train = []
    subjects_train = data_processor.train_subjects_list
    X_embedding_test = np.empty((0, parameters['population_output_dim']))
    activities_test = []
    subjects_test = data_processor.test_subjects_list

    all_models = {
        'base_net': base_net,
        'population_encoder': population_encoder,
        'personalized_encoder': personalized_encoder}
    all_models_names = list(all_models.keys())
    # put the models in CUDA
    for m in range(len(all_models_names)):
        model = all_models[all_models_names[m]]
        if (model.train_on_gpu):
            model.cuda()

    with torch.no_grad():
        for inputs, labels in trainloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            # Forward pass
            base_net_repr = base_net(inputs)

            if analysis_encoder == 'population':
                repr = population_encoder(base_net_repr)
            else:  # analysis_encoder == "personalized":
                repr = personalized_encoder(base_net_repr)
            labels = torch.argmax(labels, dim=1)
            # append repr, activities and subjects
            X_embedding_train = np.vstack((X_embedding_train, repr.detach().cpu().numpy()))
            activities_train.extend(list(labels.detach().cpu().numpy()))

        for inputs, labels in testloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            # Forward pass
            base_net_repr = base_net(inputs)
            if analysis_encoder == 'population':
                repr = population_encoder(base_net_repr)
            else:  # analysis_encoder == "personalized":
                repr = personalized_encoder(base_net_repr)

            labels = torch.argmax(labels, dim=1)
            # append repr, activities and subjects
            X_embedding_test = np.vstack((X_embedding_test, repr.detach().cpu().numpy()))
            activities_test.extend(list(labels.detach().cpu().numpy()))

        y_test = np.array(activities_test)
        y_train = np.array(activities_train)
        print("X train shape: ", X_embedding_train.shape)
        print("X test shape: ", X_embedding_test.shape)
        print("y train shape: ", y_train.shape)
        print("y test shape: ", y_test.shape)

        # clf = LinearSVC(dual="auto", random_state=0, tol=1e-5)
        clf = RandomForestClassifier(n_estimators=500,max_depth=6, random_state=0)
        # clf = KNeighborsClassifier(n_neighbors=3)
        # clf = GradientBoostingClassifier(n_estimators=200)
        clf.fit(X_embedding_train, y_train)
        y_pred_train = clf.predict(X_embedding_train)
        y_pred_test = clf.predict(X_embedding_test)


        # Calculate accuracy for training set
        train_accuracy = accuracy_score(y_train, y_pred_train)
        # Calculate F1 score for training set
        train_f1 = f1_score(y_train, y_pred_train, average='macro')

        # Calculate accuracy for training set
        test_accuracy = accuracy_score(y_test, y_pred_test)
        # Calculate F1 score for training set
        test_f1 = f1_score(y_test, y_pred_test, average='macro')

        # Print statistics
        print(f"Train Accuracy: {100 * train_accuracy:.2f}%,  Train F1: {train_f1:.4f},"
              f"Test Accuracy: {100 * test_accuracy:.2f}%,  Test F1: {test_f1:.4f},")

        plot_confusion_matrix(confusion_matrix(y_test, y_pred_test),
                              data_processor.get_activity_names(),
                              title='Confusion matrix - Test',
                              cmap=None,
                              normalize=False,
                              save_path=None)
        plot_confusion_matrix(confusion_matrix(y_train, y_pred_train),
                              data_processor.get_activity_names(),
                              title='Confusion matrix - Train',
                              cmap=None,
                              normalize=False,
                              save_path=None)











