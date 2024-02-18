from Utils.HelperFunctions import convert_to_torch
from Utils.Visualizers import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from time import localtime, strftime
from DataProcesses import DownstreamDataProcessor
from Utils.HelperFunctions import plot_loss_curves

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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches


def get_activity_names():
    return ['lying', 'sitting', 'standing', 'walking', 'running', 'cycling',
                'nordic walking', 'ascending stairs', 'descending stairs', 'vacuum cleaning',
                'ironing', 'rope jumping']

def get_custom_legend_patches(colors, labels):
    patches = []
    for j in range(len(colors)):
        patches.append(mpatches.Patch(color=colors[j], label=labels[j]))
    return patches

def get_TSNE_plot_error_amplified(X, is_error_data, activities, subjects, title, save_dir=None):
    # Define colors and markers for subjects and activities
    # errors is a binary array - if it is 1, prediction is correct, if it is 0 it is wrong

    activity_colors = ['indianred', 'royalblue', 'seagreen', 'coral', 'darkmagenta', 'gold', 'cyan', 'orchid', 'burlywood', 'pink', 'chocolate', 'gray']
    subject_markers = ['o', 's', '^', 'v', 'D', 'P', '*', '+']
    tsne = TSNE(n_components=2, random_state=42)
    X_embedding = tsne.fit_transform(X)
    # Plot the t-SNE embedded data with different colors for each subject and markers for each activity
    plt.figure(figsize=(12, 8))
    for activity_id in np.unique(activities):
        for subject_id, marker in zip(np.unique(subjects), subject_markers):

            mask_correct = (subjects == subject_id) & (activities == activity_id) & (np.invert(is_error_data))
            mask_wrong = (subjects == subject_id) & (activities == activity_id) & (is_error_data)

            plt.scatter(X_embedding[mask_correct, 0], X_embedding[mask_correct, 1], label=f'Subject {subject_id}, Activity {activity_id}',
                        color=activity_colors[activity_id - 1], marker=marker, alpha=0.25, s = 15)
            plt.scatter(X_embedding[mask_wrong, 0], X_embedding[mask_wrong, 1],
                        label=f'Subject {subject_id}, Activity {activity_id}',
                        color=activity_colors[activity_id - 1], marker=marker, alpha=0.5, s = 45)

    plt.title(f't-SNE Visualization with Subject and Activity Differentiation - {title}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    patches = get_custom_legend_patches(colors=activity_colors, labels=get_activity_names())
    plt.legend(handles=patches, title="Activities")

    # plt.legend()
    if save_dir is not None:
        image_format = 'png'  # e.g .png, .svg, etc
        plt.savefig(os.path.join(save_dir, title + '.png'), format=image_format, dpi=1200)
    # plt.show()


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
    include_personalized = False

    # Load parameter dictionary from the text file
    parameters = read_params_dict(analysis_dir)
    print("parameters: ", parameters)

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
    trainloader = Data.DataLoader(dataset=trainset, batch_size=64,
                                  shuffle=True, num_workers=0, drop_last=True)
    testset = Data.TensorDataset(test_X, test_y)
    testloader = Data.DataLoader(dataset=testset, batch_size=64,
                                 shuffle=False, num_workers=0, drop_last=True)

    model_dir = os.path.join(analysis_dir, f"S{target_subject}")
    # load models etc.
    base_net = CNNBaseNet(input_dim=parameters['num_modalities'],
                          output_channels=parameters['base_net_out_channel'],
                          output_dim=parameters['base_net_output_dim'],
                          num_time_steps=parameters['window_size'])
    population_encoder = PopulationEncoder(input_dim=parameters['base_net_output_dim'],
                                           hidden_1=256, output_dim=parameters['population_output_dim'], # was 128
                                           train_on_gpu=True)
    personalized_encoder = PersonalizedEncoder(input_dim=parameters['base_net_output_dim'],
                                       hidden_1=256, output_dim=parameters['personalized_output_dim'], # was 128
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


    if include_personalized:
        classification_net = ClassifierNet(input_dim=parameters['population_output_dim'] + parameters['personalized_output_dim'],
                                           hidden_1=128,
                                           output_dim=parameters['num_activities'])
    else:
        classification_net = ClassifierNet(input_dim=parameters['population_output_dim'],
                                       hidden_1=128,
                                       output_dim=parameters['num_activities'])

    all_models = {
        'base_net': base_net,
        'population_encoder': population_encoder,
        'personalized_encoder':personalized_encoder,
        'classifier': classification_net,

    }

    all_models_names = list(all_models.keys())
    # put the models in CUDA
    for m in range(len(all_models_names)):
        model = all_models[all_models_names[m]]
        if (model.train_on_gpu):
            model.cuda()

    # Define your loss function
    criterion = nn.CrossEntropyLoss()
    # Define your optimizer
    optimizer = optim.SGD((list(classification_net.parameters()) + list(base_net.parameters()) + list(population_encoder.parameters())),
                                 lr=0.00001, weight_decay=1e-4, momentum=0.9)
    # Define number of epochs
    num_epochs = 100
    best_train_preds = None
    best_train_gt = None
    best_test_preds = None
    best_test_gt = None
    best_classifier_state_dict = classification_net.state_dict()
    # todo: note that you are not storing best other models since you assume that population and personalized encoders are fixed
    best_test_f1_score = -np.inf
    # Training loop
    train_epoch_losses = []
    test_epoch_losses = []
    train_epoch_acc = []
    test_epoch_acc = []
    train_epoch_f1 = []
    test_epoch_f1 = []
    for epoch in range(num_epochs):
        # Training
        for model in list(all_models.values()):
            model.train()
        running_loss = 0.0
        y_true_train = []
        y_pred_train = []
        for inputs, labels in trainloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            base_net_repr = base_net(inputs)
            pop_net_repr = population_encoder(base_net_repr)
            if include_personalized:
                per_net_repr = personalized_encoder(base_net_repr)
                concatted = torch.concat((per_net_repr, pop_net_repr), 1)
                outputs = classification_net(concatted)
            else:
                outputs = classification_net(pop_net_repr)
            # Compute the loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Optimize
            optimizer.step()
            # Print statistics
            running_loss += loss.item()
            # Calculate training accuracy
            _, predicted_train = torch.max(outputs, 1)

            labels = torch.argmax(labels, dim=1)

            # Collect true and predicted labels for F1 score calculation
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(predicted_train.cpu().numpy())
        train_loss = running_loss / len(trainloader)
        train_epoch_losses.append(train_loss)
        # Calculate accuracy for training set
        train_accuracy = accuracy_score(y_true_train, y_pred_train)
        train_epoch_acc.append(train_accuracy)
        # Calculate F1 score for training set
        train_f1 = f1_score(y_true_train, y_pred_train, average='macro')
        train_epoch_f1.append(train_f1)


        # Validation
        y_true_test = []
        y_pred_test = []
        for model in list(all_models.values()):
            model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in testloader:
                base_net_repr = base_net(inputs)
                pop_net_repr = population_encoder(base_net_repr)
                if include_personalized:
                    per_net_repr = personalized_encoder(base_net_repr)
                    concatted = torch.concat((per_net_repr, pop_net_repr), 1)
                    outputs = classification_net(concatted)
                else:
                    outputs = classification_net(pop_net_repr)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()
                # Calculate validation accuracy
                _, predicted_val = torch.max(outputs, 1)
                labels = torch.argmax(labels, dim=1)

                # Collect true and predicted labels for F1 score calculation
                y_true_test.extend(labels.cpu().numpy())
                y_pred_test.extend(predicted_val.cpu().numpy())

        val_loss = running_val_loss / len(testloader)
        test_epoch_losses.append(val_loss)

        # Calculate accuracy for training set
        test_accuracy = accuracy_score(y_true_test, y_pred_test)
        test_epoch_acc.append(test_accuracy)
        # Calculate F1 score for training set
        test_f1 = f1_score(y_true_test, y_pred_test, average='macro')
        test_epoch_f1.append(test_f1)
        if test_f1 > best_test_f1_score:
            print("Best model is updated...")
            best_test_f1_score = test_f1
            best_classifier_state_dict = classification_net.state_dict()
            best_test_gt = y_true_test
            best_test_preds = y_pred_test
            best_train_gt = y_true_train
            best_train_preds = y_pred_train

        # Print statistics
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {100 * train_accuracy:.2f}%,  Train F1: {train_f1:.4f},"
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {100 * test_accuracy:.2f}%,  Test F1: {test_f1:.4f},")
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")
    print('Finished Training')
    plot_loss_curves(train_epoch_losses, test_epoch_losses,
                     save_loc=os.path.join(analysis_dir, f'S{target_subject}'), show_fig=False, title='Classification Loss Curves')
    plot_loss_curves(train_epoch_acc, test_epoch_acc,
                     save_loc=os.path.join(analysis_dir, f'S{target_subject}'), show_fig=False,
                     title='Classification Accuracy Curves')
    plot_loss_curves(train_epoch_f1, test_epoch_f1,
                     save_loc=os.path.join(analysis_dir, f'S{target_subject}'), show_fig=False,
                     title='Classification F1 Score Curves')

    # test_cm = confusion_matrix(best_test_gt, best_test_preds)
    # np.savetxt(os.path.join(save_dir, 'confusion.txt'), test_cm, fmt='%d')
    plot_confusion_matrix(confusion_matrix(best_test_gt, best_test_preds),
                          data_processor.get_activity_names(),
                          title='Confusion matrix - Test',
                          cmap=None,
                          normalize=False,
                          save_path=os.path.join(analysis_dir, f'S{target_subject}', 'test confusion mat.png'))
    plot_confusion_matrix(confusion_matrix(best_train_gt, best_train_preds),
                          data_processor.get_activity_names(),
                          title='Confusion matrix - Train',
                          cmap=None,
                          normalize=False,
                          save_path=os.path.join(analysis_dir, f'S{target_subject}', 'train confusion mat.png'))

    classification_net.load_state_dict(best_classifier_state_dict)
    analysis_encoders = ['population', 'personalized']
    tsne_save_dir = os.path.join(analysis_dir, f'S{target_subject}', "TSNEPlotsErrorAmplified")
    if not os.path.exists(tsne_save_dir):
        os.makedirs(tsne_save_dir)
    # put the models in CUDA
    for m in range(len(all_models_names)):
        model = all_models[all_models_names[m]]
        if (model.train_on_gpu):
            model.cuda()

    is_error_data = []  # this only looks at population encoder out
    with torch.no_grad():
        for analysis_encoder in analysis_encoders:
            X_embedding_train = np.empty((0, parameters['population_output_dim']))
            activities_train = []
            subjects_train = data_processor.train_subjects_list
            X_embedding_test = np.empty((0, parameters['population_output_dim']))
            activities_test = []
            subjects_test = data_processor.test_subjects_list

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

                outputs = classification_net(repr)
                _, predicted_val = torch.max(outputs, 1)
                is_error = predicted_val != labels
                if analysis_encoder == 'population':
                    is_error_data.extend(list(is_error.detach().cpu().numpy()))

            if analysis_encoder == 'population':
                get_TSNE_plot_error_amplified(X_embedding_test, is_error_data, np.array(activities_test), np.array(subjects_test)[:len(is_error_data)],
                                 f'test - {analysis_encoder}',
                                 save_dir=os.path.join(analysis_dir, f'S{target_subject}', "TSNEPlotsErrorAmplified"))
            else:
                get_TSNE_plot_error_amplified(X_embedding_test, is_error_data, np.array(activities_test), np.array(subjects_test)[:len(is_error_data)],
                                 f'test - {analysis_encoder}',
                                 save_dir=os.path.join(analysis_dir, f'S{target_subject}', "TSNEPlotsErrorAmplified"))
        plt.show()

    #
    # all_models = [base_net, common_net, classification_net]
    # all_models_names = ['base_net', 'common_net', 'classification_net']
    # for m in range(len(all_models)):
    #     model = all_models[m]
    #     model_name = all_models_names[m]
    #     if (model.train_on_gpu):
    #         model.cuda()
    #
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(list(classification_net.parameters()),
    #                              lr=parameters_dict['lr'], weight_decay=parameters_dict['weight_decay'])
    #
    # epoch_losses_train = []
    # epoch_losses_test = []
    # test_results = []
    # train_results = []
    # # save configuration
    # timestring = strftime("%Y-%m-%d_%H-%M-%S", localtime()) + "_Subject%s" % str(
    #     target_subject)
    # save_dir = os.path.join(BASE_DIR, 'CommonInformation', 'ClassifierNet', data_name, timestring)
    # train_result_csv = os.path.join(save_dir, 'train_classification_results.csv')
    # test_result_csv = os.path.join(save_dir, 'test_classification_results.csv')
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #
    #
    # for epoch in range(parameters_dict['num_epochs']):
    #     #enable train mode
    #     for model in all_models:
    #         model.train()
    #     batch_losses, all_ground_truth_train, all_predictions_train = train_one_epoch(trainloader, optimizer,
    #                                                                       base_net, common_net, classification_net, criterion, mode='train')
    #     # add losses
    #     epoch_losses_train.append(np.mean(batch_losses))
    #     # performance metrics
    #     f1_train = metrics.f1_score(all_ground_truth_train, all_predictions_train, average='macro')
    #     acc_train = metrics.accuracy_score(all_ground_truth_train, all_predictions_train)
    #     train_results.append([acc_train, f1_train, epoch+1])
    #     result_np = np.array(train_results, dtype=float)
    #     np.savetxt(train_result_csv, result_np, fmt='%.4f', delimiter=',')
    #
    #     # enable test mode
    #     for model in all_models:
    #         model.eval()
    #     batch_losses, all_ground_truth_test, all_predictions_test = train_one_epoch(trainloader, optimizer,
    #                                                                       base_net, common_net, classification_net, criterion, mode='test')
    #     # add losses
    #     epoch_losses_test.append(np.mean(batch_losses))
    #     # performance metrics
    #     f1_test = metrics.f1_score(all_ground_truth_test, all_predictions_test, average='macro')
    #     acc_test = metrics.accuracy_score(all_ground_truth_test, all_predictions_test)
    #     test_results.append([acc_test, f1_test, epoch + 1])
    #     result_np = np.array(test_results, dtype=float)
    #     np.savetxt(test_result_csv, result_np, fmt='%.4f', delimiter=',')
    #
    #     print(f'Epoch {epoch}: Train loss: {epoch_losses_train[-1]} Test: {epoch_losses_test[-1]}')
    #     print(f"Train f1: {f1_train},  Train acc: {acc_train}")
    #     print(f"Test f1: {f1_test}, test_acc:{acc_test}")
    #     print()












