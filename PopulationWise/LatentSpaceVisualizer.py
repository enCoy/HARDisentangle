import os
import numpy as np
import torch
import pandas as pd
import re
from DataProcesses import DownstreamDataProcessor
from Models.DisentangleNetModels import CNNBaseNet, PopulationEncoder, PersonalizedEncoder
from Models.ClassifierNet import ClassifierNet
from Utils.HelperFunctions import convert_to_torch
import torch.utils.data as Data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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


def get_TSNE_plot(X, activities, subjects, title, save_dir=None):
    # Define colors and markers for subjects and activities
    subject_colors = ['indianred', 'royalblue', 'seagreen', 'coral', 'darkmagenta', 'gold', 'cyan', 'orchid']
    activity_markers = ['o', 's', '^', 'v', 'D', 'P', '*', '+', '8', 'h', 'x', '1']
    tsne = TSNE(n_components=2, random_state=42)
    X_embedding = tsne.fit_transform(X)
    # Plot the t-SNE embedded data with different colors for each subject and markers for each activity
    plt.figure(figsize=(8, 8))
    for subject_id in np.unique(subjects):
        for activity_id, marker in zip(np.unique(activities), activity_markers):
            mask = (subjects == subject_id) & (activities == activity_id)
            plt.scatter(X_embedding[mask, 0], X_embedding[mask, 1], label=f'Subject {subject_id}, Activity {activity_id}',
                        color=subject_colors[subject_id - 1], marker=marker, alpha=0.5)

    plt.title(f't-SNE Visualization with Subject and Activity Differentiation - {title}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    patches = get_custom_legend_patches(colors=subject_colors, labels=list(np.arange(len(subject_colors)) + 1))
    plt.legend(handles=patches, title="Subjects")
    # plt.legend()
    if save_dir is not None:
        image_format = 'png'  # e.g .png, .svg, etc
        plt.savefig(os.path.join(save_dir, title + '.png'), format=image_format, dpi=1200)
    # plt.show()

def get_TSNE_plot_v2(X, activities, subjects, title, save_dir=None):
    # Define colors and markers for subjects and activities
    activity_colors = ['indianred', 'royalblue', 'seagreen', 'coral', 'darkmagenta', 'gold', 'cyan', 'orchid', 'burlywood', 'pink', 'chocolate', 'gray']
    subject_markers = ['o', 's', '^', 'v', 'D', 'P', '*', '+']
    tsne = TSNE(n_components=2, random_state=42)
    X_embedding = tsne.fit_transform(X)
    # Plot the t-SNE embedded data with different colors for each subject and markers for each activity
    plt.figure(figsize=(12, 8))
    for activity_id in np.unique(activities):
        for subject_id, marker in zip(np.unique(subjects), subject_markers):
            mask = (subjects == subject_id) & (activities == activity_id)
            plt.scatter(X_embedding[mask, 0], X_embedding[mask, 1], label=f'Subject {subject_id}, Activity {activity_id}',
                        color=activity_colors[activity_id - 1], marker=marker, alpha=0.5)

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
    analysis_dir = os.path.join(base_dir, r"Logging", r"2024-02-09_18-12-52")
    target_subject = 2

    output_dir = os.path.join(analysis_dir, f"S{target_subject}", "TSNEPlots")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    batch_size = 128
    trainset = Data.TensorDataset(train_X, train_y)
    trainloader = Data.DataLoader(dataset=trainset, batch_size=batch_size,
                                  shuffle=False, num_workers=0, drop_last=False)
    testset = Data.TensorDataset(test_X, test_y)
    testloader = Data.DataLoader(dataset=testset, batch_size=batch_size,
                                 shuffle=False, num_workers=0, drop_last=False)

    model_dir = os.path.join(analysis_dir, f"S{target_subject}")
    # load models etc.
    base_net = CNNBaseNet(input_dim=parameters['num_modalities'],
                          output_channels=parameters['base_net_out_channel'],
                          output_dim=parameters['base_net_output_dim'],
                          num_time_steps=parameters['window_size'])
    population_encoder = PopulationEncoder(input_dim=parameters['base_net_output_dim'],
                                           hidden_1=512, output_dim=parameters['population_output_dim'],
                                           train_on_gpu=True)
    personalized_encoder = PersonalizedEncoder(input_dim=parameters['base_net_output_dim'],
                                           hidden_1=512, output_dim=parameters['personalized_output_dim'],
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

    analysis_encoder = "personalized"

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
        print("X_embedding shape: ", X_embedding_train.shape)
        print("activities_train shape: ", len(activities_train))
        #
        if analysis_encoder == 'population':
            get_TSNE_plot_v2(X_embedding_train, np.array(activities_train), np.array(subjects_train), f'train - {analysis_encoder}',
                         save_dir=output_dir)
        else:
            get_TSNE_plot(X_embedding_train, np.array(activities_train), np.array(subjects_train),
                             f'train - {analysis_encoder}',
                             save_dir=output_dir)


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

        if analysis_encoder == 'population':
            get_TSNE_plot_v2(X_embedding_test, np.array(activities_test), np.array(subjects_test), f'test - {analysis_encoder}',
                         save_dir=output_dir)
        else:
            get_TSNE_plot_v2(X_embedding_test, np.array(activities_test), np.array(subjects_test),
                             f'test - {analysis_encoder}',
                             save_dir=output_dir)
        plt.show()


    #     # Validation
    #     y_true_test = []
    #     y_pred_test = []
    #     for model in list(all_models.values()):
    #         model.eval()
    #     running_val_loss = 0.0
    #
    #
    #         for inputs, labels in testloader:
    #             base_net_repr = base_net(inputs)
    #             pop_net_repr = population_encoder(base_net_repr)
    #             if include_personalized:
    #                 per_net_repr = personalized_encoder(base_net_repr)
    #                 concatted = torch.concat((per_net_repr, pop_net_repr), 1)
    #                 outputs = classification_net(concatted)
    #             else:
    #                 outputs = classification_net(pop_net_repr)
    #             val_loss = criterion(outputs, labels)
    #             running_val_loss += val_loss.item()
    #             # Calculate validation accuracy
    #             _, predicted_val = torch.max(outputs, 1)
    #             labels = torch.argmax(labels, dim=1)
    #
    #             # Collect true and predicted labels for F1 score calculation
    #             y_true_test.extend(labels.cpu().numpy())
    #             y_pred_test.extend(predicted_val.cpu().numpy())
    #
    #     val_loss = running_val_loss / len(testloader)
    #     # Calculate accuracy for training set
    #     test_accuracy = accuracy_score(y_true_test, y_pred_test)
    #     # Calculate F1 score for training set
    #     test_f1 = f1_score(y_true_test, y_pred_test, average='macro')
    #     if test_f1 > best_test_f1_score:
    #         best_test_gt = y_true_test
    #         best_test_preds = y_pred_test
    #         best_train_gt = y_true_train
    #         best_train_preds = y_pred_train
    #
    #     # Print statistics
    #     print(f"Epoch {epoch + 1}/{num_epochs}, "
    #           f"Train Loss: {train_loss:.4f}, Train Accuracy: {100 * train_accuracy:.2f}%,  Train F1: {train_f1:.4f},"
    #           f"Val Loss: {val_loss:.4f}, Val Accuracy: {100 * test_accuracy:.2f}%,  Test F1: {test_f1:.4f},")
    #     print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")
    # print('Finished Training')

