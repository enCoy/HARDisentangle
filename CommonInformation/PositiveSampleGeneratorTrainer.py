from Main import BASE_DIR, get_parameters
import os
from Utils.HelperFunctions import  convert_to_torch, plot_loss_curves
from DataProcesses import DataProcessor
from time import localtime, strftime

from CommonInformation.Models.LSTMAE import LSTM_AE, train
import torch
import torch.utils.data as Data
import warnings
warnings.filterwarnings('ignore')


if __name__ =="__main__":
    data_name = 'pamap2'  # 'pamap2' or 'real'
    parameters_dict = get_parameters(data_name)
    target_subject = 1

    data_processor = DataProcessor(data_dir=parameters_dict['data_dir'],
                                   data_name=data_name,
                                   target_subject_num=target_subject,
                                   num_subjects=parameters_dict['num_subjects'],
                                   num_activities=parameters_dict['num_activities'],
                                   window_size=parameters_dict['window_size'],
                                   num_modalities=parameters_dict['num_modalities'],
                                   sampling_rate=parameters_dict['sampling_rate'],
                                   sliding_window_overlap_ratio=parameters_dict['sliding_window_overlap_ratio'])

    train_X, train_y, test_X, test_y = data_processor.generate_positive_samples(data_processor.data_dict,
                                                            data_processor.modalities)
    train_X, train_y, test_X, test_y = convert_to_torch(train_X, train_y, test_X, test_y)

    trainset = Data.TensorDataset(train_X, train_y)
    trainloader = Data.DataLoader(dataset=trainset, batch_size=parameters_dict['batch_size'],
                                  shuffle=True, num_workers=0, drop_last=True)

    testset = Data.TensorDataset(test_X, test_y)
    testloader = Data.DataLoader(dataset=testset, batch_size=parameters_dict['batch_size'],
                                 shuffle=False, num_workers=0, drop_last=True)

    # call model
    if data_name == 'pamap2':
        input_size = 9
    else:
        input_size = 6

    model = LSTM_AE(input_size, parameters_dict['embedding_dim'],
                    use_bidirectional = parameters_dict['use_bidirectional'],
                    num_layers=parameters_dict['num_lstm_layers'])
    model, best_epoch, [train_losses, test_losses] =  train(model, trainloader, testloader, parameters_dict)

    timestring = strftime("%Y-%m-%d_%H-%M-%S", localtime()) + "_Subject%s" % str(
        target_subject)

    save_dir = os.path.join(BASE_DIR, 'CommonInformation', 'PositiveSampleGenerator', data_name, timestring)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # save state dict
    torch.save(model.state_dict(), os.path.join(save_dir, f"positiveGeneratorStateDict.pth"))
    # save model
    torch.save(model, os.path.join(save_dir, f"positiveGenerator.pth"))
    # save training result
    plot_loss_curves(train_losses, test_losses, save_loc=save_dir, show_fig=False)
    # save hyperparams
    parameters_dict['best_epoch'] = best_epoch
    parameters_dict['input_size'] = input_size
    with open(os.path.join(save_dir, 'parameters.txt'), 'w') as f:
        print(parameters_dict, file=f)
    print("END!")







