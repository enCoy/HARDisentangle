import numpy as np
import os
from DataProcesses import DataProcessor
import warnings
warnings.filterwarnings('ignore')



BASE_DIR = r"C:\Users\Cem Okan\Dropbox (GaTech)\DisentangledHAR"

def get_parameters(data_name):
    parameters_dict = {}
    parameters_dict['window_size'] = 50  # 50 hz = 1 sec
    parameters_dict['sampling_rate'] = 50 # hz
    parameters_dict['sliding_window_overlap_ratio'] = 0.5
    if data_name == 'pamap2':
        parameters_dict['data_dir'] = os.path.join(BASE_DIR, r"PAMAP2_Dataset\PAMAP2_Dataset\Processed50Hz")
        parameters_dict['num_modalities'] = 52  # number of sensor channels
        parameters_dict['num_activities'] = 12
        parameters_dict['num_subjects'] = 8
    elif data_name == 'real':
        parameters_dict['data_dir'] = os.path.join(BASE_DIR, r"realworld2016_dataset\Processed")
        parameters_dict['num_modalities'] = 42  # number of sensor channels
        parameters_dict['num_activities'] = 8
        parameters_dict['num_subjects'] = 15
    else:
        print("Error! Data does not exist!!")
    return parameters_dict

if __name__ == "__main__":

    data_name = 'pamap2'  # 'pamap2' or 'real'
    parameters_dict = get_parameters(data_name)

    data_processor = DataProcessor(data_dir=parameters_dict['data_dir'],
                                   data_name=data_name,
                                   target_subject_num=1,
                                   num_subjects=parameters_dict['num_subjects'],
                    num_activities=parameters_dict['num_activities'],
                        window_size=parameters_dict['window_size'],
                                   num_modalities=parameters_dict['num_modalities'],
                 sampling_rate=parameters_dict['sampling_rate'],
                    sliding_window_overlap_ratio=parameters_dict['sliding_window_overlap_ratio'])

    train_X, train_y, test_X, test_y = data_processor.get_train_test_subjects_data()


    data_processor.generate_positive_samples(data_processor.data_dict,
                                             data_processor.modalities)





