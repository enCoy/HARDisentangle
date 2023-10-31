import numpy as np
import os
import pickle
import pandas as pd
from HelperFunctions import sliding_window, calculate_amount_of_slide, get_activity_columns
import scipy


class DataProcessor():
    def __init__(self, data_dir, data_name, target_subject_num, num_subjects, num_activities, window_size, num_modalities,
                 sampling_rate, sliding_window_overlap_ratio):

        self.data_dir = data_dir
        self.target_subject_num = target_subject_num
        self.num_subjects = num_subjects
        self.num_activities = num_activities
        self.window_size = window_size   # in terms of number of samples
        self.num_modalities = num_modalities
        self.sliding_window_overlap_ratio = sliding_window_overlap_ratio  # in [0, 1] 1 means complete overlap, 0 means no overlap between windows
        self.sampling_rate = sampling_rate  # in Hz
        self.data_name = data_name  # 'real', 'pamap2'

        if self.data_name == 'pamap2':
            self.num_modalities_per_location = 9  # two acc (2x3 = 6) + gyro (3)
        elif self.data_name == 'real':
            self.num_modalities_per_location = 6  # acc (3) + gyro (3)


        self.modality_indices, self.modalities = self.get_modality_indices()
        self.data_dict = self.prepare_data_dict()



    def generate_positive_samples(self, data_dict, modalities):
        # will create two arrays -
        # input = time series modality_i window_m
        # output - positive sample = time series modality_-i window_m
        # positive samples for modality_i, window_m = window_m modality_-i
        train_X = np.empty((0, self.window_size, self.num_modalities_per_location))
        # total number of samples = ((P-1) x N_windows_per_subject) x M x M-1 ..from P-1 non-target subjects, N windows, M modalities, M-1 other modalities
        train_y = np.empty((0, self.window_size, self.num_modalities_per_location))
        test_X = np.empty((0, self.window_size, self.num_modalities_per_location))
        test_y = np.empty((0, self.window_size, self.num_modalities_per_location))

        for subject_idx in range(1, self.num_subjects + 1):
            for modality in modalities:
                other_modalities = [mm for mm in modalities if mm != modality]
                for other_modality in other_modalities:
                    # take all the windows of current modality
                    current_modality_data = data_dict[(subject_idx, modality)][0]
                    other_modality_data = data_dict[(subject_idx, other_modality)][0]
                    if subject_idx == self.target_subject_num:
                        test_X = np.vstack((test_X, current_modality_data))
                        test_y = np.vstack((test_y, other_modality_data))
                    else:
                        train_X = np.vstack((train_X, current_modality_data))
                        train_y = np.concatenate((train_y, other_modality_data))
        print("Positive samples are generated!")
        print(f'Positive samples train X shape ->', train_X.shape)
        print(f'Positive samples train y shape ->', train_y.shape)
        print(f'Positive samples test X shape ->', test_X.shape)
        print(f'Positive samples test y shape ->', test_y.shape)
        return train_X, train_y, test_X, test_y

    def generate_autoregressive_negative_samples(self, data_dict, modalities):
        # will create two arrays -
        # input = time series modality_i window_m
        # output - negative sample = time series modality_i window_m+1
        train_X = np.empty((0, self.window_size, self.num_modalities_per_location))
        # total number of samples = ((P-1) x N_windows_per_subject) x M x M-1 ..from P-1 non-target subjects, N windows, M modalities, M-1 other modalities
        train_y = np.empty((0, self.window_size, self.num_modalities_per_location))
        test_X = np.empty((0, self.window_size, self.num_modalities_per_location))
        test_y = np.empty((0, self.window_size, self.num_modalities_per_location))

        for subject_idx in range(1, self.num_subjects + 1):
            for modality in modalities:
                # take all the windows of current modality
                current_modality_data = data_dict[(subject_idx, modality)][0]
                if subject_idx == self.target_subject_num:
                    test_X = np.vstack((test_X, current_modality_data[:-1, :, :])) # samples until last time step
                    test_y = np.vstack((test_y, current_modality_data[1:, :, :]))  # starting from 1
                    # this ensured that test data is one time step forward shifted version of training data
                else:
                    train_X = np.vstack((train_X, current_modality_data[:-1, :, :]))
                    train_y = np.concatenate((train_y, current_modality_data[1:, :, :]))
        print("Negative samples are generated!")
        print(f'Negative samples train X shape ->', train_X.shape)
        print(f'Negative samples train y shape ->', train_y.shape)
        print(f'Negative samples test X shape ->', test_X.shape)
        print(f'Negative samples test y shape ->', test_y.shape)
        return train_X, train_y, test_X, test_y





    def prepare_data_dict(self):
        # keys will be (subject_id, modality name) values will be (data=all windows, labels=activities)
        data_dict = {}
        for i in range(self.num_subjects):
            if self.data_name == 'pamap2':
                data, one_hot_labels = self.read_pamap2_single_subject(i+1)
            elif self.data_name == 'real':
                data, one_hot_labels = self.read_realworld_single_subject(i + 1)

            for modality in self.modalities:
                data_dict[(i+1, modality)] = (data[:, :, self.modality_indices[modality]],
                                                             one_hot_labels)
        return data_dict

    def get_modality_indices(self):
        # NOTE: THESE VALUES ARE BASED ON THE PAMAP2 PROCESSING WITH 52 FEATURES
        modality_indices = {}
        modalities = None
        if self.data_name == 'pamap2':
            modality_indices['hand'] = np.arange(start=2, stop=10+1, step=1)  # 2 to 10 are acc and gyro data
            modality_indices['chest'] = np.arange(start=19, stop=27+1, step=1)  # 19 to 27 are acc and gyro data
            modality_indices['ankle'] = np.arange(start=36, stop=44+1, step=1)  # 36 to 44 are acc and gyro data
            modalities = list(modality_indices.keys())
        elif self.data_name == 'real':
            modality_indices['chest'] = np.arange(start=0, stop=5+1, step=1)
            modality_indices['forearm'] = np.arange(start=6, stop=11+1, step=1)
            modality_indices['head'] = np.arange(start=12, stop=17+1, step=1)
            modality_indices['shin'] = np.arange(start=18, stop=23+1, step=1)
            modality_indices['thigh'] = np.arange(start=24, stop=29+1, step=1)
            modality_indices['upperarm'] = np.arange(start=30, stop=35+1, step=1)
            modality_indices['waist'] = np.arange(start=36, stop=41+1, step=1)
            modalities = list(modality_indices.keys())
        return modality_indices, modalities

    def get_train_test_subjects_data(self):
        # X shape (N, W, M)  N:number of windows of all subjects except training, W:window size, M:num modalities
        # train_y shape: (N, D)  D: one hot encoding of number of activities
        train_X = None
        train_y = None
        test_X = None
        test_y = None
        if self.data_name == 'pamap2':
            train_X, train_y, test_X, test_y = self.get_pamap2_train_test_data()
        elif self.data_name == 'real':
            train_X, train_y, test_X, test_y = self.get_realworld_train_test_data()
        else:
            print("Dataset does not exist... Error!")
        return train_X, train_y, test_X, test_y

    def get_pamap2_train_test_data(self):
        # this is going to separate training and test subjects' data - test will include only target subject
        # training will include other subjects
        train_X = np.empty((0, self.window_size, self.num_modalities))
        train_y = np.empty((0, self.num_activities))
        test_X = np.empty((0, self.window_size, self.num_modalities))
        test_y = np.empty((0, self.num_activities))
        # traverse subjects, reserve target subject for testing, others for training
        for i in range(self.num_subjects):
            data_path = os.path.join(self.data_dir + f'subject10{i + 1}' + '.pickle')
            data, one_hot_labels = self.read_pamap2_single_subject(i+1)
            if (i + 1) == self.target_subject_num:
                test_X = data
                test_y = one_hot_labels
            else:
                train_X = np.vstack((train_X, data))
                train_y = np.concatenate((train_y, one_hot_labels))

        print(f'PAMAP2 test user ->', self.target_subject_num)
        print(f'PAMAP2 train X shape ->', train_X.shape)
        print(f'PAMAP2 train y shape ->', train_y.shape)
        print(f'PAMAP2 test X shape ->', test_X.shape)
        print(f'PAMAP2 test y shape ->', test_y.shape)
        return train_X, train_y, test_X, test_y

    def get_realworld_train_test_data(self):
        train_X = np.empty((0, self.window_size, self.num_modalities))
        train_y = np.empty((0, self.num_activities))
        test_X = np.empty((0, self.window_size, self.num_modalities))
        test_y = np.empty((0, self.num_activities))
        for i in range(self.num_subjects):
            data, one_hot_labels = self.read_realworld_single_subject(i+1)
            if (i + 1) == self.target_subject_num:
                test_X = data
                test_y = one_hot_labels
            else:
                train_X = np.vstack((train_X, data))
                train_y = np.concatenate((train_y, one_hot_labels))
        print(f'REALWORLD test user ->', self.target_subject_num)
        print(f'REALWORLD train X shape ->', train_X.shape)
        print(f'REALWORLD train y shape ->', train_y.shape)
        print(f'REALWORLD test X shape ->', test_X.shape)
        print(f'REALWORLD test y shape ->', test_y.shape)
        return train_X, train_y, test_X, test_y


    def read_pamap2_single_subject(self, subject_idx):
        data_path = os.path.join(self.data_dir, f'subject10{subject_idx}' + '.pickle')
        with open(data_path, 'rb') as handle:
            whole_data = pickle.load(handle)
            data = whole_data['data']
            label = whole_data['label']
            data = sliding_window(data, ws=(self.window_size, data.shape[1]),
                                  ss=(calculate_amount_of_slide(self.window_size,
                                                                self.sliding_window_overlap_ratio),
                                      1))  # 50 corresponds to 1 secs,
            label = sliding_window(label, ws=self.window_size,
                                   ss=calculate_amount_of_slide(self.window_size,
                                                                self.sliding_window_overlap_ratio))
            # take the most frequent activity within a window for labeling
            label = np.squeeze(scipy.stats.mode(label, axis=1)[0])  # axis=1 is for the window axis
            one_hot_labels = np.zeros((len(label), self.num_activities))
            one_hot_labels[np.arange(len(label)), label] = 1
        return data, one_hot_labels

    def read_realworld_single_subject(self, subject_idx):
        if subject_idx == 4 or subject_idx == 7:  # these subjects have multiple sessions of climbing up and down:
            activity_names = ['climbingdown_1', 'climbingdown_2', 'climbingdown_3',
                              'climbingup_1', 'climbingup_2', 'climbingup_3',
                              'jumping', 'lying', 'running', 'sitting', 'standing', 'walking']
        else:
            activity_names = ['climbingdown', 'climbingup', 'jumping', 'lying', 'running', 'sitting',
                              'standing', 'walking']
        data = None
        one_hot_labels = None
        for activity_name in activity_names:
            data_dir = os.path.join(self.data_dir, f'subject{subject_idx}', activity_name + '.csv')
            if os.path.exists(data_dir):
                activity_df = pd.read_csv(data_dir)
                columns = get_activity_columns(['acc', 'gyr'])
                data_i = activity_df[columns].values
                label_i = activity_df['activity_id'].values.astype(int)
                data_i = sliding_window(data_i, ws=(self.window_size, data_i.shape[1]),
                                        ss=(
                                        calculate_amount_of_slide(self.window_size, self.sliding_window_overlap_ratio),
                                        1))  # 50 corresponds to 1 secs, 50 - 11 -> %50 overlap
                label_i = sliding_window(label_i, ws=self.window_size,
                                         ss=calculate_amount_of_slide(self.window_size,
                                                                      self.sliding_window_overlap_ratio))  # 50 corresponds to 1 secs, 50 - 11 -> %50 overlap
                label_i = np.squeeze(scipy.stats.mode(label_i, axis=1)[0])  # axis=1 is for the window axis
                one_hot_labels_i = np.zeros((len(label_i), self.num_activities))
                one_hot_labels_i[np.arange(len(label_i)), label_i] = 1
                if data is None:
                    data = data_i
                    one_hot_labels = one_hot_labels_i
                else:  # concatenate raw files
                    data = np.concatenate((data, data_i), axis=0)
                    one_hot_labels = np.concatenate((one_hot_labels, one_hot_labels_i), axis=0)
            else:
                print("Not existing data: ", data_dir)
                print("Data does not exist... Continuing")
                continue
        return data, one_hot_labels


