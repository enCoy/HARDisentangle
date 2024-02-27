import numpy as np
import os
import pickle
from Utils.HelperFunctions import sliding_window, calculate_amount_of_slide, get_standardizer, get_activity_columns
import scipy
import pandas as pd
from torch.utils.data import Dataset
import json
import time

# This class is especially for downstream task data loading
# for self supervised disentangled learning, there is another
class DownstreamDataProcessor():
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
        self.train_subjects_list = []
        self.test_subjects_list = []

    def get_activity_names(self):
        if self.data_name == 'pamap2':
            return ['lying', 'sitting', 'standing', 'walking', 'running', 'cycling',
                    'nordic walking', 'ascending stairs', 'descending stairs', 'vacuum cleaning',
                    'ironing', 'rope jumping']
        else:
            return None

    def get_train_test_subjects_data(self):
        # X shape (N, W, M)  N:number of windows of all subjects except training, W:window size, M:num modalities
        # train_y shape: (N, D)  D: one hot encoding of number of activities
        train_X = None
        train_y = None
        test_X = None
        test_y = None
        if self.data_name == 'pamap2':
            train_X, train_y, test_X, test_y = self.get_pamap2_train_test_data()
        # elif self.data_name == 'real':
        #     train_X, train_y, test_X, test_y = self.get_realworld_train_test_data()
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
        for i in range(1, self.num_subjects + 1):
            data, one_hot_labels = read_pamap2_single_subject(self.data_dir, i, self.window_size,
                               self.sliding_window_overlap_ratio, num_activities=self.num_activities)
            if i == self.target_subject_num:
                test_X = data
                test_y = one_hot_labels
                self.test_subjects_list.extend([i for m in range(data.shape[0])])
            else:
                train_X = np.vstack((train_X, data))
                train_y = np.concatenate((train_y, one_hot_labels))
                self.train_subjects_list.extend([i for m in range(data.shape[0])])
        # standardize data
        mean_vals, std_vals = get_standardizer(train_X)
        self.mean_vals = mean_vals
        self.std_vals = std_vals
        train_X = (train_X - self.mean_vals) / self.std_vals
        test_X = (test_X - self.mean_vals) / self.std_vals
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
        for i in range(1, self.num_subjects + 1):
            data, one_hot_labels = read_realworld_single_subject(self.data_dir, i, self.window_size,
                               self.sliding_window_overlap_ratio, num_activities=self.num_activities)
            if i == self.target_subject_num:
                test_X = data
                test_y = one_hot_labels
            else:
                train_X = np.vstack((train_X, data))
                train_y = np.concatenate((train_y, one_hot_labels))
        # standardize data
        mean_vals, std_vals = get_standardizer(train_X)
        self.mean_vals = mean_vals
        self.std_vals = std_vals
        train_X = (train_X - self.mean_vals) / self.std_vals
        test_X = (test_X - self.mean_vals) / self.std_vals
        print(f'REALWORLD test user ->', self.target_subject_num)
        print(f'REALWORLD train X shape ->', train_X.shape)
        print(f'REALWORLD train y shape ->', train_y.shape)
        print(f'REALWORLD test X shape ->', test_X.shape)
        print(f'REALWORLD test y shape ->', test_y.shape)
        return train_X, train_y, test_X, test_y


class SelfSupervisedDataProcessor(Dataset):
    def __init__(self, dataframe_dir, data_name, target_subject_num, num_subjects, window_size, num_modalities,
                 sampling_rate, sliding_window_overlap_ratio, num_neg_samples=3):
        self.dataframe_dir = dataframe_dir   # data
        self.num_subjects = num_subjects
        self.window_size = window_size   # in terms of number of samples
        self.num_modalities = num_modalities
        self.sliding_window_overlap_ratio = sliding_window_overlap_ratio  # in [0, 1] 1 means complete overlap, 0 means no overlap between windows
        self.sampling_rate = sampling_rate  # in Hz
        self.data_name = data_name  # 'real', 'pamap2'
        self.num_neg_samples = num_neg_samples

        self.dataframe = pd.read_csv(os.path.join(dataframe_dir, 'dataframe_all_subjects.csv'))
        # the following line is crucial for saving list of lists - otherwise, it converts the list into strings
        self.dataframe['data'] = self.dataframe ['data'].apply(lambda x: json.loads(x))
        # remove target subject num - it is for testing
        self.dataframe = self.dataframe[self.dataframe['subj_id'] != target_subject_num]

        # stacked data array
        stacked_data = np.stack(self.dataframe['data'].values)
        print("stacked_data.shape: ", stacked_data.shape)
        mean_vals, std_vals = get_standardizer(stacked_data)
        self.mean_vals = mean_vals
        self.std_vals = std_vals

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        anchor_sample = np.array(self.dataframe.iloc[idx]['data'], dtype=float)
        anchor_subj = int(self.dataframe.iloc[idx]['subj_id'])
        anchor_activity = int(self.dataframe.iloc[idx]['act_name'])

        all_people_same_activity = self.dataframe[self.dataframe['act_name'] == anchor_activity]
        all_people_other_activities = self.dataframe[self.dataframe['act_name'] != anchor_activity]
        all_activities_same_person = self.dataframe[self.dataframe['subj_id'] == anchor_subj]
        all_activities_other_people = self.dataframe[self.dataframe['subj_id'] != anchor_subj]

        # retrieve positive samples
        # population_encoder positive sample is same activity - from all people - attracts representations of all people as long as it is the same activity
        # get a random sample from all performing same activity
        pop_enc_pos_sample = all_people_same_activity['data'].sample(n=1).to_numpy()[0]
        pop_enc_pos_sample = np.array(pop_enc_pos_sample, dtype=float)  # shaped (WindowSize, FeatureDim)

        # retrieve negative samples
        # population_encoder negative sample is different activities - the same person - repels other activities
        # Sample K rows from the DataFrame
        sampled_rows = all_people_other_activities.sample(self.num_neg_samples)
        # Concatenate the 'data' values of the sampled rows
        pop_enc_neg_sample = np.stack(sampled_rows['data'].values, axis=0)
        # same_person_different_activity = np.stack(all_people_other_activities['data'].values)  # this is too slow

        # personalized_encoder positive sample is the same person, any activity - attracts representations of the same person as long as it is the same person
        # get a random sample
        person_enc_pos_sample = all_activities_same_person['data'].sample(n=1).to_numpy()[0]
        person_enc_pos_sample = np.array(person_enc_pos_sample, dtype=float)  # shaped (WindowSize, FeatureDim)

        # personalized_encoder negative sample is the same activity - different people - repels different people
        # Sample K rows from the DataFrame
        sampled_rows = all_activities_other_people.sample(self.num_neg_samples)
        # Concatenate the 'data' values of the sampled rows
        person_enc_neg_sample = np.stack(sampled_rows['data'].values, axis=0)

        # standardize data
        anchor_sample = (anchor_sample - self.mean_vals) / self.std_vals
        pop_enc_pos_sample = (pop_enc_pos_sample - self.mean_vals) / self.std_vals
        pop_enc_neg_sample = (pop_enc_neg_sample - self.mean_vals) / self.std_vals
        person_enc_pos_sample = (person_enc_pos_sample - self.mean_vals) / self.std_vals
        person_enc_neg_sample = (person_enc_neg_sample - self.mean_vals) / self.std_vals

        sample = {'anchor': np.asarray(anchor_sample, dtype=np.float32),
                  'pop_enc_pos': np.asarray(pop_enc_pos_sample, dtype=np.float32),
                  'pop_enc_neg': np.asarray(pop_enc_neg_sample, dtype=np.float32),
                  'person_enc_pos': np.asarray(person_enc_pos_sample, dtype=np.float32),
                  'person_enc_neg': np.asarray(person_enc_neg_sample, dtype=np.float32)}
        return sample


class SelfSupervisedDataProcessorForTesting(Dataset):
    def __init__(self, dataframe_dir, data_name, target_subject_num, num_subjects, window_size, num_modalities,
                 sampling_rate, sliding_window_overlap_ratio, num_neg_samples=3):
        self.dataframe_dir = dataframe_dir   # data
        self.num_subjects = num_subjects
        self.window_size = window_size   # in terms of number of samples
        self.num_modalities = num_modalities
        self.sliding_window_overlap_ratio = sliding_window_overlap_ratio  # in [0, 1] 1 means complete overlap, 0 means no overlap between windows
        self.sampling_rate = sampling_rate  # in Hz
        self.data_name = data_name  # 'real', 'pamap2'
        self.num_neg_samples = num_neg_samples

        self.dataframe = pd.read_csv(os.path.join(dataframe_dir, 'dataframe_all_subjects.csv'))
        # the following line is crucial for saving list of lists - otherwise, it converts the list into strings
        self.dataframe['data'] = self.dataframe ['data'].apply(lambda x: json.loads(x))
        # remove target subject num - it is for testing
        self.target_dataframe = self.dataframe[self.dataframe['subj_id'] == target_subject_num]
        self.training_subjects_dataframe = self.dataframe[self.dataframe['subj_id'] != target_subject_num]

        # stacked data array for other subjects
        stacked_data = np.stack(self.training_subjects_dataframe['data'].values)
        print("stacked_data.shape: ", stacked_data.shape)
        mean_vals, std_vals = get_standardizer(stacked_data)
        self.mean_vals = mean_vals
        self.std_vals = std_vals

    def __len__(self):
        return len(self.target_dataframe)

    def __getitem__(self, idx):
        anchor_sample = np.array(self.target_dataframe.iloc[idx]['data'], dtype=float)
        anchor_subj = int(self.target_dataframe.iloc[idx]['subj_id'])
        anchor_activity = int(self.target_dataframe.iloc[idx]['act_name'])

        all_people_same_activity = self.training_subjects_dataframe[self.training_subjects_dataframe['act_name'] == anchor_activity]
        all_people_other_activities = self.training_subjects_dataframe[self.training_subjects_dataframe['act_name'] != anchor_activity]
        all_activities_same_person = self.target_dataframe[self.target_dataframe['subj_id'] == anchor_subj]
        all_activities_other_people = self.training_subjects_dataframe[self.training_subjects_dataframe['subj_id'] != anchor_subj]

        # retrieve positive samples
        # population_encoder positive sample is same activity - from other people
        # get a random sample from other people performing same activity
        pop_enc_pos_sample = all_people_same_activity['data'].sample(n=1).to_numpy()[0]
        pop_enc_pos_sample = np.array(pop_enc_pos_sample, dtype=float)  # shaped (WindowSize, FeatureDim)

        # personalized_encoder positive sample is the same person, different activity
        # get a random sample
        person_enc_pos_sample = all_activities_same_person['data'].sample(n=1).to_numpy()[0]
        person_enc_pos_sample = np.array(person_enc_pos_sample, dtype=float) #  shaped (WindowSize, FeatureDim)

        # retrieve negative samples
        # population_encoder negative sample is different activities - the same person
        # Sample K rows from the DataFrame
        sampled_rows = all_people_other_activities.sample(self.num_neg_samples)
        # Concatenate the 'data' values of the sampled rows
        pop_enc_neg_sample = np.stack(sampled_rows['data'].values, axis=0)
        # same_person_different_activity = np.stack(all_people_other_activities['data'].values)  # this is too slow


        # personalized_encoder negative sample is the same activity - different people
        # Sample K rows from the DataFrame
        sampled_rows = all_activities_other_people.sample(self.num_neg_samples)
        # Concatenate the 'data' values of the sampled rows
        person_enc_neg_sample = np.stack(sampled_rows['data'].values, axis=0)

        # standardize data
        anchor_sample = (anchor_sample - self.mean_vals) / self.std_vals
        pop_enc_pos_sample = (pop_enc_pos_sample - self.mean_vals) / self.std_vals
        pop_enc_neg_sample = (pop_enc_neg_sample - self.mean_vals) / self.std_vals
        person_enc_pos_sample = (person_enc_pos_sample - self.mean_vals) / self.std_vals
        person_enc_neg_sample = (person_enc_neg_sample - self.mean_vals) / self.std_vals

        sample = {'anchor': np.asarray(anchor_sample, dtype=np.float32),
                  'pop_enc_pos': np.asarray(pop_enc_pos_sample, dtype=np.float32),
                  'pop_enc_neg': np.asarray(pop_enc_neg_sample, dtype=np.float32),
                  'person_enc_pos': np.asarray(person_enc_pos_sample, dtype=np.float32),
                  'person_enc_neg': np.asarray(person_enc_neg_sample, dtype=np.float32)}
        return sample


###CLASS UNRELATED FUNCTIONS

#################PAMAP2
# dataframe for facilitating self supervised learning dataloader
def get_pamap2_dataframe(data_dir, num_subjects, window_size, sliding_overlap, num_activities=12):
    # create a dataframe with the following columns: idx, subj_id, act_name, data
    # data will contain time series data
    file_path = os.path.join(data_dir, 'dataframe_all_subjects.csv')
    if os.path.exists(file_path):
        print(f"The file at {file_path} exists.")
    else:
        print(f"The file at {file_path} does not exist.")
        all_column_names = ['data_idx', 'subj_id', 'act_name', 'data']
        dataframe = pd.DataFrame(columns=all_column_names)
        data_idx = 0  # this should be window index
        for i in range(1, num_subjects + 1):
            dict_subj = {}
            data, one_hot_labels = read_pamap2_single_subject(data_dir, i, window_size,
                               sliding_overlap, num_activities=num_activities)
            # data is shaped N_i, x WindowSize x FeatureDim where N_i is the number of windows for that subject
            class_values = np.argmax(one_hot_labels, axis=1).tolist()   # N_i,
            # data has shape N x WindowSize x FeatureDim
            # one hot labels has shape N x NumClasses
            print(f"Subject {i} data shape: ", data.shape)
            # create vectors of subject id
            subj_id_vec = (np.ones(data.shape[0]) * i).tolist()  # shape N_i,
            data_indices = (np.arange(data.shape[0]) + data_idx).tolist()  # shape N_i,
            data_idx += data.shape[0]
            # concat all data in the order of
            dict_subj['data_idx'] = data_indices
            dict_subj['subj_id'] = subj_id_vec
            dict_subj['act_name'] = class_values
            # dict_subj['data'] = data
            # list_of_lists = [arr.tolist() for arr in data]
            list_of_lists = [[list(row) for row in matrix] for matrix in data]
            # print("aha sorun: ", list_of_lists[0:10])
            dict_subj['data'] = list_of_lists

            df_to_append = pd.DataFrame(dict_subj, columns=dataframe.columns)
            # Append the DataFrame to the empty DataFrame
            dataframe = pd.concat([dataframe, df_to_append], ignore_index=True)
        # the following line is crucial for saving list of lists - otherwise, it converts the list into strings
        dataframe['data'] = dataframe['data'].apply(lambda x: json.dumps(x))
        dataframe.to_csv(os.path.join(data_dir, 'dataframe_all_subjects.csv'), index=False)
        print("Dataframe is successfully created")

def read_pamap2_single_subject(data_dir, subject_idx, window_size,
                               sliding_overlap, num_activities=12):
    data_path = os.path.join(data_dir, f'subject10{subject_idx}' + '.pickle')
    modality_indices, modalities = get_modality_indices('pamap2')
    with open(data_path, 'rb') as handle:
        whole_data = pickle.load(handle)
        data = whole_data['data']
        # only use acc and gyro data
        data = data[:, modality_indices]
        label = whole_data['label']
        data = sliding_window(data, ws=(window_size, data.shape[1]),
                              ss=(calculate_amount_of_slide(window_size,
                                                            sliding_overlap),
                                  1))  # 50 corresponds to 1 secs,
        label = sliding_window(label, ws=window_size,
                               ss=calculate_amount_of_slide(window_size,
                                                            sliding_overlap))
        # take the most frequent activity within a window for labeling
        label = np.squeeze(scipy.stats.mode(label, axis=1)[0])  # axis=1 is for the window axis
        one_hot_labels = np.zeros((len(label), num_activities))
        one_hot_labels[np.arange(len(label)), label] = 1
    return data, one_hot_labels  # data shape: (N, WindowLength, FeatureLength), one_hot_labels shape: (N, NumClasses)
#################PAMAP2

#################REALWORLD
def read_realworld_single_subject(data_dir, subject_idx, window_size,
                               sliding_overlap, num_activities=8):
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
        data_dir = os.path.join(data_dir, f'subject{subject_idx}', activity_name + '.csv')
        if os.path.exists(data_dir):
            activity_df = pd.read_csv(data_dir)
            columns = get_activity_columns(['acc', 'gyr'])
            data_i = activity_df[columns].values
            label_i = activity_df['activity_id'].values.astype(int)
            data_i = sliding_window(data_i, ws=(window_size, data_i.shape[1]),
                                    ss=(
                                    calculate_amount_of_slide(window_size, sliding_overlap),
                                    1))  # 50 corresponds to 1 secs, 50 - 11 -> %50 overlap
            label_i = sliding_window(label_i, ws=window_size,
                                     ss=calculate_amount_of_slide(window_size,
                                                                  sliding_overlap))  # 50 corresponds to 1 secs, 50 - 11 -> %50 overlap
            label_i = np.squeeze(scipy.stats.mode(label_i, axis=1)[0])  # axis=1 is for the window axis
            one_hot_labels_i = np.zeros((len(label_i), num_activities))
            one_hot_labels_i[np.arange(len(label_i)), label_i] = 1
            if data is None:
                data = data_i
                one_hot_labels = one_hot_labels_i
            else:  # concatenate raw files
                data = np.concatenate((data, data_i), axis=0)
                one_hot_labels = np.concatenate((one_hot_labels, one_hot_labels_i), axis=0)
        else:
            continue
    return data, one_hot_labels
#################REALWORLD

def get_modality_indices(data_name):
    # NOTE: THESE VALUES ARE BASED ON THE PAMAP2 PROCESSING WITH 52 FEATURES
    modalities = None
    modality_indices = []
    if data_name == 'pamap2':
        modality_indices = [2, 3, 4, 5, 6, 7, 8, 9, 10,   # 2 to 10 are acc and gyro data for hand
                            19, 20, 21, 22, 23, 24, 25, 26, 27, # 19 to 27 are acc and gyro data for chest
                            36, 37, 38, 39, 40, 41, 42, 43, 44]  # 36 to 44 are acc and gyro data for ankle
        modalities = ['hand', 'chest', 'ankle']
    elif data_name == 'real':
        modality_indices = [0, 1, 2, 3, 4, 5,  # chest
                            6, 7, 8, 9, 10, 11,  # forearm
                            12, 13, 14, 15, 16, 17,  # head
                            18, 19, 20, 21, 22, 23,  # shin
                            24, 25, 26, 27, 28, 29,  # thigh
                            30, 31, 32, 33, 34, 35,  # upper arm
                            36, 37, 38, 39, 40, 41]  # waist
        modalities = ['chest', 'forearm', 'head', 'shin', 'thigh', 'upperarm', 'waist']
    return modality_indices, modalities



#################MOTIONSENSE
# dataframe for facilitating self supervised learning dataloader
def get_motionsense_dataframe(data_dir, num_subjects, window_size, sliding_overlap, num_activities=12):
    # create a dataframe with the following columns: idx, subj_id, act_name, data
    # data will contain time series data
    file_path = os.path.join(data_dir, 'dataframe_all_subjects.csv')
    if os.path.exists(file_path):
        print(f"The file at {file_path} exists.")
    else:
        print(f"The file at {file_path} does not exist.")
        all_column_names = ['data_idx', 'subj_id', 'act_name', 'data']
        dataframe = pd.DataFrame(columns=all_column_names)
        data_idx = 0  # this should be window index
        for i in range(1, num_subjects + 1):
            dict_subj = {}
            data, one_hot_labels = read_motionsense_single_subject(data_dir, i, window_size,
                               sliding_overlap, num_activities=num_activities)
            # data is shaped N_i, x WindowSize x FeatureDim where N_i is the number of windows for that subject
            class_values = np.argmax(one_hot_labels, axis=1).tolist()   # N_i,
            # data has shape N x WindowSize x FeatureDim
            # one hot labels has shape N x NumClasses
            print(f"Subject {i} data shape: ", data.shape)
            # create vectors of subject id
            subj_id_vec = (np.ones(data.shape[0]) * i).tolist()  # shape N_i,
            data_indices = (np.arange(data.shape[0]) + data_idx).tolist()  # shape N_i,
            data_idx += data.shape[0]
            # concat all data in the order of
            dict_subj['data_idx'] = data_indices
            dict_subj['subj_id'] = subj_id_vec
            dict_subj['act_name'] = class_values
            # dict_subj['data'] = data
            # list_of_lists = [arr.tolist() for arr in data]
            list_of_lists = [[list(row) for row in matrix] for matrix in data]
            # print("aha sorun: ", list_of_lists[0:10])
            dict_subj['data'] = list_of_lists

            df_to_append = pd.DataFrame(dict_subj, columns=dataframe.columns)
            # Append the DataFrame to the empty DataFrame
            dataframe = pd.concat([dataframe, df_to_append], ignore_index=True)
        # the following line is crucial for saving list of lists - otherwise, it converts the list into strings
        dataframe['data'] = dataframe['data'].apply(lambda x: json.dumps(x))
        dataframe.to_csv(os.path.join(data_dir, 'dataframe_all_subjects.csv'), index=False)
        print("Dataframe is successfully created")

def read_motionsense_single_subject(data_dir, subject_idx, window_size,
                               sliding_overlap, num_activities=12):
    data_path = os.path.join(data_dir, f'subj_{subject_idx}.csv')
    with open(data_path, 'rb') as handle:
        whole_data = pd.read_csv(data_path)
        data = whole_data[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']].values
        label = whole_data['activity_id'].values
        data = sliding_window(data, ws=(window_size, data.shape[1]),
                              ss=(calculate_amount_of_slide(window_size,
                                                            sliding_overlap),
                                  1))
        label = sliding_window(label, ws=window_size,
                               ss=calculate_amount_of_slide(window_size,
                                                            sliding_overlap))
        # take the most frequent activity within a window for labeling
        label = np.squeeze(scipy.stats.mode(label, axis=1)[0]).astype(int)  # axis=1 is for the window axis
        one_hot_labels = np.zeros((len(label), num_activities))
        one_hot_labels[np.arange(len(label)), label] = 1
    return data, one_hot_labels  # data shape: (N, WindowLength, FeatureLength), one_hot_labels shape: (N, NumClasses)



