import numpy as np
import torch
import torch.utils.data as data
from scipy.io import loadmat
from tool.utils import sliding_window
import pickle
import os

# base_dir = '../data/'
machine = 'other'
if machine == 'linux':
    base_dir = r'/home/cmyldz/Dropbox (GaTech)/DisentangledHAR/'
else:
    base_dir = r'C:\Users\Cem Okan\Dropbox (GaTech)\DisentangledHAR/'
pamap2_dir = 'PAMAP2_Dataset/PAMAP2_Dataset/Processed/'
dsads_dir = 'DSADS_Dataset/Processed/'
gotov_dir = 'GOTOV_Dataset/Processed/'
mhealth_dir = 'MHEALTH_Dataset/Processed/'

PAMAP2_DATA_FILES = ['subject101',
                     'subject102',
                     'subject103',
                     'subject104',
                     'subject105',
                     'subject106',
                     'subject107',
                     'subject108']

MHEALTH_DATA_FILES = ['subject_1',
                      'subject_2',
                      'subject_3',
                      'subject_4',
                      'subject_5',
                      'subject_6',
                      'subject_7',
                      'subject_8',
                      'subject_9',
                      'subject_10']

DSADS_DATA_FILE = ['subject_1',
                   'subject_2',
                   'subject_3',
                   'subject_4',
                   'subject_5',
                   'subject_6',
                   'subject_7',
                   'subject_8']

GOTOV_DATA_FILE = ['GOTOV02', 'GOTOV03', 'GOTOV04', 'GOTOV05', 'GOTOV06', 'GOTOV07', 'GOTOV08',
                   'GOTOV09', 'GOTOV10', 'GOTOV11', 'GOTOV12', 'GOTOV13', 'GOTOV14', 'GOTOV15', 'GOTOV16',
                   'GOTOV17', 'GOTOV18', 'GOTOV19', 'GOTOV20', 'GOTOV21', 'GOTOV22', 'GOTOV23', 'GOTOV24',
                   'GOTOV25', 'GOTOV26', 'GOTOV27', 'GOTOV28', 'GOTOV29', 'GOTOV30', 'GOTOV31', 'GOTOV32',
                   'GOTOV33', 'GOTOV34', 'GOTOV35', 'GOTOV36']

GOTOV_DATA_FILE_Train = ['GOTOV02', 'GOTOV03', 'GOTOV04', 'GOTOV05', 'GOTOV06', 'GOTOV07', 'GOTOV08',
                         'GOTOV09', 'GOTOV10', 'GOTOV11', 'GOTOV12', 'GOTOV13', 'GOTOV14', 'GOTOV15', 'GOTOV16',
                         'GOTOV17', 'GOTOV18', 'GOTOV19', 'GOTOV20', 'GOTOV21', 'GOTOV22', 'GOTOV23', 'GOTOV24',
                         'GOTOV25', 'GOTOV26', 'GOTOV27', 'GOTOV28', 'GOTOV29']

GOTOV_DATA_FILE_Test = ['GOTOV30', 'GOTOV31', 'GOTOV32', 'GOTOV33', 'GOTOV34', 'GOTOV35', 'GOTOV36']


def load_pamap2(candidate_number):
    global target_train_label, target_train, target_test, target_test_label
    target_user = candidate_number

    candidate_list = PAMAP2_DATA_FILES

    train_X = np.empty((0, 52))
    train_d = np.empty((0))
    train_y = np.empty((0))

    for i in range(0, len(candidate_list)):
        # candidate = loadmat(base_dir + pamap2_dir + candidate_list[i])
        with open(os.path.join(base_dir + pamap2_dir + candidate_list[i] + '.pickle'), 'rb') as handle:
            candidate = pickle.load(handle)
        if (i + 1) == target_user:
            test_X = candidate["data"]
            test_y = candidate["label"].reshape(-1)
            test_d = np.ones(test_y.shape) * i
        else:
            train_X = np.vstack((train_X, candidate["data"]))
            train_y = np.concatenate((train_y, candidate["label"].reshape(-1)))
            train_d = np.concatenate((train_d, np.ones(train_y.shape) * i))

    print('pamap2 test user ->', target_user)
    print('pamap2 train X shape ->', train_X.shape)
    print('pamap2 train y shape ->', train_y.shape)
    print('pamap2 test X shape ->', test_X.shape)
    print('pamap2 test y shape ->', test_y.shape)

    return train_X, train_y, train_d, test_X, test_y, test_d

def get_activity_columns(data_of_interest = ['acc', 'gyr']):
    body_device_locations = ['chest', 'forearm', 'head', 'shin', 'thigh', 'upperarm', 'waist']
    column_list = []
    for device_loc in body_device_locations:
        for data_name in data_of_interest:
            column_list.append(device_loc + '_' + data_name + '_x')
            column_list.append(device_loc + '_' + data_name + '_y')
            column_list.append(device_loc + '_' + data_name + '_z')
    return column_list

def load_real(candidate_number):
    import pandas as pd
    global target_train_label, target_train, target_test, target_test_label
    target_user = candidate_number
    number_of_subjects = 15
    har_data_dir = 'realworld2016_dataset/Processed/'

    train_X = np.empty((0, 42))
    train_d = np.empty((0))
    train_y = np.empty((0))

    for i in range(number_of_subjects):

        if i + 1 == 4 or i + 1 == 7:  # these subjects have multiple sessions of climbing up and down:
            activity_names = ['climbingdown_1', 'climbingdown_2', 'climbingdown_3',
                              'climbingup_1', 'climbingup_2', 'climbingup_3',
                              'jumping', 'lying', 'running', 'sitting', 'standing', 'walking']
        else:
            activity_names = ['climbingdown', 'climbingup', 'jumping', 'lying', 'running', 'sitting', 'standing',
                              'walking']

        data = None
        label = None
        for activity_name in activity_names:
            data_dir = os.path.join(base_dir + har_data_dir + f'subject{i + 1}', activity_name + '.csv')
            if os.path.exists(data_dir):
                activity_df = pd.read_csv(data_dir)
                columns = get_activity_columns(['acc', 'gyr'])
                data_i = activity_df[columns].values
                label_i = activity_df['activity_id'].values.astype(int)

                if data is None:
                    data = data_i
                    label = label_i
                else:  # concatenate raw files
                    data = np.concatenate((data, data_i), axis=0)
                    label = np.concatenate((label, label_i), axis=0)
            else:
                print("Not existing data: ", data_dir)
                print("Data does not exist... Continuing")
                continue

        if (i + 1) == target_user:
            test_X = data
            test_y = label
            test_d = np.ones(test_y.shape) * i
        else:
            train_X = np.vstack((train_X, data))
            train_y = np.concatenate((train_y, label))
            train_d = np.concatenate((train_d, np.ones(train_y.shape) * i))

    print('real test user ->', target_user)
    print('real train X shape ->', train_X.shape)
    print('real train y shape ->', train_y.shape)
    print('real test X shape ->', test_X.shape)
    print('real test y shape ->', test_y.shape)

    return train_X, train_y, train_d, test_X, test_y, test_d


def load_mhealth(candidate_number):
    global target_train_label, target_train, target_test, target_test_label
    target_user = candidate_number
    candidate_list = MHEALTH_DATA_FILES

    train_X = np.empty((0, 23))
    test_X = np.empty((0, 23))
    train_d = np.empty((0))
    train_y = np.empty((0))
    test_y = np.empty((0))

    for i in range(0, len(candidate_list)):
        candidate = loadmat(base_dir + mhealth_dir + candidate_list[i])
        if (i + 1) == target_user:
            test_X = candidate["data"]
            test_y = candidate["label"].reshape(-1)
            test_d = np.ones(test_y.shape) * i
        else:
            train_X = np.vstack((train_X, candidate["data"]))
            train_y = np.concatenate((train_y, candidate["label"].reshape(-1)))
            train_d = np.concatenate((train_d, np.ones(train_y.shape) * i))

    print('mhealth test user ->', target_user)
    print('mhealth train X shape ->', train_X.shape)
    print('mhealth train y shape ->', train_y.shape)
    print('mhealth test X shape ->', test_X.shape)
    print('mhealth test y shape ->', test_y.shape)

    return train_X, train_y, train_d, test_X, test_y, test_d


def load_dsads(candidate_number):
    global target_train_label, target_train, target_test, target_test_label
    target_user = candidate_number
    candidate_list = DSADS_DATA_FILE

    train_X = np.empty((0, 45))
    test_X = np.empty((0, 45))
    train_d = np.empty((0))
    train_y = np.empty((0))
    test_y = np.empty((0))

    for i in range(0, len(candidate_list)):
        candidate = loadmat(base_dir + dsads_dir + candidate_list[i])
        if (i + 1) == target_user:
            test_X = candidate["data"]
            test_y = candidate["label"].reshape(-1)
            test_d = np.ones(test_y.shape) * i
        else:
            train_X = np.vstack((train_X, candidate["data"]))
            train_y = np.concatenate((train_y, candidate["label"].reshape(-1)))
            train_d = np.concatenate((train_d, np.ones(train_y.shape) * i))

    print('dsads test user ->', target_user)
    print('dsads train X shape ->', train_X.shape)
    print('dsads train y shape ->', train_y.shape)
    print('dsads test X shape ->', test_X.shape)
    print('dsads test y shape ->', test_y.shape)

    return train_X, train_y, train_d, test_X, test_y, test_d


def load_gotov(candidate_number, position):
    global target_train_label, target_train, target_test, target_test_label
    target_user = candidate_number
    candidate_list = GOTOV_DATA_FILE
    position = position

    train_X = np.empty((0, 3))
    test_X = np.empty((0, 3))
    train_d = np.empty((0))
    train_y = np.empty((0))
    test_y = np.empty((0))

    for i in range(0, len(candidate_list)):
        candidate = loadmat(base_dir + gotov_dir + candidate_list[i])
        if (i + 1) == target_user:
            test_X = candidate[position + '_x']
            test_y = candidate[position + '_y'].reshape(-1)
            test_d = np.ones(test_y.shape) * i
        else:
            train_X = np.vstack((train_X, candidate[position + '_x']))
            train_y = np.concatenate((train_y, candidate[position + '_y'].reshape(-1)))
            train_d = np.concatenate((train_d, np.ones(train_y.shape) * i))

    print('gotov test user ->', target_user)
    print('gotov train X shape ->', train_X.shape)
    print('gotov train y shape ->', train_y.shape)
    print('gotov test X shape ->', test_X.shape)
    print('gotov test y shape ->', test_y.shape)

    return train_X, train_y, train_d, test_X, test_y, test_d


class Dataset(data.Dataset):
    def __init__(self, data, label, domain, win_len=50, step_len=32, dim=None):
        self.data = data
        self.label = label
        self.domain = domain  # basically subject num array
        self.window_len = win_len
        self.step_len = step_len
        self.dim = dim
        self.slide_X = sliding_window(self.data, (self.window_len, data.shape[1]), (self.step_len, 1))
        self.slide_y = np.asarray([[i[-1]] for i in sliding_window(self.label, self.window_len, self.step_len)])
        self.slide_d = np.asarray([[i[-1]] for i in sliding_window(self.domain, self.window_len, self.step_len)])
        self.slide_X = self.slide_X.reshape((-1, self.window_len, self.dim))
        self.slide_y = self.slide_y.reshape(len(self.slide_y))
        self.slide_d = self.slide_d.reshape(len(self.slide_d))

    def __getitem__(self, index):
        X = self.slide_X[index]
        y = self.slide_y[index]
        d = self.slide_d[index]

        return X.astype(np.float32), y.astype(np.uint8), d.astype(np.uint8)

    def __len__(self):
        return len(self.slide_X)


class DataLoader():
    def initialize(self, data, label, domain, batch_size=64, win_len=50, step_len=32, dim=None):
        dataset = Dataset(data, label, domain, win_len, step_len, dim)

        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0)

    def load_data(self):
        return self.data_loader


def dataset_read(dataset, batch_size, dim, candidate, position=None):
    if dataset == 'pamap2':
        tr_X, tr_y, tr_d, te_X, te_y, te_d = load_pamap2(candidate)
    elif dataset == 'real':
        tr_X, tr_y, tr_d, te_X, te_y, te_d = load_real(candidate)
    elif dataset == 'mhealth':
        tr_X, tr_y, tr_d, te_X, te_y, te_d = load_mhealth(candidate)
    elif dataset == 'dsads':
        tr_X, tr_y, tr_d, te_X, te_y, te_d = load_dsads(candidate)
    elif dataset == 'gotov':
        tr_X, tr_y, tr_d, te_X, te_y, te_d = load_gotov(candidate, position)
    train_loader = DataLoader()
    train_loader.initialize(data=tr_X, label=tr_y, domain=tr_d, batch_size=batch_size, dim=dim)
    test_loader = DataLoader()
    test_loader.initialize(data=te_X, label=te_y, domain=te_d, batch_size=batch_size, dim=dim)

    dataset_train, dataset_test = train_loader.load_data(), test_loader.load_data()

    return [dataset_train], dataset_test


def dataset_selection(args):
    source_loaders, target_loader = dataset_read(args.dataset, args.batch_size, args.input_dim, int(args.target_domain),
                                                 args.position)

    return source_loaders, target_loader
