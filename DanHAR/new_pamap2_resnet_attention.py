# THIS IS THE ONE I SHOULD RUN
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import torch
print("If I do not print this, it cannot use correct device: ", torch.cuda.is_available())
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
from time import gmtime, strftime
from torch.utils.data import Dataset,DataLoader
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib
import os
import argparse
from torchstat import stat
from scipy.stats import mode
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,confusion_matrix
import sklearn.metrics as sm
WINDOW_SIZE = 50
ACTIVITY_NUM = 12
def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.
    Parameters
        shape - an int, or a tuple of ints
    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')

def sliding_window(a, ws, ss=None, flatten=True):
    '''
    Return a sliding window over a in any number of dimensions
    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.
    Returns
        an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError( \
            'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError( \
            'ws cannot be larger than a in any dimension.\
     a.shape was %s and ws was %s' % (str(a.shape), str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a, shape=newshape, strides=newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    #     dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)

PAMAP2_DATA_FILES = ['subject101',
                     'subject102',
                     'subject103',
                     'subject104',
                     'subject105',
                     'subject106',
                     'subject107',
                     'subject108']

# def get_train_test_data(target_subject, number_of_subjects=8):
#         train_X = np.empty((0, WINDOW_SIZE, 52))
#         train_y = np.empty((0, ACTIVITY_NUM))
#         test_X = np.empty((0, WINDOW_SIZE, 52))
#         test_y = np.empty((0, ACTIVITY_NUM))
#         for i in range(number_of_subjects):
#             with open(os.path.join(base_dir + pamap2_dir + f'subject10{i+1}' + '.pickle'), 'rb') as handle:
#                 whole_data = pickle.load(handle)
#                 data = whole_data['data']
#                 label = whole_data['label']
#                 data = sliding_window(data, ws=(WINDOW_SIZE, data.shape[1]), ss=(11, 1))  # 50 corresponds to 1 secs, 50 - 11 -> %78 overlap
#                 label = sliding_window(label, ws=WINDOW_SIZE, ss=11)
#                 # take the most frequent activity within a window for labeling
#                 label = np.squeeze(mode(label, axis=1)[0])  # axis=1 is for the window axis
#                 one_hot_labels = np.zeros((len(label), ACTIVITY_NUM))
#                 one_hot_labels[np.arange(len(label)), label] = 1
#
#             if (i + 1) == target_subject:
#                 test_X = data
#                 test_y = one_hot_labels
#             else:
#                 train_X = np.vstack((train_X, data))
#                 train_y = np.concatenate((train_y, one_hot_labels))
#
#         print('pamap2 test user ->', target_subject)
#         print('pamap2 train X shape ->', train_X.shape)
#         print('pamap2 train y shape ->', train_y.shape)
#         print('pamap2 test X shape ->', test_X.shape)
#         print('pamap2 test y shape ->', test_y.shape)
#         return train_X, train_y, test_X, test_y

def get_activity_columns(data_of_interest = ['acc', 'gyr']):
    body_device_locations = ['chest', 'forearm', 'head', 'shin', 'thigh', 'upperarm', 'waist']
    column_list = []
    for device_loc in body_device_locations:
        for data_name in data_of_interest:
            column_list.append(device_loc + '_' + data_name + '_x')
            column_list.append(device_loc + '_' + data_name + '_y')
            column_list.append(device_loc + '_' + data_name + '_z')
    return column_list

def get_train_test_data(target_subject, number_of_subjects=8, dataset_of_interest='pamap2'):
    train_X = np.empty((0, WINDOW_SIZE, num_modalities))
    train_y = np.empty((0, ACTIVITY_NUM))
    test_X = np.empty((0, WINDOW_SIZE, num_modalities))
    test_y = np.empty((0, ACTIVITY_NUM))
    for i in range(number_of_subjects):
        if dataset_of_interest == 'pamap2':
            with open(os.path.join(base_dir + har_data_dir + f'subject10{i + 1}' + '.pickle'), 'rb') as handle:
                whole_data = pickle.load(handle)
                data = whole_data['data']
                label = whole_data['label']
                data = sliding_window(data, ws=(WINDOW_SIZE, data.shape[1]),
                                      ss=(25, 1))  # 50 corresponds to 1 secs, 50 - 11 -> %50 overlap
                label = sliding_window(label, ws=WINDOW_SIZE,
                                       ss=25)

                # take the most frequent activity within a window for labeling
                label = np.squeeze(mode(label, axis=1)[0])  # axis=1 is for the window axis
                one_hot_labels = np.zeros((len(label), ACTIVITY_NUM))
                one_hot_labels[np.arange(len(label)), label] = 1
        else:
            if i+1 == 4 or i+1 == 7:  # these subjects have multiple sessions of climbing up and down:
                activity_names =  ['climbingdown_1', 'climbingdown_2', 'climbingdown_3',
                                   'climbingup_1', 'climbingup_2', 'climbingup_3',
                                   'jumping', 'lying', 'running', 'sitting', 'standing', 'walking']
            else:
                activity_names =  ['climbingdown', 'climbingup', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking']
            data = None
            one_hot_labels = None
            for activity_name in activity_names:
                data_dir = os.path.join(base_dir + har_data_dir + f'subject{i+1}', activity_name + '.csv')
                if os.path.exists(data_dir):
                    activity_df = pd.read_csv(data_dir)
                    columns = get_activity_columns(['acc', 'gyr'])
                    data_i = activity_df[columns].values
                    label_i = activity_df['activity_id'].values.astype(int)
                    data_i = sliding_window(data_i, ws=(WINDOW_SIZE, data_i.shape[1]),
                                          ss=(25, 1))  # 50 corresponds to 1 secs, 50 - 11 -> %50 overlap
                    label_i = sliding_window(label_i, ws=WINDOW_SIZE,
                                          ss=25)  # 50 corresponds to 1 secs, 50 - 11 -> %50 overlap
                    label_i = np.squeeze(mode(label_i, axis=1)[0])  # axis=1 is for the window axis
                    one_hot_labels_i = np.zeros((len(label_i), ACTIVITY_NUM))
                    one_hot_labels_i[np.arange(len(label_i)), label_i] = 1
                    if data is None:
                        data = data_i
                        one_hot_labels = one_hot_labels_i
                    else: # concatenate raw files
                        data = np.concatenate((data, data_i), axis=0)
                        one_hot_labels = np.concatenate((one_hot_labels, one_hot_labels_i), axis=0)
                else:
                    print("Not existing data: ", data_dir)
                    print("Data does not exist... Continuing")
                    continue
        if (i + 1) == target_subject:
            test_X = data
            test_y = one_hot_labels
        else:
            train_X = np.vstack((train_X, data))
            train_y = np.concatenate((train_y, one_hot_labels))

    print(f'{dataset_of_interest} test user ->', target_subject)
    print(f'{dataset_of_interest} train X shape ->', train_X.shape)
    print(f'{dataset_of_interest} train y shape ->', train_y.shape)
    print(f'{dataset_of_interest} test X shape ->', test_X.shape)
    print(f'{dataset_of_interest} test y shape ->', test_y.shape)
    return train_X, train_y, test_X, test_y

machine = 'windows'
if machine == 'linux':
    base_dir = r'/home/cmyldz/Dropbox (GaTech)/DisentangledHAR/'
else:
    base_dir = r'C:\Users\Cem Okan\Dropbox (GaTech)\DisentangledHAR/'

har_data_name = 'real'  # 'real' or 'pamap2'
if har_data_name == 'pamap2':
    har_data_dir = 'PAMAP2_Dataset/PAMAP2_Dataset/Processed50Hz/'
    num_modalities = 52  # number of sensor channels
    ACTIVITY_NUM = 12  # pamap2
elif har_data_name == 'real':
    har_data_dir = 'realworld2016_dataset/Processed/'
    num_modalities = 42  # number of sensor channels
    ACTIVITY_NUM = 8

target_subject = 1
os.environ['CUDA_VISIBLE_DEVICES']='1'

parser = argparse.ArgumentParser(description='PyTorch Har Training')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

timestring = strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + "_%s" % str(
            target_subject)
logdir = os.path.join('./logs', f'{har_data_name}_danhar', timestring)
if not os.path.exists(logdir):
    os.makedirs(logdir)

result_csv = logdir + '/danhar_results.csv'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')


train_x, train_y, test_x, test_y = get_train_test_data(target_subject, number_of_subjects=8, dataset_of_interest=har_data_name)
# train_x = np.load('/home/gaowenbing/desktop/dd/Torch_Har_cbam/HAR_Dataset/pamap2_/train_x.npy')
shape = train_x.shape
train_x = torch.from_numpy(np.reshape(train_x.astype(float), [shape[0], 1, shape[1], shape[2]]))
train_x = train_x.type(torch.FloatTensor).cuda()

# train_y = (np.load('/home/gaowenbing/desktop/dd/Torch_Har_cbam/HAR_Dataset/pamap2_/train_y_p.npy'))
train_y = torch.from_numpy(train_y)
train_y = train_y.type(torch.FloatTensor).cuda()


# test_x = np.load('/home/gaowenbing/desktop/dd/Torch_Har_cbam/HAR_Dataset/pamap2_/test_x.npy')
test_x = torch.from_numpy(np.reshape(test_x.astype(float), [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]))
test_x = test_x.type(torch.FloatTensor)

# test_y = np.load('/home/gaowenbing/desktop/dd/Torch_Har_cbam/HAR_Dataset/pamap2_/test_y_p.npy')
test_y = torch.from_numpy(test_y.astype(np.float32))
test_y = test_y.type(torch.FloatTensor)

print("Here are the shapes!")
print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
# torch.Size([5568, 1, 171, 40]) torch.Size([5568, 18])  18 IS POSSIBLY NUMBER OF ACTIVITIES
# torch.Size([2048, 1, 171, 40]) torch.Size([2048, 18])
trainset = Data.TensorDataset(train_x, train_y)
trainloader = Data.DataLoader(dataset=trainset, batch_size=300, shuffle=True, num_workers=0)

testset = Data.TensorDataset(test_x, test_y)
testloader = Data.DataLoader(dataset=testset, batch_size=300, shuffle=True, num_workers=0)

test_results = []

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=(7,1)):
        super(SpatialAttention, self).__init__()

        assert kernel_size in ((3,1), (7,1)), 'kernel size must be 3 or 7'
        padding = (3,0) if kernel_size == (7,1) else (1,0)

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def plot_confusion(comfusion,class_data):
    plt.figure(figsize=(12,9))
    plt.rcParams['font.family'] = ['Times New Roman']
    classes = class_data
    plt.imshow(comfusion, interpolation='nearest', cmap=plt.cm.Reds)  # 按照像素显示出矩阵
    plt.title('confusion_matrix',fontsize = 12)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,rotation=315)
    plt.yticks(tick_marks, classes)
    plt.tick_params(labelsize=12)
    thresh = comfusion.max() / 2.
    print("comfusion: ", comfusion)
    # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
    # ij配对，遍历矩阵迭代器
    print("len classes: ", len(classes))
    iters = np.reshape([[[i, j] for j in range(len(classes))] for i in range(len(classes))], (comfusion.size, 2))
    for i, j in iters:
        plt.text(j, i, format(comfusion[i, j]),verticalalignment="center",horizontalalignment="center")  # 显示对应的数字

    plt.ylabel('Real label',fontsize = 12)
    plt.xlabel('Predicted label',fontsize = 12)

    plt.tight_layout()
    # plt.savefig('/home/gaowenbing/desktop/dd/Torch_Har_cbam/store_visual/confusion_matrix/pamap2_resnet_cbam/pamap2_resnet_cbam.png')
    plt.show()


class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()

        # print(channel_in, channel_out,  kernel, stride, bias,'channel_in, channel_out, kernel, stride, bias')
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
        )
        self.ca1 = ChannelAttention(128)
        self.sa1 = SpatialAttention()

        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
        )
        self.ca2 = ChannelAttention(256)
        self.sa2 = SpatialAttention()

        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(384),
            nn.ReLU(True)
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(384),
        )
        self.ca3 = ChannelAttention(384)
        self.sa3 = SpatialAttention()

        self.fc = nn.Sequential(
            # nn.Linear(76800, 18)  # for some reason they took all 18 actions and took only 40 inputs
            # nn.Linear(99840, 12)  # 5.12 seconds - 171 samples window length
            nn.Linear(384 * num_modalities, ACTIVITY_NUM)  # 50 hz, 50 samples = 1 sec signal
        )

    def forward(self, x):
        # print(x.shape)
        h1 = self.Block1(x)
        # print(h1.shape)
        r = self.shortcut1(x)
        # print(r.shape)
        h1 = self.ca1(h1) * h1
        h1=  self.sa1(h1) * h1
        h1 = h1 + r
        # print(h1.shape)
        h2 = self.Block2(h1)
        # print(h2.shape)
        r = self.shortcut2(h1)
        # print(r.shape)
        h2 = self.ca2(h2) * h2
        h2 = self.sa2(h2) * h2
        h2 = h2 + r
        # print(h2.shape)
        h3 = self.Block3(h2)
        # print(h3.shape)
        r = self.shortcut3(h2)
        # print(r.shape)
        h3 = self.ca3(h3) * h3
        h3 = self.sa3(h3) * h3
        h3 = h3 + r
        x = h3.view(h3.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        x = nn.LayerNorm(x.size())(x.cpu())
        x = x.cuda()
        return x


# Model
print('==> Building model..')

net = resnet()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/ckpt.pth')
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=5e-4,weight_decay=1e-3)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
#
def flat(data):
    data=np.argmax(data,axis=1)
    return data

epoch_list=[]
error_list=[]

# Training


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    total = 0
    total=total
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx % 30 == 0:
            print(f"Batch {batch_idx} is started!")
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        inputs=inputs.type(torch.FloatTensor)
        inputs,targets=inputs.cuda(),targets
        outputs = net(inputs)
        # targets=torch.max(targets, 1)[1]
        # print(targets)
        loss = criterion(outputs,torch.max(targets, 1)[1].long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        # predicted = torch.max(predicted, 1)[1].cuda()
        targets=torch.max(targets, 1)[1].cuda()
        predicted=predicted
        taccuracy = (torch.sum(predicted == targets.long()).type(torch.FloatTensor) / targets.size(0)).cuda()
        # print(type(predicted),type(targets),predicted,targets,'type(predicted),type(targets)')
        # correct += predicted.eq(targets).sum().item()
        train_error = 1 - taccuracy.item()
    return train_loss

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    avg_acc = 0
    all_ground_truth = []
    all_predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.type(torch.FloatTensor)
            inputs, targets = inputs.cuda(), targets
            outputs = net(inputs)
            # targets=torch.max(targets, 1)[1]
            # print(targets)
            loss = criterion(outputs, torch.max(targets, 1)[1].long())
            scheduler.step()
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            # predicted = torch.max(predicted, 1)[1].cuda()
            targets = torch.max(targets, 1)[1].cuda()  # [1] for taking arg_max
            taccuracy = (torch.sum(predicted == targets.long()).type(torch.FloatTensor) / targets.size(0)).cuda()
            avg_acc += taccuracy.item()
            all_ground_truth = all_ground_truth + targets.view(-1).tolist()
            all_predictions =  all_predictions + predicted.view(-1).tolist()
            # print(type(predicted),type(targets),predicted,targets,'type(predicted),type(targets)')
            # correct += predicted.eq(targets).sum().item()
            test_error=1-taccuracy.item()
            # print('test:', taccuracy.item(), '||', test_error)
            epoch_list.append(epoch)
            # print(epoch_list)
            # accuracy_list.append(taccuracy.item())
            error_list.append(test_error)
            # confusion = sm.confusion_matrix(targets.cpu().numpy(), predicted.cpu().numpy())
            # print('The confusion matrix is：', confusion, sep='\n')
            # plot_confusion(confusion,
            #                ['Lying', 'Sitting', 'Standing', 'Walking', 'Running', 'Cycling', 'Nordic walking',
            #                 'Ascending stairs', 'Descending stairs', 'Vacuum cleaning', 'Ironing', 'Rope jumping'])
            # print(error_list)
            # np.save('/home/gaowenbing/desktop/dd/Torch_Har_cbam/store_visual/pamap2/epoch_resnet_att_1.npy',epoch_list)
            # np.save('/home/gaowenbing/desktop/dd/Torch_Har_cbam/store_visual/pamap2/error_resnet_att_1.npy',error_list)

    avg_acc /= len(testloader)
    f1 = f1_score(all_ground_truth, all_predictions, average='macro')
    print(f"avg acc: {avg_acc}")
    print(f"f1 score: {f1}")

    test_results.append([avg_acc, f1, epoch])
    result_np = np.array(test_results, dtype=float)
    np.savetxt(result_csv, result_np, fmt='%.4f', delimiter=',')
    return test_loss

for epoch in range(start_epoch, start_epoch+500):
    train_loss = train(epoch)
    print(f"Train loss: {train_loss}")
    test_loss = test(epoch)
    print(f"Test loss: {test_loss}")


