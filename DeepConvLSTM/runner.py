import os
import torch
from time import gmtime, strftime
import numpy as np
import pickle
from numpy.lib.stride_tricks import as_strided as ast
from scipy.stats import mode
import torch.utils.data as Data
import torch.nn as nn
import sklearn.metrics as metrics


from DeepConvLSTMModel import HARModel, init_weights

# window should be 500 ms.. We are going to use 33.3 hz



def get_train_test_data(target_subject, number_of_subjects=8):
    train_X = np.empty((0, WINDOW_SIZE, 52))
    train_y = np.empty((0, ACTIVITY_NUM))
    test_X = np.empty((0, WINDOW_SIZE, 52))
    test_y = np.empty((0, ACTIVITY_NUM))
    for i in range(number_of_subjects):
        with open(os.path.join(base_dir + pamap2_dir + f'subject10{i + 1}' + '.pickle'), 'rb') as handle:
            whole_data = pickle.load(handle)
            data = whole_data['data']
            label = whole_data['label']
            data = sliding_window(data, ws=(WINDOW_SIZE, data.shape[1]),
                                  ss=(38, 1))  # 171 corresponds to 5.12 secs, 171 - 38 -> %78 overlap
            label = sliding_window(label, ws=WINDOW_SIZE,
                                   ss=38)  # 171 corresponds to 5.12 secs, 171 - 38 -> %78 overlap
            # take the most frequent activity within a window for labeling
            label = np.squeeze(mode(label, axis=1)[0])  # axis=1 is for the window axis
            one_hot_labels = np.zeros((len(label), ACTIVITY_NUM))
            one_hot_labels[np.arange(len(label)), label] = 1

        if (i + 1) == target_subject:
            test_X = data
            test_y = one_hot_labels
        else:
            train_X = np.vstack((train_X, data))
            train_y = np.concatenate((train_y, one_hot_labels))

    print('pamap2 test user ->', target_subject)
    print('pamap2 train X shape ->', train_X.shape)
    print('pamap2 train y shape ->', train_y.shape)
    print('pamap2 test X shape ->', test_X.shape)
    print('pamap2 test y shape ->', test_y.shape)
    return train_X, train_y, test_X, test_y

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


def train(net, train_loader, test_loader, epochs=10, batch_size=100, lr=0.001):
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    if (train_on_gpu):
        net.cuda()

    for e in range(epochs):

        # initialize hidden state
        h = net.init_hidden(batch_size)
        train_losses = []
        net.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if (train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])
            # zero accumulated gradients
            opt.zero_grad()

            # get the output from the model
            output, h = net(inputs, h, batch_size)
            loss = criterion(output, targets)
            train_losses.append(loss.item())
            loss.backward()
            opt.step()

        val_h = net.init_hidden(batch_size)
        val_losses = []

        net.eval()

        all_ground_truth = []
        all_predictions = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                val_h = tuple([each.data for each in val_h])

                if (train_on_gpu):
                    inputs, targets = inputs.cuda(), targets.cuda()

                output, val_h = net(inputs, val_h, batch_size)

                val_loss = criterion(output, targets)
                val_losses.append(val_loss.item())

                top_p, top_class = output.topk(1, dim=1)
                # equals = top_class == targets.view(*top_class.shape).long()
                all_ground_truth = all_ground_truth + torch.max(targets, 1)[1].cuda().view(-1).tolist()
                all_predictions = all_predictions + top_class.view(-1).tolist()
                # accuracy += torch.mean(equals.type(torch.FloatTensor))
                # f1score += metrics.f1_score(top_class.cpu(), targets.view(*top_class.shape).long().cpu(),
                #                             average='weighted')

        net.train()  # reset to train mode after iterationg through validation data
        f1 = metrics.f1_score(all_ground_truth, all_predictions, average='macro')
        acc = metrics.accuracy_score(all_ground_truth, all_predictions)
        print("Epoch: ", e)
        print(f"avg acc: {acc}")
        print(f"f1 score: {f1}")
        test_results.append([acc, f1, e])
        result_np = np.array(test_results, dtype=float)
        np.savetxt(result_csv, result_np, fmt='%.4f', delimiter=',')


if __name__ == "__main__":

    machine = 'windows'
    if machine == 'linux':
        base_dir = r'/home/cmyldz/Dropbox (GaTech)/DisentangledHAR/'
    else:
        base_dir = r'C:\Users\Cem Okan\Dropbox (GaTech)\DisentangledHAR/'

    pamap2_dir = 'PAMAP2_Dataset/PAMAP2_Dataset/Processed50Hz/'
    target_subject = 1

    num_modalities = 52  # number of sensor channels
    WINDOW_SIZE = 50  # 1 sec with 50 Hz
    ACTIVITY_NUM = 12  # pamap2
    test_results = []

    timestring = strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + "_%s" % str(
        target_subject)
    logdir = os.path.join('./logs', 'pamap2_conv2lstm', timestring)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    result_csv = logdir + '/conv2lstm_results.csv'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')

    train_x, train_y, test_x, test_y = get_train_test_data(target_subject, number_of_subjects=8)
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

    # Data is reshaped    -     basically an image of sensor signals
    # where rows are time steps, columns are modalities
    train_x = train_x.reshape((-1, WINDOW_SIZE, num_modalities))  # for input to Conv1D
    test_x = test_x.reshape((-1, WINDOW_SIZE, num_modalities))  # for input to Conv1D
    trainset = Data.TensorDataset(train_x, train_y)
    trainloader = Data.DataLoader(dataset=trainset, batch_size=100, shuffle=True, num_workers=0, drop_last=True)

    testset = Data.TensorDataset(test_x, test_y)
    testloader = Data.DataLoader(dataset=testset, batch_size=100, shuffle=True, num_workers=0, drop_last=True)

    print("Here are the shapes!")
    print(train_x.shape, train_y.shape)
    print(test_x.shape, test_y.shape)

    net = HARModel(num_sensor_channels=num_modalities,
                   window_length=WINDOW_SIZE,
                   n_classes=12)
    net.apply(init_weights)

    ## check if GPU is available
    train_on_gpu = torch.cuda.is_available()
    if (train_on_gpu):
        print('Training on GPU!')
    else:
        print('No GPU available, training on CPU; consider making n_epochs very small.')

    train(net, trainloader, testloader, epochs=150, batch_size=100, lr=0.001)