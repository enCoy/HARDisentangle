import torch
import numpy as np
from Utils.Visualizers import plot_single_continuous_plot
import os
from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import as_strided as ast

# PAMAP2 SPECIFIC
def get_activity_columns(data_of_interest = ['acc', 'gyr']):
    body_device_locations = ['chest', 'forearm', 'head', 'shin', 'thigh', 'upperarm', 'waist']
    column_list = []
    for device_loc in body_device_locations:
        for data_name in data_of_interest:
            column_list.append(device_loc + '_' + data_name + '_x')
            column_list.append(device_loc + '_' + data_name + '_y')
            column_list.append(device_loc + '_' + data_name + '_z')
    return column_list



# GENERAL
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

def calculate_amount_of_slide(window_num_samples, sliding_window_overlap_ratio):
    amount_overlapped = int(window_num_samples) *sliding_window_overlap_ratio
    return int(window_num_samples - amount_overlapped)


def standardize_data(data_to_standardize, standardizer, is_generated):
    # data to standardize should be shaped (N x T x F)  T: num_windows F: Feature size
    # reshape it into (N x F)
    previous_shape = data_to_standardize.shape
    data_to_standardize = np.reshape(data_to_standardize, (-1, data_to_standardize.shape[-1]))
    if not is_generated:
        standardizer.fit(data_to_standardize)
    data_to_standardize = standardizer.transform(data_to_standardize)
    data_to_standardize = np.reshape(data_to_standardize, previous_shape)
    return data_to_standardize, standardizer

def convert_to_torch(train_X, train_y, test_X, test_y):
    # train X
    shape = train_X.shape
    train_X = torch.from_numpy(train_X.astype(float))
    train_X = train_X.type(torch.FloatTensor).cuda()
    # train Y
    train_y = torch.from_numpy(train_y.astype(float))
    train_y = train_y.type(torch.FloatTensor).cuda()
    # test X
    test_X = torch.from_numpy(test_X.astype(float))
    test_X = test_X.type(torch.FloatTensor).cuda()
    # test y
    test_y = torch.from_numpy(test_y.astype(float))
    test_y = test_y.type(torch.FloatTensor).cuda()
    print("Shapes after converting to torch:")
    print("Train x shape: ", train_X.size())
    print("Train y shape: ", train_y.size())
    print("Test x shape: ", test_X.size())
    print("Test y shape: ", test_y.size())
    return train_X, train_y, test_X, test_y

def convert_to_torch_v2(train_X, train_y, test_X, test_y):
    # train X
    shape = train_X.shape  # 4 dimensional
    train_X = torch.from_numpy(np.reshape(train_X.astype(float), [shape[0], shape[1], shape[2], shape[3]]))
    train_X = train_X.type(torch.FloatTensor).cuda()
    # train Y
    train_y = torch.from_numpy(train_y)
    train_y = train_y.type(torch.FloatTensor).cuda()
    # test X
    shape = test_X.shape
    test_X = torch.from_numpy(np.reshape(test_X.astype(float), [shape[0], shape[1], shape[2], shape[3]]))
    test_X = test_X.type(torch.FloatTensor).cuda()
    # test y
    test_y = torch.from_numpy(test_y.astype(np.float32))
    test_y = test_y.type(torch.FloatTensor).cuda()
    print("Shapes after converting to torch:")
    print(f"Train x shape: ", train_X.size())
    print(f"Train y shape: ", train_y.size())
    print(f"Test x shape: ", test_X.size())
    print(f"Test y shape: ", test_y.size())
    return train_X, train_y, test_X, test_y

def plot_loss_curves(train_loss, test_loss, save_loc=None, show_fig=True, title='Loss Curve'):
    epochs = np.arange(len(train_loss)) + 1
    plot_single_continuous_plot(epochs, train_loss, title, 'Epoch', 'Loss', color='tab:red', label='train')
    plot_single_continuous_plot(epochs, test_loss, title, 'Epoch', 'Loss', color='navy',
                                show_enable=show_fig, hold_on=True, legend_enable=True, label='test',
                                save_path=os.path.join(save_loc, f"{title}.png"))

def save_models(models, save_dir):
    # models is a dict --- key: name of the model, value: model itself
    # save glob model
    model_names = list(models.keys())
    for model_name in model_names:
        model = models[model_name]
        # save state dict
        torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_stateDict.pth"))
        # save model
        torch.save(model, os.path.join(save_dir, f"{model_name}.pth"))

def save_best_models(models, save_dir):
    # models is a dict --- key: name of the model, value: model itself
    # save glob model
    model_names = list(models.keys())
    for model_name in model_names:
        model = models[model_name]
        # save state dict
        torch.save(model.state_dict(), os.path.join(save_dir, f"best_{model_name}_stateDict.pth"))
        # save model
        torch.save(model, os.path.join(save_dir, f"best_{model_name}.pth"))


from pathlib import Path
def convert_win_path(win_path):
    path_universal = Path(win_path)
    return path_universal


# custom loss functions
def contrastive_loss_criterion(x1, x2, label, margin: float = 3):
    """
    Computes Contrastive Loss
    """
    # label = 1 means positive samples, 0 negative samples
    dist = torch.nn.functional.pairwise_distance(x1, x2)
    # if the difference is above margin, that means they are separated enough and no more separation will occur
    loss = (1 - label) * torch.pow(dist, 2) + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)

    return loss

def mi_estimator(mine, x, z, z_marg):
    joint, marginal = mine(x, z), mine(x, z_marg)
    return torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))

def mutual_information(mine, x, z, z_marg):
    joint, marginal = mine(x, z), mine(x, z_marg)
    et = torch.exp(marginal)
    mi_lb = torch.mean(joint) - torch.log(torch.mean(et))
    return mi_lb, joint, et


# fusion of predictions
def fuse_predictions(predictions, confidences, modalities, num_classes, fusion_mode='majority_vote'):
    if fusion_mode == 'majority_vote':
        fused = fuse_predictions_from_multiple_modalities_majority_vote(predictions, modalities, num_classes)
    elif fusion_mode == 'confidence_weighted':
        fused =  fuse_predictions_from_multiple_modalities_based_confidence_weight(predictions, confidences, modalities, num_classes)
    elif fusion_mode == 'max_confidence':
        fused = fuse_predictions_from_multiple_modalities_based_max_confidence(predictions, confidences, modalities, num_classes)
    else:
        print("FUSION MODE IS NOT VALID!")
        fused = None
    return fused


def fuse_predictions_from_multiple_modalities_majority_vote(predictions, modalities, num_classes):
    # fuse predictions - take the majority
    fused_predictions = []
    for j in range(len(predictions[modalities[0]])):  # over all examples
        classes_predicted = np.zeros(num_classes)
        modality_predictions = np.array([predictions[m][j] for m in modalities])
        # print("modality predictions: ", modality_predictions)
        for kk in range(len(modality_predictions)):
            classes_predicted[modality_predictions[kk]] += 1
        # print("class predicted: ", classes_predicted)
        fused_predictions.append(np.argmax(classes_predicted))
    return np.array(fused_predictions)

def fuse_predictions_from_multiple_modalities_based_max_confidence(predictions, confidences, modalities, num_classes):
    # fuse predictions - take the majority
    fused_predictions = []
    for j in range(len(predictions[modalities[0]])):  # over all examples
        classes_predicted = np.zeros(num_classes)
        modality_predictions = np.array([predictions[m][j] for m in modalities])
        modality_predictions_confidences = np.array([confidences[m][j] for m in modalities])
        # for kk in range(len(modality_predictions)):
        #     classes_predicted[modality_predictions[kk]] += modality_predictions_confidences[kk]
        # # print("class predicted: ", classes_predicted)
        # fused_predictions.append(np.argmax(classes_predicted))
        max_confidence_idx = np.argmax(modality_predictions_confidences)
        fused_predictions.append(modality_predictions[max_confidence_idx])
    return np.array(fused_predictions)

def fuse_predictions_from_multiple_modalities_based_confidence_weight(predictions, confidences, modalities, num_classes):
    # fuse predictions - take the majority
    fused_predictions = []
    for j in range(len(predictions[modalities[0]])):  # over all examples
        classes_predicted = np.zeros(num_classes)
        modality_predictions = np.array([predictions[m][j] for m in modalities])
        modality_predictions_confidences = np.array([confidences[m][j] for m in modalities])
        for kk in range(len(modality_predictions)):
            classes_predicted[modality_predictions[kk]] += modality_predictions_confidences[kk]
        fused_predictions.append(np.argmax(classes_predicted))
    return np.array(fused_predictions)

def get_standardizer(data):
    # assuming that data has shape N x T x F, this one standardizes over F
    # Calculate mean and standard deviation along the feature axis
    mean_values = np.mean(data, axis=(0, 1), keepdims=True)
    std_values = np.std(data, axis=(0, 1), keepdims=True)
    # Standardize the dataset
    return mean_values, std_values
