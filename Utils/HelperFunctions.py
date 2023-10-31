import torch
import numpy as np
from Utils.Visualizers import plot_single_continuous_plot
import os

def convert_to_torch(train_X, train_y, test_X, test_y):
    # train X
    shape = train_X.shape
    train_X = torch.from_numpy(np.reshape(train_X.astype(float), [shape[0], shape[1], shape[2]]))
    train_X = train_X.type(torch.FloatTensor).cuda()
    # train Y
    train_y = torch.from_numpy(train_y)
    train_y = train_y.type(torch.FloatTensor).cuda()
    # test X
    test_X = torch.from_numpy(np.reshape(test_X.astype(float), [test_X.shape[0], test_X.shape[1], test_X.shape[2]]))
    test_X = test_X.type(torch.FloatTensor)
    # test y
    test_y = torch.from_numpy(test_y.astype(np.float32))
    test_y = test_y.type(torch.FloatTensor)
    print("Shapes after converting to torch:")
    print("Train x shape: ", train_X.size())
    print("Train y shape: ", train_y.size())
    print("Test x shape: ", test_X.size())
    print("Test y shape: ", test_y.size())
    return train_X, train_y, test_X, test_y

def plot_loss_curves(train_loss, test_loss, save_loc=None, show_fig=True):
    epochs = np.arange(len(train_loss)) + 1
    plot_single_continuous_plot(epochs, train_loss, 'Loss curve', 'Epoch', 'Loss', color='tab:red', label='train')
    plot_single_continuous_plot(epochs, test_loss, 'Loss curve', 'Epoch', 'Loss', color='navy',
                                show_enable=show_fig, hold_on=True, legend_enable=True, label='test',
                                save_path=os.path.join(save_loc, f"LossCurve.png"))