from matplotlib import pyplot as plt
import random
import numpy as np

def plot_single_continuous_plot(x_axis, y_axis, title, x_label_pick, y_label_pick, x_lim=None, y_lim=None, color=None, linestyle=None,
                                show_enable=False, hold_on=False, legend_enable=False, label=None, save_path=None,
                                only_curve_visible=False):
    """
    Plots the given data
    :param x_axis: What will be the data on the x-axis?
    :type x_axis: 1D array
    :param y_axis: What will be the data on the y-axis?
    :type y_axis: 1D array
    :param title: title of the plot
    :type title: str
    :param x_label_pick: x-label name of the plot
    :type x_label_pick: str
    :param y_label_pick: y-label name of the plot
    :type y_label_pick: str
    :param color: specific color wanted. If none, it chooses randomly from the list below
    :type color: str
    :param linestyle: specific linestyle wanted. If none, it chooses randomly from the list below
    :type linestyle: str
    :param show_enable: True if you want to display. If you want to use hold on option, set it to false for the image
                        at the background
    :type show_enable: bool
    :param hold_on: True if you want to plot the current plot on top of the previous one
    :type hold_on: bool
    :param legend_enable: True if you want legend. Useful if you use hold on.
    :type legend_enable: bool
    :param label: Label of the plot that will be seen in legend
    :type label: str
    """
    COLORS = ["rosybrown", "indianred", "firebrick", "dimgrey", "tab:orange", "tab:red", "tab:blue", "tab:purple",
              "tab:green", "tab:brown", "forestgreen", "seagreen", "darkolivegreen", "darkslategrey", "teal",
              "darkcyan",
              "deepskyblue", "crimson", "mediumvioletred", "darkslateblue", "navy", "midnightblue", "cornflowerblue",
              "chocolate", "darkorange", "olivedrab", "darkred", "maroon", "royalblue", "mediumseagreen", "tomato",
              "palevioletred"]
    LINESTYLES = ["solid", "dashed", "dotted", "dashdot"]
    FIGSIZE = (10, 7)
    TITLE_FONT_SIZE = 16
    LABEL_FONT_SIZE = 12
    if color is None:
        color_pick = random.choice(COLORS)
        print("Chosen color is: ", color_pick)
    else:
        color_pick = color
    if linestyle is None:
        linestyle = LINESTYLES[0]  # Solid is the ordınary choice

    if not hold_on:
        plt.figure(figsize=FIGSIZE)
    plt.title(title, fontsize=TITLE_FONT_SIZE, fontweight="bold")
    plt.xlabel(x_label_pick, fontsize=LABEL_FONT_SIZE, fontweight="normal")
    plt.ylabel(y_label_pick, fontsize=LABEL_FONT_SIZE, fontweight="normal")
    if x_axis is None:  # just use arange
        x_axis = np.arange(len(y_axis))
    plt.plot(x_axis, y_axis, color=color_pick, linestyle=linestyle, label=label, alpha=0.5)

    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)

    if only_curve_visible:
        # disable ticks
        plt.xticks([])
        plt.yticks([])
        # remove title
        plt.title("")
        # remove labels
        plt.xlabel("")
        plt.ylabel("")
        # remove the box
        plt.box(False)

    if legend_enable:
        plt.legend()
    if save_path is not None:
        image_format = 'png'  # e.g .png, .svg, etc
        plt.savefig(save_path, format=image_format, dpi=1200)
    if show_enable:
        plt.show()


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          save_path=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()