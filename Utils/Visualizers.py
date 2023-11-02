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