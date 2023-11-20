import numpy as np
from matplotlib import pyplot as plt
from math import tan, atan2, pi, radians as rad, degrees as deg
# Import original modules
from utilities import classify_dict_kargs


# Frequently used plot variables
cb_rgb = ["#FF4B00", "#03AF7A", "#0041FF"]
cb_rgby = ["#FF4B00", "#03AF7A", "#0041FF", "#FFF100"]
tab_rgb = ["tab:red", "tab:green", "tab:blue"]
tab_rgbo = ["tab:red", "tab:green", "tab:blue", "tab:orange"]
trans_labels = ["x", "y", "z"]
trans_labels_w_tgt = [trans_labels, [label + "-tgt" for label in trans_labels]]
orien_labels = ["r", "p", "y"]
orien_labels_w_tgt = [orien_labels, [label + "-tgt" for label in orien_labels]]
act_tgt_ls = ["-", "--"]


def ax_plot_lines(ax, x, ys, ylabel, c=cb_rgb, alpha=0.5, **kargs):
    # Update default plot variables with kargs
    defaults = {"c": c, "alpha": alpha}
    defaults.update(kargs)
    # Separate keyword arguments into array like ones (excluding strings) and
    # the others
    arr_like, others = classify_dict_kargs(defaults)

    ys = np.asarray(ys)
    ys = np.atleast_2d(ys) if 1 == len(ys.shape) else ys.T
    len_x = len(x)
    len_y = ys.shape[1]
    clip = min((len_x, len_y))

    for i, y in enumerate(ys):
        ax.plot(
            x[:clip],
            y[:clip],
            **others,
            **{k: v[i % len(v)] for k, v in arr_like.items()},
            )

    _, labels = plt.gca().get_legend_handles_labels()

    ax.set(ylabel=ylabel)
    if labels:
        ax.legend()

    return 1 if clip < len_y else 2 if clip < len_x else 0


def ax_plot_lines_w_tgt(ax, time, act, tgt, ylabel, **kargs):
    for ys, ls in zip((act, tgt), act_tgt_ls):
        ax_plot_lines(ax, time, ys, ylabel, ls=ls, **kargs)


def axes_plot_frc(axes, time, act, tgt, ylabel=None, **kargs):
    ylabel = "frc [N]" if ylabel is None else ylabel
    ax_plot_lines_w_tgt(
        axes[0], time, act[:, :2], tgt[:, :2], ylabel, c=cb_rgb[:2], **kargs)
    ax_plot_lines_w_tgt(
        axes[1], time, act[:, 2:], tgt[:, 2:], ylabel, c=cb_rgb[2:], **kargs)
