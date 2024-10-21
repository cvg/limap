# Author: Rémi Pautrat

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_matches(kpts0, kpts1, color=None, lw=1.5, ps=4, indices=(0, 1)):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
    """
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    ax0, ax1 = ax[indices[0]], ax[indices[1]]
    fig.canvas.draw()

    assert len(kpts0) == len(kpts1)
    if color is None:
        color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    # transform the points into the figure coordinate system
    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax0.transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax1.transData.transform(kpts1))
    fig.lines += [
        matplotlib.lines.Line2D(
            (fkpts0[i, 0], fkpts1[i, 0]),
            (fkpts0[i, 1], fkpts1[i, 1]),
            zorder=1,
            transform=fig.transFigure,
            c=color[i],
            linewidth=lw,
        )
        for i in range(len(kpts0))
    ]

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_lines(
    lines,
    line_colors="orange",
    point_color="cyan",
    ps=4,
    lw=2,
    indices=(0, 1),
    alpha=1,
):
    """Plot lines and endpoints for existing images.
    Args:
        lines: list of ndarrays of size (N, 2, 2).
        line_colors: string, or list of list of tuples (one for per line).
        point_color: unique color for all endpoints.
        ps: size of the keypoints as float pixels.
        lw: line width as float pixels.
        indices: indices of the images to draw the matches on.
        alpha: alpha transparency.
    """
    if not isinstance(line_colors, list):
        line_colors = [[line_colors] * len(line) for line in lines]
    for i in range(len(lines)):
        if (not isinstance(line_colors[i], list)) and (
            not isinstance(line_colors[i], np.ndarray)
        ):
            line_colors[i] = [line_colors[i]] * len(lines[i])

    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]
    fig.canvas.draw()

    # Plot the lines and junctions
    for a, line, lc in zip(axes, lines, line_colors):
        for i in range(len(line)):
            mpl_line = matplotlib.lines.Line2D(
                (line[i, 0, 0], line[i, 1, 0]),
                (line[i, 0, 1], line[i, 1, 1]),
                zorder=1,
                c=lc[i],
                linewidth=lw,
                alpha=alpha,
            )
            a.add_line(mpl_line)
        pts = line.reshape(-1, 2)
        a.scatter(
            pts[:, 0],
            pts[:, 1],
            c=point_color,
            s=ps,
            linewidths=0,
            zorder=2,
            alpha=alpha,
        )


def plot_color_line_matches(lines, correct_matches=None, lw=2, indices=(0, 1)):
    """Plot line matches for existing images with multiple colors.
    Args:
        lines: list of ndarrays of size (N, 2, 2).
        correct_matches: bool array of size (N,) indicating correct matches.
        lw: line width as float pixels.
        indices: indices of the images to draw the matches on.
    """
    n_lines = len(lines[0])
    colors = sns.color_palette("husl", n_colors=n_lines)
    np.random.shuffle(colors)
    alphas = np.ones(n_lines)
    # If correct_matches is not None, display wrong matches with a low alpha
    if correct_matches is not None:
        alphas[~np.array(correct_matches)] = 0.2

    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]
    fig.canvas.draw()

    # Plot the lines
    for a, line in zip(axes, lines):
        # Transform the points into the figure coordinate system
        transFigure = fig.transFigure.inverted()
        endpoint0 = transFigure.transform(a.transData.transform(line[:, 0]))
        endpoint1 = transFigure.transform(a.transData.transform(line[:, 1]))
        fig.lines += [
            matplotlib.lines.Line2D(
                (endpoint0[i, 0], endpoint1[i, 0]),
                (endpoint0[i, 1], endpoint1[i, 1]),
                zorder=1,
                transform=fig.transFigure,
                c=colors[i],
                alpha=alphas[i],
                linewidth=lw,
            )
            for i in range(n_lines)
        ]


def plot_color_lines(
    lines, correct_matches, wrong_matches, lw=2, indices=(0, 1)
):
    """Plot line matches for existing images with multiple colors:
    green for correct matches, red for wrong ones, and blue for the rest.
    Args:
        lines: list of ndarrays of size (N, 2, 2).
        correct_matches: list of bool arrays of size N with correct matches.
        wrong_matches: list of bool arrays of size (N,) with correct matches.
        lw: line width as float pixels.
        indices: indices of the images to draw the matches on.
    """
    # palette = sns.color_palette()
    palette = sns.color_palette("hls", 8)
    blue = palette[5]  # palette[0]
    red = palette[0]  # palette[3]
    green = palette[2]  # palette[2]
    colors = [np.array([blue] * len(line)) for line in lines]
    for i, c in enumerate(colors):
        c[np.array(correct_matches[i])] = green
        c[np.array(wrong_matches[i])] = red

    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]
    fig.canvas.draw()

    # Plot the lines
    for a, line, c in zip(axes, lines, colors):
        # Transform the points into the figure coordinate system
        transFigure = fig.transFigure.inverted()
        endpoint0 = transFigure.transform(a.transData.transform(line[:, 0]))
        endpoint1 = transFigure.transform(a.transData.transform(line[:, 1]))
        fig.lines += [
            matplotlib.lines.Line2D(
                (endpoint0[i, 0], endpoint1[i, 0]),
                (endpoint0[i, 1], endpoint1[i, 1]),
                zorder=1,
                transform=fig.transFigure,
                c=c[i],
                linewidth=lw,
            )
            for i in range(len(line))
        ]


def save_plot(path, **kw):
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
