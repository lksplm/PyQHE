from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

def spectrum_spin(L, Eint, S, ax=None, integer=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = None

    Splt = S.copy()
    Splt[np.isnan(Splt)] = -1
    Splt[np.abs(Splt) < 1.e-5] = 0
    if integer:
        Splt = np.array(np.round(Splt), dtype=np.int)
    else:
        Splt = np.round(Splt, 5)
    Slbl = np.unique(Splt)
    print(Slbl)
    Splt2 = Splt.copy()
    for i, spi in enumerate(Slbl):
        Splt[Splt2 == spi] = i

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    marker = ['^', 'v', '*', 's', '+', 'X', 'h', 'o', 'p']
    cmap = ListedColormap(cycle)
    norm = BoundaryNorm(np.arange(len(Slbl) + 1) - 0.5, ncolors=len(Slbl))

    if Splt.dtype== np.int:
        legend_elements = [plt.Line2D([0], [0], color=cycle[i], marker=marker[i], label='S={:d}'.format(s)) for i, s in
                       enumerate(Slbl)]
    else:
        legend_elements = [plt.Line2D([0], [0], color=cycle[i], marker=marker[i], label='S={:.3f}'.format(s)) for i, s in
                           enumerate(Slbl)]

    for i, s in enumerate(Slbl):
        idx = (Splt == i)
        ax.scatter(L[idx], Eint[idx], c=Splt[idx], marker=marker[i], facecolor='none', alpha=0.5, cmap=cmap,
                    norm=norm)
    ax.legend(handles=legend_elements)

    return fig, ax


# Adopted from the SciPy Cookbook.
def _blob(x, y, w, w_max, area, cmap=None):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])

    plt.fill(xcorners, ycorners,
             color=cmap(int((w + w_max) * 256 / (2 * w_max))))

# Adopted from the SciPy Cookbook.
def hinton(W, xlabels=None, ylabels=None, title=None, ax=None, cmap=None, label_top=True):

    cmap =  cm.RdBu
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = None

    if not (xlabels or ylabels):
        ax.axis('off')

    ax.axis('equal')
    ax.set_frame_on(False)

    height, width = W.shape

    w_max = 1.25 * max(abs(np.diag(np.matrix(W))))
    if w_max <= 0.0:
        w_max = 1.0

    ax.fill(np.array([0, width, width, 0]), np.array([0, 0, height, height]),
            color=cmap(128))
    for x in range(width):
        for y in range(height):
            _x = x + 1
            _y = y + 1
            if np.real(W[x, y]) !=0.0:
                if np.real(W[x, y]) > 0.0:
                    _blob(_x - 0.5, height - _y + 0.5, abs(W[x,
                          y]), w_max, min(1, abs(W[x, y]) / w_max), cmap=cmap)
                else:
                    _blob(_x - 0.5, height - _y + 0.5, -abs(W[
                          x, y]), w_max, min(1, abs(W[x, y]) / w_max), cmap=cmap)

    # color axis
    norm = mpl.colors.Normalize(-abs(W).max(), abs(W).max())
    cax, kw = mpl.colorbar.make_axes(ax, shrink=0.75, pad=.1)
    mpl.colorbar.ColorbarBase(cax, norm=norm, cmap=cmap)

    # x axis
    ax.xaxis.set_major_locator(plt.IndexLocator(1, 0.5))
    if xlabels:
        ax.set_xticklabels(xlabels)
        if label_top:
            ax.xaxis.tick_top()
    ax.tick_params(axis='x', labelsize=7)

    # y axis
    ax.yaxis.set_major_locator(plt.IndexLocator(1, 0.5))
    if ylabels:
        ax.set_yticklabels(list(reversed(ylabels)))
    ax.tick_params(axis='y', labelsize=7)

    return fig, ax

def hinton_fast(W, xlabels=None, ylabels=None, ax=None, label_top=False):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = None

    if not (xlabels or ylabels):
        ax.axis('off')

    ax.axis('equal')
    ax.set_frame_on(False)

    ampl = np.max(np.abs(W))+0.25
    ax.imshow(W, cmap=cm.RdBu, vmin=-ampl, vmax=ampl)
    # color axis
    norm = mpl.colors.Normalize(-abs(W).max(), abs(W).max())
    cax, kw = mpl.colorbar.make_axes(ax, shrink=0.75, pad=.1)
    mpl.colorbar.ColorbarBase(cax, norm=norm, cmap=cm.RdBu)

    # x axis
    ax.xaxis.set_major_locator(plt.IndexLocator(1, 0.5))
    if xlabels:
        ax.set_xticklabels(xlabels)
        if label_top:
            ax.xaxis.tick_top()
    ax.tick_params(axis='x', labelsize=7)

    # y axis
    ax.yaxis.set_major_locator(plt.IndexLocator(1, 0.5))
    if ylabels:
        ax.set_yticklabels(list(reversed(ylabels)))
    ax.tick_params(axis='y', labelsize=7)

    return fig, ax


def hinton_nolabel(W, xlabels=None, ylabels=None, ax=None, label_top=False):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = None

    if not (xlabels or ylabels):
        ax.axis('off')

    ax.axis('equal')
    ax.set_frame_on(False)

    ampl = np.max(W)
    ax.imshow(W, cmap=cm.RdBu, vmin=-ampl, vmax=ampl)
    # color axis
    norm = mpl.colors.Normalize(-abs(W).max(), abs(W).max())
    cax, kw = mpl.colorbar.make_axes(ax, shrink=0.75, pad=.1)
    mpl.colorbar.ColorbarBase(cax, norm=norm, cmap=cm.RdBu)

    # x axis
    ax.xaxis.set_major_locator(plt.IndexLocator(1, 0.5))
    if xlabels:
        ax.set_xticklabels(xlabels)
        if label_top:
            ax.xaxis.tick_top()
    ax.tick_params(axis='x', labelsize=7)

    # y axis
    ax.yaxis.set_major_locator(plt.IndexLocator(1, 0.5))
    if ylabels:
        ax.set_yticklabels(list(reversed(ylabels)))
    ax.tick_params(axis='y', labelsize=7)

    return fig, ax
