from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from fractions import Fraction
from itertools import cycle


class listcycle:
    def __init__(self, lst):
        self.lst = lst
        self.N = len(lst)

    def __getitem__(self, n):
        i = n % self.N
        return self.lst[i]

def format_spin(s):
    """
    Format a spin (multiples of 0.5) as a fraction n/2 for nice display
    :param s: float number
    :return: string "n/2"
    """
    f = Fraction(int(s*2), 2)
    return "{:d}/{:d}".format(f.numerator, f.denominator)

def unique_close(a,rtol=1.0e-5, atol=0.0, return_index=False):
    out = []
    outind = []
    a = np.array(a)
    c = a.copy()
    cind = np.arange(len(c))
    while len(c)>0:
        b = c[0]
        out.append(b)
        outind.append(cind[0])
        ind = np.where(np.abs(c- b)<=(atol + rtol * np.abs(b)))[0]
        c = np.delete(c, ind)
        cind = np.delete(cind, ind)
    if return_index:
        return np.array(out), np.array(outind)
    else:
        return np.array(out)


def spectrum_spin_mod(L, Eint, S, ax=None, integer=True, rdigits=5, atol=1.0e-6, rtol=1e-2):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = None

    ls = L.flatten().round().astype(np.int)
    es = Eint.flatten()
    ss = S.flatten()
    mask = np.isfinite(ls) & (ls >= 0) & np.isfinite(es) & np.isfinite(ss)
    ls, es, ss = ls[mask], es[mask], ss[mask]

    if integer:
        ss = ss.round().astype(np.int)
    else:
        ss = ss.round(rdigits)

    lu = np.unique(ls)
    su = np.unique(ss)
    print("Found L: ", lu, " and S: ", su)
    eun = []
    lun = []
    sun = []
    # for each spin and l
    for j, s in enumerate(su):
        for i, l in enumerate(lu):
            ind = np.where((ls == l) & (ss == s))[0]
            if len(ind) > 0:
                etmp, indu = unique_close(es[ind], return_index=True, atol=atol, rtol=rtol)
                eun.append(etmp)
                lun.append(ls[ind][indu])
                sun.append(ss[ind][indu])

    eun = np.concatenate(eun)
    lun = np.concatenate(lun)
    sun = np.concatenate(sun)

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    marker = ['^', 'v', 'o', '*', 's', '+', 'X', 'h', 'p']
    offs = np.linspace(-0.2, 0.2, len(su), endpoint=True)

    if ss.dtype == np.int:
        legend_elements = [plt.Line2D([0], [0], color=cycle[i], marker=marker[i], label='S={:d}'.format(s)) for i, s in
                           enumerate(su)]
    else:
        legend_elements = [plt.Line2D([0], [0], color=cycle[i], marker=marker[i], label='S={}'.format(format_spin(s)))
                           for i, s in
                           enumerate(su)]

    for i, s in enumerate(su):
        idx = (sun == s)
        ax.plot(lun[idx] + offs[i], eun[idx], color=cycle[i], marker=marker[i], markerfacecolor='None', ls='None')

    ax.legend(handles=legend_elements)

    return fig, ax

def spectrum_spin_sz(L, Eint, S, Sz, ax=None, integer=True, rdigits=5, atol=1.0e-6, rtol=1e-2):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = None

    ls = L.flatten().round().astype(np.int)
    es = Eint.flatten()
    ss = S.flatten()
    ssz = Sz.flatten()
    mask = np.isfinite(ls) & (ls >= 0) & np.isfinite(es) & np.isfinite(ss) & np.isfinite(ssz)
    ls, es, ss, ssz = ls[mask], es[mask], ss[mask], ssz[mask]

    if integer:
        ss = ss.round().astype(np.int)
        ssz = ssz.round().astype(np.int)
    else:
        ss = ss.round(rdigits)
        ssz = ssz.round(rdigits)

    lu = np.unique(ls)
    su = np.unique(ss)
    szu = np.unique(ssz)
    print("Found L: ", lu, " and S: ", su, " and Sz: ", szu)
    eun = []
    lun = []
    sun = []
    szun = []
    # for each spin and l
    for k, sz in enumerate(szu):
        for j, s in enumerate(su):
            for i, l in enumerate(lu):
                ind = np.where((ls == l) & (ss == s) & (ssz == sz))[0]
                if len(ind) > 0:
                    etmp, indu = unique_close(es[ind], return_index=True, atol=atol, rtol=rtol)
                    eun.append(etmp)
                    lun.append(ls[ind][indu])
                    sun.append(ss[ind][indu])
                    szun.append(ssz[ind][indu])

    eun = np.concatenate(eun)
    lun = np.concatenate(lun)
    sun = np.concatenate(sun)
    szun = np.concatenate(szun)

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    marker = ['^', 'v', 'o', '*', 's', '+', 'X', 'h', 'p']
    offs = np.linspace(-0.2, 0.2, len(su), endpoint=True)
    offsy = np.linspace(-0.05, 0.05, len(szu), endpoint=True)

    if ss.dtype == np.int:
        legend_elements = [plt.Line2D([0], [0], color=cycle[i], marker='o', label='S={:d}'.format(s)) for i, s in
                           enumerate(su)] + [plt.Line2D([0], [0], color='k', marker=marker[k], label='Sz={:d}'.format(sz)) for k, sz in enumerate(szu)]
    else:
        legend_elements = [plt.Line2D([0], [0], color=cycle[i], marker='o', label='S={}'.format(format_spin(s)))
                           for i, s in enumerate(su)] + [plt.Line2D([0], [0], color='k', marker=marker[i], label='Sz={}'.format(format_spin(sz))) for k, sz in enumerate(szu)]

    for k, sz in enumerate(szu):
        for i, s in enumerate(su):
            idx = (sun == s) & (szun == sz)
            ax.plot(lun[idx] + offs[i], eun[idx] + offsy[k], color=cycle[i], marker=marker[k], ls='None')#markerfacecolor='None',

    ax.legend(handles=legend_elements)

    return fig, ax

def energy_spin(alpha, E, S, Mshow=10, ax=None, integer=True, sort=True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = None

    S = np.squeeze(S)
    E = np.squeeze(E)

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
    cmap = ListedColormap(cycle)
    norm = BoundaryNorm(np.arange(len(Slbl) + 1) - 0.5, ncolors=len(Slbl))

    if Splt.dtype== np.int:
        legend_elements = [plt.Line2D([0], [0], color=cycle[i], marker='o', label='S={:d}'.format(s)) for i, s in
                       enumerate(Slbl)]
    else:
        legend_elements = [plt.Line2D([0], [0], color=cycle[i], marker='o', label='S={}'.format(format_spin(s))) for i, s in
                           enumerate(Slbl)]

    Esort = E.copy()
    Ssort = Splt.copy()
    if sort:
        for i in range(E.shape[1]):
            idx = np.argsort(Esort[:, i])
            Esort[:, i] = Esort[idx, i]
            Ssort[:, i] = Ssort[idx, i]

    for i in range(Mshow):
        e = Esort[i, :]
        s = Ssort[i, :]
        points = np.array([alpha, e]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, **kwargs)
        lc.set_array(s)
        #lc.set_linewidth(1.0)
        plt.gca().add_collection(lc)

    ax.set_xlim(alpha.min(), alpha.max())
    ax.set_ylim(0, 1.2*E[0:Mshow,:].max())  # 2:3.5, 3: 6.5, 4:10.5
    ax.legend(handles=legend_elements, loc=2)

    return fig, ax

def energy_spin_sz(alpha, E, S, Sz, Mshow=10, ax=None, integer=True, rdigits=5, sort=True, offset=0, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = None

    S = np.squeeze(S)
    E = np.squeeze(E)
    Sz = np.squeeze(Sz)

    Splt = S.copy()
    Splt[np.isnan(Splt)] = -1
    Splt[np.abs(Splt) < 1.e-5] = 0
    if integer:
        Splt = np.array(np.round(Splt), dtype=np.int)
    else:
        Splt = np.round(Splt, 5)
    Slbl = np.unique(Splt)
    su = Slbl
    print(su)
    Splt2 = Splt.copy()
    for i, spi in enumerate(Slbl):
        Splt[Splt2 == spi] = i

    Sz[np.isnan(Sz)] = -1
    Sz[np.abs(Sz) < 1.e-5] = 0
    if integer:
        Sz = np.array(np.round(Sz), dtype=np.int)
    else:
        Sz = np.round(Sz, 5)

    szu = np.unique(Sz)
    print(szu)

    ccycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cmap = ListedColormap(ccycle)
    norm = BoundaryNorm(np.arange(len(su) + 1) - 0.5, ncolors=len(su))
    linestyles = listcycle(['solid', 'dashed', 'dashdot', 'dotted']) #['-', '--', '-.', ':']
    lss = listcycle(['-', '--', '-.', ':', '-.'])
    if Splt2.dtype == np.int:
        legend_elements = [plt.Line2D([0], [0], color=ccycle[i], label='S={:d}'.format(s)) for i, s in enumerate(su)] + [plt.Line2D([0], [0], color='k', linestyle=lss[k], label='Sz={:d}'.format(sz)) for k, sz in enumerate(szu)]
    else:
        legend_elements = [plt.Line2D([0], [0], color=ccycle[i], label='S={}'.format(format_spin(s))) for i, s in enumerate(su)] + [plt.Line2D([0], [0], color='k', linestyle=lss[k], label='Sz={}'.format(format_spin(sz))) for k, sz in enumerate(szu)]

    Esort = E.copy()
    Ssort = Splt.copy()
    Szsort = Sz.copy()
    if sort:
        for i in range(E.shape[1]):
            idx = np.argsort(Esort[:, i])
            Esort[:, i] = Esort[idx, i]
            Ssort[:, i] = Ssort[idx, i]
            Szsort[:, i] = Szsort[idx, i]

    for k, szz in enumerate(szu):
        for i in range(Mshow):
            e = Esort[i, :]
            s = Ssort[i, :]
            em = np.ma.masked_where(Szsort[i, :] == szz, e)
            points = np.ma.array([alpha, em]).T.reshape(-1, 1, 2)
            print(szz)
            segments = np.ma.concatenate([points[:-1], points[1:]], axis=1)
            print(segments)
            lc = LineCollection(segments, cmap=cmap, norm=norm, linestyles=linestyles[k], **kwargs)
            lc.set_array(s)
            #lc.set_linewidth(1.0)
            plt.gca().add_collection(lc)

    ax.set_xlim(alpha.min(), alpha.max())
    ax.set_ylim(0, 1.2*E[0:Mshow,:].max())  # 2:3.5, 3: 6.5, 4:10.5
    ax.legend(handles=legend_elements, loc=2)

    return fig, ax

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
        legend_elements = [plt.Line2D([0], [0], color=cycle[i], marker=marker[i], label='S={}'.format(format_spin(s))) for i, s in
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
