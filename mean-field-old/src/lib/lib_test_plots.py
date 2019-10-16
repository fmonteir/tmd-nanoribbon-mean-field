import numpy as np
import sys
project_directory = '/Users/franciscobrito/projects/'
sys.path.append(project_directory + 'tmd-nanoribbon/mean-field-old/src/lib')
from lib.lib_tmd_model import HribbonRealSpace, setParams, HribbonKSpace, HribbonKSpace2sublat
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("white")
sns.set_palette("GnBu_d")
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

def plotFillings(nOrb, Nx, Ny, SAVE):
    '''
    Plots the filling as a function of the Fermi energy for TMD nanoribbons with different
    transition metal and chalcogens.
    '''

    fig = plt.figure(figsize=(16,8))
    fig.subplots_adjust(hspace=0.1, wspace=0.15)
    nCols = 3
    tmds = ['MoS2', 'WS2', 'MoSe2', 'WSe2', 'MoTe2', 'WTe2']
    tmdLatex = [r'$MoS_2$', r'$WS_2$', r'$MoSe_2$', r'$WSe_2$', r'$MoTe_2$', r'$WTe_2$']

    for idx, tmd in enumerate(tmds):
        abs_t0, e1, e2, t0, t1, t2, t11, t12, t22, \
        E0, E1, E2, E3, E4, E5, E6 = setParams(tmd)
        ens = la.eigvalsh( HribbonRealSpace(nOrb, Nx, Ny, E0, E1, E2, E3, E4, E5, E6) )
        ax = fig.add_subplot(len(tmds) / nCols, nCols, idx + 1)
        # The factor of 2 below is due to spin degeneracy of the bands
        ax.plot(ens * abs_t0, np.arange(ens.size) / ens.size * 2,\
                label= str(Nx) + r'$\times$' + str(Ny) + ' ' +\
                tmdLatex[idx], linewidth = 0.7, markersize = 1,\
               markeredgewidth = 6, color = 'r')
        if (idx + 1) > len(tmds) - nCols:
            ax.set_xlabel(r'$\varepsilon_F [eV]$')
        if (idx + nCols) % nCols == 0:
            ax.set_ylabel(r'$\left\langle n \right\rangle$')
        else:
            ax.set_yticklabels([])
        ax.set_xlim(-1, 4)
        ax.plot([-1, 3.5], [2 / 3, 2 / 3], linewidth = 1,\
                c = 'green', linestyle = '--', label = 'Charge neutrality')
        lgd = ax.legend(fontsize = 'large')

    if SAVE == True:
        plt.savefig("../plots/lib_test/" + "filling.png", dpi = 200)

def plotBands(nOrb, Ny, SAVE):
    # Number of k-points taken when computing the band structure
    nks = 1024
    ks = np.linspace(-np.pi,np.pi, nks)
    bandEn = np.zeros( ( nks, Ny * nOrb ) )

    fig = plt.figure(figsize=(9,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.02)
    nCols = 2
    tmds = ['MoS2', 'WS2', 'MoSe2', 'WSe2', 'MoTe2', 'WTe2']
    tmdLatex = [r'$MoS_2$', r'$WS_2$', r'$MoSe_2$', r'$WSe_2$', r'$MoTe_2$', r'$WTe_2$']

    for idx, tmd in enumerate(tmds):
        abs_t0, e1, e2, t0, t1, t2, t11, t12, t22, \
        E0, E1, E2, E3, E4, E5, E6 = setParams(tmd)
        for en, k in enumerate(ks):
            bandEn[en,:] = la.eigvalsh( HribbonKSpace(k, nOrb, Ny, e1, e2, t0, t1, t2, t11, t12, t22) ) * abs_t0

        ax = fig.add_subplot(len(tmds) / nCols, nCols, idx + 1)
        ax.plot(ks, bandEn, c = 'Darkblue',\
                label= tmdLatex[idx], linewidth = 0.3, markersize = 0.05, marker = 'o',\
               markeredgewidth = 0.5)
        lgd = ax.legend(fontsize = 16, loc = 1)
        if (idx + 1) > len(tmds) - nCols:
            ax.set_xlabel(r'$k \, a$', fontsize = 20)
        if (idx + nCols) % nCols == 0:
            ax.set_ylabel(r'$\varepsilon_{k} [eV]$',\
            fontsize = 20, labelpad = 10)
        else:
            ax.set_yticklabels([])
        if idx > 3:
            ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$" , r"$\pi/2$", r'$\pi$'])
        else:
            ax.set_xticklabels([])
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12)

        ax.set_ylim(-1.2, 5.5)
    if SAVE == True:
        plt.savefig("../plots/lib_test/" + "bandStructures.png", dpi = 600)

def plotTB1D(SAVE):
    plt.plot(np.linspace(-np.pi, np.pi),\
    -np.cos(np.linspace(-np.pi, np.pi)),c = 'Darkblue', linestyle = '-',\
    label = 'Normal unit cell', linewidth = 0.5)
    plt.plot(np.linspace(-np.pi / 2, np.pi / 2),\
    -np.cos(np.linspace(-np.pi / 2, np.pi / 2)),c = 'Darkred',\
    linestyle = '--', linewidth = 1, label = 'Doubled unit cell',)
    plt.plot(np.linspace(-np.pi / 2, np.pi / 2),\
    np.cos(np.linspace(-np.pi / 2, np.pi / 2)), c = 'Darkred',\
    linestyle = '--', linewidth = 1, label = 'Doubled unit cell')
    plt.xticks([-np.pi, - np.pi / 2, 0, np.pi / 2, np.pi],\
               [r"$-\pi$", r"$-\pi/2$", r"$0$" , r"$\pi/2$", r'$\pi$'])
    plt.xlabel(r'$k$')
    plt.ylabel(r'$\varepsilon_k$')
    plt.legend()
    plt.title(r'1D tight-binding bands')
    if SAVE == True:
        plt.savefig("../plots/lib_test/" + "1d-tight-binding.png", dpi = 300)

def plot_matToDiagonalize(nOrb, Ny, k, SAVE):
    tmd = 'MoS2'
    abs_t0, e1, e2, t0, t1, t2,\
    t11, t12, t22, E0, E1, E2, E3, E4, E5, E6\
    = setParams(tmd)
    mat = HribbonKSpace2sublat(k, nOrb, Ny, E0, E1, E2, E3, E4)
    plt.figure(1, figsize=(9,9))
    plt.imshow(np.absolute(mat), interpolation='nearest')
    plt.title(r'$\mathbf{H}(k)$')
    plt.xlabel(r'$(\alpha, n, o)$')
    plt.ylabel(r'$(\beta, n\prime, o\prime)$')
    if SAVE == True:
        plt.savefig("../plots/lib_test/" + "matrix-to-diagonalize-double-unit-cell.png", dpi = 300)

def plotHalfBands(tmd, nOrb, Ny, SAVE):
    abs_t0, e1, e2, t0, t1, t2,\
    t11, t12, t22, E0, E1, E2, E3, E4, E5, E6\
    = setParams(tmd)
    nks = 512
    ks = np.linspace(-np.pi, np.pi, nks)
    ksF = np.concatenate( ( np.linspace(0, - np.pi / 2, int(nks / 4) ) ,\
                          np.linspace(-np.pi / 2 , np.pi / 2, int(nks / 2) ),\
                          np.linspace(np.pi / 2, 0,  int(nks / 4) )) )
    ksAF = np.linspace(-np.pi / 2 , np.pi / 2, nks)
    bandEnF = np.zeros( ( nks, Ny * nOrb ) )
    bandEnAF = np.zeros( ( nks, 2 * Ny * nOrb ) )

    for en, k in enumerate(ks):
        bandEnF[en,:] = la.eigvalsh(\
                                    HribbonKSpace(k, nOrb, Ny, e1, e2, t0, t1, t2, t11, t12, t22)\
                                   ) * abs_t0

    for en, k in enumerate(ksAF):
        bandEnAF[en,:] = la.eigvalsh(\
                                     HribbonKSpace2sublat(k, nOrb, Ny, E0, E1, E2, E3, E4)\
                                    ) * abs_t0
    fig = plt.figure(1, figsize = (8,6))
    ax = fig.add_subplot(1,1,1)

    ax.plot(ksF, bandEnF, c = "darkblue", linestyle = '--', label =\
            'Normal unit cell - ' + r'$\frac{1}{2}$' + 'FBZ')
    ax.plot(ksAF, bandEnAF, c = "darkred", linestyle = '-', linewidth = 0.58, label =\
            'Doubled unit cell - ' + 'FBZ')
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$\varepsilon_k \, [eV]$')
    ax.set_xticks([- np.pi / 2, - np.pi / 3, 0, np.pi / 3, np.pi / 2])
    ax.set_xticklabels([r"$-\pi/2$", r"$-\pi/3$", r"$0$" , r"$\pi/3$", r"$\pi/2$"])
    ax.legend(bbox_to_anchor = (1,1))
    if SAVE == True:
        plt.savefig("../plots/lib_test/" + "free-bands-doubled-unit-cell.png", dpi = 300, bbox_inches = 'tight')

    plt.figure(2)
    plt.plot(np.sort(bandEnF.flatten()), np.linspace(0, 2, num=nks * Ny * nOrb),\
         c = 'darkblue', marker = 'o', markersize = 0.1, markeredgewidth = 6,\
         linewidth = 0., label = 'Normal unit cell', linestyle = '--')
    plt.plot(np.sort(bandEnAF.flatten()), np.linspace(0, 2, num=2 * nks * Ny * nOrb ),\
             c = 'darkred', marker = '^', markersize = 0.01, markeredgewidth = 2,\
             linewidth = 0., label = 'Double unit cell', linestyle = '--')
    plt.plot([-0.7, 3.6], [2 / 3, 2 / 3], c = 'seagreen', linewidth = 1, label = r'Charge neutrality')
    plt.xlim(-0.7, 3.6)
    plt.xlabel(r'$\varepsilon_F [eV]$')
    plt.ylabel(r'$\left\langle n \right\rangle$')
    plt.legend(bbox_to_anchor = (1, 1))
    if SAVE == True:
        plt.savefig("../plots/lib_test/" + "filling-doubled-unit-cell.png", dpi = 300, bbox_inches = 'tight')

def plotFreeBands(tmd, nOrb, Ny, SAVE):
    abs_t0, e1, e2, t0, t1, t2,\
    t11, t12, t22, E0, E1, E2, E3, E4, E5, E6\
    = setParams(tmd)
    nks = 512
    ksAF = np.linspace(-np.pi / 2 , np.pi / 2, nks)
    bandEnAF = np.zeros( ( nks, 2 * Ny * nOrb ) )

    for en, k in enumerate(ksAF):
        bandEnAF[en,:] = la.eigvalsh(\
                                     HribbonKSpace2sublat(k, nOrb, Ny, E0, E1, E2, E3, E4)\
                                    ) * abs_t0
    fig = plt.figure(1, figsize = (7,6))
    ax = fig.add_subplot(1,1,1)

    ax.plot(ksAF, bandEnAF, c = "darkblue", linestyle = '-', linewidth = 1)
    # ax.axvline(np.pi / 4, linewidth = 0.8)
    # ax.axhline(0.95, linewidth = 0.8)
#    ax.plot(ksAF, bandEnAF[:,:2*Ny-2], c = "darkblue", linestyle = '-', linewidth = 0.58, label =\
#            'Doubled unit cell - ' + 'FBZ', alpha = 0.1)
#    ax.plot(ksAF, bandEnAF[:,2*Ny-2:2*Ny+1], c = "darkblue", linestyle = '-', linewidth = 0.58, label =\
#            'Doubled unit cell - ' + 'FBZ')
#    ax.plot(ksAF, bandEnAF[:,2*Ny+1:2*Ny+25], c = "darkblue", linestyle = '-', linewidth = 0.58, label =\
#            'Doubled unit cell - ' + 'FBZ')
#    ax.plot(ksAF, bandEnAF[:,2*Ny+25:6*Ny], c = "darkblue", linestyle = '-', linewidth = 0.58, label =\
#            'Doubled unit cell - ' + 'FBZ', alpha = 0.1)
    ax.set_xlabel(r'$k \, a$', fontsize = 20)
    ax.set_ylabel(r'$\varepsilon_{k, \sigma} \,\, [eV]$',\
        fontsize = 20, labelpad = 10)
    ax.set_xticks([-np.pi / 2, - np.pi / 4, 0,\
    np.pi / 4, np.pi / 2])
    ax.set_xticklabels([r"$-\pi/2$", r"$-\pi/4$",\
    r"$0$" , r"$\pi/4$", r"$\pi/2$"], fontsize = 12)
    fig.tight_layout()
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    if SAVE == True:
        plt.savefig("../plots/lib_test/" + "free-bands-halved.png", dpi = 600, bbox_inches = 'tight')
