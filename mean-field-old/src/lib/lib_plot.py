##
##  lib_plot.py
##
##
##  Created by Francisco Brito on 16/11/2018.
##
##  This library defines some graphic representations related to the solution
##  of the mean field intra-orbital Hubbard model for a TMD
##  nanoribbon.
##
##

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator \
import inset_axes, zoomed_inset_axes, mark_inset
import seaborn as sns
sns.set()
sns.set_style("white")
sns.set_palette("GnBu_d")
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

def showLattice(nUp, nDown, Nx, Ny, nOrb, dotscale, SAVE, name):
    nUpSite = np.zeros(Nx * Ny)
    nDownSite = np.zeros(Nx * Ny)
    for i in range(Nx * Ny):
        nUpSite[i] = nUp[nOrb * i] + nUp[nOrb * i + 1] + nUp[nOrb * i + 2]
        nDownSite[i] = nDown[nOrb * i] + nDown[nOrb * i + 1] + nDown[nOrb * i + 2]
    clr = np.chararray((Nx * Ny), itemsize = 10)
    a1 = np.arange(Nx)
    a2 = np.arange(Ny)
    vs = np.zeros((Nx * Ny , 2))
    lat = np.zeros((Nx * Ny))
    v1 = np.array([1, 0])
    v2 = np.array([1 / 2, np.sqrt(3) / 2])
    for i in range(Nx):
        for j in range(Ny):
                vs[Nx * j + i, :] =\
                a1[i] * v1 + a2[j] * v2
                lat[Nx * j + i]\
                =  dotscale * (nUpSite[Nx * j + i]\
                          - nDownSite[Nx * j + i] )
                if (nUpSite[Nx * j + i]\
                          - nDownSite[Nx * j + i] ) < 0 :
                    clr[Nx * j + i] = "#95a5a6"
                else:
                    clr[Nx * j + i] = "#e74c3c"

    fig = plt.figure(1, figsize = (Nx, Ny))
    ax = fig.add_subplot(111)
    ax.scatter(vs[:, 0], vs[:, 1], s = abs(lat),\
               c = clr.decode('UTF-8'), alpha = 0.8, edgecolors = None)
    ax.axis('equal')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    if SAVE == True:
        plt.savefig(name + ".png", dpi = 400)

def showLatticeORBdz2(nUp, nDown, Nx, Ny, nOrb, dotscale, SAVE, name):
    nUpSite = np.zeros(Nx * Ny)
    nDownSite = np.zeros(Nx * Ny)
    for i in range(Nx * Ny):
        nUpSite[i] = nUp[nOrb * i]
        nDownSite[i] = nDown[nOrb * i]
    clr = np.chararray((Nx * Ny), itemsize = 10)
    a1 = np.arange(Nx)
    a2 = np.arange(Ny)
    vs = np.zeros((Nx * Ny , 2))
    lat = np.zeros((Nx * Ny))
    v1 = np.array([1, 0])
    v2 = np.array([1 / 2, np.sqrt(3) / 2])
    for i in range(Nx):
        for j in range(Ny):
                vs[Nx * j + i, :] =\
                a1[i] * v1 + a2[j] * v2
                lat[Nx * j + i]\
                =  dotscale * (nUpSite[Nx * j + i]\
                          - nDownSite[Nx * j + i] )
                if (nUpSite[Nx * j + i]\
                          - nDownSite[Nx * j + i] ) < 0 :
                    clr[Nx * j + i] = "#95a5a6"
                else:
                    clr[Nx * j + i] = "#e74c3c"

    fig = plt.figure(1, figsize = (Nx, Ny))
    ax = fig.add_subplot(111)
    ax.scatter(vs[:, 0], vs[:, 1], s = abs(lat),\
               c = clr.decode('UTF-8'), alpha = 0.8, edgecolors = None)
    ax.axis('equal')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    if SAVE == True:
        plt.savefig(name + ".png", dpi = 400)

def showLatticeORBdxy(nUp, nDown, Nx, Ny, nOrb, dotscale, SAVE, name):
    nUpSite = np.zeros(Nx * Ny)
    nDownSite = np.zeros(Nx * Ny)
    for i in range(Nx * Ny):
        nUpSite[i] = nUp[nOrb * i + 1]
        nDownSite[i] = nDown[nOrb * i + 1]
    clr = np.chararray((Nx * Ny), itemsize = 10)
    a1 = np.arange(Nx)
    a2 = np.arange(Ny)
    vs = np.zeros((Nx * Ny , 2))
    lat = np.zeros((Nx * Ny))
    v1 = np.array([1, 0])
    v2 = np.array([1 / 2, np.sqrt(3) / 2])
    for i in range(Nx):
        for j in range(Ny):
                vs[Nx * j + i, :] =\
                a1[i] * v1 + a2[j] * v2
                lat[Nx * j + i]\
                =  dotscale * (nUpSite[Nx * j + i]\
                          - nDownSite[Nx * j + i] )
                if (nUpSite[Nx * j + i]\
                          - nDownSite[Nx * j + i] ) < 0 :
                    clr[Nx * j + i] = "#95a5a6"
                else:
                    clr[Nx * j + i] = "#e74c3c"

    fig = plt.figure(1, figsize = (Nx, Ny))
    ax = fig.add_subplot(111)
    ax.scatter(vs[:, 0], vs[:, 1], s = abs(lat),\
               c = clr.decode('UTF-8'), alpha = 0.8, edgecolors = None)
    ax.axis('equal')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    if SAVE == True:
        plt.savefig(name + ".png", dpi = 400)

def showGrandpotentialMinimization(itSwitch, lastIt, energies):
    lastNit = 3
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(np.arange(lastIt - itSwitch), energies[itSwitch:lastIt],\
            marker = 'o', markersize = 0.5, markeredgewidth = 2, color = "#e74c3c", linewidth = 0.5)
    ax.set_xlabel(r'$Iteration$')
    ax.set_ylabel(r'$\frac{\Omega}{N} [eV]$')
    #axins1 = zoomed_inset_axes(ax, 2.5)
    #x1, x2, y1, y2 = lastIt - lastNit, lastIt - 1, energies[lastIt-1]-4, energies[lastIt-1]+4
    #axins1.set_xlim(x1, x2 + 0.5) # apply the x-limits
    #axins1.set_ylim(y1, y2) # apply the y-limits
    #axins1.set_xlabel(r'$Iteration$')
    #axins1.set_ylabel(r'$\frac{\omega}{| t_0 |}$')
    #mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="0.1")
    #axins1.plot(np.arange(energies[x1:x2+1].size) + lastIt - lastNit,\
    #        energies[x1:x2+1], color = "#e74c3c", linewidth = 1)

def showBandStructure(Nk, abs_t0, eUp, eDown, mu):
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(1, 1, 1)
    sns.set_style("white")
    ks = np.linspace(-np.pi,np.pi, num=Nk, endpoint=False)
    ax.plot(ks, abs_t0*eUp, markersize=2, linewidth = 0.5, c = 'r', label = r'$\uparrow$')
    ax.plot(ks, abs_t0*eDown, markersize=2, linewidth = 0.5, c = 'b', label = r'$\downarrow$')
    ax.plot(ks, abs_t0*np.ones(Nk)*mu, c = 'seagreen', linewidth = 1, label = r'$\varepsilon_F$')
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$\varepsilon_{k, \sigma}$')
    ax.set_xticks([-np.pi, -2 * np.pi / 3, - np.pi / 3, 0, np.pi / 3, 2 * np.pi / 3, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-2\pi/3$", r"$-\pi/3$", r"$0$" , r"$\pi/3$", r"$2\pi/3$", r'$\pi$'])
    ax.legend(bbox_to_anchor=(1,1))
    ax.set_ylim(0, 5)

def showHalfBandStructure(Nk, abs_t0, eUp, eDown, mu):
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(1, 1, 1)

    sns.set_style("white")
    ks = np.linspace(-np.pi / 2 , np.pi / 2, Nk, endpoint = False)
    ax.plot(ks, abs_t0*eUp, markersize=2, linewidth = 0.5, c = 'r', label = r'$\uparrow$')
    ax.plot(ks, abs_t0*eDown, markersize=2, linewidth = 0.5, c = 'b', label = r'$\downarrow$')
    ax.plot(ks, abs_t0*np.ones(Nk)*mu, c = 'seagreen', linewidth = 1, label = r'$\varepsilon_F$')
    ax.axhline(np.pi / 3)
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$\varepsilon_{k, \sigma}$')
    ax.set_xticks([-np.pi / 2, - np.pi / 3, 0, np.pi / 3, np.pi / 2])
    ax.set_xticklabels([r"$-\pi/2$", r"$-\pi/3$", r"$0$" , r"$\pi/3$", r"$\pi/2$"])
    ax.legend(bbox_to_anchor=(1,1))
    ax.set_ylim(0, 5)

def showMagProf(nUp, nDown, Ny, nOrb, dotscale, longLength):
    nUpSite = np.zeros(Ny)
    nDownSite = np.zeros(Ny)
    for i in range(Ny):
        nUpSite[i] = nUp[nOrb * i] + nUp[nOrb * i + 1] + nUp[nOrb * i + 2]
        nDownSite[i] = nDown[nOrb * i] + nDown[nOrb * i + 1] + nDown[nOrb * i + 2]

    fig, ax = plt.subplots()

    ax.plot((nUpSite-nDownSite) / nOrb, markersize=4, linewidth = 0.5, marker = 'o', label = 'TMDNR',\
           color = sns.color_palette("Paired")[1])
    ax.set_ylabel(r'$\left\langle m \right\rangle ( y )$')
    ax.set_xlabel(r'$y$')
    ax.legend()

    clr = np.chararray((longLength * Ny), itemsize = 10)
    a1 = np.arange(longLength)
    a2 = np.arange(Ny)
    vs = np.zeros((longLength * Ny , 2))
    lat = np.zeros((longLength * Ny))
    v1 = np.array([1, 0])
    v2 = np.array([1 / 2, np.sqrt(3) / 2])
    for i in range(longLength):
        for j in range(Ny):
                vs[longLength * j + i, :] = \
                a1[i] * v1 + a2[j] * v2
                lat[longLength * j + i] \
                =  dotscale * (nUpSite[j]\
                               - nDownSite[j] )
                if (nUpSite[j]\
                    - nDownSite[j] ) < 0 :
                    clr[longLength * j + i] = "#95a5a6"
                else:
                    clr[longLength * j + i] = "#e74c3c"

    fig1 = plt.figure(2, figsize=(longLength, Ny))
    ax1 = fig1.add_subplot(111)
    ax1.scatter(vs[:, 0], vs[:, 1], s = abs(lat), c = clr.decode('UTF-8'), alpha = 0.8, edgecolors = None)
    ax1.axis('equal')
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])

def showMagProf2sublat(nUp, nDown, Ny, nOrb, dotscale, longLength):
    nUpSite = np.zeros(Ny)
    nDownSite = np.zeros(Ny)
    for i in range(Ny):
        nUpSite[i] = nUp[nOrb * i] + nUp[nOrb * i + 1] + nUp[nOrb * i + 2] \
        + nUp[nOrb * i + Ny * nOrb] + nUp[nOrb * i + 1 + Ny * nOrb]\
        + nUp[nOrb * i + 2 + Ny * nOrb]
        nDownSite[i] = nDown[nOrb * i] + nDown[nOrb * i + 1]\
         + nDown[nOrb * i + 2] + nDown[nOrb * i + Ny * nOrb]\
         + nDown[nOrb * i + 1 + Ny * nOrb] + nDown[nOrb * i + 2 + Ny * nOrb]

    fig, ax = plt.subplots()

    ax.plot((nUpSite-nDownSite) / 2 / nOrb, markersize=4, linewidth = 0.5,\
     marker = 'o', label = 'TMDNR',\
     color = sns.color_palette("Paired")[1])
    ax.set_ylabel(r'$\left\langle m_{st} \right\rangle ( y )$')
    ax.set_xlabel(r'$y$')
    ax.legend()

    clr = np.chararray((longLength * Ny), itemsize = 10)
    a1 = np.arange(longLength)
    a2 = np.arange(Ny)
    vs = np.zeros((longLength * Ny , 2))
    lat = np.zeros((longLength * Ny))
    v1 = np.array([1, 0])
    v2 = np.array([1 / 2, np.sqrt(3) / 2])
    for i in range(longLength):
        for j in range(Ny):
                vs[longLength * j + i, :] = \
                a1[i] * v1 + a2[j] * v2
                if i % 2 == 0:
                    lat[longLength * j + i] \
                    =  dotscale * longLength *\
                    ( nUp[nOrb * j]\
                    + nUp[nOrb * j + 1]\
                    + nUp[nOrb * j + 2]\
                    - nDown[nOrb * j]\
                    - nDown[nOrb * j + 1]\
                    - nDown[nOrb * j + 2] )
                else:
                    lat[longLength * j + i] \
                    =  dotscale * longLength *\
                    ( nUp[nOrb * j + Ny * nOrb]\
                    + nUp[nOrb * j + 1 + Ny * nOrb]\
                    + nUp[nOrb * j + 2 + Ny * nOrb]\
                    - nDown[nOrb * j + Ny * nOrb]\
                    - nDown[nOrb * j + 1 + Ny * nOrb]\
                    - nDown[nOrb * j + 2 + Ny * nOrb] )
                if lat[longLength * j + i] < 0 :
                    clr[longLength * j + i] = "#95a5a6"
                else:
                    clr[longLength * j + i] = "#e74c3c"

    fig1 = plt.figure(2, figsize=(longLength, Ny))
    ax1 = fig1.add_subplot(111)
    ax1.scatter(vs[:, 0], vs[:, 1], s = abs(lat), c = clr.decode('UTF-8'), alpha = 0.8, edgecolors = None)
    ax1.axis('equal')
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])

def showWF(wfUp, wfDown, Nk, Ny, nOrb, Kpoint, Ktol, Eedge, Etol, eUp, eDown, abs_t0):
    ks = np.linspace(-np.pi,np.pi, num=Nk, endpoint=False)
    idxK = np.where(np.isclose(ks, Kpoint, atol=Ktol) == True)[0][0]
    idxE = np.where(np.isclose(eUp[idxK, :] * abs_t0, Eedge, atol=Etol) == True)[0][0]
    wfUp = wfUp.reshape(Nk, nOrb * Ny, nOrb * Ny)
    wfDown = wfDown.reshape(Nk, nOrb * Ny, nOrb * Ny)
    statesUp = wfUp[idxK, :, :]
    statesDown = wfDown[idxK, :, :]

    dz2Up = np.zeros(Ny)
    dxyUp = np.zeros(Ny)
    dx2y2Up = np.zeros(Ny)

    for i in range(Ny):
        dz2Up[i] = statesUp[nOrb * i, idxE]
        dxyUp[i] = statesUp[nOrb * i + 1, idxE]
        dx2y2Up[i] = statesUp[nOrb * i + 2, idxE]

    plt.figure(1)
    plt.plot(dz2Up, \
             label = \
             r'$| \psi_{k, d_{z^2}, \sigma} (y) |^2$',\
            linewidth = 0.7, marker = 'o', markersize = 4,
            )
    plt.plot(dxyUp, \
            label = \
             r'$| \psi_{k, d_{xy}, \sigma} (y) |^2$',\
            linewidth = 0.7, marker = 'o', markersize = 4)
    plt.plot(dx2y2Up, \
            label = \
             r'$| \psi_{k, d_{x^2 - y^2}, \sigma} (y) |^2$',\
            linewidth = 0.7, marker = 'o', markersize = 4)
    plt.xlabel(r'$y$')
    plt.ylabel(r'$| \psi_{k, \alpha, \sigma} (y) |^2$')
    plt.legend()
