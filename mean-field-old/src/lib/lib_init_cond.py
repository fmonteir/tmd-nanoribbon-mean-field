##
##  lib_init_cond.py
##
##
##  Created by Francisco Brito on 16/11/2018.
##
##  This library defines some possible initial conditions used to solve
##  the mean field intra-orbital Hubbard model for a TMD nanoribbon.
##
##

import numpy as np
import warnings
from scipy.optimize import bisect

# random initial condition
def random(filling, N, seed):
    '''
    Random initial condition.
    filling : number between 0 and 2 specifying the filling of the lattice.
    N : total number of sites (including different orbitals).
    seed : seed for the random number generation.
    '''
    np.random.seed(seed)
    nUp = np.random.rand(N)
    nDown = np.random.rand(N)
    nSum = nUp.sum() + nDown.sum()
    nUp = nUp / nSum * filling * N
    nDown = nDown / nSum * filling * N
    return nUp, nDown

# antiferromagnetic initial condition

def antiferro(filling, N):
    '''
    Fully antiferromagnetic initial condition.
    filling : number between 0 and 2 specifying the filling of the lattice.
    N : total number of sites (including different orbitals).
    '''
    nUp = np.array([1. , 0])
    for i in range(int(N/2) - 1):
        nUp = np.concatenate((nUp, np.array([1., 0]) ))
    nDown = np.array([0, 1])
    for i in range(int(N/2) - 1):
        nDown = np.concatenate((nDown, np.array([0, 1.]) ))
    nSum = nUp.sum() + nDown.sum()
    nUp = nUp / nSum * filling * N
    nDown = nDown / nSum * filling * N
    return nUp, nDown

# ferromagnetic initial condition
def ferro(filling, N, seed):
    '''
    Fully ferromagnetic initial condition.
    filling : number between 0 and 2 specifying the filling of the lattice.
    N : total number of sites (including different orbitals).
    '''
    np.random.seed(seed)
    nUp = np.ones(N) - 0.01 * np.random.rand(N)
    nDown = np.zeros(N) + 0.01 * np.random.rand(N)
    nSum = nUp.sum() + nDown.sum()
    nUp = nUp / nSum * filling * N
    nDown = nDown / nSum * filling * N
    return nUp, nDown

# ferromagnetic initial condition larger at the edges
def ferro_decaying(filling, N):
    '''
    Decaying ferromagnetic initial condition.
    filling : number between 0 and 2 specifying the filling of the lattice.
    N : total number of sites (including different orbitals).
    '''
    indices1 = np.arange(int(N/2))
    indices2 = np.arange(int(N/2), 0, -1)
    delta = 10
    nUp = np.zeros(N)
    nDown = np.zeros(N)
    nUp[:int(N/2)] = np.ones(int(N/2)) * np.exp(-indices1 / delta) - 0.001 * np.random.rand(int(N/2))
    nDown[:int(N/2)] = np.zeros(int(N/2)) + 0.001 * np.random.rand(int(N/2))
    nUp[int(N/2):N] = np.ones(int(N/2)) * np.exp(-indices2/ delta) - 0.001 * np.random.rand(int(N/2))
    nDown[int(N/2):N] = np.zeros(int(N/2)) + 0.001 * np.random.rand(int(N/2))
    nUp *= filling
    nDown *= filling
    return nUp, nDown

# ferromagnetic initial condition switching signs
def ferro_sign_switch(filling, N):
    '''
    Sign switching ferromagnetic initial condition.
    filling : number between 0 and 2 specifying the filling of the lattice.
    N : total number of sites (including different orbitals).
    '''
    indices1 = np.arange(int(N/2))
    indices2 = np.arange(int(N/2), 0, -1)
    delta = 10
    nUp = np.zeros(N)
    nDown = np.zeros(N)
    nUp[:int(N/2)] = np.zeros(int(N/2)) + 0.05 * np.random.rand(int(N/2))
    nDown[:int(N/2)] = np.ones(int(N/2)) * np.exp(-indices1 / delta) - 0.05 * np.random.rand(int(N/2))
    nUp[int(N/2):N] = np.ones(int(N/2)) * np.exp(-indices2/ delta) - 0.05 * np.random.rand(int(N/2))
    nDown[int(N/2):N] = np.zeros(int(N/2)) + 0.05 * np.random.rand(int(N/2))
    nUp *= filling
    nDown *= filling
    return nUp, nDown

# row ferromagnetic initial condition
def row_ferro(filling, N, Nx, nOrb):
    '''
    Row ferromagnetic initial condition.
    The sign of the magnetization alternates between rows
    filling : number between 0 and 2 specifying the filling of the lattice.
    N : total number of sites (including different orbitals).
    Nx : number of atoms along the x, longitudinal direction.
    nOrb : number of atomic orbitals included in the TB model.
    '''
    nUp = np.zeros(N)
    nDown = np.zeros(N)
    spinFlipper = 0.5
    for i in range(N):
        if i % (nOrb * Nx) == 0:
            spinFlipper *= -1
        nUp[i] = 0.5 + spinFlipper
        nDown[i] = 0.5 - spinFlipper
    nUp *= filling
    nDown *= filling
    return nUp, nDown

# row antiferromagnetic initial condition
def row_antiferro(filling, N, Nx, nOrb):
    '''
    Row antiferromagnetic initial condition.
    The sign of the magnetization alternates between rows.
    filling : number between 0 and 2 specifying the filling of the lattice.
    N : total number of sites (including different orbitals).
    Nx : number of atoms along the x, longitudinal direction.
    nOrb : number of atomic orbitals included in the TB model.
    '''
    nUp = np.zeros(N)
    nDown = np.zeros(N)
    spinFlipper = 0.5
    for i in range(N):
        if i % (nOrb * Nx) == 0:
            spinFlipper *= -1
        nUp[i] = 0.5 + spinFlipper * (-1)**i
        nDown[i] = 0.5 - spinFlipper * (-1)**i
    nUp *= filling
    nDown *= filling
    return nUp, nDown

# row antiferromagnetic initial condition
def antiferro_along_row(filling, Ny, nOrb):
    '''
    Antiferromagnetic along each row initial condition.
    The sign of the magnetization alternates between atoms in a row.
    filling : number between 0 and 2 specifying the filling of the lattice.
    Ny : number of atoms along the y, transverse, direction.
    nOrb : number of atomic orbitals included in the TB model.
    '''
    nUp = np.zeros(2 * Ny * nOrb)
    nDown = np.zeros(2 * Ny * nOrb)
    for i in range(2 * Ny * nOrb):
        if i >= Ny * nOrb:
            nDown[i] = 1
        else:
            nUp[i] = 1
    nUp -= 0.01 * np.random.rand(2 * Ny * nOrb)
    nDown += 0.01 * np.random.rand(2 * Ny * nOrb)
    nUp *= filling
    nDown *= filling
    return nUp, nDown
