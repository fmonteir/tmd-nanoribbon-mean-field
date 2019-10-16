import os
import sys
project_directory = '/Users/franciscobrito/projects/'
sys.path.append(project_directory + 'tmd-nanoribbon/mean-field/src/lib')
import numpy as np
from lib.lib_tmd_model import setParams, HribbonRealSpace
from lib.lib_solvers import solve_self_consistent_real_space
from lib.lib_init_cond import random

# Set the desired parameters

# tmd : choose the TMD (MoS2, WS2, MoSe2, WSe2, MoTe2, WTe2)
#
# Nk : Number of ks
#
# Ny : Number of atoms along the transverse direction
#
# nHole : Density of holes (to study a hole-doped system)
#
# invTemp : Inverse temperature (if we set it very high, we get T = 0)
#
# betaStart : Inverse temperature at which the annealing starts
#
# betaSpeed : This parameter (> 1) regulates the speed of the annealing
#
# betaThreshold: The point at which annealing stops (and after which we jump to the desired temperature - maybe zero!)
#
# U : On-site interaction
#
# itMax : Maximum allowed number of iterations
#
# singleExcitationFreq : How often to shake up the Markovian dynamics
#
# dampFreq : Frequency of the damping
#
# delta : Tolerance for updated densities convergence
#
# nUp, nDown = ferro(...) : Initial condition

# Number of orbitals in the model
nOrb = 3

# Dimensions of the ribbon
Nx = int(sys.argv[1])
Ny = int(sys.argv[2])

# Model parameters
t = 1
U = int(sys.argv[3])

# Choose TMD
tmd = 'MoS2'
abs_t0, e1, e2, t0, t1, t2, t11, t12, t22, \
E0, E1, E2, E3, E4, E5, E6 = setParams(tmd)
K = HribbonRealSpace(nOrb, Nx, Ny, E0, E1, E2, E3, E4, E5, E6)

# For a hole-doped system (0 means no holes)
nHole = 0

# Self-explanatory
anneal_or_not = True
osc = False

# Inverse temperature and annealing parameters
invTemp = 2
betaStart = 0.02
betaSpeed = 1.12
betaThreshold = 20

# Solver parameters
itMax = 100
dampFreq = 1
delta = 1e-6
singleExcitationFreq = itMax + 1
dyn = 'mixed'

# Initial conditions
seed = abs(int(np.loadtxt("seeds.csv")[int(sys.argv[4]) - 1]))
print("seed : ", seed)
nUp, nDown = random(2 / 3, Nx * Ny * nOrb, seed)

nUp, nDown, energies,\
lastGrandpotential, itSwitch, lastIt, mu,\
eUp, eDown, wfUp, wfDown\
= solve_self_consistent_real_space(Nx, Ny, nOrb, nHole, invTemp, betaStart, betaSpeed, betaThreshold,\
anneal_or_not, t, U, itMax, dampFreq, dyn, singleExcitationFreq, osc,\
K, abs_t0, delta, nUp, nDown)

SAVEDIR = tmd + "-U" + str(U) + "NY" + str(Ny)\
+ "BETA" + str(invTemp) + "real_random_ann"
if not os.path.exists(SAVEDIR):
        os.makedirs(SAVEDIR)

file = open(SAVEDIR + "/energies.csv", "a")

file.write(str(seed) + ',')
file.write(str(lastGrandpotential) + '\n')

file.close()
