import os
import sys
project_directory = '/Users/franciscobrito/projects/'
sys.path.append(project_directory + 'tmd-nanoribbon/mean-field/src/lib')
import numpy as np
from lib.lib_tmd_model import setParams
from lib.lib_solvers import solve_self_consistent_k_space
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

# Number of k-points
Nk = 512

# Dimensions of the ribbon
Ny = int(sys.argv[1])

# Model parameters
t = 1
U = int(sys.argv[2])

# Choose TMD
tmd = 'MoS2'
abs_t0, e1, e2, t0, t1, t2, t11, t12, t22, \
E0, E1, E2, E3, E4, E5, E6 = setParams(tmd)

# For a hole-doped system (0 means no holes)
nHole = 0

# Self-explanatory
anneal_or_not = False
osc = False

# Inverse temperature and annealing parameters
invTemp = 'infty'
betaStart = 0.1
betaSpeed = 1.25
betaThreshold = 20

# Solver parameters
itMax = 50
dampFreq = 1
delta = 1e-5
singleExcitationFreq = 4
dyn = 'wait'

# Initial conditions
seed = abs(int(np.loadtxt("seeds.csv")[int(sys.argv[3]) - 1]))
print("seed : ", seed)
nUp, nDown = random(2 / 3, Ny * nOrb, seed)

nUp, nDown, energies,\
lastGrandpotential, itSwitch, lastIt, mu, abs_t0,\
eUp, eDown, wfUp, wfDown\
= solve_self_consistent_k_space(abs_t0, e1, e2, t0, t1, t2, t11, t12, t22,\
                                  Nk, Ny, nOrb, nHole,\
                                  invTemp, betaStart, betaSpeed, betaThreshold,\
                                  anneal_or_not, U, itMax, dampFreq, dyn,\
                                  singleExcitationFreq, osc, delta, nUp, nDown)

SAVEDIR = tmd + "-U" + str(U) + "NY" + str(Ny)\
+ "BETA" + str(invTemp) + "-normal-unit-cell-random-minimal"
if not os.path.exists(SAVEDIR):
        os.makedirs(SAVEDIR)

file = open(SAVEDIR + "/energies.csv", "a")

file.write(str(seed) + ',')
file.write(str(lastGrandpotential) + '\n')

file.close()
