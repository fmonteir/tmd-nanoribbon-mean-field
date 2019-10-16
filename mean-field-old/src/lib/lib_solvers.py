##
##  lib_solvers.py
##
##
##  Created by Francisco Brito on 16/11/2018.
##
##  This library solves the mean field intra-orbital Hubbard model for a TMD
##  nanoribbon self-consistently. It uses a 3-band tight-binding model
##  (Liu et al., Phys Rev B 88, 085433 (2013)) and adds an on-site intra-orbital
##  interaction U treated in mean field.
##  These calculations are to be compared with unbiased determinant QMC which
##  treat the interaction more precisely.
##
##

import numpy as np
import numpy.linalg as la
import sys
project_directory = '/Users/franciscobrito/projects/'
sys.path.append(project_directory + 'tmd-nanoribbon/mean-field-old/src/lib')
from lib.lib_tmd_model import HribbonKSpace, HribbonKSpace2sublat
from lib.lib_solver_helpers import *
import warnings
from scipy.optimize import bisect
import time

def solve_self_consistent_real_space\
(Nx, Ny, nOrb, nHole, invTemp, betaStart, betaSpeed, betaThreshold,\
anneal_or_not, t, U, itMax, dampFreq, dyn, singleExcitationFreq, osc,\
K, abs_t0, delta, nUp, nDown):
    '''
    Solves the self-consistent equation in real space.
    '''
    nUpBest, nDownBest = nUp, nDown
    # Total number of sites + orbitals
    N = nOrb * Nx * Ny
    # Initialize deltas for the tolerance check (ensure that it does not stop
    # at the first step)
    deltaUp = delta + 1
    deltaDown = delta + 1
    # Initialize inverse temperature for the annealing
    beta = betaStart
    # Initialize energies
    energies = np.zeros(itMax)
    bestGrandpotential = 1e100
    # Initialize iteration
    it = 0
    # Initialize iteration at which we finish annealing
    itSwitch = 0
    # How many iterations to wait between dynamic kicks
    itWait = 3
    # lbda is a parameter that reduces the weight
    # on the density obtained in the previous iteration.
    # the factor multiplied by itMax impedes that P ( I ) < delta
    # initially, we give equal weights and then progressively more
    # to the new configuration
    factor = 1.2
    lbda = 0.5 / (factor * itMax)
    # This ensures that we do not stop at the first step
    energies[-1] = 1e100
    # Print frequency
    printFreq = 10

    if anneal_or_not == True:
        print("Started annealing.\n")

    while loopCondition(it, itMax, deltaUp, deltaDown,\
                        delta, beta, invTemp, betaThreshold):

        # Annealing
        if anneal_or_not == True:
            beta = anneal(invTemp, betaStart, betaSpeed, beta,\
            betaThreshold, it, osc)
        else:
            beta = noAnneal(invTemp, beta)

        # Define the MF Hamiltonian for this iteration
        C, Hup, Hdown = hamiltonian(nUp, nDown, K, U, N)

        # Diagonalize
        eUp, wUp = la.eigh(Hup)
        eDown, wDown = la.eigh(Hdown)

        # Save the previous fields to compare to test convergence
        nUpOld = nUp.copy()
        nDownOld = nDown.copy()

        # Compute the chemical potential implicitly
        interval_limits = 50
        mu = bisect(rootToChem, -interval_limits, interval_limits, \
        args = (eUp, eDown, beta, Nx, Ny, nHole) )

        # Update fields
        nUp, nDown = update(nUp, nDown, N, wUp, wDown, eUp, eDown, mu, beta)

        # Grandpotential per site
        energy = grandpotential(U, nUp, nDown, nUpOld, nDownOld, \
        nOrb, Nx, Ny, mu, invTemp, eUp, eDown, abs_t0)

        # Damping
        nUp, nDown = damp(it, dampFreq, nUp, nDown, nUpOld, nDownOld, lbda)

        # Relative difference between current and previous fields
        deltaUp = np.dot(nUp - nUpOld, nUp - nUpOld) \
        / np.dot(nUp, nUp)
        deltaDown = np.dot(nDown - nDownOld, nDown - nDownOld) \
        / np.dot(nDown, nDown)

        if it % printFreq == 0:
            print("\niteration: ", it)
            print("deltaUp: ", deltaUp)
            print("deltaDown: ", deltaDown, "\n")

        if ( it + 1 ) % singleExcitationFreq == 0 :
            if dyn == 'local' or dyn == 'wait':
                for attempt in range(nOrb * N):
                    i = int( np.random.random() * N )
                    nUpTmp, nDownTmp = nUp[i], nDown[i]
                    nUp[i], nDown[i] = updateLocal(i, nUp, nDown, N, wUp, wDown,\
                    eUp, eDown, mu, beta)

                    # Define the MF Hamiltonian for this iteration
                    C, Hup, Hdown = hamiltonian(nUp, nDown, K, U, N)

                    # Diagonalize
                    eUp, wUp = la.eigh(Hup)
                    eDown, wDown = la.eigh(Hdown)

                    # Save the previous fields to compare to test convergence
                    nUpOld = nUp.copy()
                    nDownOld = nDown.copy()

                    # Compute the chemical potential implicitly
                    interval_limits = 50
                    mu = bisect(rootToChem, -interval_limits, interval_limits, \
                    args = (eUp, eDown, beta, Nx, Ny, nHole) )

                    # Update fields
                    nUp, nDown = update(nUp, nDown, N, wUp, wDown, eUp, eDown, mu, beta)

                    # Grandpotential per site
                    energyTmp = grandpotential(U, nUp, nDown, nUpOld, nDownOld, \
                    nOrb, Nx, Ny, mu, invTemp, eUp, eDown, abs_t0)

                    if energyTmp > energy:
                        nUp[i], nDown[i] = nUpTmp, nDownTmp
                    else:
                        energy = energyTmp

            elif dyn == 'kick':
                for attempt in range(nOrb * N):
                    i = int( np.random.random() * N )
                    nUpTmp, nDownTmp = nUp[i], nDown[i]
                    nUp[i], nDown[i] = singleExcitation(nUp, nDown, i)

                    # Define the MF Hamiltonian for this iteration
                    C, Hup, Hdown = hamiltonian(nUp, nDown, K, U, N)

                    # Diagonalize
                    eUp, wUp = la.eigh(Hup)
                    eDown, wDown = la.eigh(Hdown)

                    # Save the previous fields to compare to test convergence
                    nUpOld = nUp.copy()
                    nDownOld = nDown.copy()

                    # Compute the chemical potential implicitly
                    interval_limits = 50
                    mu = bisect(rootToChem, -interval_limits, interval_limits, \
                    args = (eUp, eDown, beta, Nx, Ny, nHole) )

                    # Update fields
                    nUp, nDown = update(nUp, nDown, N, wUp, wDown, eUp, eDown, mu, beta)

                    # Grandpotential per site
                    energyTmp = grandpotential(U, nUp, nDown, nUpOld, nDownOld, \
                    nOrb, Nx, Ny, mu, invTemp, eUp, eDown, abs_t0)

                    if energyTmp > energy:
                        nUp[i], nDown[i] = nUpTmp, nDownTmp
                    else:
                        energy = energyTmp

            elif dyn == 'mixed':
                for attempt in range(nOrb * N):
                    i = int( np.random.random() * N )
                    nUpTmp, nDownTmp = nUp[i], nDown[i]
                    nUp[i], nDown[i] = updateLocal(i, nUp, nDown, N, wUp, wDown,\
                    eUp, eDown, mu, beta)
                    nUp[i], nDown[i] = singleExcitation(nUp, nDown, i)

                    # Define the MF Hamiltonian for this iteration
                    C, Hup, Hdown = hamiltonian(nUp, nDown, K, U, N)

                    # Diagonalize
                    eUp, wUp = la.eigh(Hup)
                    eDown, wDown = la.eigh(Hdown)

                    # Save the previous fields to compare to test convergence
                    nUpOld = nUp.copy()
                    nDownOld = nDown.copy()

                    # Compute the chemical potential implicitly
                    interval_limits = 50
                    mu = bisect(rootToChem, -interval_limits, interval_limits, \
                    args = (eUp, eDown, beta, Nx, Ny, nHole) )

                    # Update fields
                    nUp, nDown = update(nUp, nDown, N, wUp, wDown, eUp, eDown, mu, beta)

                    # Grandpotential per site
                    energyTmp = grandpotential(U, nUp, nDown, nUpOld, nDownOld, \
                    nOrb, Nx, Ny, mu, invTemp, eUp, eDown, abs_t0)

                    if energyTmp > energy:
                        nUp[i], nDown[i] = nUpTmp, nDownTmp
                    else:
                        energy = energyTmp
        if dyn == 'wait':
            if ( it + 1 + itWait ) % singleExcitationFreq == 0 :
                for attempt in range(nOrb * N):
                    i = int( np.random.random() * N )
                    nUpTmp, nDownTmp = nUp[i], nDown[i]
                    nUp[i], nDown[i] = singleExcitation(nUp, nDown, i)

                    # Define the MF Hamiltonian for this iteration
                    C, Hup, Hdown = hamiltonian(nUp, nDown, K, U, N)

                    # Diagonalize
                    eUp, wUp = la.eigh(Hup)
                    eDown, wDown = la.eigh(Hdown)

                    # Save the previous fields to compare to test convergence
                    nUpOld = nUp.copy()
                    nDownOld = nDown.copy()

                    # Compute the chemical potential implicitly
                    mu = -1
                    # interval_limits = 50
                    # mu = bisect(rootToChem, -interval_limits, interval_limits, \
                    # args = (eUp, eDown, beta, Nx, Ny, nHole) )

                    # Update fields
                    nUp, nDown = update(nUp, nDown, N, wUp, wDown, eUp, eDown, mu, beta)

                    # Grandpotential per site
                    energyTmp = grandpotential(U, nUp, nDown, nUpOld, nDownOld, \
                    nOrb, Nx, Ny, mu, invTemp, eUp, eDown, abs_t0)

                    if energyTmp > energy:
                        nUp[i], nDown[i] = nUpTmp, nDownTmp
                    else:
                        energy = energyTmp

        energies[it] = energy

        if invTemp == 'infty':
            if energy < bestGrandpotential and beta >= betaThreshold:
                bestGrandpotential = energy
                nUpBest, nDownBest = nUp, nDown
                eUpBest, eDownBest = eUp, eDown
                wUpBest, wDownBest = wUp, wDown
        else:
            if energy < bestGrandpotential and beta == invTemp:
                bestGrandpotential = energy
                nUpBest, nDownBest = nUp, nDown
                eUpBest, eDownBest = eUp, eDown
                wUpBest, wDownBest = wUp, wDown

        # Move to the next iteration
        it += 1

    # Save the last iteration
    lastIt = it
    print("\nTotal number of iterations: ", lastIt, "\n")
    return nUpBest, nDownBest, energies, bestGrandpotential,\
    itSwitch, lastIt, mu, eUpBest, eDownBest,\
    np.absolute(wUpBest.flatten('C'))**2, np.absolute(wDownBest.flatten('C'))**2

def solve_self_consistent_k_space(abs_t0, e1, e2, t0, t1, t2, t11, t12, t22,\
                                  Nk, Ny, nOrb, nHole,\
                                  invTemp, betaStart, betaSpeed, betaThreshold,\
                                  anneal_or_not, U, itMax, dampFreq, dyn, \
                                  singleExcitationFreq, osc, delta, nUp, nDown):
    '''
    Solves the self-consistent equation in momentum space along the
    longitudinal direction.
    '''
    # Initialize deltas for the tolerance check and
    # beta for the annealing.
    deltaUp = delta + 1
    deltaDown = delta + 1
    beta = betaStart
    # Initialize energies
    energies = np.zeros(itMax)
    # Initialize iteration
    it = 0
    # Initialize iteration at which we finish annealing
    itSwitch = 0
    # How many iterations to wait between kicks
    itWait = 3
    # lbda is a parameter that reduces weight
    # on the density obtained in the previous iteration.
    # the factor multiplied by itMax impedes that P ( I ) < delta
    # initially, we give equal weights and then progressively more
    # to the new configuration
    factor = 1.2
    lbda = 0.5 / (factor * itMax)
    # This ensures that we do not stop at the first step
    energies[-1] = 1e100
    # Print frequency
    printFreq = 10

    # Initialize arrays to store energies and eigenstates
    eUp = np.zeros((Nk, nOrb * Ny))
    wUp = np.zeros((Nk, nOrb * Ny, nOrb * Ny), dtype=np.complex64)
    eDown = np.zeros((Nk, nOrb * Ny))
    wDown = np.zeros((Nk, nOrb * Ny, nOrb * Ny), dtype=np.complex64)
    eUpBest = np.zeros((Nk, nOrb * Ny))
    wUpBest = np.zeros((Nk, nOrb * Ny, nOrb * Ny), dtype=np.complex64)
    eDownBest = np.zeros((Nk, nOrb * Ny))
    wDownBest = np.zeros((Nk, nOrb * Ny, nOrb * Ny), dtype=np.complex64)
    ks = np.linspace(-np.pi,np.pi, num=Nk, endpoint=False)

    if anneal_or_not == True:
        print("Started annealing.\n")

    nUpBest, nDownBest = nUp, nDown
    bestGrandpotential = 1e100

    while loopCondition(it, itMax, deltaUp, deltaDown,\
                        delta, beta, invTemp, betaThreshold):

        if anneal_or_not == True:
            beta = anneal(invTemp, betaStart, betaSpeed, beta,\
            betaThreshold, it, osc)
        else:
            beta = noAnneal(invTemp, beta)

        for kCount, k in enumerate(ks):

            # Define the MF Hamiltonian for this iteration and k-point
            K = HribbonKSpace(k, nOrb, Ny, e1, e2, t0, t1, t2, t11, t12, t22)
            Hup = K + U * np.eye( nOrb * Ny ) * nDown
            Hdown = K + U * np.eye( nOrb * Ny ) * nUp

            # Diagonalize
            eUp[kCount, :], wUp[kCount, :, :] = la.eigh(Hup)
            eDown[kCount, :], wDown[kCount, :, :] = la.eigh(Hdown)

        # Save the previous fields to compare to test convergence
        nUpOld = nUp.copy()
        nDownOld = nDown.copy()

        # Compute the chemical potential implicitly
        #mu = bisect(rootToChem, -50, 50,\
        #args = (eUp, eDown, beta, Nk, Ny, nHole) )
        mu = 12.5
        # Update fields
        nUp, nDown = updateKSpace(nUp, nDown, Nk, Ny,\
         wUp, wDown, eUp, eDown, mu, beta, nOrb)

        # Grandpotential per site
        energy = grandpotentialKSpace(U, nUp, nDown, nUpOld, nDownOld, invTemp,\
        nOrb, Nk, Ny, eUp, eDown, wUp, wDown, mu, abs_t0)

        # Damping
        nUp, nDown = damp(it, dampFreq, nUp, nDown, nUpOld, nDownOld, lbda)

        # Relative difference between current and previous fields
        deltaUp = np.dot(nUp - nUpOld, nUp - nUpOld)\
         / np.dot(nUpOld, nUpOld)
        deltaDown = np.dot(nDown - nDownOld, nDown - nDownOld)\
         / np.dot(nDownOld, nDownOld)

        if it % printFreq == 0:
            print("iteration: ", it)
            print("deltaUp: ", deltaUp)
            print("deltaDown: ", deltaDown)

        if ( it + 1 ) % singleExcitationFreq == 0 :
            if dyn == 'local' or dyn == 'wait':
                for attempt in range(nOrb * Ny):
                    i = int( np.random.random() * nOrb * Ny )
                    nUpTmp, nDownTmp = nUp[i], nDown[i]
                    nUp[i], nDown[i] = updateLocalKSpace(i, nUp, nDown, Nk, Ny,\
                    wUp, wDown, eUp, eDown, mu, beta, nOrb)
                    for kCount, k in enumerate(ks):
                        # Define the MF Hamiltonian for this
                        # iteration and k-point

                        K = HribbonKSpace(k, nOrb, Ny, e1, e2,\
                        t0, t1, t2, t11, t12, t22)
                        Hup = K + U * np.eye( nOrb * Ny ) * nDown
                        Hdown = K + U * np.eye( nOrb * Ny ) * nUp

                        # Diagonalize
                        eUp[kCount, :], wUp[kCount, :, :] = la.eigh(Hup)
                        eDown[kCount, :], wDown[kCount, :, :] = la.eigh(Hdown)

                    # Save the previous fields to compare to test convergence
                    nUpOld = nUp.copy()
                    nDownOld = nDown.copy()

                    # Compute the chemical potential implicitly
                    mu = bisect(rootToChem, -50, 50,\
                     args = (eUp, eDown, beta, Nk, Ny, nHole) )

                    # Update fields
                    nUp, nDown = updateKSpace(nUp, nDown, Nk, Ny,\
                     wUp, wDown, eUp, eDown, mu, beta, nOrb)

                    # Grandpotential per site
                    energyTmp = grandpotentialKSpace(U, nUp, nDown, nUpOld,\
                    nDownOld, invTemp, nOrb, Nk, Ny, eUp, eDown, mu, abs_t0)

                    if energyTmp > energy:
                        nUp[i], nDown[i] = nUpTmp, nDownTmp
                    else:
                        energy = energyTmp

            elif dyn == 'kick':
                for attempt in range(nOrb * Ny):
                    i = int( np.random.random() * nOrb * Ny )
                    nUpTmp, nDownTmp = nUp[i], nDown[i]
                    nUp[i], nDown[i] = singleExcitation(nUp, nDown, i)
                    for kCount, k in enumerate(ks):
                        # Define the MF Hamiltonian for this
                        # iteration and k-point

                        K = HribbonKSpace(k, nOrb, Ny, e1, e2,\
                        t0, t1, t2, t11, t12, t22)
                        Hup = K + U * np.eye( nOrb * Ny ) * nDown
                        Hdown = K + U * np.eye( nOrb * Ny ) * nUp

                        # Diagonalize
                        eUp[kCount, :], wUp[kCount, :, :] = la.eigh(Hup)
                        eDown[kCount, :], wDown[kCount, :, :] = la.eigh(Hdown)

                    # Save the previous fields to compare to test convergence
                    nUpOld = nUp.copy()
                    nDownOld = nDown.copy()

                    # Compute the chemical potential implicitly
                    mu = bisect(rootToChem, -50, 50,\
                     args = (eUp, eDown, beta, Nk, Ny, nHole) )

                    # Update fields
                    nUp, nDown = updateKSpace(nUp, nDown, Nk, Ny,\
                     wUp, wDown, eUp, eDown, mu, beta, nOrb)

                    # Grandpotential per site
                    energyTmp = grandpotentialKSpace(U, nUp, nDown, nUpOld,\
                    nDownOld, invTemp, nOrb, Nk, Ny, eUp, eDown, mu, abs_t0)

                    if energyTmp > energy:
                        nUp[i], nDown[i] = nUpTmp, nDownTmp
                    else:
                        energy = energyTmp
            elif dyn == 'mixed':
                for attempt in range(nOrb * Ny):
                    i = int( np.random.random() * nOrb * Ny )
                    nUpTmp, nDownTmp = nUp[i], nDown[i]
                    nUp[i], nDown[i] = updateLocalKSpace(i, nUp, nDown, Nk, Ny,\
                    wUp, wDown, eUp, eDown, mu, beta, nOrb)
                    nUp[i], nDown[i] = singleExcitation(nUp, nDown, i)
                    for kCount, k in enumerate(ks):
                        # Define the MF Hamiltonian for this
                        # iteration and k-point

                        K = HribbonKSpace(k, nOrb, Ny, e1, e2,\
                        t0, t1, t2, t11, t12, t22)
                        Hup = K + U * np.eye( nOrb * Ny ) * nDown
                        Hdown = K + U * np.eye( nOrb * Ny ) * nUp

                        # Diagonalize
                        eUp[kCount, :], wUp[kCount, :, :] = la.eigh(Hup)
                        eDown[kCount, :], wDown[kCount, :, :] = la.eigh(Hdown)

                    # Save the previous fields to compare to test convergence
                    nUpOld = nUp.copy()
                    nDownOld = nDown.copy()

                    # Compute the chemical potential implicitly
                    mu = bisect(rootToChem, -50, 50,\
                     args = (eUp, eDown, beta, Nk, Ny, nHole) )

                    # Update fields
                    nUp, nDown = updateKSpace(nUp, nDown, Nk, Ny,\
                     wUp, wDown, eUp, eDown, mu, beta, nOrb)

                    # Grandpotential per site
                    energyTmp = grandpotentialKSpace(U, nUp, nDown, nUpOld,\
                    nDownOld, invTemp, nOrb, Nk, Ny, eUp, eDown, mu, abs_t0)

                    if energyTmp > energy:
                        nUp[i], nDown[i] = nUpTmp, nDownTmp
                    else:
                        energy = energyTmp
        if dyn == 'wait':
            if ( it + 1 + itWait ) % singleExcitationFreq == 0 :
                for attempt in range(nOrb * Ny):
                    i = int( np.random.random() * nOrb * Ny )
                    nUpTmp, nDownTmp = nUp[i], nDown[i]
                    nUp[i], nDown[i] = singleExcitation(nUp, nDown, i)
                    for kCount, k in enumerate(ks):
                        # Define the MF Hamiltonian for this
                        # iteration and k-point

                        K = HribbonKSpace(k, nOrb, Ny, e1, e2,\
                        t0, t1, t2, t11, t12, t22)
                        Hup = K + U * np.eye( nOrb * Ny ) * nDown
                        Hdown = K + U * np.eye( nOrb * Ny ) * nUp

                        # Diagonalize
                        eUp[kCount, :], wUp[kCount, :, :] = la.eigh(Hup)
                        eDown[kCount, :], wDown[kCount, :, :] = la.eigh(Hdown)

                    # Save the previous fields to compare to test convergence
                    nUpOld = nUp.copy()
                    nDownOld = nDown.copy()

                    # Compute the chemical potential implicitly
                    mu = bisect(rootToChem, -50, 50,\
                     args = (eUp, eDown, beta, Nk, Ny, nHole) )

                    # Update fields
                    nUp, nDown = updateKSpace(nUp, nDown, Nk, Ny,\
                     wUp, wDown, eUp, eDown, mu, beta, nOrb)

                    # Grandpotential per site
                    energyTmp = grandpotentialKSpace(U, nUp, nDown, nUpOld,\
                    nDownOld, invTemp, nOrb, Nk, Ny, eUp, eDown, mu, abs_t0)

                    if energyTmp > energy:
                        nUp[i], nDown[i] = nUpTmp, nDownTmp
                    else:
                        energy = energyTmp

        energies[it] = energy

        bestGrandpotential = energy

        # if invTemp == 'infty':
        #     if energy < bestGrandpotential and beta >= betaThreshold:
        #         bestGrandpotential = energy
        #         nUpBest, nDownBest = nUp, nDown
        #         eUpBest, eDownBest = eUp, eDown
        #         wUpBest, wDownBest = wUp, wDown
        # else:
        #     if energy < bestGrandpotential and beta == invTemp:
        #         bestGrandpotential = energy
        #         nUpBest, nDownBest = nUp, nDown
        #         eUpBest, eDownBest = eUp, eDown
        #         wUpBest, wDownBest = wUp, wDown

        # Move to the next iteration
        it += 1

    # Save the last iteration
    lastIt = it
    return nUp, nDown, energies, bestGrandpotential, itSwitch,\
     lastIt, mu, abs_t0, eUp, eDown,\
     np.absolute(wUp.flatten('C'))**2, np.absolute(wDown.flatten('C'))**2

def solve_self_consistent_k_space_2sublat\
(abs_t0, E0, E1, E2, E3, E4,\
Nk, Ny, nOrb, nHole, invTemp, betaStart, betaSpeed, betaThreshold,anneal_or_not,\
U, itMax, dampFreq, dyn, singleExcitationFreq, osc, delta, nUp, nDown):
    '''
    Solves the self-consistent equation in momentum space.
    '''
    # Initialize deltas for the tolerance check and
    # beta for the annealing.
    deltaUp = delta + 1
    deltaDown = delta + 1
    beta = betaStart
    # Initialize energies
    energies = np.zeros(itMax)
    # Initialize iteration
    it = 0
    # Initialize iteration at which we finish annealing
    itSwitch = 0
    # How many iterations to wait between kicks
    itWait = 3
    # lbda is a parameter that reduces weight
    # on the density obtained in the previous iteration.
    # the factor multiplied by itMax impedes that P ( I ) < delta
    # initially, we give equal weights and then progressively more
    # to the new configuration
    factor = 1.2
    lbda = 0.5 / (factor * itMax)
    # This ensures that we do not stop at the first step
    energies[-1] = 1e100
    # Print frequency
    printFreq = 10

    # Initialize arrays to store energies and eigenstates
    eUp = np.zeros((Nk, 2 * nOrb * Ny))
    wUp = np.zeros((Nk, 2 * nOrb * Ny, 2 * nOrb * Ny), dtype=np.complex64)
    eDown = np.zeros((Nk, 2 * nOrb * Ny))
    wDown = np.zeros((Nk, 2 * nOrb * Ny, 2 * nOrb * Ny), dtype=np.complex64)
    eUpBest = np.zeros((Nk, 2 * nOrb * Ny))
    wUpBest = np.zeros((Nk, 2 * nOrb * Ny, 2 * nOrb * Ny), dtype=np.complex64)
    eDownBest = np.zeros((Nk, 2 * nOrb * Ny))
    wDownBest = np.zeros((Nk, 2 * nOrb * Ny, 2 * nOrb * Ny), dtype=np.complex64)
    ks = np.linspace(-np.pi / 2,np.pi / 2, num=Nk, endpoint=False)

    if anneal_or_not == True:
        print("Started annealing.\n")

    nUpBest, nDownBest = nUp, nDown
    bestGrandpotential = 1e100

    while loopCondition(it, itMax, deltaUp, deltaDown,\
                        delta, beta, invTemp, betaThreshold):

        # Annealing
        if anneal_or_not == True:
            beta = anneal(invTemp, betaStart, betaSpeed, beta,\
            betaThreshold, it, osc)
        else:
            beta = noAnneal(invTemp, beta)

        for kCount, k in enumerate(ks):
            # Define the MF Hamiltonian for this iteration and k-point
            K = HribbonKSpace2sublat(k, nOrb, Ny, E0, E1, E2, E3, E4)
            Hup = K + U * np.eye(2 * nOrb * Ny) * nDown
            Hdown = K + U * np.eye(2 * nOrb * Ny) * nUp

            # Diagonalize
            eUp[kCount, :], wUp[kCount, :, :] = la.eigh(Hup)
            eDown[kCount, :], wDown[kCount, :, :] = la.eigh(Hdown)

        # Save the previous fields to compare to test convergence
        nUpOld = nUp.copy()
        nDownOld = nDown.copy()

        # Compute the chemical potential implicitly
        mu = bisect(rootToChem2sublat, -50, 50,\
         args = (eUp, eDown, beta, Nk, Ny, nHole) )

        # Update fields
        nUp, nDown = update2sublat(nUp, nDown, Nk, Ny,\
         wUp, wDown, eUp, eDown, mu, beta, nOrb)

        # Damping
        nUp, nDown = damp(it, dampFreq, nUp, nDown, nUpOld, nDownOld, lbda)

        # Relative difference between current and previous fields
        deltaUp = np.dot(nUp - nUpOld, nUp - nUpOld)\
         / np.dot(nUpOld, nUpOld)
        deltaDown = np.dot(nDown - nDownOld, nDown - nDownOld)\
         / np.dot(nDownOld, nDownOld)

        if it % printFreq == 0:
            print("iteration: ", it)
            print("deltaUp: ", deltaUp)
            print("deltaDown: ", deltaDown)

        # Grandpotential per site
        energy = grandpotentialKSpace2sublat(U, nUp, nDown, nUpOld, nDownOld,\
         invTemp, Nk, Ny, eUp, eDown, mu, abs_t0)
        if it % singleExcitationFreq == 0 :
            for attempt in range(2 * nOrb * Ny):
                i = int( np.random.random() * 2 * nOrb * Ny )
                nUpTmp, nDownTmp = nUp[i], nDown[i]
                nUp[i], nDown[i] = updateLocal2sublat(i, nUp, nDown, Nk, Ny,\
                wUp, wDown, eUp, eDown, mu, beta, nOrb)
                energyTmp = grandpotentialKSpace2sublat(U, nUp, nDown,\
                nUpOld, nDownOld, invTemp, Nk, Ny, eUp, eDown, mu, abs_t0)
                if energyTmp > energy:
                    nUp[i], nDown[i] = nUpTmp, nDownTmp
                else:
                    energy = energyTmp

        if ( it + 1 ) % singleExcitationFreq == 0 :
            for attempt in range(nOrb * 2 * Ny):
                i = int( np.random.random() * 2 * nOrb * Ny )
                nUpTmp, nDownTmp = nUp[i], nDown[i]
                if dyn == 'local' or dyn == 'wait':
                    nUp[i], nDown[i] = updateLocalKSpace(i, nUp, nDown, Nk, Ny,\
                    wUp, wDown, eUp, eDown, mu, beta, nOrb)
                    # We do not take steps that increase energy
                    energyTmp = grandpotentialKSpace2sublat(U, nUp, nDown,\
                    nUpOld, nDownOld, invTemp, Nk, Ny, eUp, eDown, mu, abs_t0)
                    if energyTmp > energy:
                        nUp[i], nDown[i] = nUpTmp, nDownTmp
                    else:
                        energy = energyTmp
                if dyn == 'kick':
                    nUp[i], nDown[i] = singleExcitation(nUp, nDown, i)
                    # We do not take steps that increase energy
                    energyTmp = grandpotentialKSpace2sublat(U, nUp, nDown,\
                    nUpOld, nDownOld, invTemp, Nk, Ny, eUp, eDown, mu, abs_t0)
                    if energyTmp > energy:
                        nUp[i], nDown[i] = nUpTmp, nDownTmp
                    else:
                        energy = energyTmp
                if dyn == 'mixed':
                    nUp[i], nDown[i] = updateLocalKSpace(i, nUp, nDown, Nk, Ny,\
                    wUp, wDown, eUp, eDown, mu, beta, nOrb)
                    # We do not take steps that increase energy
                    energyTmp = grandpotentialKSpace2sublat(U, nUp, nDown,\
                    nUpOld, nDownOld, invTemp, Nk, Ny, eUp, eDown, mu, abs_t0)
                    if energyTmp > energy:
                        nUp[i], nDown[i] = nUpTmp, nDownTmp
                    else:
                        energy = energyTmp
                    nUp[i], nDown[i] = singleExcitation(nUp, nDown, i)
                    # We do not take steps that increase energy
                    energyTmp = grandpotentialKSpace2sublat(U, nUp, nDown,\
                    nUpOld, nDownOld, invTemp, Nk, Ny, eUp, eDown, mu, abs_t0)
                    if energyTmp > energy:
                        nUp[i], nDown[i] = nUpTmp, nDownTmp
                    else:
                        energy = energyTmp
        if dyn == 'wait':
            if ( it + 1 + itWait ) % singleExcitationFreq == 0 :
                for attempt in range(nOrb * 2 * Ny):
                    i = int( np.random.random() * 2 * nOrb * Ny )
                    nUpTmp, nDownTmp = nUp[i], nDown[i]
                    nUp[i], nDown[i] = singleExcitation(nUp, nDown, i)
                    # We do not take steps that increase energy
                    energyTmp = grandpotentialKSpace2sublat(U, nUp, nDown,\
                    nUpOld, nDownOld, invTemp, Nk, Ny, eUp, eDown, mu, abs_t0)
                    if energyTmp > energy:
                        nUp[i], nDown[i] = nUpTmp, nDownTmp
                    else:
                        energy = energyTmp

        ## HERE

        if ( it + 1 ) % singleExcitationFreq == 0 :
            if dyn == 'local' or dyn == 'wait':
                for attempt in range(2 * nOrb * Ny):
                    i = int( np.random.random() * 2 * nOrb * Ny )
                    nUpTmp, nDownTmp = nUp[i], nDown[i]
                    nUp[i], nDown[i] = updateLocal2sublat(i, nUp, nDown, Nk, Ny,\
                    wUp, wDown, eUp, eDown, mu, beta, nOrb)
                    for kCount, k in enumerate(ks):
                        # Define the MF Hamiltonian for this iteration and k-point
                        K = HribbonKSpace2sublat(k, nOrb, Ny, E0, E1, E2, E3, E4)
                        Hup = K + U * np.eye(2 * nOrb * Ny) * nDown
                        Hdown = K + U * np.eye(2 * nOrb * Ny) * nUp

                        # Diagonalize
                        eUp[kCount, :], wUp[kCount, :, :] = la.eigh(Hup)
                        eDown[kCount, :], wDown[kCount, :, :] = la.eigh(Hdown)

                    # Save the previous fields to compare to test convergence
                    nUpOld = nUp.copy()
                    nDownOld = nDown.copy()

                    # Compute the chemical potential implicitly
                    mu = bisect(rootToChem2sublat, -50, 50,\
                     args = (eUp, eDown, beta, Nk, Ny, nHole) )

                    # Update fields
                    nUp, nDown = update2sublat(nUp, nDown, Nk, Ny,\
                     wUp, wDown, eUp, eDown, mu, beta, nOrb)

                    # Grandpotential per site
                    energyTmp = grandpotentialKSpace2sublat(U, nUp, nDown,\
                    nUpOld, nDownOld, invTemp, Nk, Ny, eUp, eDown, mu, abs_t0)

                    if energyTmp > energy:
                        nUp[i], nDown[i] = nUpTmp, nDownTmp
                    else:
                        energy = energyTmp

            elif dyn == 'kick':
                for attempt in range(2 * nOrb * Ny):
                    i = int( np.random.random() * 2 * nOrb * Ny )
                    nUpTmp, nDownTmp = nUp[i], nDown[i]
                    nUp[i], nDown[i] = singleExcitation(nUp, nDown, i)
                    for kCount, k in enumerate(ks):
                        # Define the MF Hamiltonian for this iteration and k-point
                        K = HribbonKSpace2sublat(k, nOrb, Ny, E0, E1, E2, E3, E4)
                        Hup = K + U * np.eye(2 * nOrb * Ny) * nDown
                        Hdown = K + U * np.eye(2 * nOrb * Ny) * nUp

                        # Diagonalize
                        eUp[kCount, :], wUp[kCount, :, :] = la.eigh(Hup)
                        eDown[kCount, :], wDown[kCount, :, :] = la.eigh(Hdown)

                    # Save the previous fields to compare to test convergence
                    nUpOld = nUp.copy()
                    nDownOld = nDown.copy()

                    # Compute the chemical potential implicitly
                    mu = bisect(rootToChem2sublat, -50, 50,\
                     args = (eUp, eDown, beta, Nk, Ny, nHole) )

                    # Update fields
                    nUp, nDown = update2sublat(nUp, nDown, Nk, Ny,\
                     wUp, wDown, eUp, eDown, mu, beta, nOrb)

                    # Grandpotential per site
                    energyTmp = grandpotentialKSpace2sublat(U, nUp, nDown,\
                    nUpOld, nDownOld, invTemp, Nk, Ny, eUp, eDown, mu, abs_t0)

                    if energyTmp > energy:
                        nUp[i], nDown[i] = nUpTmp, nDownTmp
                    else:
                        energy = energyTmp
            elif dyn == 'mixed':
                for attempt in range(2 * nOrb * Ny):
                    i = int( np.random.random() * 2 * nOrb * Ny )
                    nUpTmp, nDownTmp = nUp[i], nDown[i]
                    nUp[i], nDown[i] = updateLocal2sublat(i, nUp, nDown, Nk, Ny,\
                    wUp, wDown, eUp, eDown, mu, beta, nOrb)
                    nUp[i], nDown[i] = singleExcitation(nUp, nDown, i)
                    for kCount, k in enumerate(ks):
                        # Define the MF Hamiltonian for this iteration and k-point
                        K = HribbonKSpace2sublat(k, nOrb, Ny, E0, E1, E2, E3, E4)
                        Hup = K + U * np.eye(2 * nOrb * Ny) * nDown
                        Hdown = K + U * np.eye(2 * nOrb * Ny) * nUp

                        # Diagonalize
                        eUp[kCount, :], wUp[kCount, :, :] = la.eigh(Hup)
                        eDown[kCount, :], wDown[kCount, :, :] = la.eigh(Hdown)

                    # Save the previous fields to compare to test convergence
                    nUpOld = nUp.copy()
                    nDownOld = nDown.copy()

                    # Compute the chemical potential implicitly
                    mu = bisect(rootToChem2sublat, -50, 50,\
                     args = (eUp, eDown, beta, Nk, Ny, nHole) )

                    # Update fields
                    nUp, nDown = update2sublat(nUp, nDown, Nk, Ny,\
                     wUp, wDown, eUp, eDown, mu, beta, nOrb)

                    # Grandpotential per site
                    energyTmp = grandpotentialKSpace2sublat(U, nUp, nDown,\
                    nUpOld, nDownOld, invTemp, Nk, Ny, eUp, eDown, mu, abs_t0)

                    if energyTmp > energy:
                        nUp[i], nDown[i] = nUpTmp, nDownTmp
                    else:
                        energy = energyTmp
        if dyn == 'wait':
            if ( it + 1 + itWait ) % singleExcitationFreq == 0 :
                for attempt in range(nOrb * Ny * 2):
                    i = int( np.random.random() * nOrb * Ny * 2 )
                    nUpTmp, nDownTmp = nUp[i], nDown[i]
                    nUp[i], nDown[i] = singleExcitation(nUp, nDown, i)
                    for kCount, k in enumerate(ks):
                        # Define the MF Hamiltonian for this iteration and k-point
                        K = HribbonKSpace2sublat(k, nOrb, Ny, E0, E1, E2, E3, E4)
                        Hup = K + U * np.eye(2 * nOrb * Ny) * nDown
                        Hdown = K + U * np.eye(2 * nOrb * Ny) * nUp

                        # Diagonalize
                        eUp[kCount, :], wUp[kCount, :, :] = la.eigh(Hup)
                        eDown[kCount, :], wDown[kCount, :, :] = la.eigh(Hdown)

                    # Save the previous fields to compare to test convergence
                    nUpOld = nUp.copy()
                    nDownOld = nDown.copy()

                    # Compute the chemical potential implicitly
                    mu = bisect(rootToChem2sublat, -50, 50,\
                     args = (eUp, eDown, beta, Nk, Ny, nHole) )

                    # Update fields
                    nUp, nDown = update2sublat(nUp, nDown, Nk, Ny,\
                     wUp, wDown, eUp, eDown, mu, beta, nOrb)

                    # Grandpotential per site
                    energyTmp = grandpotentialKSpace2sublat(U, nUp, nDown,\
                    nUpOld, nDownOld, invTemp, Nk, Ny, eUp, eDown, mu, abs_t0)

                    if energyTmp > energy:
                        nUp[i], nDown[i] = nUpTmp, nDownTmp
                    else:
                        energy = energyTmp

        energies[it] = energy
        if invTemp == 'infty':
            if energy < bestGrandpotential and beta >= betaThreshold:
                bestGrandpotential = energy
                nUpBest, nDownBest = nUp, nDown
                eUpBest, eDownBest = eUp, eDown
                wUpBest, wDownBest = wUp, wDown
        else:
            if energy < bestGrandpotential and beta == invTemp:
                bestGrandpotential = energy
                nUpBest, nDownBest = nUp, nDown
                eUpBest, eDownBest = eUp, eDown
                wUpBest, wDownBest = wUp, wDown

        # Move to the next iteration
        it += 1

    # Save the last iteration
    lastIt = it
    return nUpBest, nDownBest, energies, bestGrandpotential, itSwitch,\
     lastIt, mu, abs_t0, eUpBest, eDownBest,\
     np.absolute(wUpBest.flatten('C'))**2, np.absolute(wDownBest.flatten('C'))**2
