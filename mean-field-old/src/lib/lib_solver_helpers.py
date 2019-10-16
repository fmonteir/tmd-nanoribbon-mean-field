##
##  lib_solver_helpers.py
##
##
##  Created by Francisco Brito on 16/11/2018.
##
##  This library defines useful functions to help in the solution of the
##  mean field intra-orbital Hubbard model for a TMD nanoribbon
##  self-consistently.
##
##

import numpy as np
import warnings
from scipy.optimize import bisect

def fermi(e, mu, beta):
    '''
    Fermi distribution for a numpy array of energies, e.
    If beta is high enough, it returns the step function.
    '''
    warnings.filterwarnings('error')
    if beta == 'infty':
        return (e < mu).astype(int)
    try:
        return 1 / ( 1 + np.exp( beta * ( e - mu ) ) )
    except:
        return (e < mu).astype(int)

def rootToChem(chemPot, eUp, eDown, beta, Nx, Ny, nHole):
    '''
    The root of this function is computed to obtain the
    chemical potential implicitly.
    '''
    return ( np.sum(fermi(eUp, chemPot, beta )) \
    +  np.sum(fermi(eDown, chemPot, beta )) ) / Nx / Ny - (2 - nHole)

def rootToChem2sublat(chemPot, eUp, eDown, beta, Nx, Ny, nHole):
    '''
    The root of this function is computed to obtain the
    chemical potential implicitly.
    '''
    return ( np.sum(fermi(eUp, chemPot, beta )) \
    +  np.sum(fermi(eDown, chemPot, beta )) ) / Nx / Ny / 2 - (2 - nHole)

def anneal(invTemp, betaStart, betaSpeed, beta, betaThreshold, it, osc):
    '''
    Computes the inverse temperature for the annealing phase.
    '''
    inftyCutOff = 1e10  # very large number, doesn't matter (see fermi function)
                        # at this point, fermi is basically a step function
    itPeriod = 5
    if invTemp == 'infty':
        if beta < betaThreshold:
            if osc == True:
                beta = ( betaStart - 1 + betaSpeed ** it )\
                * ( 1.85 + np.cos( 2 * np.pi * it / itPeriod ) ) / 2.85
                print("Inverse temperature: ", beta)
                if beta > betaThreshold:
                    itSwitch = it
                    beta = betaThreshold
                    print("\nFinished annealing.\n")
            else:
                beta = ( betaStart - 1 + betaSpeed ** it )
                print("Inverse temperature: ", beta)
            if beta > betaThreshold:
                itSwitch = it
                beta = betaThreshold
                print("\nFinished annealing.\n")
        else:
            beta = inftyCutOff
        return beta
    if beta < invTemp:
        if osc == True:
            beta = ( betaStart - 1 + betaSpeed ** it )\
            * ( 1.85 + np.cos( 2 * np.pi * it / itPeriod ) ) / 2.85
            print("Inverse temperature: ", beta)
            if beta > betaThreshold:
                itSwitch = it
                beta = betaThreshold
                print("\nFinished annealing.\n")
        else:
            beta = ( betaStart - 1 + betaSpeed ** it )
            print("Inverse temperature: ", beta)
        if beta > invTemp:
            itSwitch = it
            beta = invTemp
            print("\nFinished annealing.\n")
    else:
        beta = invTemp
    return beta

def noAnneal(invTemp, beta):
    '''
    Computes the inverse temperature for the annealing phase.
    '''
    inftyCutOff = 1e10 # very large number, doesn't matter (see fermi function)
    if invTemp == 'infty':
        beta = inftyCutOff
    else:
        beta = invTemp
    return beta

def hamiltonian(nUp, nDown, K, U, N):
    '''
    Computes the new mean field Hamiltonian at each step.
    '''
    return - nUp * nDown, K + U * np.eye(N) * nDown,\
K + U * np.eye(N) * nUp

def loopCondition(it, itMax, deltaUp, deltaDown, delta,\
beta, invTemp, betaThreshold):
    '''
    Verifies whether the main loop should stop.
    '''
    if it < itMax and (deltaUp > delta or deltaDown > delta):
        return True
    else:
        if (invTemp == 'infty' and beta >= betaThreshold)\
        or (invTemp != 'infty' and beta >= invTemp ):
            return False
        else:
            return True


def damp(it, dampFreq, nUp, nDown, nUpOld, nDownOld, lbda):
    '''
    Introduces damping by varying the weight given to
    the newly calculated fields.
    '''
    if it % dampFreq == 0:
        nUp = ( 1 / 2 + lbda * it ) * nUp\
        + ( 1 / 2 - lbda * it) * nUpOld
        nDown = ( 1 / 2 + lbda * it ) * nDown\
        + ( 1 / 2 - lbda * it) * nDownOld
    return nUp, nDown

def update(nUp, nDown, N, wUp, wDown, eUp, eDown, mu, beta):
    for i in range(N):
        nUp[i] = np.sum( np.absolute(wUp[i, :])**2 * fermi(eUp, mu , beta) )
        nDown[i] = np.sum( np.absolute(wDown[i, :])**2 * fermi(eDown, mu, beta) )
    return nUp, nDown

def updateKSpace(nUp, nDown, Nk, Ny, wUp, wDown, eUp, eDown, mu, beta, nOrb):
    for i in range(nOrb * Ny):
        nUp[i] = ( np.absolute(wUp[:, i, :])**2\
        * fermi(eUp.real, mu , beta) ).sum() / Nk
        nDown[i] = ( np.absolute(wDown[:, i, :])**2\
        * fermi(eDown.real, mu , beta) ).sum() / Nk
    print(nUp[0])
    return nUp, nDown

def update2sublat(nUp, nDown, Nk, Ny, wUp, wDown, eUp, eDown, mu, beta, nOrb):
    for i in range(2 * nOrb * Ny):
        nUp[i] = ( np.absolute(wUp[:, i, :])**2\
        * fermi(eUp.real, mu , beta) ).sum() / Nk
        nDown[i] = ( np.absolute(wDown[:, i, :])**2\
        * fermi(eDown.real, mu , beta) ).sum() / Nk
    return nUp, nDown

def updateLocal(i, nUp, nDown, N, wUp, wDown, eUp, eDown, mu, beta):
    nUp[i] = np.sum( np.absolute(wUp[i, :])**2 * fermi(eUp, mu , beta) ) / N
    nDown[i] = np.sum( np.absolute(wDown[i, :])**2 * fermi(eDown, mu, beta) ) / N
    return nUp[i], nDown[i]

def updateLocalKSpace(i, nUp, nDown,\
Nk, Ny, wUp, wDown, eUp, eDown, mu, beta, nOrb):
    nUp[i] = np.sum( np.absolute(wUp[:, i, :])**2\
    * fermi(eUp, mu , beta) ) / Nk
    nDown[i] = np.sum( np.absolute(wDown[:, i, :])**2\
    * fermi(eDown, mu, beta) ) / Nk
    return nUp[i], nDown[i]

def updateLocal2sublat(i, nUp, nDown,\
Nk, Ny, wUp, wDown, eUp, eDown, mu, beta, nOrb):
    nUp[i] = ( np.absolute(wUp[:, i, :])**2\
    * fermi(eUp, mu , beta) ).sum() / ( Nk / 2 )
    nDown[i] = ( np.absolute(wDown[:, i, :])**2\
    * fermi(eDown, mu , beta) ).sum() / ( Nk / 2 )/ Nk
    return nUp[i], nDown[i]

def singleExcitation(nUp, nDown, i):
    frac = 0.1
    nUp[i] += 0.1 / 2 * np.random.random() - 0.1
    nDown[i] += 0.1 / 2 * np.random.random() - 0.1
    return nUp[i], nDown[i]

def grandpotential\
(U, nUp, nDown, nUpOld, nDownOld, nOrb, Nx, Ny,\
mu, invTemp, eUp, eDown, abs_t0):
    '''
    Computes the grandpotential functional to be minimized.
    '''


    if invTemp != 'infty':

        dH = U  * ( np.dot(nUp, nDown) - np.dot(nUp, nDownOld) - np.dot(nDown, nUpOld) )

        OmegaMF = - 1 / invTemp * ( ( np.log( 1 + np.exp( - invTemp * ( eUp - mu ) ) ) ).sum() + \
        (np.log( 1 + np.exp( - invTemp * ( eDown - mu ) ) ) ).sum() )

        return ( dH + OmegaMF ) * abs_t0 / ( nOrb * Nx * Ny ) + 2 * mu / 3 * abs_t0
    else:

        return ( U  * np.dot(nUp, nDown)) * abs_t0 / ( nOrb * Nx * Ny ) + 2 * mu / 3 * abs_t0

def grandpotentialKSpace(U, nUp, nDown, nUpOld, nDownOld, invTemp,\
nOrb, Nk, Ny, eUp, eDown, wUp, wDown, mu, abs_t0):
    '''
    Computes the grandpotential functional to be minimized.
    '''
    for i in range(nOrb * Ny):
        nUp[i] = ( np.absolute(wUp[:, i, :])**2\
        * fermi(eUp.real, mu , invTemp) ).sum() / Nk
        nDown[i] = ( np.absolute(wDown[:, i, :])**2\
        * fermi(eDown.real, mu , invTemp) ).sum() / Nk
    if invTemp != 'infty':
        return (\
        U * Nk * ( np.dot(nUp, nDown)\
        - np.dot(nUp, nDownOld) - np.dot(nDown, nUpOld) )\

        - 1 / invTemp * ( ( np.log( 1 + np.exp( - invTemp * ( eUp - mu ) ) ) ).sum()\
        + (np.log( 1 + np.exp( - invTemp * ( eDown - mu ) ) ) ).sum() )\

        ) * abs_t0 / Nk / Ny \

#        + mu * abs_t0 * (nUp.sum() + nDown.sum() ) / ( nOrb * Ny )
    else:
        return  U * ( np.dot(nUp, nDown)\
         # - np.dot(nUp, nDownOld) - np.dot(nDown, nUpOld) \
         ) * abs_t0 / Ny

def grandpotentialKSpace2sublat(U, nUp, nDown, nUpOld, nDownOld, invTemp,\
Nx, Ny, eUp, eDown, mu, abs_t0):
    '''
    Computes the grandpotential functional to be minimized.
    '''
    if invTemp != 'infty':
        return (  U * ( np.dot(nUp, nDown) - np.dot(nUp, nDownOld) - np.dot(nDown, nUpOld) )\
                - 1 / invTemp / Nx * ( ( np.log( 1 + np.exp( - invTemp * ( eUp - mu ) ) ) ).sum() + \
                                   (np.log( 1 + np.exp( - invTemp * ( eDown - mu ) ) ) ).sum() )\
               ) * abs_t0 / ( 3 * Ny * 2) \
               + mu * abs_t0 * (nUp.sum() + nDown.sum() ) / ( 3 * Ny * 2)
    else:
        return  U * ( np.dot(nUp, nDown)\
         - np.dot(nUp, nDownOld) - np.dot(nDown, nUpOld) ) * abs_t0 / ( 3 * Ny * 2)
