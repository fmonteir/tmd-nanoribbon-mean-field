##
##  lib_tmd_model.py
##
##
##  Created by Francisco Brito on 16/11/2018.
##
##  This library defines the parameters of a 3-band model of a transition metal
##  dichalcogenide (Liu et al., Phys Rev B 88, 085433 (2013)).
##  The Hamiltonian is defined in real space, and for a nanoribbon with single
##  and doubled unit cells along the longitudinal, periodic direction.
##

import numpy as np

def setParams(tmd):
    if tmd == 'MoS2' :
        abs_t0 = 0.184
        e1 = 1.046 / abs_t0
        e2 = 2.104 / abs_t0
        t0 = - 1
        t1 = 0.401 / abs_t0
        t2 = 0.507 / abs_t0
        t11 = 0.218 / abs_t0
        t12 = 0.338 / abs_t0
        t22 = 0.057 / abs_t0

    if tmd == 'WS2' :
        abs_t0 = 0.206
        e1 = 1.130 / abs_t0
        e2 = 2.275 / abs_t0
        t0 = - 1
        t1 = 0.567 / abs_t0
        t2 = 0.536 / abs_t0
        t11 = 0.286 / abs_t0
        t12 = 0.384 / abs_t0
        t22 = -0.061 / abs_t0

    if tmd == 'MoSe2' :
        abs_t0 = 0.188
        e1 = 0.919 / abs_t0
        e2 = 2.065 / abs_t0
        t0 = - 1
        t1 = 0.317 / abs_t0
        t2 = 0.456 / abs_t0
        t11 = 0.211 / abs_t0
        t12 = 0.290 / abs_t0
        t22 = 0.130 / abs_t0

    if tmd == 'WSe2' :
        abs_t0 = 0.207
        e1 = 0.943 / abs_t0
        e2 = 2.179 / abs_t0
        t0 = - 1
        t1 = 0.457 / abs_t0
        t2 = 0.486 / abs_t0
        t11 = 0.263 / abs_t0
        t12 = 0.329 / abs_t0
        t22 = 0.034 / abs_t0

    if tmd == 'MoTe2' :
        abs_t0 = 0.169
        e1 = 0.605 / abs_t0
        e2 = 1.972 / abs_t0
        t0 = - 1
        t1 = 0.228 / abs_t0
        t2 = 0.390 / abs_t0
        t11 = 0.207 / abs_t0
        t12 = 0.239 / abs_t0
        t22 = 0.252 / abs_t0

    if tmd == 'WTe2' :
        abs_t0 = 0.175
        e1 = 0.606 / abs_t0
        e2 = 2.102 / abs_t0
        t0 = - 1
        t1 = 0.342 / abs_t0
        t2 = 0.410 / abs_t0
        t11 = 0.233 / abs_t0
        t12 = 0.270 / abs_t0
        t22 = 0.190 / abs_t0

    E0 = np.array([[e1, 0, 0],
                   [0, e2, 0],
                   [0, 0, e2]])

    E1 = np.array([[t0, t1, t2],
                   [-t1, t11, t12],
                   [t2, -t12, t22]])

    E2 = np.array([

                   [t0,\
                   0.5 * t1 - np.sqrt(3) / 2 * t2,\
                   - np.sqrt(3) / 2 * t1 - 0.5 * t2],

                   [-0.5 * t1 - np.sqrt(3) / 2 * t2,\
                    0.25 * ( t11 + 3 * t22 ),\
                     np.sqrt(3) / 4 * ( t22 - t11 ) - t12],

                   [np.sqrt(3) / 2 * t1 - 0.5 * t2,\
                    np.sqrt(3) / 4 * ( t22 - t11 ) + t12,\
                     ( 3 * t11 + t22) / 4 ]

                     ])

    E3 = np.array([

                  [t0,\
                  - 0.5 * t1 + np.sqrt(3) / 2 * t2,\
                  -np.sqrt(3) / 2 * t1 - 0.5 * t2],

                  [0.5 * t1 + np.sqrt(3) / 2 * t2,\
                  0.25 * ( t11 + 3 * t22 ),\
                  -np.sqrt(3) / 4 * ( t22 - t11 ) + t12],

                  [np.sqrt(3) / 2 * t1 - 0.5 * t2,\
                  -np.sqrt(3) / 4 * ( t22 - t11 ) - t12,\
                  ( 3 * t11 + t22) / 4 ]

                  ])


    E4 = np.array([[t0, -t1, t2],
                   [t1, t11, -t12],
                   [t2, t12, t22]])

    E5 = np.array([

                  [t0,\
                   - 0.5 * t1 - np.sqrt(3) / 2 * t2,\
                    np.sqrt(3) / 2 * t1 - 0.5 * t2],

                  [0.5 * t1 - np.sqrt(3) / 2 * t2,\
                   0.25 * ( t11 + 3 * t22 ),\
                    np.sqrt(3) / 4 * ( t22 - t11 ) + t12],

                  [-np.sqrt(3) / 2 * t1 - 0.5 * t2,\
                   np.sqrt(3) / 4 * ( t22 - t11 ) - t12,\
                    ( 3 * t11 + t22) / 4 ]

                    ])

    E6 = np.array([

                  [t0,\
                  0.5 * t1 + np.sqrt(3) / 2 * t2,\
                  np.sqrt(3) / 2 * t1 - 0.5 * t2],

                  [-0.5 * t1 + np.sqrt(3) / 2 * t2,\
                  0.25 * ( t11 + 3 * t22 ),\
                  -np.sqrt(3) / 4 * ( t22 - t11 ) - t12],

                  [-np.sqrt(3) / 2 * t1 - 0.5 * t2,\
                  -np.sqrt(3) / 4 * ( t22 - t11 ) + t12,\
                   ( 3 * t11 + t22) / 4 ]])

    return abs_t0, e1, e2, t0, t1, t2, t11, t12, t22, E0, E1, E2, E3, E4, E5, E6

def siteLabel(x, y, Nx, nOrb):
    return nOrb * ( Nx * y + x )

def HribbonRealSpace(nOrb, Nx, Ny, E0, E1, E2, E3, E4, E5, E6):
    K = np.zeros( ( nOrb * Nx * Ny, nOrb * Nx * Ny ) )
    for x in range(Nx):
        for y in range(Ny):
            # Diagonal term
            K[siteLabel(x, y, Nx, nOrb):siteLabel(x, y, Nx, nOrb) + nOrb,\
              siteLabel(x, y, Nx, nOrb):siteLabel(x, y, Nx, nOrb) + nOrb]\
            = E0

            # E1
            K[siteLabel(x, y, Nx, nOrb):siteLabel(x, y, Nx, nOrb) + nOrb,\
              siteLabel((x + 1) % Nx , y, Nx, nOrb):\
              siteLabel((x + 1) % Nx , y, Nx, nOrb) + nOrb]\
            = E1

            # E4
            K[siteLabel((x + 1) % Nx , y, Nx, nOrb):\
            siteLabel((x + 1) % Nx , y, Nx, nOrb) + nOrb,\
            siteLabel(x, y, Nx, nOrb):siteLabel(x, y, Nx, nOrb) + nOrb] = E4

            if y == 0:
                K[siteLabel(x, 0, Nx, nOrb):siteLabel(x, 0, Nx, nOrb) + nOrb,\
                  siteLabel(x, 1, Nx, nOrb):siteLabel(x, 1, Nx, nOrb) + nOrb]\
                = E6

                K[siteLabel(x, 1, Nx, nOrb):siteLabel(x, 1, Nx, nOrb) + nOrb,\
                  siteLabel(x, 0, Nx, nOrb):siteLabel(x, 0, Nx, nOrb) + nOrb]\
                = E3

                if x == 0:

                    K[siteLabel(x, 0, Nx, nOrb)\
                    :siteLabel(x, 0, Nx, nOrb) + nOrb,\
                    siteLabel(Nx - 1, 1, Nx, nOrb)\
                    :siteLabel(Nx - 1, 1, Nx, nOrb) + nOrb]\
                    = E5

                    K[siteLabel(Nx - 1, 1, Nx, nOrb)\
                    :siteLabel(Nx - 1, 1, Nx, nOrb) + nOrb,\
                    siteLabel(x, 0, Nx, nOrb)\
                    :siteLabel(x, 0, Nx, nOrb) + nOrb]\
                    = E2
                else:
                    K[siteLabel(x, 0, Nx, nOrb)\
                    :siteLabel(x, 0, Nx, nOrb) + nOrb,\
                    siteLabel(x - 1, 1, Nx, nOrb)\
                    :siteLabel(x - 1, 1, Nx, nOrb) + nOrb]\
                    = E5

                    K[siteLabel(x - 1, 1, Nx, nOrb)\
                    :siteLabel(x - 1, 1, Nx, nOrb) + nOrb,\
                    siteLabel(x, 0, Nx, nOrb)\
                    :siteLabel(x, 0, Nx, nOrb) + nOrb]\
                    = E2
            else:
                if y == Ny - 1:

                    K[siteLabel(x, Ny - 1, Nx, nOrb)\
                    :siteLabel(x, Ny - 1, Nx, nOrb) + nOrb,\
                    siteLabel( ( x + 1 ) % Nx, Ny - 2, Nx, nOrb):\
                    siteLabel( ( x + 1 ) % Nx, Ny - 2, Nx, nOrb) + nOrb]\
                    = E2

                    K[siteLabel( ( x + 1 ) % Nx, Ny - 2, Nx, nOrb):\
                    siteLabel( ( x + 1 ) % Nx, Ny - 2, Nx, nOrb) + nOrb,\
                    siteLabel(x, Ny - 1, Nx, nOrb)\
                    :siteLabel(x, Ny - 1, Nx, nOrb) + nOrb]\
                    = E5

                    K[siteLabel(x, Ny - 1, Nx, nOrb)\
                    :siteLabel(x, Ny - 1, Nx, nOrb) + nOrb,\
                    siteLabel(x, Ny - 2, Nx, nOrb)\
                    :siteLabel(x, Ny - 2, Nx, nOrb) + nOrb]\
                    = E3

                    K[siteLabel(x, Ny - 2, Nx, nOrb)\
                    :siteLabel(x, Ny - 2, Nx, nOrb) + nOrb,\
                    siteLabel(x, Ny - 1, Nx, nOrb)\
                    :siteLabel(x, Ny - 1, Nx, nOrb) + nOrb]\
                    = E6

                else:
                    K[siteLabel(x, y, Nx, nOrb)\
                    :siteLabel(x, y, Nx, nOrb) + nOrb,\
                    siteLabel((x + 1) % Nx , y - 1, Nx, nOrb):\
                    siteLabel((x + 1) % Nx , y - 1, Nx, nOrb) + nOrb]\
                    = E2

                    K[siteLabel((x + 1) % Nx , y - 1, Nx, nOrb):\
                    siteLabel((x + 1) % Nx , y - 1, Nx, nOrb) + nOrb,\
                    siteLabel(x, y, Nx, nOrb)\
                    :siteLabel(x, y, Nx, nOrb) + nOrb]\
                    = E5

                    K[siteLabel(x, y, Nx, nOrb)\
                    :siteLabel(x, y, Nx, nOrb) + nOrb,\
                    siteLabel(x, y - 1, Nx, nOrb)\
                    :siteLabel(x, y - 1, Nx, nOrb) + nOrb]\
                    = E3

                    K[siteLabel(x, y - 1, Nx, nOrb)\
                    :siteLabel(x, y - 1, Nx, nOrb) + nOrb,\
                    siteLabel(x, y, Nx, nOrb)\
                    :siteLabel(x, y, Nx, nOrb) + nOrb]\
                    = E6

                    if x == 0:

                        K[siteLabel(x, y, Nx, nOrb)\
                        :siteLabel(x, y, Nx, nOrb) + nOrb,\
                        siteLabel(Nx - 1, y + 1, Nx, nOrb):\
                        siteLabel(Nx - 1, y + 1, Nx, nOrb) + nOrb]\
                        = E5

                        K[siteLabel(Nx - 1, y + 1, Nx, nOrb):\
                        siteLabel(Nx - 1, y + 1, Nx, nOrb) + nOrb,\
                        siteLabel(x, y, Nx, nOrb)\
                        :siteLabel(x, y, Nx, nOrb) + nOrb]\
                        = E2

                    else:

                        K[siteLabel(x, y, Nx, nOrb)\
                        :siteLabel(x, y, Nx, nOrb) + nOrb,\
                        siteLabel(x - 1, y + 1, Nx, nOrb):\
                        siteLabel(x - 1, y + 1, Nx, nOrb) + nOrb]\
                        = E5

                        K[siteLabel(x - 1, y + 1, Nx, nOrb):\
                        siteLabel(x - 1, y + 1, Nx, nOrb) + nOrb,\
                        siteLabel(x, y, Nx, nOrb)\
                        :siteLabel(x, y, Nx, nOrb) + nOrb]\
                        = E2

    return K

def HribbonKSpace(k, nOrb, Ny, e1, e2, t0, t1, t2, t11, t12, t22):

    Hrib = np.zeros((nOrb * Ny, nOrb * Ny), dtype=np.complex64)

    h1 = np.array([

                [e1 + 2 * t0 * np.cos(k),
                2.j * np.sin(k) * t1,
                2 * t2 * np.cos(k)],

                [-2.j * np.sin(k) * t1,
                e2 + 2 * t11 * np.cos(k),
                2.j * np.sin(k) * t12],

                [2 * t2 * np.cos(k),
                -2.j * np.sin(k) * t12,
                e2 + 2 * t22 * np.cos(k)]

               ], dtype=np.complex64)

    h2 = np.array([

          [ 2 * t0 * np.cos(k/2) ,
           1.j * np.sin(k/2) * ( t1 - np.sqrt(3) * t2 ) ,
           -1. * np.cos(k/2) * ( np.sqrt(3) * t1 + t2 )] ,

          [ -1.j * np.sin(k/2) * ( t1 + np.sqrt(3) * t2 ),
           0.5 * np.cos(k/2) * ( t11 + 3 * t22 ),
           1.j * np.sin(k/2) * ( np.sqrt(3) / 2 * ( t22 - t11 ) - 2 * t12 ) ],

          [ np.cos(k/2) * ( np.sqrt(3) * t1 - t2 ),
           1.j * np.sin(k/2) * ( np.sqrt(3)/2 * ( t22 - t11 ) + 2 * t12 ),
           0.5 * np.cos(k/2) * ( 3 * t11 + t22 ) ]

          ], dtype=np.complex64)

    for y in range(1, Ny - 1):

        Hrib[nOrb * y\
        :nOrb * ( y + 1 ), \
        nOrb * y\
        :nOrb * ( y + 1 )] = h1

        Hrib[nOrb * ( y - 1 )\
        :nOrb * y, nOrb * y\
        :nOrb * ( y + 1 )] = (h2.conj()).T

        Hrib[nOrb * ( y + 1 )\
        :nOrb * ( y + 2 ), \
        nOrb * y\
        :nOrb * ( y + 1 )] = h2

    Hrib[0:nOrb, 0:nOrb] = h1

    Hrib[nOrb * ( Ny - 1 )\
    :nOrb * Ny, \
    nOrb * ( Ny - 1 )\
    :nOrb * Ny] = h1
    Hrib[nOrb * ( Ny - 2 )\
    :nOrb * ( Ny - 1 ),\
    nOrb * ( Ny - 1 )\
    :nOrb * Ny] = (h2.conj()).T

    Hrib[nOrb:2 * nOrb, :nOrb] = h2

    return Hrib

def HribbonKSpace2sublat(k, nOrb, Ny, E0, E1, E2, E3, E4):

    Hrib = np.zeros((2 * nOrb * Ny, 2 * nOrb * Ny), dtype=np.complex64)

    h1 = E0

    h2 = np.exp(1.j * k / 2) * E3

    h3 = np.exp(1.j * k) * E4 + np.exp(-1.j * k) * E1

    h4 = np.exp(-1.j * k / 2) * E2


    for y in range(1, Ny - 1):
        Hrib[nOrb * y:nOrb * ( y + 1 ), nOrb * y:nOrb * ( y + 1 ) ]\
         = h1

        Hrib[nOrb * ( y - 1 ):nOrb * y, nOrb * y:nOrb * ( y + 1 ) ]\
         = h2

        Hrib[nOrb * ( y + 1 ):nOrb * ( y + 2 ), nOrb * y:nOrb * ( y + 1 ) ]\
         = (h2.conj()).T

        Hrib[nOrb * ( y + Ny ):nOrb * ( y + Ny + 1 ),\
         nOrb * ( y + Ny ):nOrb * ( y + Ny + 1 ) ]\
         = h1

        Hrib[nOrb * ( y + Ny - 1):nOrb * ( y + Ny ),\
         nOrb * ( y + Ny ):nOrb * ( y + Ny + 1 ) ]\
         = h2

        Hrib[nOrb * ( y + Ny + 1 ):nOrb * ( y + 2 + Ny ),\
         nOrb * ( y + Ny ):nOrb * ( y + Ny + 1 ) ]\
         = (h2.conj()).T

        Hrib[nOrb * ( y + Ny ):nOrb * ( y + 1 + Ny ),\
         nOrb * y:nOrb * ( y + 1 ) ]\
         = h3

        Hrib[nOrb * ( y - 1 + Ny ):nOrb * ( y + Ny ),\
         nOrb * y:nOrb * ( y + 1 ) ]\
         = h4

        Hrib[nOrb * ( y + 1 + Ny ):nOrb * ( y + 2 + Ny ),\
         nOrb * y:nOrb * ( y + 1 ) ]\
         = (h4.conj()).T

        Hrib[nOrb * y:nOrb * ( y + 1 ),\
         nOrb * ( y + Ny ):nOrb * ( y + 1 + Ny )]\
         = h3

        Hrib[nOrb * y:nOrb * ( y + 1 ),\
         nOrb * ( y - 1 + Ny ):nOrb * ( y + Ny )]\
         = (h4.conj()).T

        Hrib[nOrb * y:nOrb * ( y + 1 ),\
         nOrb * ( y + 1 + Ny ):nOrb * ( y + 2 + Ny )]\
         = h4


    Hrib[:nOrb, :nOrb]\
     = h1

    Hrib[nOrb * ( Ny - 1 ):nOrb * Ny, nOrb * ( Ny - 1 ):nOrb * Ny ]\
     = h1

    Hrib[nOrb * Ny:nOrb * ( Ny + 1 ),\
     nOrb * Ny:nOrb * ( Ny + 1 ) ]\
    = h1

    Hrib[2 * nOrb * Ny - nOrb:2 * nOrb * Ny,\
     2 * Ny * nOrb - nOrb:2 * nOrb * Ny]\
     = h1

    Hrib[nOrb:nOrb *  2 , :nOrb ]\
     = (h2.conj()).T

    Hrib[nOrb * ( Ny - 2 ):nOrb * ( Ny - 1), nOrb * ( Ny - 1 ):nOrb * Ny ]\
     = h2

    Hrib[nOrb * ( Ny + 1 ):nOrb * ( 2 + Ny ),\
     nOrb * ( Ny ):nOrb * ( Ny + 1 ) ]\
     = (h2.conj()).T

    Hrib[nOrb * ( Ny + Ny - 2):nOrb * ( Ny - 1 + Ny ),\
     nOrb * ( Ny - 1 + Ny ):nOrb * ( Ny + Ny ) ]\
     = h2


    Hrib[nOrb * Ny:nOrb * ( 1 + Ny ),\
     :nOrb]\
     = h3

    Hrib[nOrb * ( 2 * Ny - 1 ):nOrb * 2 * Ny,\
     nOrb * ( Ny - 1):nOrb * Ny ]\
     = h3

    Hrib[nOrb * ( 2 * Ny - 2 ):nOrb * ( 2 * Ny - 1 ),\
     nOrb * ( Ny - 1):nOrb * Ny ]\
     = h4

    Hrib[nOrb * ( 1 + Ny ):nOrb * ( 2 + Ny ),\
     :nOrb]\
     = (h4.conj()).T

    Hrib[:nOrb,\
    nOrb * Ny:nOrb * ( 1 + Ny )]\
     = h3

    Hrib[nOrb * ( Ny - 1):nOrb * Ny,\
     nOrb * ( 2 * Ny - 1 ):nOrb * 2 * Ny]\
     = h3

    Hrib[nOrb * ( Ny - 1):nOrb * Ny,\
     nOrb * ( 2 * Ny - 2 ):nOrb * ( 2 * Ny - 1 )]\
     = (h4.conj()).T

    Hrib[:nOrb,\
     nOrb * ( 1 + Ny ):nOrb * ( 2 + Ny )]\
     = h4

    return Hrib
