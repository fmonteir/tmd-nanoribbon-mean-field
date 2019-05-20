#include <vector>
#include <complex>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "aux.hpp"
#include "model.hpp"
#ifndef NTH
#define NTH 2
#endif
#ifndef NK
#define NK 512
#endif
#ifndef NA
#define NA 4
#endif
#ifndef NX
#define NX 12
#endif
#ifndef NY
#define NY 5
#endif
#ifndef BETA_START
#define BETA_START 0.01
#endif
#ifndef BETA_SPEED
#define BETA_SPEED 1.1
#endif
#ifndef BETA_THRESHOLD
#define BETA_THRESHOLD 50
#endif
#ifndef DELTA
#define DELTA 0.000001
#endif
#ifndef DAMP_FREQ
#define DAMP_FREQ 1
#endif
#ifndef MAX_IT
#define MAX_IT 500
#endif
#ifndef NORB
#define NORB 3
#endif

int main(int argc, char **argv)
{
    if ( argc != 6) //  tmd, u, beta, mu, number of independent seeds
    {
        std::cout << "Not enough arguments given to simulation."
        << std::endl << std::endl;
        return -1;
    }

    int tmd = atoi(argv[1]);
    double u = atof(argv[2]);
    double beta = atof(argv[3]);
    double mu = atof(argv[4]);
    int noSeeds = atoi(argv[5]);

    unsigned seeds[50] =
    {1987923818, 412767627, 525529884, 1391258185, 468262125,
     1793170370, 2136503712, 1903040763, 1099057133, 466750251,
     738248922, 1400172130, 2014015198, 552281714, 1596777150,
     1922631459, 715888080, 574751449, 999922481, 1148627831,
     1084434028, 1751040488, 1216104707, 1675889077, 1785103677,
     718632744, 787572230, 1614514171, 646786096, 1802582656,
     664598139, 1317824688, 1206060368, 1105735043, 1897642531,
     975233512, 574589901, 487397491, 1503108649, 14377998,
     53507733, 689926744, 1120258236, 1137701721, 878774344,
     1204014339, 899882252, 1649945363, 885341924, 1188020622};

    unsigned bestSeed = seeds[0];

    //

    model_k_space solver(tmd, u, mu, beta);
    solver.TMDnanoribbon();
    int seed = seeds[0];
    solver.init_para(seed);

    unsigned it = 0;
    while ( solver.loop_condition(it) )
    {
        solver.anneal(it);
        solver.diagonalize();
        solver.fermi();
        solver.update();
        solver.damp(it);
        solver.tolerance_check();
        it++;
    }
    solver.compute_grand_potential();

    double bestVarGrandPot = solver.grand_potential();
    double bestElDens = solver.filling();
    Eigen::VectorXd best_nUp = solver.UpField();
    Eigen::VectorXd best_nDw = solver.DwField();
    Eigen::MatrixXd best_bandsUp = solver.MFbandsUp();
    Eigen::MatrixXd best_bandsDw = solver.MFbandsDw();

    //

    double VarGrandPotArr [noSeeds];
    double ElDensArr [noSeeds];
    Eigen::VectorXd nUpArr [noSeeds];
    Eigen::VectorXd nDwArr [noSeeds];
    Eigen::MatrixXd bandsUpArr [noSeeds];
    Eigen::MatrixXd bandsDwArr [noSeeds];

    #pragma omp parallel for num_threads(NTH)
    for (int indSols=0; indSols < noSeeds; indSols++)
    {
        model_k_space solverRd(tmd, u, mu, beta);
        solverRd.TMDnanoribbon();
        seed = seeds[indSols];
        solverRd.init_random(seed);
        unsigned it = 0;
        while ( solverRd.loop_condition(it) )
        {
            solverRd.anneal(it);
            solverRd.diagonalize();
            solverRd.fermi();
            solverRd.update();
            solverRd.damp(it);
            solverRd.tolerance_check();
            it++;
        }
        solverRd.compute_grand_potential();
        VarGrandPotArr[indSols] = solverRd.grand_potential();
        ElDensArr[indSols] = solverRd.filling();
        nUpArr[indSols] = solverRd.UpField();
        nDwArr[indSols] = solverRd.DwField();
        bandsUpArr[indSols] = solverRd.MFbandsUp();
        bandsDwArr[indSols] = solverRd.MFbandsDw();
    }

    for (int indSols=0; indSols < noSeeds; indSols++)
    {
        if (VarGrandPotArr[indSols] < bestVarGrandPot)
        {
            bestVarGrandPot = VarGrandPotArr[indSols];
            bestElDens = ElDensArr[indSols];
            best_nUp = nUpArr[indSols];
            best_nDw = nDwArr[indSols];
            best_bandsUp = bandsUpArr[indSols];
            best_bandsDw = bandsDwArr[indSols];
            bestSeed = seeds[indSols];
        }
    }

    //  SAVE OUTPUT
    save(tmd, u, beta, mu, noSeeds, bestSeed, bestElDens,
         bestVarGrandPot, best_nUp, best_nDw, best_bandsUp, best_bandsDw);
    return 0;
}
