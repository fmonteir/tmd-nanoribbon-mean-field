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
    if ( argc != 9) //  tmd, u, beta, muLow, muHigh, target filling, tol,
                    //  number of independent seeds
    {
        std::cout << "Not enough arguments given to simulation."
        << std::endl << std::endl;
        return -1;
    }

    int tmd = atoi(argv[1]);
    double u = atof(argv[2]);
    double beta = atof(argv[3]);
    double muLow = atof(argv[4]);
    double muHigh = atof(argv[5]);
    double fill_target = atof(argv[6]);
    double tol = atof(argv[7]);
    int noSeeds = atoi(argv[8]);

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
    double mu = 0.0;
    double bestVarGrandPot = 0.0;
    double bestElDensLow = 0.0;
    double bestElDensHigh = 0.0;
    int it_bissect = 0;
    int it_bissect_max = 50;
    double bissect_tol = 1E-6;
    double bestElDens = 0.0;
    int bestSeed = 1;
    Eigen::VectorXd best_nUp;
    Eigen::VectorXd best_nDw;
    Eigen::MatrixXd best_bandsUp;
    Eigen::MatrixXd best_bandsDw;

    //

    model_k_space solverLow(tmd, u, muLow, beta);
    solverLow.TMDnanoribbon();
    int seed = seeds[0];
    solverLow.init_para(seed);
    unsigned it = 0;
    while ( solverLow.loop_condition(it) )
    {
        solverLow.anneal(it);
        solverLow.diagonalize();
        solverLow.fermi();
        solverLow.update();
        solverLow.damp(it);
        solverLow.tolerance_check();
        it++;
    }
    solverLow.compute_grand_potential();
    bestVarGrandPot = solverLow.grand_potential();
    bestElDensLow = solverLow.filling();

    double VarGrandPotArr[noSeeds];
    double ElDensArr[noSeeds];

    #pragma omp parallel for num_threads(NTH)
    for (int indSols=0; indSols < noSeeds; indSols++)
    {
        model_k_space solverLowRd(tmd, u, muLow, beta);
        solverLowRd.TMDnanoribbon();
        seed = seeds[indSols];
        solverLowRd.init_random(seed);
        unsigned it = 0;
        while ( solverLowRd.loop_condition(it) )
        {
            solverLowRd.anneal(it);
            solverLowRd.diagonalize();
            solverLowRd.fermi();
            solverLowRd.update();
            solverLowRd.damp(it);
            solverLowRd.tolerance_check();
            it++;
        }
        solverLowRd.compute_grand_potential();
        VarGrandPotArr[indSols] = solverLowRd.grand_potential();
        ElDensArr[indSols] = solverLowRd.filling();
    }

    for (int indSols=0; indSols < noSeeds; indSols++)
    {
        if (VarGrandPotArr[indSols] < bestVarGrandPot)
        {
            bestVarGrandPot = VarGrandPotArr[indSols];
            bestElDensLow = ElDensArr[indSols];
        }
    }

    //

    //

    model_k_space solverHigh(tmd, u, muHigh, beta);
    solverHigh.TMDnanoribbon();
    seed = seeds[0];
    solverHigh.init_para(seed);
    it = 0;
    while ( solverHigh.loop_condition(it) )
    {
        solverHigh.anneal(it);
        solverHigh.diagonalize();
        solverHigh.fermi();
        solverHigh.update();
        solverHigh.damp(it);
        solverHigh.tolerance_check();
        it++;
    }
    solverHigh.compute_grand_potential();
    bestVarGrandPot = solverHigh.grand_potential();
    bestElDensHigh = solverHigh.filling();

    #pragma omp parallel for num_threads(NTH)
    for (int indSols=0; indSols < noSeeds; indSols++)
    {
        model_k_space solverHighRd(tmd, u, muHigh, beta);
        solverHighRd.TMDnanoribbon();
        seed = seeds[indSols];
        solverHighRd.init_random(seed);
        unsigned it = 0;
        while ( solverHighRd.loop_condition(it) )
        {
            solverHighRd.anneal(it);
            solverHighRd.diagonalize();
            solverHighRd.fermi();
            solverHighRd.update();
            solverHighRd.damp(it);
            solverHighRd.tolerance_check();
            it++;
        }
        solverHighRd.compute_grand_potential();
        VarGrandPotArr[indSols] = solverHighRd.grand_potential();
        ElDensArr[indSols] = solverHighRd.filling();
    }

    for (int indSols=0; indSols < noSeeds; indSols++)
    {
        if (VarGrandPotArr[indSols] < bestVarGrandPot)
        {
            bestVarGrandPot = VarGrandPotArr[indSols];
            bestElDensHigh = ElDensArr[indSols];
        }
    }

    //

    // std::cout << bestElDensLow << std::endl << std::endl;
    // std::cout << bestElDensHigh << std::endl << std::endl;

    Eigen::VectorXd nUpArr [noSeeds];
    Eigen::VectorXd nDwArr [noSeeds];
    Eigen::MatrixXd bandsUpArr [noSeeds];
    Eigen::MatrixXd bandsDwArr [noSeeds];

    if ( (bestElDensLow - fill_target) * (bestElDensHigh - fill_target) < 0 )
    {
        while ( it_bissect < it_bissect_max )
        {
            // std::cout << it_bissect << std::endl << std::endl;
            mu = ( muLow + muHigh ) / 2;

            //

            bestSeed = seeds[0];
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
            bestVarGrandPot = solver.grand_potential();
            bestElDens = solver.filling();
            best_nUp = solver.UpField();
            best_nDw = solver.DwField();
            best_bandsUp = solver.MFbandsUp();
            best_bandsDw = solver.MFbandsDw();
            bestSeed = 0;

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

            // std::cout << bestElDens << std::endl << std::endl;
            //
            // std::cout << fabs(bestElDens - fill_target) << std::endl << std::endl;
            //
            // std::cout << ( muHigh - muLow ) / 2 << std::endl << std::endl;

            if ( (fabs(bestElDens - fill_target) < tol) ||
                ( muHigh - muLow ) / 2 < bissect_tol )
                break;

            it_bissect++;
            if ( ( bestElDensLow - fill_target ) * ( bestElDens - fill_target ) > 0 )
            {
                muLow = mu;
            }
            else
            {
                muHigh = mu;
            }
        }
    }

    save(tmd, u, beta, mu, noSeeds, bestSeed, bestElDens,
         bestVarGrandPot, best_nUp, best_nDw, best_bandsUp, best_bandsDw);

    return 0;
}
