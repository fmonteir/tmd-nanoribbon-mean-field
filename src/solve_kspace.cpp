#include <vector>
#include <complex>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "model_kspace.hpp"
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
#define BETA_THRESHOLD 20
#endif
#ifndef DELTA
#define DELTA 0.00001
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
    if ( argc != 7) //  tmd, u, beta, mu, seed, init.cond.
    {
        std::cout << "Not enough arguments given to simulation."
        << std::endl << std::endl;
        return -1;
    }

    int tmd = atoi(argv[1]);
    double u = atof(argv[2]);
    double beta = atof(argv[3]);
    double mu = atof(argv[4]);
    double seed = atoi(argv[5]);
    int init_cond = atoi(argv[6]);

    double abs_t0;
    if (tmd == 1) //  MoS2
    {abs_t0 = 0.184;}
    if (tmd == 2) //  WS2
    {abs_t0 = 0.206;}
    if (tmd == 3) //  MoSe2
    {abs_t0 = 0.188;}
    if (tmd == 4) //  WSe2
    {abs_t0 = 0.207;}
    if (tmd == 5) //  MoTe2
    {abs_t0 = 0.169;}
    if (tmd == 6) //  WTe2
    {abs_t0 = 0.175;}
    model solver(tmd, u, mu, beta);
    solver.TMDnanoribbon();

    if (init_cond == 1)
    {
        solver.init_random(seed);
    }
    if (init_cond == 2)
    {
        solver.init_para(seed);
    }

    unsigned it = 0;
    while ( solver.loop_condition(it) )
    {
        solver.anneal(it);
        solver.diagonalize();
        solver.fermi();
        solver.update();
        solver.compute_grand_potential();
        solver.save_grand_potential(it);
        solver.damp(it);
        solver.tolerance_check();
        it++;
    }

    //  SAVE OUTPUT
    int precision = 10;
    std::ofstream parameters("temp-data/parameters.csv");
    if (parameters.is_open())
    {
        parameters << "U" << ',' << u << '\n';
        parameters << "BETA" << ',' << beta << '\n';
        parameters << "MU" << ',' << mu << '\n';
        parameters << "SEED" << ',' << seed << '\n';
        parameters << "FILLING" << ',' << solver.filling() << '\n';
        parameters << "FINAL GRAND POTENTIAL" << ',' << solver.grand_potential() << '\n';
        parameters << "FINAL IT" << ',' << it << '\n';
    }
    parameters.close();
    std::ofstream bands("temp-data/free-bands.csv");
    if (bands.is_open())
    {
        bands << std::setprecision(precision) << solver.TBbands() << '\n';
    }
    bands.close();
    std::ofstream nUp("temp-data/nUp.csv");
    if (nUp.is_open())
    {
        nUp << std::setprecision(precision) << solver.UpField() << '\n';
    }
    nUp.close();
    std::ofstream nDw("temp-data/nDw.csv");
    if (nDw.is_open())
    {
        nDw << std::setprecision(precision) << solver.DwField() << '\n';
    }
    nDw.close();
    std::ofstream grand_potentials("temp-data/grand_potential_evol.csv");
    if (grand_potentials.is_open())
    {
        grand_potentials << std::setprecision(precision)
        << solver.grand_potential_evol() << '\n';
    }
    grand_potentials.close();
    std::ofstream bandsUp("temp-data/bandsUp.csv");
    if (bandsUp.is_open())
    {
        bandsUp << std::setprecision(precision)
        << solver.MFbandsUp() << '\n';
    }
    bandsUp.close();
    std::ofstream bandsDw("temp-data/bandsDw.csv");
    if (bandsDw.is_open())
    {
        bandsDw << std::setprecision(precision)
        << solver.MFbandsDw() << '\n';
    }
    bandsDw.close();
    return 0;
}
