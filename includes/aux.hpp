//
//  aux.hpp
//
//
//  Created by Francisco Brito on 13/05/2019.
//

#ifndef aux_hpp
#define aux_hpp

void save(int tmd, double u, double beta, double mu, double noSeeds,
          int bestSeed, double bestElDens, double bestVarGrandPot,
          Eigen::VectorXd best_nUp, Eigen::VectorXd best_nDw,
          Eigen::MatrixXd best_bandsUp, Eigen::MatrixXd best_bandsDw)
{
    //  SAVE OUTPUT
    int precision = 10;
    std::ofstream parameters("temp-data/parameters.csv");
    if (parameters.is_open())
    {
        parameters << "U" << ',' << u << '\n';
        parameters << "BETA" << ',' << beta << '\n';
        parameters << "MU" << ',' << mu << '\n';
        parameters << "SEED" << ',' << bestSeed << '\n';
        parameters << "FILLING" << ',' << bestElDens << '\n';
        parameters << "FINAL GRAND POTENTIAL" << ',' << bestVarGrandPot << '\n';
        parameters << "NUMBER OF SEEDS" << ',' << noSeeds << '\n';
    }
    parameters.close();
    std::ofstream nUp("temp-data/nUp.csv");
    if (nUp.is_open())
    {
        nUp << std::setprecision(precision) << best_nUp << '\n';
    }
    nUp.close();
    std::ofstream nDw("temp-data/nDw.csv");
    if (nDw.is_open())
    {
        nDw << std::setprecision(precision) << best_nDw << '\n';
    }
    nDw.close();
    std::ofstream bandsUp("temp-data/bandsUp.csv");
    if (bandsUp.is_open())
    {
        bandsUp << std::setprecision(precision)
        << best_bandsUp << '\n';
    }
    bandsUp.close();
    std::ofstream bandsDw("temp-data/bandsDw.csv");
    if (bandsDw.is_open())
    {
        bandsDw << std::setprecision(precision)
        << best_bandsDw << '\n';
    }
    bandsDw.close();
}

#endif /* aux_hpp */
