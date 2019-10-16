//
//  seed_generator.cpp
//
//
//  Created by Francisco Brito on 05/12/2018.
//
//  This program generates seeds for a sequence of random numbers such that
//  they are uncorrelated when running the same process simultaneously.
//  This is needed because if processes are started simultaneously, the seeds
//  may be the same.
//

#include <fstream>
#include <random>
#include <array>

int main(int argc, char **argv)
{
    const int nSeeds = 1000000;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::random_device r;
    std::array<int, nSeeds> seed_data;
    std::generate(seed_data.begin(), seed_data.end(), std::ref(r));
    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
    gen.seed(seq);

    std::ofstream file("../seeds.csv");
    if (file.is_open())
    {
        for (int i = 0; i < nSeeds; i++)
            file << seed_data[i] << '\n';
    }
    file.close();
}
