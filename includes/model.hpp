//
//  model.hpp
//
//
//  Created by Francisco Brito on 17/04/2019.
//

struct node {
    //  This structure defines the model. A point or node on the M-atom
    //  triangular lattice has 3 d-orbitals from which the electrons hop.
    //  There are hoppings to the 3 orbitals of each of the 6 nearest neighbors.
    //  There are also on-site terms. See Phys. Rev. B 88, 085433 (2013)

    //  Number of orbitals in the tight-binding model
    int Norb;
    //  Number of hoppings from a point on the lattice
    unsigned int *NHoppings;
    //  Longitudinal distance between sites
    std::vector<int> *x_idxs;
    //  Transverse distance between sites
    std::vector<int> *y_idxs;
    //  d-orbital index (0, 1,..., Norb) of the orbital to which we hop
    std::vector<unsigned int> *end_orbs;
    //  Hopping
    std::vector<std::complex<double>> *hoppings;

    node(int number_of_orbitals) {
      Norb = number_of_orbitals;
      NHoppings = new unsigned[Norb];
      for (int o = 0; o < Norb; o++)
      {
          NHoppings[o] = 0;
      }
      x_idxs = new std::vector<int>[Norb];
      y_idxs = new std::vector<int>[Norb];
      end_orbs = new std::vector<unsigned int>[Norb];
      hoppings = new std::vector< std::complex<double> >[Norb];
    }

    void add_hopping(unsigned start_orb, unsigned end_orb,
      int x_idx, int y_idx, std::complex<double> hopping)
    {
        NHoppings[start_orb]++; //  increment neighbor
        end_orbs[start_orb].push_back(end_orb); //  write d-orbital
        x_idxs[start_orb].push_back(x_idx); //  write longitudinal distance
        y_idxs[start_orb].push_back(y_idx); //  write transverse distance
        hoppings[start_orb].push_back(hopping); //  write hopping
    }
};

class model_k_space
{
    double t0;
    double abs_t0;
    double e0;
    double e1;
    double t1;
    double t2;
    double t11;
    double t12;
    double t22;
    double lbda;
    double Bc;
    double Bv;
    double u;
    double mu;
    double beta;
    double beta_target;
    double energy;
    Eigen::VectorXd energies;
    Eigen::Matrix<std::complex<double>, -1, -1> *TB_HamiltonianUp;
    Eigen::Matrix<std::complex<double>, -1, -1> *TB_HamiltonianDw;
    Eigen::VectorXd nUp;
    Eigen::VectorXd nDw;
    Eigen::VectorXd nUpAux;
    Eigen::VectorXd nDwAux;
    Eigen::VectorXd nUpOld;
    Eigen::VectorXd nDwOld;
    double deltaUp;
    double deltaDw;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix
      <std::complex<double>, -1, -1>> *KsolUp;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix
        <std::complex<double>, -1, -1>> *KsolDw;
    Eigen::MatrixXd FermiDistUp;
    Eigen::MatrixXd FermiDistDw;
    Eigen::MatrixXd freeBands;
    Eigen::MatrixXd bandsUp;
    Eigen::MatrixXd bandsDw;
public:
    model_k_space(int tmd, double on_site_interaction,
      double chem_pot, double beta_f)
    {
        t0 = -1;
        // Define model
        // TMD : 1 - MoS2, 2 - WS2, 3 - MoSe2, 4 - WSe2, 5 - MoTe2, 6 - WTe2
        if (tmd == 1) //  MoS2
        {abs_t0 = 0.184; e0 = 1.046 / abs_t0; e1 = 2.104 / abs_t0;
         t1 = 0.401 / abs_t0; t2 = 0.507 / abs_t0; t11 = 0.218 / abs_t0;
         t12 = 0.338 / abs_t0; t22 = 0.057 / abs_t0; lbda = 0.073 / abs_t0;
         Bc = 0; Bv = 0;}
        if (tmd == 2) //  WS2
        {abs_t0 = 0.206; e0 = 1.130 / abs_t0; e1 = 2.275 / abs_t0;
         t1 = 0.567 / abs_t0; t2 = 0.536 / abs_t0; t11 = 0.286 / abs_t0;
         t12 = 0.384 / abs_t0; t22 = -0.061 / abs_t0; lbda = 0.211 / abs_t0;
         Bc = 0; Bv = 0;}
        if (tmd == 3) //  MoSe2
        {abs_t0 = 0.188;   e0 = 0.919 / abs_t0; e1 = 2.065 / abs_t0;
         t1 = 0.317 / abs_t0; t2 = 0.456 / abs_t0; t11 = 0.211 / abs_t0;
         t12 = 0.290 / abs_t0; t22 = 0.130 / abs_t0; lbda = 0.091 / abs_t0;
         Bc = 0; Bv = 0;}
        if (tmd == 4) //  WSe2
        {abs_t0 = 0.207; e0 = 0.943 / abs_t0; e1 = 2.179 / abs_t0;
         t1 = 0.457 / abs_t0; t2 = 0.486 / abs_t0; t11 = 0.263 / abs_t0;
         t12 = 0.329 / abs_t0;t22 = 0.034 / abs_t0; lbda = 0.228 / abs_t0;
         Bc = 0; Bv = 0;}
        if (tmd == 5) //  MoTe2
        {abs_t0 = 0.169; e0 = 0.605 / abs_t0; e1 = 1.972 / abs_t0;
         t1 = 0.228 / abs_t0; t2 = 0.390 / abs_t0; t11 = 0.207 / abs_t0;
         t12 = 0.239 / abs_t0; t22 = 0.252 / abs_t0; lbda = 0.107 / abs_t0;
         Bc = 0.206 / abs_t0; Bv = 0.170 / abs_t0;}
        if (tmd == 6) //  WTe2
        {abs_t0 = 0.175; e0 = 0.606 / abs_t0; e1 = 2.102 / abs_t0;
         t1 = 0.342 / abs_t0; t2 = 0.410 / abs_t0; t11 = 0.233 / abs_t0;
         t12 = 0.270 / abs_t0; t22 = 0.190 / abs_t0; lbda = 0.237 / abs_t0;
         Bc = 0; Bv = 0;}
        TB_HamiltonianUp = new Eigen::Matrix<std::complex<double>, -1, -1>[NK];
        TB_HamiltonianDw = new Eigen::Matrix<std::complex<double>, -1, -1>[NK];
        KsolUp = new Eigen::SelfAdjointEigenSolver<Eigen::Matrix
          <std::complex<double>, -1, -1>>[NK];
        KsolDw = new Eigen::SelfAdjointEigenSolver<Eigen::Matrix
          <std::complex<double>, -1, -1>>[NK];
        FermiDistUp = Eigen::MatrixXd(NK, NA * NORB * NY);
        FermiDistDw = Eigen::MatrixXd(NK, NA * NORB * NY);
        freeBands = Eigen::MatrixXd(NK, NA * NORB * NY);
        bandsUp = Eigen::MatrixXd(NK, NA * NORB * NY);
        bandsDw = Eigen::MatrixXd(NK, NA * NORB * NY);
        beta = BETA_START;
        beta_target = beta_f;
        mu = chem_pot;
        u = on_site_interaction;
        energy = 0.0;
        energies = Eigen::VectorXd(MAX_IT);
        deltaUp = DELTA + 1.0;
        deltaDw = DELTA + 1.0;
    }
    void TMDnanoribbon();
    void TMDnanoribbonSOC();
    void TMDnanoribbonSOC_Mag();
    void reset(double on_site_interaction, double chem_pot, double beta_f);
    void diagonalize();
    void fermi();
    void update();
    void compute_grand_potential();
    void save_grand_potential(unsigned it);
    double grand_potential();
    void anneal(unsigned it);
    void tolerance_check();
    void damp(unsigned it);
    bool loop_condition(unsigned it);
    Eigen::VectorXd UpField();
    Eigen::VectorXd DwField();
    Eigen::MatrixXd TBbandsUp();
    Eigen::MatrixXd TBbandsDw();
    Eigen::MatrixXd MFbandsUp();
    Eigen::MatrixXd MFbandsDw();
    Eigen::VectorXd grand_potential_evol();
    double filling();
    void init_random(int seed);
    void init_para(int seed);
    Eigen::Matrix<std::complex<double>, -1, -1> hopMat(int idx_k);
};

void model_k_space::TMDnanoribbon()
{
    node nUp(NORB);
    node nDw(NORB);

    // Add hoppings
    nUp.add_hopping(0, 0, 0, 0, std::complex<double>(e0,0));
    nUp.add_hopping(1, 1, 0, 0, std::complex<double>(e1,0));
    nUp.add_hopping(2, 2, 0, 0, std::complex<double>(e1,0));
    // R1
    nUp.add_hopping(0, 0, 1, 0, std::complex<double>(t0,0));
    nUp.add_hopping(1, 1, 1, 0, std::complex<double>(t11,0));
    nUp.add_hopping(2, 2, 1, 0, std::complex<double>(t22,0));
    nUp.add_hopping(0, 1, 1, 0, std::complex<double>(t1,0));
    nUp.add_hopping(0, 2, 1, 0, std::complex<double>(t2,0));
    nUp.add_hopping(1, 2, 1, 0, std::complex<double>(t12,0));
    nUp.add_hopping(1, 0, 1, 0, std::complex<double>(-t1,0));
    nUp.add_hopping(2, 0, 1, 0, std::complex<double>(t2,0));
    nUp.add_hopping(2, 1, 1, 0, std::complex<double>(-t12,0));
    // R4
    nUp.add_hopping(0, 0, -1, 0, std::complex<double>(t0,0));
    nUp.add_hopping(1, 1, -1, 0, std::complex<double>(t11,0));
    nUp.add_hopping(2, 2, -1, 0, std::complex<double>(t22,0));
    nUp.add_hopping(0, 1, -1, 0, std::complex<double>(-t1,0));
    nUp.add_hopping(0, 2, -1, 0, std::complex<double>(t2,0));
    nUp.add_hopping(1, 2, -1, 0, std::complex<double>(-t12,0));
    nUp.add_hopping(1, 0, -1, 0, std::complex<double>(t1,0));
    nUp.add_hopping(2, 0, -1, 0, std::complex<double>(t2,0));
    nUp.add_hopping(2, 1, -1, 0, std::complex<double>(t12,0));
    // R2
    nUp.add_hopping(0, 0, 1, -1, std::complex<double>(t0,0));
    nUp.add_hopping(1, 1, 1, -1, std::complex<double>((t11 + 3 * t22)/4,0));
    nUp.add_hopping(2, 2, 1, -1, std::complex<double>(( 3 * t11 + t22 ) / 4,0));
    nUp.add_hopping(0, 1, 1, -1, std::complex<double>(t1/2 - sqrt(3)*t2/2,0));
    nUp.add_hopping(0, 2, 1, -1, std::complex<double>(-sqrt(3)*t1/2 - t2/2,0));
    nUp.add_hopping(1, 2, 1, -1, std::complex<double>(sqrt(3)*(t22-t11)/4-t12,0));
    nUp.add_hopping(1, 0, 1, -1, std::complex<double>(-t1/2-sqrt(3)*t2/2,0));
    nUp.add_hopping(2, 0, 1, -1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(2, 1, 1, -1, std::complex<double>(sqrt(3)*(t22-t11)/4+t12,0));
    // R5
    nUp.add_hopping(0, 0, -1, 1, std::complex<double>(t0,0));
    nUp.add_hopping(1, 1, -1, 1, std::complex<double>((t11 + 3 * t22 ) / 4,0));
    nUp.add_hopping(2, 2, -1, 1, std::complex<double>((3 * t11 + t22 ) / 4,0));
    nUp.add_hopping(0, 1, -1, 1, std::complex<double>(-t1/2-sqrt(3)*t2/2,0));
    nUp.add_hopping(0, 2, -1, 1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(1, 2, -1, 1, std::complex<double>(sqrt(3)*(t22-t11)/4+t12,0));
    nUp.add_hopping(1, 0, -1, 1, std::complex<double>(t1/2-sqrt(3)*t2/2,0));
    nUp.add_hopping(2, 0, -1, 1, std::complex<double>(-sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(2, 1, -1, 1, std::complex<double>(sqrt(3)*(t22-t11)/4-t12,0));
    // R3
    nUp.add_hopping(0, 0, 0, -1, std::complex<double>(t0,0));
    nUp.add_hopping(1, 1, 0, -1, std::complex<double>(( t11 + 3 * t22 ) / 4,0));
    nUp.add_hopping(2, 2, 0, -1, std::complex<double>(( 3 * t11 + t22 ) / 4,0));
    nUp.add_hopping(0, 1, 0, -1, std::complex<double>(-t1/2+sqrt(3)*t2/2,0));
    nUp.add_hopping(0, 2, 0, -1, std::complex<double>(-sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(1, 2, 0, -1, std::complex<double>(-sqrt(3)*(t22-t11)/4+t12,0));
    nUp.add_hopping(1, 0, 0, -1, std::complex<double>(t1/2+sqrt(3)*t2/2,0));
    nUp.add_hopping(2, 0, 0, -1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(2, 1, 0, -1, std::complex<double>(-sqrt(3)*(t22-t11)/4-t12,0));
    // R6
    nUp.add_hopping(0, 0, 0, 1, std::complex<double>(t0,0));
    nUp.add_hopping(1, 1, 0, 1, std::complex<double>(( t11 + 3 * t22 ) / 4,0));
    nUp.add_hopping(2, 2, 0, 1, std::complex<double>(( 3 * t11 + t22 ) / 4,0));
    nUp.add_hopping(0, 1, 0, 1, std::complex<double>(t1/2+sqrt(3)*t2/2,0));
    nUp.add_hopping(0, 2, 0, 1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(1, 2, 0, 1, std::complex<double>(-sqrt(3)*(t22-t11)/4-t12,0));
    nUp.add_hopping(1, 0, 0, 1, std::complex<double>(-t1/2+sqrt(3)*t2/2,0));
    nUp.add_hopping(2, 0, 0, 1, std::complex<double>(-sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(2, 1, 0, 1, std::complex<double>(-sqrt(3)*(t22-t11)/4+t12,0));

    // Add hoppings
    nDw.add_hopping(0, 0, 0, 0, std::complex<double>(e0,0));
    nDw.add_hopping(1, 1, 0, 0, std::complex<double>(e1,0));
    nDw.add_hopping(2, 2, 0, 0, std::complex<double>(e1,0));
    // R1
    nDw.add_hopping(0, 0, 1, 0, std::complex<double>(t0,0));
    nDw.add_hopping(1, 1, 1, 0, std::complex<double>(t11,0));
    nDw.add_hopping(2, 2, 1, 0, std::complex<double>(t22,0));
    nDw.add_hopping(0, 1, 1, 0, std::complex<double>(t1,0));
    nDw.add_hopping(0, 2, 1, 0, std::complex<double>(t2,0));
    nDw.add_hopping(1, 2, 1, 0, std::complex<double>(t12,0));
    nDw.add_hopping(1, 0, 1, 0, std::complex<double>(-t1,0));
    nDw.add_hopping(2, 0, 1, 0, std::complex<double>(t2,0));
    nDw.add_hopping(2, 1, 1, 0, std::complex<double>(-t12,0));
    // R4
    nDw.add_hopping(0, 0, -1, 0, std::complex<double>(t0,0));
    nDw.add_hopping(1, 1, -1, 0, std::complex<double>(t11,0));
    nDw.add_hopping(2, 2, -1, 0, std::complex<double>(t22,0));
    nDw.add_hopping(0, 1, -1, 0, std::complex<double>(-t1,0));
    nDw.add_hopping(0, 2, -1, 0, std::complex<double>(t2,0));
    nDw.add_hopping(1, 2, -1, 0, std::complex<double>(-t12,0));
    nDw.add_hopping(1, 0, -1, 0, std::complex<double>(t1,0));
    nDw.add_hopping(2, 0, -1, 0, std::complex<double>(t2,0));
    nDw.add_hopping(2, 1, -1, 0, std::complex<double>(t12,0));
    // R2
    nDw.add_hopping(0, 0, 1, -1, std::complex<double>(t0,0));
    nDw.add_hopping(1, 1, 1, -1, std::complex<double>((t11 + 3 * t22)/4,0));
    nDw.add_hopping(2, 2, 1, -1, std::complex<double>(( 3 * t11 + t22 ) / 4,0));
    nDw.add_hopping(0, 1, 1, -1, std::complex<double>(t1/2 - sqrt(3)*t2/2,0));
    nDw.add_hopping(0, 2, 1, -1, std::complex<double>(-sqrt(3)*t1/2 - t2/2,0));
    nDw.add_hopping(1, 2, 1, -1, std::complex<double>(sqrt(3)*(t22-t11)/4-t12,0));
    nDw.add_hopping(1, 0, 1, -1, std::complex<double>(-t1/2-sqrt(3)*t2/2,0));
    nDw.add_hopping(2, 0, 1, -1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(2, 1, 1, -1, std::complex<double>(sqrt(3)*(t22-t11)/4+t12,0));
    // R5
    nDw.add_hopping(0, 0, -1, 1, std::complex<double>(t0,0));
    nDw.add_hopping(1, 1, -1, 1, std::complex<double>((t11 + 3 * t22 ) / 4,0));
    nDw.add_hopping(2, 2, -1, 1, std::complex<double>((3 * t11 + t22 ) / 4,0));
    nDw.add_hopping(0, 1, -1, 1, std::complex<double>(-t1/2-sqrt(3)*t2/2,0));
    nDw.add_hopping(0, 2, -1, 1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(1, 2, -1, 1, std::complex<double>(sqrt(3)*(t22-t11)/4+t12,0));
    nDw.add_hopping(1, 0, -1, 1, std::complex<double>(t1/2-sqrt(3)*t2/2,0));
    nDw.add_hopping(2, 0, -1, 1, std::complex<double>(-sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(2, 1, -1, 1, std::complex<double>(sqrt(3)*(t22-t11)/4-t12,0));
    // R3
    nDw.add_hopping(0, 0, 0, -1, std::complex<double>(t0,0));
    nDw.add_hopping(1, 1, 0, -1, std::complex<double>(( t11 + 3 * t22 ) / 4,0));
    nDw.add_hopping(2, 2, 0, -1, std::complex<double>(( 3 * t11 + t22 ) / 4,0));
    nDw.add_hopping(0, 1, 0, -1, std::complex<double>(-t1/2+sqrt(3)*t2/2,0));
    nDw.add_hopping(0, 2, 0, -1, std::complex<double>(-sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(1, 2, 0, -1, std::complex<double>(-sqrt(3)*(t22-t11)/4+t12,0));
    nDw.add_hopping(1, 0, 0, -1, std::complex<double>(t1/2+sqrt(3)*t2/2,0));
    nDw.add_hopping(2, 0, 0, -1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(2, 1, 0, -1, std::complex<double>(-sqrt(3)*(t22-t11)/4-t12,0));
    // R6
    nDw.add_hopping(0, 0, 0, 1, std::complex<double>(t0,0));
    nDw.add_hopping(1, 1, 0, 1, std::complex<double>(( t11 + 3 * t22 ) / 4,0));
    nDw.add_hopping(2, 2, 0, 1, std::complex<double>(( 3 * t11 + t22 ) / 4,0));
    nDw.add_hopping(0, 1, 0, 1, std::complex<double>(t1/2+sqrt(3)*t2/2,0));
    nDw.add_hopping(0, 2, 0, 1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(1, 2, 0, 1, std::complex<double>(-sqrt(3)*(t22-t11)/4-t12,0));
    nDw.add_hopping(1, 0, 0, 1, std::complex<double>(-t1/2+sqrt(3)*t2/2,0));
    nDw.add_hopping(2, 0, 0, 1, std::complex<double>(-sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(2, 1, 0, 1, std::complex<double>(-sqrt(3)*(t22-t11)/4+t12,0));

    // Build the Tight-binding part of the Hamiltonian
    double k;
    for (int idx_k = 0; idx_k < NK; idx_k++ )
    {
        k = 2 * M_PI / NK * idx_k - M_PI;
        TB_HamiltonianUp[idx_k] =
        Eigen::Matrix<std::complex<double>, -1, -1>::Zero
          (NA * NY * NORB, NA * NY * NORB);
        TB_HamiltonianDw[idx_k] =
        Eigen::Matrix<std::complex<double>, -1, -1>::Zero
          (NA * NY * NORB, NA * NY * NORB);
        for (int atom = 0; atom < NA; atom++ )
            for (int y = 0; y < NY; y++ )
              for (int orb = 0; orb < NORB; orb++ )
              {
                  unsigned start_idx = orb + NORB * ( NY * atom + y );
                  for (unsigned neighbor_idx = 0;
                      neighbor_idx < nUp.NHoppings[orb]; neighbor_idx++)
                  {
                      unsigned end_atom =
                        ( atom + nUp.x_idxs[orb].at(neighbor_idx) + NA ) % NA;
                      int end_y =
                        y + nUp.y_idxs[orb].at(neighbor_idx);
                      unsigned end_idx =
                        nUp.end_orbs[orb].at(neighbor_idx) + NORB
                        * ( NY * end_atom + end_y );
                      int cell_distance =
                        (atom + nUp.x_idxs[orb].at(neighbor_idx) + NA) / NA - 1;
                      if ( end_y >= 0 && end_y < NY )
                      {
                          TB_HamiltonianUp[idx_k](start_idx, end_idx)
                             += nUp.hoppings[orb].at(neighbor_idx)
                             * exp(std::complex<double>(0, k * cell_distance ));
                          TB_HamiltonianDw[idx_k](start_idx, end_idx)
                             += nDw.hoppings[orb].at(neighbor_idx)
                             * exp(std::complex<double>(0, k * cell_distance ));
                      }

                  }
              }
      }
}

void model_k_space::TMDnanoribbonSOC()
{
    node nUp(NORB);
    node nDw(NORB);

    // Add hoppings
    nUp.add_hopping(0, 0, 0, 0, std::complex<double>(e0,0));
    nUp.add_hopping(1, 1, 0, 0, std::complex<double>(e1,0));
    nUp.add_hopping(2, 2, 0, 0, std::complex<double>(e1,0));
    nUp.add_hopping(1, 2, 0, 0, std::complex<double>(0,lbda));
    nUp.add_hopping(2, 1, 0, 0, std::complex<double>(0,-lbda));
    // R1
    nUp.add_hopping(0, 0, 1, 0, std::complex<double>(t0,0));
    nUp.add_hopping(1, 1, 1, 0, std::complex<double>(t11,0));
    nUp.add_hopping(2, 2, 1, 0, std::complex<double>(t22,0));
    nUp.add_hopping(0, 1, 1, 0, std::complex<double>(t1,0));
    nUp.add_hopping(0, 2, 1, 0, std::complex<double>(t2,0));
    nUp.add_hopping(1, 2, 1, 0, std::complex<double>(t12,0));
    nUp.add_hopping(1, 0, 1, 0, std::complex<double>(-t1,0));
    nUp.add_hopping(2, 0, 1, 0, std::complex<double>(t2,0));
    nUp.add_hopping(2, 1, 1, 0, std::complex<double>(-t12,0));
    // R4
    nUp.add_hopping(0, 0, -1, 0, std::complex<double>(t0,0));
    nUp.add_hopping(1, 1, -1, 0, std::complex<double>(t11,0));
    nUp.add_hopping(2, 2, -1, 0, std::complex<double>(t22,0));
    nUp.add_hopping(0, 1, -1, 0, std::complex<double>(-t1,0));
    nUp.add_hopping(0, 2, -1, 0, std::complex<double>(t2,0));
    nUp.add_hopping(1, 2, -1, 0, std::complex<double>(-t12,0));
    nUp.add_hopping(1, 0, -1, 0, std::complex<double>(t1,0));
    nUp.add_hopping(2, 0, -1, 0, std::complex<double>(t2,0));
    nUp.add_hopping(2, 1, -1, 0, std::complex<double>(t12,0));
    // R2
    nUp.add_hopping(0, 0, 1, -1, std::complex<double>(t0,0));
    nUp.add_hopping(1, 1, 1, -1, std::complex<double>((t11 + 3 * t22)/4,0));
    nUp.add_hopping(2, 2, 1, -1, std::complex<double>(( 3 * t11 + t22 ) / 4,0));
    nUp.add_hopping(0, 1, 1, -1, std::complex<double>(t1/2 - sqrt(3)*t2/2,0));
    nUp.add_hopping(0, 2, 1, -1, std::complex<double>(-sqrt(3)*t1/2 - t2/2,0));
    nUp.add_hopping(1, 2, 1, -1, std::complex<double>(sqrt(3)*(t22-t11)/4-t12,0));
    nUp.add_hopping(1, 0, 1, -1, std::complex<double>(-t1/2-sqrt(3)*t2/2,0));
    nUp.add_hopping(2, 0, 1, -1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(2, 1, 1, -1, std::complex<double>(sqrt(3)*(t22-t11)/4+t12,0));
    // R5
    nUp.add_hopping(0, 0, -1, 1, std::complex<double>(t0,0));
    nUp.add_hopping(1, 1, -1, 1, std::complex<double>((t11 + 3 * t22 ) / 4,0));
    nUp.add_hopping(2, 2, -1, 1, std::complex<double>((3 * t11 + t22 ) / 4,0));
    nUp.add_hopping(0, 1, -1, 1, std::complex<double>(-t1/2-sqrt(3)*t2/2,0));
    nUp.add_hopping(0, 2, -1, 1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(1, 2, -1, 1, std::complex<double>(sqrt(3)*(t22-t11)/4+t12,0));
    nUp.add_hopping(1, 0, -1, 1, std::complex<double>(t1/2-sqrt(3)*t2/2,0));
    nUp.add_hopping(2, 0, -1, 1, std::complex<double>(-sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(2, 1, -1, 1, std::complex<double>(sqrt(3)*(t22-t11)/4-t12,0));
    // R3
    nUp.add_hopping(0, 0, 0, -1, std::complex<double>(t0,0));
    nUp.add_hopping(1, 1, 0, -1, std::complex<double>(( t11 + 3 * t22 ) / 4,0));
    nUp.add_hopping(2, 2, 0, -1, std::complex<double>(( 3 * t11 + t22 ) / 4,0));
    nUp.add_hopping(0, 1, 0, -1, std::complex<double>(-t1/2+sqrt(3)*t2/2,0));
    nUp.add_hopping(0, 2, 0, -1, std::complex<double>(-sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(1, 2, 0, -1, std::complex<double>(-sqrt(3)*(t22-t11)/4+t12,0));
    nUp.add_hopping(1, 0, 0, -1, std::complex<double>(t1/2+sqrt(3)*t2/2,0));
    nUp.add_hopping(2, 0, 0, -1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(2, 1, 0, -1, std::complex<double>(-sqrt(3)*(t22-t11)/4-t12,0));
    // R6
    nUp.add_hopping(0, 0, 0, 1, std::complex<double>(t0,0));
    nUp.add_hopping(1, 1, 0, 1, std::complex<double>(( t11 + 3 * t22 ) / 4,0));
    nUp.add_hopping(2, 2, 0, 1, std::complex<double>(( 3 * t11 + t22 ) / 4,0));
    nUp.add_hopping(0, 1, 0, 1, std::complex<double>(t1/2+sqrt(3)*t2/2,0));
    nUp.add_hopping(0, 2, 0, 1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(1, 2, 0, 1, std::complex<double>(-sqrt(3)*(t22-t11)/4-t12,0));
    nUp.add_hopping(1, 0, 0, 1, std::complex<double>(-t1/2+sqrt(3)*t2/2,0));
    nUp.add_hopping(2, 0, 0, 1, std::complex<double>(-sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(2, 1, 0, 1, std::complex<double>(-sqrt(3)*(t22-t11)/4+t12,0));

    // Add hoppings
    nDw.add_hopping(0, 0, 0, 0, std::complex<double>(e0,0));
    nDw.add_hopping(1, 1, 0, 0, std::complex<double>(e1,0));
    nDw.add_hopping(2, 2, 0, 0, std::complex<double>(e1,0));
    nDw.add_hopping(1, 2, 0, 0, std::complex<double>(0,-lbda));
    nDw.add_hopping(2, 1, 0, 0, std::complex<double>(0,lbda));
    // R1
    nDw.add_hopping(0, 0, 1, 0, std::complex<double>(t0,0));
    nDw.add_hopping(1, 1, 1, 0, std::complex<double>(t11,0));
    nDw.add_hopping(2, 2, 1, 0, std::complex<double>(t22,0));
    nDw.add_hopping(0, 1, 1, 0, std::complex<double>(t1,0));
    nDw.add_hopping(0, 2, 1, 0, std::complex<double>(t2,0));
    nDw.add_hopping(1, 2, 1, 0, std::complex<double>(t12,0));
    nDw.add_hopping(1, 0, 1, 0, std::complex<double>(-t1,0));
    nDw.add_hopping(2, 0, 1, 0, std::complex<double>(t2,0));
    nDw.add_hopping(2, 1, 1, 0, std::complex<double>(-t12,0));
    // R4
    nDw.add_hopping(0, 0, -1, 0, std::complex<double>(t0,0));
    nDw.add_hopping(1, 1, -1, 0, std::complex<double>(t11,0));
    nDw.add_hopping(2, 2, -1, 0, std::complex<double>(t22,0));
    nDw.add_hopping(0, 1, -1, 0, std::complex<double>(-t1,0));
    nDw.add_hopping(0, 2, -1, 0, std::complex<double>(t2,0));
    nDw.add_hopping(1, 2, -1, 0, std::complex<double>(-t12,0));
    nDw.add_hopping(1, 0, -1, 0, std::complex<double>(t1,0));
    nDw.add_hopping(2, 0, -1, 0, std::complex<double>(t2,0));
    nDw.add_hopping(2, 1, -1, 0, std::complex<double>(t12,0));
    // R2
    nDw.add_hopping(0, 0, 1, -1, std::complex<double>(t0,0));
    nDw.add_hopping(1, 1, 1, -1, std::complex<double>((t11 + 3 * t22)/4,0));
    nDw.add_hopping(2, 2, 1, -1, std::complex<double>(( 3 * t11 + t22 ) / 4,0));
    nDw.add_hopping(0, 1, 1, -1, std::complex<double>(t1/2 - sqrt(3)*t2/2,0));
    nDw.add_hopping(0, 2, 1, -1, std::complex<double>(-sqrt(3)*t1/2 - t2/2,0));
    nDw.add_hopping(1, 2, 1, -1, std::complex<double>(sqrt(3)*(t22-t11)/4-t12,0));
    nDw.add_hopping(1, 0, 1, -1, std::complex<double>(-t1/2-sqrt(3)*t2/2,0));
    nDw.add_hopping(2, 0, 1, -1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(2, 1, 1, -1, std::complex<double>(sqrt(3)*(t22-t11)/4+t12,0));
    // R5
    nDw.add_hopping(0, 0, -1, 1, std::complex<double>(t0,0));
    nDw.add_hopping(1, 1, -1, 1, std::complex<double>((t11 + 3 * t22 ) / 4,0));
    nDw.add_hopping(2, 2, -1, 1, std::complex<double>((3 * t11 + t22 ) / 4,0));
    nDw.add_hopping(0, 1, -1, 1, std::complex<double>(-t1/2-sqrt(3)*t2/2,0));
    nDw.add_hopping(0, 2, -1, 1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(1, 2, -1, 1, std::complex<double>(sqrt(3)*(t22-t11)/4+t12,0));
    nDw.add_hopping(1, 0, -1, 1, std::complex<double>(t1/2-sqrt(3)*t2/2,0));
    nDw.add_hopping(2, 0, -1, 1, std::complex<double>(-sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(2, 1, -1, 1, std::complex<double>(sqrt(3)*(t22-t11)/4-t12,0));
    // R3
    nDw.add_hopping(0, 0, 0, -1, std::complex<double>(t0,0));
    nDw.add_hopping(1, 1, 0, -1, std::complex<double>(( t11 + 3 * t22 ) / 4,0));
    nDw.add_hopping(2, 2, 0, -1, std::complex<double>(( 3 * t11 + t22 ) / 4,0));
    nDw.add_hopping(0, 1, 0, -1, std::complex<double>(-t1/2+sqrt(3)*t2/2,0));
    nDw.add_hopping(0, 2, 0, -1, std::complex<double>(-sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(1, 2, 0, -1, std::complex<double>(-sqrt(3)*(t22-t11)/4+t12,0));
    nDw.add_hopping(1, 0, 0, -1, std::complex<double>(t1/2+sqrt(3)*t2/2,0));
    nDw.add_hopping(2, 0, 0, -1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(2, 1, 0, -1, std::complex<double>(-sqrt(3)*(t22-t11)/4-t12,0));
    // R6
    nDw.add_hopping(0, 0, 0, 1, std::complex<double>(t0,0));
    nDw.add_hopping(1, 1, 0, 1, std::complex<double>(( t11 + 3 * t22 ) / 4,0));
    nDw.add_hopping(2, 2, 0, 1, std::complex<double>(( 3 * t11 + t22 ) / 4,0));
    nDw.add_hopping(0, 1, 0, 1, std::complex<double>(t1/2+sqrt(3)*t2/2,0));
    nDw.add_hopping(0, 2, 0, 1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(1, 2, 0, 1, std::complex<double>(-sqrt(3)*(t22-t11)/4-t12,0));
    nDw.add_hopping(1, 0, 0, 1, std::complex<double>(-t1/2+sqrt(3)*t2/2,0));
    nDw.add_hopping(2, 0, 0, 1, std::complex<double>(-sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(2, 1, 0, 1, std::complex<double>(-sqrt(3)*(t22-t11)/4+t12,0));

    // Build the Tight-binding part of the Hamiltonian
    double k;
    for (int idx_k = 0; idx_k < NK; idx_k++ )
    {
        k = 2 * M_PI / NK * idx_k - M_PI;
        TB_HamiltonianUp[idx_k] =
        Eigen::Matrix<std::complex<double>, -1, -1>::Zero
          (NA * NY * NORB, NA * NY * NORB);
        TB_HamiltonianDw[idx_k] =
        Eigen::Matrix<std::complex<double>, -1, -1>::Zero
          (NA * NY * NORB, NA * NY * NORB);
        for (int atom = 0; atom < NA; atom++ )
            for (int y = 0; y < NY; y++ )
              for (int orb = 0; orb < NORB; orb++ )
              {
                  unsigned start_idx = orb + NORB * ( NY * atom + y );
                  for (unsigned neighbor_idx = 0;
                      neighbor_idx < nUp.NHoppings[orb]; neighbor_idx++)
                  {
                      unsigned end_atom =
                        ( atom + nUp.x_idxs[orb].at(neighbor_idx) + NA ) % NA;
                      int end_y =
                        y + nUp.y_idxs[orb].at(neighbor_idx);
                      unsigned end_idx =
                        nUp.end_orbs[orb].at(neighbor_idx) + NORB
                        * ( NY * end_atom + end_y );
                      int cell_distance =
                        (atom + nUp.x_idxs[orb].at(neighbor_idx) + NA) / NA - 1;
                      if ( end_y >= 0 && end_y < NY )
                      {
                          TB_HamiltonianUp[idx_k](start_idx, end_idx)
                             += nUp.hoppings[orb].at(neighbor_idx)
                             * exp(std::complex<double>(0, k * cell_distance ));
                          TB_HamiltonianDw[idx_k](start_idx, end_idx)
                             += nDw.hoppings[orb].at(neighbor_idx)
                             * exp(std::complex<double>(0, k * cell_distance ));
                      }

                  }
              }
      }
}

void model_k_space::TMDnanoribbonSOC_Mag()
{
    node nUp(NORB);
    node nDw(NORB);

    // Add hoppings
    nUp.add_hopping(0, 0, 0, 0, std::complex<double>(e0 - Bc,0));
    nUp.add_hopping(1, 1, 0, 0, std::complex<double>(e1 - Bv,0));
    nUp.add_hopping(2, 2, 0, 0, std::complex<double>(e1 - Bv,0));
    nUp.add_hopping(1, 2, 0, 0, std::complex<double>(0,lbda));
    nUp.add_hopping(2, 1, 0, 0, std::complex<double>(0,-lbda));
    // R1
    nUp.add_hopping(0, 0, 1, 0, std::complex<double>(t0,0));
    nUp.add_hopping(1, 1, 1, 0, std::complex<double>(t11,0));
    nUp.add_hopping(2, 2, 1, 0, std::complex<double>(t22,0));
    nUp.add_hopping(0, 1, 1, 0, std::complex<double>(t1,0));
    nUp.add_hopping(0, 2, 1, 0, std::complex<double>(t2,0));
    nUp.add_hopping(1, 2, 1, 0, std::complex<double>(t12,0));
    nUp.add_hopping(1, 0, 1, 0, std::complex<double>(-t1,0));
    nUp.add_hopping(2, 0, 1, 0, std::complex<double>(t2,0));
    nUp.add_hopping(2, 1, 1, 0, std::complex<double>(-t12,0));
    // R4
    nUp.add_hopping(0, 0, -1, 0, std::complex<double>(t0,0));
    nUp.add_hopping(1, 1, -1, 0, std::complex<double>(t11,0));
    nUp.add_hopping(2, 2, -1, 0, std::complex<double>(t22,0));
    nUp.add_hopping(0, 1, -1, 0, std::complex<double>(-t1,0));
    nUp.add_hopping(0, 2, -1, 0, std::complex<double>(t2,0));
    nUp.add_hopping(1, 2, -1, 0, std::complex<double>(-t12,0));
    nUp.add_hopping(1, 0, -1, 0, std::complex<double>(t1,0));
    nUp.add_hopping(2, 0, -1, 0, std::complex<double>(t2,0));
    nUp.add_hopping(2, 1, -1, 0, std::complex<double>(t12,0));
    // R2
    nUp.add_hopping(0, 0, 1, -1, std::complex<double>(t0,0));
    nUp.add_hopping(1, 1, 1, -1, std::complex<double>((t11 + 3 * t22)/4,0));
    nUp.add_hopping(2, 2, 1, -1, std::complex<double>(( 3 * t11 + t22 ) / 4,0));
    nUp.add_hopping(0, 1, 1, -1, std::complex<double>(t1/2 - sqrt(3)*t2/2,0));
    nUp.add_hopping(0, 2, 1, -1, std::complex<double>(-sqrt(3)*t1/2 - t2/2,0));
    nUp.add_hopping(1, 2, 1, -1, std::complex<double>(sqrt(3)*(t22-t11)/4-t12,0));
    nUp.add_hopping(1, 0, 1, -1, std::complex<double>(-t1/2-sqrt(3)*t2/2,0));
    nUp.add_hopping(2, 0, 1, -1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(2, 1, 1, -1, std::complex<double>(sqrt(3)*(t22-t11)/4+t12,0));
    // R5
    nUp.add_hopping(0, 0, -1, 1, std::complex<double>(t0,0));
    nUp.add_hopping(1, 1, -1, 1, std::complex<double>((t11 + 3 * t22 ) / 4,0));
    nUp.add_hopping(2, 2, -1, 1, std::complex<double>((3 * t11 + t22 ) / 4,0));
    nUp.add_hopping(0, 1, -1, 1, std::complex<double>(-t1/2-sqrt(3)*t2/2,0));
    nUp.add_hopping(0, 2, -1, 1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(1, 2, -1, 1, std::complex<double>(sqrt(3)*(t22-t11)/4+t12,0));
    nUp.add_hopping(1, 0, -1, 1, std::complex<double>(t1/2-sqrt(3)*t2/2,0));
    nUp.add_hopping(2, 0, -1, 1, std::complex<double>(-sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(2, 1, -1, 1, std::complex<double>(sqrt(3)*(t22-t11)/4-t12,0));
    // R3
    nUp.add_hopping(0, 0, 0, -1, std::complex<double>(t0,0));
    nUp.add_hopping(1, 1, 0, -1, std::complex<double>(( t11 + 3 * t22 ) / 4,0));
    nUp.add_hopping(2, 2, 0, -1, std::complex<double>(( 3 * t11 + t22 ) / 4,0));
    nUp.add_hopping(0, 1, 0, -1, std::complex<double>(-t1/2+sqrt(3)*t2/2,0));
    nUp.add_hopping(0, 2, 0, -1, std::complex<double>(-sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(1, 2, 0, -1, std::complex<double>(-sqrt(3)*(t22-t11)/4+t12,0));
    nUp.add_hopping(1, 0, 0, -1, std::complex<double>(t1/2+sqrt(3)*t2/2,0));
    nUp.add_hopping(2, 0, 0, -1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(2, 1, 0, -1, std::complex<double>(-sqrt(3)*(t22-t11)/4-t12,0));
    // R6
    nUp.add_hopping(0, 0, 0, 1, std::complex<double>(t0,0));
    nUp.add_hopping(1, 1, 0, 1, std::complex<double>(( t11 + 3 * t22 ) / 4,0));
    nUp.add_hopping(2, 2, 0, 1, std::complex<double>(( 3 * t11 + t22 ) / 4,0));
    nUp.add_hopping(0, 1, 0, 1, std::complex<double>(t1/2+sqrt(3)*t2/2,0));
    nUp.add_hopping(0, 2, 0, 1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(1, 2, 0, 1, std::complex<double>(-sqrt(3)*(t22-t11)/4-t12,0));
    nUp.add_hopping(1, 0, 0, 1, std::complex<double>(-t1/2+sqrt(3)*t2/2,0));
    nUp.add_hopping(2, 0, 0, 1, std::complex<double>(-sqrt(3)*t1/2-t2/2,0));
    nUp.add_hopping(2, 1, 0, 1, std::complex<double>(-sqrt(3)*(t22-t11)/4+t12,0));

    // Add hoppings
    nDw.add_hopping(0, 0, 0, 0, std::complex<double>(e0 + Bc,0));
    nDw.add_hopping(1, 1, 0, 0, std::complex<double>(e1 + Bv,0));
    nDw.add_hopping(2, 2, 0, 0, std::complex<double>(e1 + Bv,0));
    nDw.add_hopping(1, 2, 0, 0, std::complex<double>(0,-lbda));
    nDw.add_hopping(2, 1, 0, 0, std::complex<double>(0,lbda));
    // R1
    nDw.add_hopping(0, 0, 1, 0, std::complex<double>(t0,0));
    nDw.add_hopping(1, 1, 1, 0, std::complex<double>(t11,0));
    nDw.add_hopping(2, 2, 1, 0, std::complex<double>(t22,0));
    nDw.add_hopping(0, 1, 1, 0, std::complex<double>(t1,0));
    nDw.add_hopping(0, 2, 1, 0, std::complex<double>(t2,0));
    nDw.add_hopping(1, 2, 1, 0, std::complex<double>(t12,0));
    nDw.add_hopping(1, 0, 1, 0, std::complex<double>(-t1,0));
    nDw.add_hopping(2, 0, 1, 0, std::complex<double>(t2,0));
    nDw.add_hopping(2, 1, 1, 0, std::complex<double>(-t12,0));
    // R4
    nDw.add_hopping(0, 0, -1, 0, std::complex<double>(t0,0));
    nDw.add_hopping(1, 1, -1, 0, std::complex<double>(t11,0));
    nDw.add_hopping(2, 2, -1, 0, std::complex<double>(t22,0));
    nDw.add_hopping(0, 1, -1, 0, std::complex<double>(-t1,0));
    nDw.add_hopping(0, 2, -1, 0, std::complex<double>(t2,0));
    nDw.add_hopping(1, 2, -1, 0, std::complex<double>(-t12,0));
    nDw.add_hopping(1, 0, -1, 0, std::complex<double>(t1,0));
    nDw.add_hopping(2, 0, -1, 0, std::complex<double>(t2,0));
    nDw.add_hopping(2, 1, -1, 0, std::complex<double>(t12,0));
    // R2
    nDw.add_hopping(0, 0, 1, -1, std::complex<double>(t0,0));
    nDw.add_hopping(1, 1, 1, -1, std::complex<double>((t11 + 3 * t22)/4,0));
    nDw.add_hopping(2, 2, 1, -1, std::complex<double>(( 3 * t11 + t22 ) / 4,0));
    nDw.add_hopping(0, 1, 1, -1, std::complex<double>(t1/2 - sqrt(3)*t2/2,0));
    nDw.add_hopping(0, 2, 1, -1, std::complex<double>(-sqrt(3)*t1/2 - t2/2,0));
    nDw.add_hopping(1, 2, 1, -1, std::complex<double>(sqrt(3)*(t22-t11)/4-t12,0));
    nDw.add_hopping(1, 0, 1, -1, std::complex<double>(-t1/2-sqrt(3)*t2/2,0));
    nDw.add_hopping(2, 0, 1, -1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(2, 1, 1, -1, std::complex<double>(sqrt(3)*(t22-t11)/4+t12,0));
    // R5
    nDw.add_hopping(0, 0, -1, 1, std::complex<double>(t0,0));
    nDw.add_hopping(1, 1, -1, 1, std::complex<double>((t11 + 3 * t22 ) / 4,0));
    nDw.add_hopping(2, 2, -1, 1, std::complex<double>((3 * t11 + t22 ) / 4,0));
    nDw.add_hopping(0, 1, -1, 1, std::complex<double>(-t1/2-sqrt(3)*t2/2,0));
    nDw.add_hopping(0, 2, -1, 1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(1, 2, -1, 1, std::complex<double>(sqrt(3)*(t22-t11)/4+t12,0));
    nDw.add_hopping(1, 0, -1, 1, std::complex<double>(t1/2-sqrt(3)*t2/2,0));
    nDw.add_hopping(2, 0, -1, 1, std::complex<double>(-sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(2, 1, -1, 1, std::complex<double>(sqrt(3)*(t22-t11)/4-t12,0));
    // R3
    nDw.add_hopping(0, 0, 0, -1, std::complex<double>(t0,0));
    nDw.add_hopping(1, 1, 0, -1, std::complex<double>(( t11 + 3 * t22 ) / 4,0));
    nDw.add_hopping(2, 2, 0, -1, std::complex<double>(( 3 * t11 + t22 ) / 4,0));
    nDw.add_hopping(0, 1, 0, -1, std::complex<double>(-t1/2+sqrt(3)*t2/2,0));
    nDw.add_hopping(0, 2, 0, -1, std::complex<double>(-sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(1, 2, 0, -1, std::complex<double>(-sqrt(3)*(t22-t11)/4+t12,0));
    nDw.add_hopping(1, 0, 0, -1, std::complex<double>(t1/2+sqrt(3)*t2/2,0));
    nDw.add_hopping(2, 0, 0, -1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(2, 1, 0, -1, std::complex<double>(-sqrt(3)*(t22-t11)/4-t12,0));
    // R6
    nDw.add_hopping(0, 0, 0, 1, std::complex<double>(t0,0));
    nDw.add_hopping(1, 1, 0, 1, std::complex<double>(( t11 + 3 * t22 ) / 4,0));
    nDw.add_hopping(2, 2, 0, 1, std::complex<double>(( 3 * t11 + t22 ) / 4,0));
    nDw.add_hopping(0, 1, 0, 1, std::complex<double>(t1/2+sqrt(3)*t2/2,0));
    nDw.add_hopping(0, 2, 0, 1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(1, 2, 0, 1, std::complex<double>(-sqrt(3)*(t22-t11)/4-t12,0));
    nDw.add_hopping(1, 0, 0, 1, std::complex<double>(-t1/2+sqrt(3)*t2/2,0));
    nDw.add_hopping(2, 0, 0, 1, std::complex<double>(-sqrt(3)*t1/2-t2/2,0));
    nDw.add_hopping(2, 1, 0, 1, std::complex<double>(-sqrt(3)*(t22-t11)/4+t12,0));

    // Build the Tight-binding part of the Hamiltonian
    double k;
    for (int idx_k = 0; idx_k < NK; idx_k++ )
    {
        k = 2 * M_PI / NK * idx_k - M_PI;
        TB_HamiltonianUp[idx_k] =
        Eigen::Matrix<std::complex<double>, -1, -1>::Zero
          (NA * NY * NORB, NA * NY * NORB);
        TB_HamiltonianDw[idx_k] =
        Eigen::Matrix<std::complex<double>, -1, -1>::Zero
          (NA * NY * NORB, NA * NY * NORB);
        for (int atom = 0; atom < NA; atom++ )
            for (int y = 0; y < NY; y++ )
              for (int orb = 0; orb < NORB; orb++ )
              {
                  unsigned start_idx = orb + NORB * ( NY * atom + y );
                  for (unsigned neighbor_idx = 0;
                      neighbor_idx < nUp.NHoppings[orb]; neighbor_idx++)
                  {
                      unsigned end_atom =
                        ( atom + nUp.x_idxs[orb].at(neighbor_idx) + NA ) % NA;
                      int end_y =
                        y + nUp.y_idxs[orb].at(neighbor_idx);
                      unsigned end_idx =
                        nUp.end_orbs[orb].at(neighbor_idx) + NORB
                        * ( NY * end_atom + end_y );
                      int cell_distance =
                        (atom + nUp.x_idxs[orb].at(neighbor_idx) + NA) / NA - 1;
                      if ( end_y >= 0 && end_y < NY )
                      {
                          TB_HamiltonianUp[idx_k](start_idx, end_idx)
                             += nUp.hoppings[orb].at(neighbor_idx)
                             * exp(std::complex<double>(0, k * cell_distance ));
                          TB_HamiltonianDw[idx_k](start_idx, end_idx)
                             += nDw.hoppings[orb].at(neighbor_idx)
                             * exp(std::complex<double>(0, k * cell_distance ));
                      }

                  }
              }
      }
}

void model_k_space::reset(double on_site_interaction, double chem_pot, double beta_f)
{
    KsolUp = new Eigen::SelfAdjointEigenSolver<Eigen::Matrix
      <std::complex<double>, -1, -1>>[NK];
    KsolDw = new Eigen::SelfAdjointEigenSolver<Eigen::Matrix
      <std::complex<double>, -1, -1>>[NK];
    FermiDistUp = Eigen::MatrixXd(NK, NA * NORB * NY);
    FermiDistDw = Eigen::MatrixXd(NK, NA * NORB * NY);
    bandsUp = Eigen::MatrixXd(NK, NA * NORB * NY);
    bandsDw = Eigen::MatrixXd(NK, NA * NORB * NY);
    beta = BETA_START;
    beta_target = beta_f;
    mu = chem_pot;
    u = on_site_interaction;
    energy = 0.0;
    deltaUp = DELTA + 1.0;
    deltaDw = DELTA + 1.0;
}

Eigen::MatrixXd model_k_space::TBbandsUp()
{
    for (int idx_k = 0; idx_k < NK; idx_k++)
    {
        KsolUp[idx_k].compute(
          TB_HamiltonianUp[idx_k] );
        for (int idxEn = 0; idxEn < NA * NY * NORB; idxEn++)
        {
            freeBands(idx_k, idxEn) = KsolUp[idx_k].eigenvalues()(idxEn);
        }
    }
    return abs_t0 * freeBands;
}

Eigen::MatrixXd model_k_space::TBbandsDw()
{
    for (int idx_k = 0; idx_k < NK; idx_k++)
    {
        KsolDw[idx_k].compute(
          TB_HamiltonianDw[idx_k] );
        for (int idxEn = 0; idxEn < NA * NY * NORB; idxEn++)
        {
            freeBands(idx_k, idxEn) = KsolDw[idx_k].eigenvalues()(idxEn);
        }
    }
    return abs_t0 * freeBands;
}

void model_k_space::diagonalize()
{
    // #pragma omp parallel for num_threads(NTH)
    // check parallel
    for (int idx_k = 0; idx_k < NK; idx_k++)
    {
        KsolUp[idx_k].compute(
          TB_HamiltonianUp[idx_k] + u * Eigen::MatrixXd(nDw.asDiagonal()) );
        KsolDw[idx_k].compute(
          TB_HamiltonianDw[idx_k] + u * Eigen::MatrixXd(nUp.asDiagonal()) );
    }
}

Eigen::MatrixXd model_k_space::MFbandsUp()
{
    for (int idx_k = 0; idx_k < NK; idx_k++)
    {
        KsolUp[idx_k].compute(
          TB_HamiltonianUp[idx_k] + u * Eigen::MatrixXd(nDw.asDiagonal()) );
        for (int idxEn = 0; idxEn < NA * NY * NORB; idxEn++)
        {
            bandsUp(idx_k, idxEn) = KsolUp[idx_k].eigenvalues()(idxEn);
        }
    }
    return abs_t0 * bandsUp;
}

Eigen::MatrixXd model_k_space::MFbandsDw()
{
    for (int idx_k = 0; idx_k < NK; idx_k++)
    {
        KsolDw[idx_k].compute(
          TB_HamiltonianDw[idx_k] + u * Eigen::MatrixXd(nUp.asDiagonal()) );
        for (int idxEn = 0; idxEn < NA * NY * NORB; idxEn++)
        {
            bandsDw(idx_k, idxEn) = KsolDw[idx_k].eigenvalues()(idxEn);
        }
    }
    return abs_t0 * bandsDw;
}

void model_k_space::fermi()
{
    //  Fermi distribution for the spectrum for each k, spectrum.
    //  If beta is greater or equal to a threshold, it returns the step function.
    FermiDistUp = Eigen::MatrixXd::Zero(NK, NA * NORB * NY);
    FermiDistDw = Eigen::MatrixXd::Zero(NK, NA * NORB * NY);
    // #pragma omp parallel for num_threads(NTH)
    // check parallel
    for (int idx_k = 0; idx_k < NK; idx_k++ )
    {
        for (int idxEn=0; idxEn < NA * NY * NORB; idxEn++)
        {
            if (beta >= BETA_THRESHOLD)
            {
                if (KsolUp[idx_k].eigenvalues()(idxEn) < mu)
                {
                    FermiDistUp(idx_k, idxEn) = 1;
                }
                else
                {
                    if (KsolUp[idx_k].eigenvalues()(idxEn) == mu)
                    {
                        FermiDistUp(idx_k, idxEn) = 1/2;
                    }
                }
                if (KsolDw[idx_k].eigenvalues()(idxEn) < mu)
                {
                    FermiDistDw(idx_k, idxEn) = 1;
                }
                else
                {
                    if (KsolDw[idx_k].eigenvalues()(idxEn) == mu)
                    {
                        FermiDistDw(idx_k, idxEn) = 1/2;
                    }
                }
            }
            else
            {
                FermiDistUp(idx_k, idxEn)
                = 1/(1+exp(beta*(KsolUp[idx_k].eigenvalues()(idxEn)- mu)));
                FermiDistDw(idx_k, idxEn)
                = 1/(1+exp(beta*(KsolDw[idx_k].eigenvalues()(idxEn)- mu)));
            }
        }
    }
}

void model_k_space::update()
{
    nUpOld = nUp;
    nDwOld = nDw;
    nUp = Eigen::VectorXd::Zero(NA * NY * NORB);
    nDw = Eigen::VectorXd::Zero(NA * NY * NORB);
    for (int idx=0; idx < NA * NY * NORB; idx++)
    {
        for (int idx_k=0; idx_k < NK; idx_k++)
            for (int idxEn=0; idxEn < NA * NY * NORB; idxEn++)
            {
                nUp(idx) += pow(
                  std::abs(KsolUp[idx_k].eigenvectors()(idx, idxEn)), 2)
                  * FermiDistUp(idx_k, idxEn);
                nDw(idx) += pow(
                  std::abs(KsolDw[idx_k].eigenvectors()(idx, idxEn)), 2)
                  * FermiDistDw(idx_k, idxEn);
            }
        nUp(idx) /= NK;
        nDw(idx) /= NK;
    }
}

void model_k_space::compute_grand_potential()
{
    double dH;
    double OmegaMF = 0.0;
    if (beta == beta_target)
    {
        if (beta_target < BETA_THRESHOLD)
        {
            dH = u * ( nUp.dot(nDw) - nUp.dot(nDwOld) - nDw.dot(nUpOld) );
            for (int idx_k=0; idx_k < NK; idx_k++)
                for (int idxEn=0; idxEn < NA * NY * NORB; idxEn++)
                {
                    OmegaMF -= ( log( 1 + exp( -beta_target * (
                      KsolUp[idx_k].eigenvalues()(idxEn) - mu ) ) ) +
                      log( 1 + exp( -beta_target * (
                      KsolDw[idx_k].eigenvalues()(idxEn) - mu ) ) ) );
                }
            OmegaMF = OmegaMF / beta_target / NK;
            energy = ( dH + OmegaMF) * abs_t0 / NY / NA;
        }
        else
        {
            dH = u * ( nUp.dot(nDw) - nUp.dot(nDwOld) - nDw.dot(nUpOld) );
            for (int idx_k=0; idx_k < NK; idx_k++)
                for (int idxEn=0; idxEn < NA * NY * NORB; idxEn++)
                {
                    OmegaMF += KsolUp[idx_k].eigenvalues()(idxEn)
                      * FermiDistUp(idx_k, idxEn) +
                      KsolDw[idx_k].eigenvalues()(idxEn)
                      * FermiDistDw(idx_k, idxEn);
                }
            OmegaMF /= NK;
            energy = ( dH + OmegaMF
                       - mu * (nUp.sum() + nDw.sum())) * abs_t0 / NY / NA;
        }
    }
    else
    {
        if (beta_target < BETA_THRESHOLD)
        {
            nUpAux = Eigen::VectorXd::Zero(NA * NY * NORB);
            nDwAux = Eigen::VectorXd::Zero(NA * NY * NORB);
            for (int idx=0; idx < NA * NY * NORB; idx++)
            {
                for (int idx_k=0; idx_k < NK; idx_k++)
                    for (int idxEn=0; idxEn < NA * NY * NORB; idxEn++)
                    {
                        nUpAux(idx) += pow(
                          std::abs(KsolUp[idx_k].eigenvectors()(idx, idxEn)), 2)
                          * 1/(1+exp(beta_target*(KsolUp[idx_k].eigenvalues()(idxEn)- mu)));
                        nDwAux(idx) += pow(
                          std::abs(KsolDw[idx_k].eigenvectors()(idx, idxEn)), 2)
                          * 1/(1+exp(beta_target*(KsolDw[idx_k].eigenvalues()(idxEn)- mu)));
                    }
                nUpAux(idx) /= NK;
                nDwAux(idx) /= NK;
            }
            dH = u * ( nUpAux.dot(nDwAux) - nUpAux.dot(nDwOld) - nDwAux.dot(nUpOld) );
            for (int idx_k=0; idx_k < NK; idx_k++)
                for (int idxEn=0; idxEn < NA * NY * NORB; idxEn++)
                {
                    OmegaMF -= ( log( 1 + exp( -beta_target * (
                      KsolUp[idx_k].eigenvalues()(idxEn) - mu ) ) ) +
                      log( 1 + exp( -beta_target * (
                      KsolDw[idx_k].eigenvalues()(idxEn) - mu ) ) ) );
                }
            OmegaMF = OmegaMF / beta_target / NK;
            energy = ( dH + OmegaMF) * abs_t0 / NY / NA;
        }
        else
        {
            nUpAux = Eigen::VectorXd::Zero(NA * NY * NORB);
            nDwAux = Eigen::VectorXd::Zero(NA * NY * NORB);
            FermiDistUp = Eigen::MatrixXd::Zero(NK, NA * NORB * NY);
            FermiDistDw = Eigen::MatrixXd::Zero(NK, NA * NORB * NY);
            for (int idx_k=0; idx_k < NK; idx_k++)
            {
                for (int idxEn=0; idxEn < NA * NY * NORB; idxEn++)
                {
                    if (KsolUp[idx_k].eigenvalues()(idxEn) < mu)
                    {
                        FermiDistUp(idx_k, idxEn) = 1;
                    }
                    else
                    {
                        if (KsolUp[idx_k].eigenvalues()(idxEn) == mu)
                        {
                            FermiDistUp(idx_k, idxEn) = 1/2;
                        }
                    }
                    if (KsolDw[idx_k].eigenvalues()(idxEn) < mu)
                    {
                        FermiDistDw(idx_k, idxEn) = 1;
                    }
                    else
                    {
                        if (KsolDw[idx_k].eigenvalues()(idxEn) == mu)
                        {
                            FermiDistDw(idx_k, idxEn) = 1/2;
                        }
                    }
                    OmegaMF += KsolUp[idx_k].eigenvalues()(idxEn)
                      * FermiDistUp(idx_k, idxEn) +
                      KsolDw[idx_k].eigenvalues()(idxEn)
                      * FermiDistDw(idx_k, idxEn);
                    for (int idx=0; idx < NA * NY * NORB; idx++)
                    {
                        nUpAux(idx) += pow(
                          std::abs(KsolUp[idx_k].eigenvectors()(idx, idxEn)), 2)
                          * FermiDistUp(idx_k, idxEn) / NK;
                        nDwAux(idx) += pow(
                          std::abs(KsolDw[idx_k].eigenvectors()(idx, idxEn)), 2)
                          * FermiDistDw(idx_k, idxEn) / NK;
                    }
                }
            }
            dH = u * ( nUpAux.dot(nDwAux) - nUpAux.dot(nDwOld) - nDwAux.dot(nUpOld) );
            OmegaMF /= NK;
            energy = ( dH + OmegaMF
                       - mu * (nUpAux.sum() + nDwAux.sum())) * abs_t0 / NY / NA;
        }
    }
}

void model_k_space::save_grand_potential(unsigned it)
{
    energies(it) = energy;
}

double model_k_space::grand_potential()
{
    return energy;
}

Eigen::VectorXd model_k_space::grand_potential_evol()
{
    return energies;
}

void model_k_space::anneal(unsigned it)
{
    if (beta < BETA_THRESHOLD)
    {
        if (beta < beta_target)
        {
            beta = BETA_START * pow(BETA_SPEED, it) ;
        }
        else
        {
            beta = beta_target;
        }
    }
    else
    {
        if (beta_target == BETA_THRESHOLD)
        {
            beta = BETA_THRESHOLD;
        }
        else
        {
            beta = beta_target;
        }
    }
}

void model_k_space::tolerance_check()
{
    deltaUp = 0.0; deltaDw = 0.0;
    double normUp = 0.0; double normDw = 0.0;
    for (int idx=0; idx < NA * NY * NORB; idx++)
    {
        deltaUp += pow(nUp(idx) - nUpOld(idx), 2);
        deltaDw += pow(nDw(idx) - nDwOld(idx), 2);
        normUp += pow(nUp(idx), 2);
        normDw += pow(nDw(idx), 2);
    }
    deltaUp = sqrt( deltaUp / normUp );
    deltaDw = sqrt( deltaDw / normDw );
}

void model_k_space::damp(unsigned it)
{
    // Introduces damping by varying the weight given to
    // the newly calculated fields.
    double factor = 1.2;
    double lbd = 0.5 / (factor * MAX_IT);
    if (it % DAMP_FREQ == 0)
    {
        nUp = ( 0.5 + lbd * it ) * nUp\
        + ( 0.5 - lbd * it) * nUpOld;
        nDw = ( 0.5 + lbd * it ) * nDw\
        + ( 0.5 - lbd * it) * nDwOld;
    }
}

bool model_k_space::loop_condition(unsigned it)
{
    // Verifies whether the main loop should keep going.
    if ( it < MAX_IT && (deltaUp > DELTA || deltaDw > DELTA) )
    {
        return true;
    }
    else
    {
        if (beta < beta_target)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}

Eigen::VectorXd model_k_space::UpField()
{
    return nUp;
}

Eigen::VectorXd model_k_space::DwField()
{
    return nDw;
}

double model_k_space::filling()
{
    return ( nUp.sum() + nDw.sum() ) / NA / NY / NORB;
}

void model_k_space::init_random(int seed)
{
    std::srand((unsigned int) seed);
    nUp = ( Eigen::VectorXd::Random(NA * NY * NORB)
      + Eigen::VectorXd::Constant(NA * NY * NORB, 1) ) * 0.5;
    nDw = ( Eigen::VectorXd::Random(NA * NY * NORB)
      + Eigen::VectorXd::Constant(NA * NY * NORB, 1) ) * 0.5;
}

void model_k_space::init_para(int seed)
{
    std::srand((unsigned int) seed);
    nUp = ( Eigen::VectorXd::Random(NA * NY * NORB)
      + Eigen::VectorXd::Constant(NA * NY * NORB, 1) ) * 0.5;
    nDw = nUp;
}

Eigen::Matrix<std::complex<double>, -1, -1> model_k_space::hopMat(int idx_k)
{
    return TB_HamiltonianUp[idx_k];
}

class model_real_space
{
    double t0;
    double abs_t0;
    double e0;
    double e1;
    double t1;
    double t2;
    double t11;
    double t12;
    double t22;
    double u;
    double mu;
    double beta;
    double beta_target;
    double energy;
    Eigen::VectorXd energies;
    Eigen::Matrix<std::complex<double>, -1, -1> TB_Hamiltonian;
    Eigen::VectorXd nUp;
    Eigen::VectorXd nDw;
    Eigen::VectorXd nUpAux;
    Eigen::VectorXd nDwAux;
    Eigen::VectorXd nUpOld;
    Eigen::VectorXd nDwOld;
    double deltaUp;
    double deltaDw;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix
      <std::complex<double>, -1, -1>> KsolUp;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix
        <std::complex<double>, -1, -1>> KsolDw;
    Eigen::VectorXd FermiDistUp;
    Eigen::VectorXd FermiDistDw;
public:
    model_real_space(int tmd, double on_site_interaction,
      double chem_pot, double beta_f)
    {
        t0 = -1;
        // Define model
        // TMD : 1 - MoS2, 2 - WS2, 3 - MoSe2, 4 - WSe2, 5 - MoTe2, 6 - WTe2
        if (tmd == 1) //  MoS2
        {abs_t0 = 0.184; e0 = 1.046 / abs_t0; e1 = 2.104 / abs_t0;
         t1 = 0.401 / abs_t0; t2 = 0.507 / abs_t0; t11 = 0.218 / abs_t0;
         t12 = 0.338 / abs_t0; t22 = 0.057 / abs_t0;}
        if (tmd == 2) //  WS2
        {abs_t0 = 0.206; e0 = 1.130 / abs_t0; e1 = 2.275 / abs_t0;
         t1 = 0.567 / abs_t0; t2 = 0.536 / abs_t0; t11 = 0.286 / abs_t0;
         t12 = 0.384 / abs_t0; t22 = -0.061 / abs_t0;}
        if (tmd == 3) //  MoSe2
        {abs_t0 = 0.188;   e0 = 0.919 / abs_t0; e1 = 2.065 / abs_t0;
         t1 = 0.317 / abs_t0; t2 = 0.456 / abs_t0; t11 = 0.211 / abs_t0;
         t12 = 0.290 / abs_t0; t22 = 0.130 / abs_t0;}
        if (tmd == 4) //  WSe2
        {abs_t0 = 0.207; e0 = 0.943 / abs_t0; e1 = 2.179 / abs_t0;
         t1 = 0.457 / abs_t0; t2 = 0.486 / abs_t0; t11 = 0.263 / abs_t0;
         t12 = 0.329 / abs_t0;t22 = 0.034 / abs_t0;}
        if (tmd == 5) //  MoTe2
        {abs_t0 = 0.169; e0 = 0.605 / abs_t0; e1 = 1.972 / abs_t0;
         t1 = 0.228 / abs_t0; t2 = 0.390 / abs_t0; t11 = 0.207 / abs_t0;
         t12 = 0.239 / abs_t0; t22 = 0.252 / abs_t0;}
        if (tmd == 6) //  WTe2
        {abs_t0 = 0.175; e0 = 0.606 / abs_t0; e1 = 2.102 / abs_t0;
         t1 = 0.342 / abs_t0; t2 = 0.410 / abs_t0; t11 = 0.233 / abs_t0;
         t12 = 0.270 / abs_t0; t22 = 0.190 / abs_t0;}
        TB_Hamiltonian = Eigen::MatrixXd(NORB * NX * NY, NORB * NX * NY);
        FermiDistUp = Eigen::VectorXd(NORB * NX * NY);
        FermiDistDw = Eigen::VectorXd(NORB * NX * NY);
        beta = BETA_START;
        beta_target = beta_f;
        mu = chem_pot;
        u = on_site_interaction;
        energy = 0.0;
        energies = Eigen::VectorXd(MAX_IT);
        deltaUp = DELTA + 1.0;
        deltaDw = DELTA + 1.0;
    }
    void TMDnanoribbon();
    void reset(double on_site_interaction, double chem_pot, double beta_f);
    void diagonalize();
    void fermi();
    void update();
    void compute_grand_potential();
    void save_grand_potential(unsigned it);
    double grand_potential();
    void anneal(unsigned it);
    void tolerance_check();
    void damp(unsigned it);
    bool loop_condition(unsigned it);
    Eigen::VectorXd UpField();
    Eigen::VectorXd DwField();
    Eigen::VectorXd grand_potential_evol();
    double filling();
    void init_random(int seed);
    void init_para(int seed);
    void init_dimer(int seed);
    Eigen::Matrix<std::complex<double>, -1, -1> matrix();
};

void model_real_space::TMDnanoribbon()
{
    node n(NORB);

    // Add hoppings
    n.add_hopping(0, 0, 0, 0, std::complex<double>(e0,0));
    n.add_hopping(1, 1, 0, 0, std::complex<double>(e1,0));
    n.add_hopping(2, 2, 0, 0, std::complex<double>(e1,0));
    // R1
    n.add_hopping(0, 0, 1, 0, std::complex<double>(t0,0));
    n.add_hopping(1, 1, 1, 0, std::complex<double>(t11,0));
    n.add_hopping(2, 2, 1, 0, std::complex<double>(t22,0));
    n.add_hopping(0, 1, 1, 0, std::complex<double>(t1,0));
    n.add_hopping(0, 2, 1, 0, std::complex<double>(t2,0));
    n.add_hopping(1, 2, 1, 0, std::complex<double>(t12,0));
    n.add_hopping(1, 0, 1, 0, std::complex<double>(-t1,0));
    n.add_hopping(2, 0, 1, 0, std::complex<double>(t2,0));
    n.add_hopping(2, 1, 1, 0, std::complex<double>(-t12,0));
    // R2
    n.add_hopping(0, 0, 1, -1, std::complex<double>(t0,0));
    n.add_hopping(1, 1, 1, -1, std::complex<double>((t11 + 3 * t22)/4,0));
    n.add_hopping(2, 2, 1, -1, std::complex<double>(( 3 * t11 + t22 ) / 4,0));
    n.add_hopping(0, 1, 1, -1, std::complex<double>(t1/2 - sqrt(3)*t2/2,0));
    n.add_hopping(0, 2, 1, -1, std::complex<double>(-sqrt(3)*t1/2 - t2/2,0));
    n.add_hopping(1, 2, 1, -1, std::complex<double>(sqrt(3)*(t22-t11)/4-t12,0));
    n.add_hopping(1, 0, 1, -1, std::complex<double>(-t1/2-sqrt(3)*t2/2,0));
    n.add_hopping(2, 0, 1, -1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    n.add_hopping(2, 1, 1, -1, std::complex<double>(sqrt(3)*(t22-t11)/4+t12,0));
    // R3
    n.add_hopping(0, 0, 0, -1, std::complex<double>(t0,0));
    n.add_hopping(1, 1, 0, -1, std::complex<double>(( t11 + 3 * t22 ) / 4,0));
    n.add_hopping(2, 2, 0, -1, std::complex<double>(( 3 * t11 + t22 ) / 4,0));
    n.add_hopping(0, 1, 0, -1, std::complex<double>(-t1/2+sqrt(3)*t2/2,0));
    n.add_hopping(0, 2, 0, -1, std::complex<double>(-sqrt(3)*t1/2-t2/2,0));
    n.add_hopping(1, 2, 0, -1, std::complex<double>(-sqrt(3)*(t22-t11)/4+t12,0));
    n.add_hopping(1, 0, 0, -1, std::complex<double>(t1/2+sqrt(3)*t2/2,0));
    n.add_hopping(2, 0, 0, -1, std::complex<double>(sqrt(3)*t1/2-t2/2,0));
    n.add_hopping(2, 1, 0, -1, std::complex<double>(-sqrt(3)*(t22-t11)/4-t12,0));

    TB_Hamiltonian = Eigen::MatrixXd::Zero(NORB * NX * NY, NORB * NX * NY);

    // Build the Tight-binding part of the Hamiltonian
    for (int x = 0; x < NX; x++ )
        for (int y = 0; y < NY; y++ )
          for (int orb = 0; orb < NORB; orb++ )
          {
              unsigned start_idx = orb + NORB * ( NX * y + x );
              for (unsigned neighbor_idx = 0;
                  neighbor_idx < n.NHoppings[orb]; neighbor_idx++)
              {
                  unsigned end_x =
                    ( x + n.x_idxs[orb].at(neighbor_idx) + NX ) % NX;
                  int end_y =
                    y + n.y_idxs[orb].at(neighbor_idx);
                  unsigned end_idx =
                    n.end_orbs[orb].at(neighbor_idx) + NORB
                    * ( NX * end_y + end_x );
                  if ( end_y >= 0 && end_y < NY )
                  {
                      TB_Hamiltonian(start_idx, end_idx)
                         = n.hoppings[orb].at(neighbor_idx);
                      TB_Hamiltonian(end_idx, start_idx)
                         = n.hoppings[orb].at(neighbor_idx);
                  }
              }
          }
}

void model_real_space::reset(double on_site_interaction, double chem_pot, double beta_f)
{
    FermiDistUp = Eigen::VectorXd(NORB * NX * NY);
    FermiDistDw = Eigen::VectorXd(NORB * NX * NY);
    beta = BETA_START;
    beta_target = beta_f;
    mu = chem_pot;
    u = on_site_interaction;
    energy = 0.0;
    deltaUp = DELTA + 1.0;
    deltaDw = DELTA + 1.0;
}

void model_real_space::diagonalize()
{
    KsolUp.compute(
      TB_Hamiltonian + u * Eigen::MatrixXd(nDw.asDiagonal()) );
    KsolDw.compute(
      TB_Hamiltonian + u * Eigen::MatrixXd(nUp.asDiagonal()) );
}

void model_real_space::fermi()
{
    //  Fermi distribution for the spectrum for each k, spectrum.
    //  If beta is greater or equal to a threshold, it returns the step function.
    FermiDistUp = Eigen::VectorXd::Zero(NORB * NX * NY);
    FermiDistDw = Eigen::VectorXd::Zero(NORB * NX * NY);
    for (int idxEn=0; idxEn < NX * NY * NORB; idxEn++)
    {
        if (beta >= BETA_THRESHOLD)
        {
            if (KsolUp.eigenvalues()(idxEn) < mu)
            {
                FermiDistUp(idxEn) = 1;
            }
            else
            {
                if (KsolUp.eigenvalues()(idxEn) == mu)
                {
                    FermiDistUp(idxEn) = 1/2;
                }
            }
            if (KsolDw.eigenvalues()(idxEn) < mu)
            {
                FermiDistDw(idxEn) = 1;
            }
            else
            {
                if (KsolDw.eigenvalues()(idxEn) == mu)
                {
                    FermiDistDw(idxEn) = 1/2;
                }
            }
        }
        else
        {
            FermiDistUp(idxEn)
            = 1/(1+exp(beta*(KsolUp.eigenvalues()(idxEn)- mu)));
            FermiDistDw(idxEn)
            = 1/(1+exp(beta*(KsolDw.eigenvalues()(idxEn)- mu)));
        }
    }
}

void model_real_space::update()
{
    nUpOld = nUp;
    nDwOld = nDw;
    nUp = Eigen::VectorXd::Zero(NX * NY * NORB);
    nDw = Eigen::VectorXd::Zero(NX * NY * NORB);
    for (int idx=0; idx < NX * NY * NORB; idx++)
    {
        for (int idxEn=0; idxEn < NX * NY * NORB; idxEn++)
        {
            nUp(idx) += pow(
              std::abs(KsolUp.eigenvectors()(idx, idxEn)), 2)
              * FermiDistUp(idxEn);
            nDw(idx) += pow(
              std::abs(KsolDw.eigenvectors()(idx, idxEn)), 2)
              * FermiDistDw(idxEn);
        }
    }
}

void model_real_space::compute_grand_potential()
{
    double dH;
    double OmegaMF = 0.0;
    if (beta == beta_target)
    {
        if (beta_target < BETA_THRESHOLD)
        {
            dH = u * ( nUp.dot(nDw) - nUp.dot(nDwOld) - nDw.dot(nUpOld) );
            for (int idxEn=0; idxEn < NX * NY * NORB; idxEn++)
            {
                OmegaMF -= ( log( 1 + exp( -beta_target * (
                  KsolUp.eigenvalues()(idxEn) - mu ) ) ) +
                  log( 1 + exp( -beta_target * (
                  KsolDw.eigenvalues()(idxEn) - mu ) ) ) );
            }
            OmegaMF = OmegaMF / beta_target;
            energy = ( dH + OmegaMF) * abs_t0 / NY / NX;
        }
        else
        {
            dH = u * ( nUp.dot(nDw) - nUp.dot(nDwOld) - nDw.dot(nUpOld) );
            for (int idxEn=0; idxEn < NX * NY * NORB; idxEn++)
            {
                OmegaMF += KsolUp.eigenvalues()(idxEn)
                  * FermiDistUp(idxEn) +
                  KsolDw.eigenvalues()(idxEn)
                  * FermiDistDw(idxEn);
            }
            energy = ( dH + OmegaMF
                       - mu * (nUp.sum() + nDw.sum())) * abs_t0 / NY / NX;
        }
    }
    else
    {
        if (beta_target < BETA_THRESHOLD)
        {
            nUpAux = Eigen::VectorXd::Zero(NX * NY * NORB);
            nDwAux = Eigen::VectorXd::Zero(NX * NY * NORB);
            for (int idx=0; idx < NX * NY * NORB; idx++)
            {
                for (int idxEn=0; idxEn < NX * NY * NORB; idxEn++)
                {
                    nUpAux(idx) += pow(
                      std::abs(KsolUp.eigenvectors()(idx, idxEn)), 2)
                      * 1/(1+exp(beta_target*(KsolUp.eigenvalues()(idxEn)- mu)));
                    nDwAux(idx) += pow(
                      std::abs(KsolDw.eigenvectors()(idx, idxEn)), 2)
                      * 1/(1+exp(beta_target*(KsolDw.eigenvalues()(idxEn)- mu)));
                }
            }
            dH = u * ( nUpAux.dot(nDwAux) - nUpAux.dot(nDwOld) - nDwAux.dot(nUpOld) );
            for (int idxEn=0; idxEn < NX * NY * NORB; idxEn++)
            {
                OmegaMF -= ( log( 1 + exp( -beta_target * (
                  KsolUp.eigenvalues()(idxEn) - mu ) ) ) +
                  log( 1 + exp( -beta_target * (
                  KsolDw.eigenvalues()(idxEn) - mu ) ) ) );
            }
            OmegaMF = OmegaMF / beta_target;
            energy = ( dH + OmegaMF) * abs_t0 / NY / NX;
        }
        else
        {
            nUpAux = Eigen::VectorXd::Zero(NX * NY * NORB);
            nDwAux = Eigen::VectorXd::Zero(NX * NY * NORB);
            FermiDistUp = Eigen::VectorXd::Zero(NX * NORB * NY);
            FermiDistDw = Eigen::VectorXd::Zero(NX * NORB * NY);
            for (int idxEn=0; idxEn < NX * NY * NORB; idxEn++)
            {
                if (KsolUp.eigenvalues()(idxEn) < mu)
                {
                    FermiDistUp(idxEn) = 1;
                }
                else
                {
                    if (KsolUp.eigenvalues()(idxEn) == mu)
                    {
                        FermiDistUp(idxEn) = 1/2;
                    }
                }
                if (KsolDw.eigenvalues()(idxEn) < mu)
                {
                    FermiDistDw(idxEn) = 1;
                }
                else
                {
                    if (KsolDw.eigenvalues()(idxEn) == mu)
                    {
                        FermiDistDw(idxEn) = 1/2;
                    }
                }
                OmegaMF += KsolUp.eigenvalues()(idxEn)
                  * FermiDistUp(idxEn) +
                  KsolDw.eigenvalues()(idxEn)
                  * FermiDistDw(idxEn);
                for (int idx=0; idx < NX * NY * NORB; idx++)
                {
                    nUpAux(idx) += pow(
                      std::abs(KsolUp.eigenvectors()(idx, idxEn)), 2)
                      * FermiDistUp(idxEn);
                    nDwAux(idx) += pow(
                      std::abs(KsolDw.eigenvectors()(idx, idxEn)), 2)
                      * FermiDistDw(idxEn);
                }
            }
            dH = u * ( nUpAux.dot(nDwAux) - nUpAux.dot(nDwOld) - nDwAux.dot(nUpOld) );
            energy = ( dH + OmegaMF
                       - mu * (nUpAux.sum() + nDwAux.sum())) * abs_t0 / NY / NX;
        }
    }
}

void model_real_space::save_grand_potential(unsigned it)
{
    energies(it) = energy;
}

double model_real_space::grand_potential()
{
    return energy;
}

Eigen::VectorXd model_real_space::grand_potential_evol()
{
    return energies;
}

void model_real_space::anneal(unsigned it)
{
    if (beta < BETA_THRESHOLD)
    {
        if (beta < beta_target)
        {
            beta = BETA_START * pow(BETA_SPEED, it) ;
        }
        else
        {
            beta = beta_target;
        }
    }
    else
    {
        if (beta_target == BETA_THRESHOLD)
        {
            beta = BETA_THRESHOLD;
        }
        else
        {
            beta = beta_target;
        }
    }
}

void model_real_space::tolerance_check()
{
    deltaUp = 0.0; deltaDw = 0.0;
    double normUp = 0.0; double normDw = 0.0;
    for (int idx=0; idx < NX * NY * NORB; idx++)
    {
        deltaUp += pow(nUp(idx) - nUpOld(idx), 2);
        deltaDw += pow(nDw(idx) - nDwOld(idx), 2);
        normUp += pow(nUp(idx), 2);
        normDw += pow(nDw(idx), 2);
    }
    deltaUp = sqrt( deltaUp / normUp );
    deltaDw = sqrt( deltaDw / normDw );
}

void model_real_space::damp(unsigned it)
{
    // Introduces damping by varying the weight given to
    // the newly calculated fields.
    double factor = 1.2;
    double lbda = 0.5 / (factor * MAX_IT);
    if (it % DAMP_FREQ == 0)
    {
        nUp = ( 0.5 + lbda * it ) * nUp\
        + ( 0.5 - lbda * it) * nUpOld;
        nDw = ( 0.5 + lbda * it ) * nDw\
        + ( 0.5 - lbda * it) * nDwOld;
    }
}

bool model_real_space::loop_condition(unsigned it)
{
    // Verifies whether the main loop should keep going.
    if ( it < MAX_IT && (deltaUp > DELTA || deltaDw > DELTA) )
    {
        return true;
    }
    else
    {
        if (beta < beta_target)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}

Eigen::VectorXd model_real_space::UpField()
{
    return nUp;
}

Eigen::VectorXd model_real_space::DwField()
{
    return nDw;
}

double model_real_space::filling()
{
    return ( nUp.sum() + nDw.sum() ) / NX / NY / NORB;
}

void model_real_space::init_random(int seed)
{
    std::srand((unsigned int) seed);
    nUp = ( Eigen::VectorXd::Random(NX * NY * NORB)
      + Eigen::VectorXd::Constant(NX * NY * NORB, 1) ) * 0.5;
    nDw = ( Eigen::VectorXd::Random(NX * NY * NORB)
      + Eigen::VectorXd::Constant(NX * NY * NORB, 1) ) * 0.5;
    double nSum = nUp.sum() + nDw.sum();
    nUp = nUp / nSum * 2 / 3 * NORB * NX * NY;
    nDw = nDw / nSum * 2 / 3 * NORB * NX * NY;
}

void model_real_space::init_para(int seed)
{
    std::srand((unsigned int) seed);
    nUp = ( Eigen::VectorXd::Random(NX * NY * NORB)
      + Eigen::VectorXd::Constant(NX * NY * NORB, 1) ) * 0.5;
    nDw = nUp;
    double nSum = 2 * nUp.sum();
    nUp = nUp / nSum * 2 / 3 * NORB * NX * NY;
    nDw = nDw / nSum * 2 / 3 * NORB * NX * NY;
}

void model_real_space::init_dimer(int seed)
{
    std::srand((unsigned int) seed);
    nUp = ( Eigen::VectorXd::Random(NX * NY * NORB)
      + Eigen::VectorXd::Constant(NX * NY * NORB, 1) ) * 0.5;
    nDw = nUp;
    int x; int y; unsigned idx;
    for (int xId = 0; xId < (int)(NX / 4) - 1; xId++)
        for (int orb = 0; orb < NORB; orb++ )
        {
            y = 0;
            x = 4 * xId;
            idx = orb + NORB * ( NX * y + x );
            nUp(idx) = 1;
            nDw(idx) = 0;
            x = 4 * xId + 1;
            idx = orb + NORB * ( NX * y + x );
            nUp(idx) = 1;
            nDw(idx) = 0;
            x = 4 * xId + 2;
            idx = orb + NORB * ( NX * y + x );
            nUp(idx) = 0;
            nDw(idx) = 1;
            x = 4 * xId + 3;
            idx = orb + NORB * ( NX * y + x );
            nUp(idx) = 0;
            nDw(idx) = 1;
            y = NY - 1;
            x = 4 * xId;
            idx = orb + NORB * ( NX * y + x );
            nUp(idx) = 1;
            nDw(idx) = 0;
            x = 4 * xId + 1;
            idx = orb + NORB * ( NX * y + x );
            nUp(idx) = 1;
            nDw(idx) = 0;
            x = 4 * xId + 2;
            idx = orb + NORB * ( NX * y + x );
            nUp(idx) = 0;
            nDw(idx) = 1;
            x = 4 * xId + 3;
            idx = orb + NORB * ( NX * y + x );
            nUp(idx) = 0;
            nDw(idx) = 1;
        }
    double nSum = nUp.sum() + nDw.sum();
    nUp = nUp / nSum * 2 / 3 * NORB * NX * NY;
    nDw = nDw / nSum * 2 / 3 * NORB * NX * NY;
}

Eigen::Matrix<std::complex<double>, -1, -1> model_real_space::matrix()
{
    return TB_Hamiltonian;
}
