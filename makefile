# MF TMD NANORIBBON - Iterative Solution of a minimal MF Hubbard model of a
# 										transition metal dichalcogenide nanoribbon
# Created by Francisco Brito April 16, 2019

#	DEFAULT PARAMETERS

# Number of threads
nthreads=4
# Number of k-points
nk=512
# Number of atoms in the periodic cell
na=4
# Number of columns in the ribbon
nx=12
# Number of rows in the ribbon
ny=5
# Inverse temperature at which the annealing process starts
beta_start=0.01
# Speed at which the inverse temperature increases in the annealing process
beta_speed=1.1
# Inverse temperature at which T becomes 0 in the annealing process
beta_threshold=100
# Number of orbitals in the tight-binding model
norb=3
# Tolerance given for the convergence of the self-consistent field
delta=0.000001
# Frequency of damping the self-consistent fields
damp_freq=1
# Maximum number of iterations
max_it=500
# Source file
source=solve_kspace
# Linux vs Mac
sys=0

# Set parameters of the solver here.
ifeq ($(sys),0)
 CXX = g++-8 -DNTH=$(nthreads) -DNK=$(nk) -DNA=$(na) -DNX=$(nx) -DNY=$(ny)\
  -DNORB=$(norb) -DBETA_START=$(beta_start) -DBETA_SPEED=$(beta_speed)\
  -DBETA_THRESHOLD=$(beta_threshold) -DDELTA=$(delta) -DDAMP_FREQ=$(damp_freq)\
  -DMAX_IT=$(max_it) -fopenmp
endif
ifeq ($(sys),1)
 CXX = g++ -DNTH=$(nthreads) -DNK=$(nk) -DNA=$(na) -DNX=$(nx) -DNY=$(ny)\
  -DNORB=$(norb) -DBETA_START=$(beta_start) -DBETA_SPEED=$(beta_speed)\
  -DBETA_THRESHOLD=$(beta_threshold) -DDELTA=$(delta) -DDAMP_FREQ=$(damp_freq)\
  -DMAX_IT=$(max_it) -fopenmp
endif


include_dir=./includes

CXXFLAGS = -Wall -g -O3 -std=c++11 -I$(include_dir)

solver: src/$(source).o
	@echo ""
	@echo "		MF TMD NANORIBBON - Iterative Solution of a minimal MF Hubbard\
	 model of"
	@echo "		a transition metal dichalcogenide nanoribbon"
	@echo ""
	@echo "		Created by Francisco Brito (2019)"
	@echo ""
	@echo ""
	@echo "		The code has compiled successfully. To change the number of k-points, \
	number of atoms in the periodic cell, transverse length,"
	@echo "		number of orbitals, \
	starting inverse temperature, annealing speed or T=0 threshold, run"
	@echo ""
	@echo "make clean"
	@echo ""
	@echo "make nk=<Number of k-points> na=<Number of atoms in the periodic cell> \
	nx=<Longitudinal length> ny=<Transverse length> norb=<Number of orbitals>"
	@echo "beta_start=<Starting inverse \
	temperature> beta_speed=<Annealing speed> beta_threshold=<Inv. temp. at \
	at which T is set to 0 in the annealing process>"
	@echo "delta=<Tolerance for the convergence of the self-consistent fields> \
	damp_freq=<Frequency of damping the s.c. fields> max_it=<max. n. of iterations>"
	@echo "nthreads=<Number of threads>"
	@echo ""
	@echo "		To solve the MF model, simply type ./solver followed by its arguments:"
	@echo ""
	@echo "./solver <TMD> <U> <BETA> <MU> <SEED> <INIT_COND>"
	@echo ""
	@echo "->TMD=1,2,...6 (MoS2, WS2, MoSe2, WSe2, MoTe2, WTe2)"
	@echo ""
	@echo "->U, BETA, MU in units of |t0|"
	@echo ""
	@echo "->INIT_COND is chosen from the following list of initial conditions"
	@echo ""
	@echo "(1)	Random"
	@echo ""
	@echo "(2) 	Paramagnetic"
	@echo ""
	$(CXX) $(CXXFLAGS) -o solver src/$(source).o

src/$(source).o: src/$(source).cpp $(include_dir)/model.hpp\
	$(include_dir)/aux.hpp

clean:
	rm -f solver src/*.o
