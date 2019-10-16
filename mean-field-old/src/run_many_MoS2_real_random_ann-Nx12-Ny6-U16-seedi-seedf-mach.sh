#PBS -S /bin/bash
#PBS -V
#PBS -j oe
#PBS -l walltime=01:00:00


for i in $(eval echo {$1..$2})
do
	qsub -q batch -l nodes=$3.grid.fe.up.pt:ppn=1 MoS2_real_random_ann-Nx-Ny-U-seed.sh -F "12 6 16 $i"
	sleep 1s
done
