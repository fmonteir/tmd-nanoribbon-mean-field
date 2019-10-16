#PBS -S /bin/bash
#PBS -V
#PBS -j oe
#PBS -l walltime=02:00:00


for i in $(eval echo {$1..$2})
do
	qsub -q batch -l nodes=$3.grid.fe.up.pt:ppn=1 MoS2_1sub_random_ann-W-U-seed.sh -F "32 20 $i"
	sleep 1s
done
