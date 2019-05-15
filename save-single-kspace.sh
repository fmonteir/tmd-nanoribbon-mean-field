#!/bin/bash

#   This script saves the results in temp-data to examples/data and the plots
#   to examples/plots

#   NO ARGUMENTS

U=$(awk 'BEGIN{FS=","; RS=""} {print $2}' temp-data/parameters.csv)
BETA=$(awk 'BEGIN{FS=","; RS=""} {print $4}' temp-data/parameters.csv)
MU=$(awk 'BEGIN{FS=","; RS=""} {print $6}' temp-data/parameters.csv)
NA=$(awk 'BEGIN{FS=","; RS=""} {print $16}' temp-data/parameters.csv)
NY=$(awk 'BEGIN{FS=","; RS=""} {print $18}' temp-data/parameters.csv)


mkdir examples/data
mkdir examples/data/kspace/
mkdir examples/data/kspace/NA$NA-NY$NY
mkdir examples/data/kspace/NA$NA-NY$NY/U$U-BETA$BETA-MU$MU
mkdir examples/plots
mkdir examples/plots/kspace
mkdir examples/plots/kspace/NA$NA-NY$NY
mkdir examples/plots/kspace/NA$NA-NY$NY/U$U-BETA$BETA-MU$MU
cp -r temp-data/* examples/data/kspace/NA$NA-NY$NY/U$U-BETA$BETA-MU$MU
rm temp-data/*
