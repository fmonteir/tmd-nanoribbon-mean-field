#!/bin/bash

#   This script saves the results in temp-data to examples/data

#   NO ARGUMENTS

U=$(awk 'BEGIN{FS=","; RS=""} {print $2}' temp-data/parameters.csv)
BETA=$(awk 'BEGIN{FS=","; RS=""} {print $4}' temp-data/parameters.csv)
MU=$(awk 'BEGIN{FS=","; RS=""} {print $6}' temp-data/parameters.csv)
# SEED=$(awk 'BEGIN{FS=","; RS=""} {print $8}' temp-data/parameters.csv)
# FILLING=$(awk 'BEGIN{FS=","; RS=""} {print $10}' temp-data/parameters.csv)
# FINAL_GRAND_POT=$(awk 'BEGIN{FS=","; RS=""} {print $12}' temp-data/parameters.csv)
# FINAL_IT=$(awk 'BEGIN{FS=","; RS=""} {print $14}' temp-data/parameters.csv)

mkdir examples/data
mkdir examples/data/U$U-BETA$BETA-MU$MU
mkdir examples/plots
mkdir examples/plots/U$U-BETA$BETA-MU$MU
cp -r temp-data/* examples/data/U$U-BETA$BETA-MU$MU
rm temp-data/*
