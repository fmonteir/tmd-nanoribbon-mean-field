#!/bin/bash

#   This script saves the results in temp-data to examples/data

#   NO ARGUMENTS

U=$(awk 'BEGIN{FS=","; RS=""} {print $2}' temp-data/parameters.csv)
BETA=$(awk 'BEGIN{FS=","; RS=""} {print $4}' temp-data/parameters.csv)
MU=$(awk 'BEGIN{FS=","; RS=""} {print $6}' temp-data/parameters.csv)
NX=$(awk 'BEGIN{FS=","; RS=""} {print $14}' temp-data/parameters.csv)
NY=$(awk 'BEGIN{FS=","; RS=""} {print $16}' temp-data/parameters.csv)

mkdir examples/data
mkdir examples/data/real-space/
mkdir examples/data/real-space/NA$NX-NY$NY
mkdir examples/data/real-space/NA$NX-NY$NY/U$U-BETA$BETA-MU$MU
mkdir examples/plots
mkdir examples/plots/real-space/
mkdir examples/plots/real-space/NA$NX-NY$NY
mkdir examples/plots/real-space/NA$NX-NY$NY/U$U-BETA$BETA-MU$MU
cp -r temp-data/* examples/data/real-space/NA$NX-NY$NY/U$U-BETA$BETA-MU$MU
rm temp-data/*
