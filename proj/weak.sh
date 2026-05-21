#!/bin/bash

make clean
make

rm -f weak.csv

echo "processes,N,iterations,time" > weak.csv

BASE=250000

for p in 1 2 4 8
do
    N=$((BASE * p))

    echo "Running weak scaling with $p processes and N=$N"

    mpirun -n $p ./power $N
done