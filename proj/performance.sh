#!/bin/bash

make clean
make

rm -f results.csv

echo "processes,N,iteration,time" > results.csv

# strong scaling 
N = 1000000

for p in 1 2 4 8
do
    echo "Running strong scaling with $p processes"
    mpirun -n $p ./power $N
done