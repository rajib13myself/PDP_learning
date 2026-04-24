#!/bin/bash

# =========================
# Weak Scaling Experiment
# =========================

BASE_INPUT="/proj/uppmax2026-1-92/A2/input1000000.txt"
STEPS=50

OUTPUT_DIR="./weak_results"
mkdir -p $OUTPUT_DIR

echo "Processes,N,Steps,Time" > $OUTPUT_DIR/weak_results.csv

# mapping: P -> input file
declare -A INPUTS
INPUTS[1]="/proj/uppmax2026-1-92/A2/input1000000.txt"
INPUTS[2]="/proj/uppmax2026-1-92/A2/input1000000.txt"
INPUTS[4]="/proj/uppmax2026-1-92/A2/input1000000.txt"
INPUTS[8]="/proj/uppmax2026-1-92/A2/input1000000.txt"

for P in 1 2 4 8
do
    echo "Running weak scaling with P=$P"

    INPUT=${INPUTS[$P]}
    OUT_FILE="$OUTPUT_DIR/output100K_P${P}.txt"

    RESULT=$(mpirun --bind-to none -n $P ./stencil $INPUT $OUT_FILE $STEPS)

    echo "$RESULT"

    echo "$RESULT" >> $OUTPUT_DIR/weak_results.csv

done

echo "Weak scaling done. Results saved in $OUTPUT_DIR/weak_results.csv"