#!/bin/bash

# =========================
# Weak Scaling Experiment
# =========================

BASE_INPUT="../test_data/input100k.txt"
STEPS=50

OUTPUT_DIR="./weak_results"
mkdir -p $OUTPUT_DIR

echo "Processes,N,Steps,Time" > $OUTPUT_DIR/weak_results.csv

# mapping: P -> input file
declare -A INPUTS
INPUTS[1]="../test_data/input100k.txt"
INPUTS[2]="../test_data/input200k.txt"
INPUTS[4]="../test_data/input400k.txt"
INPUTS[8]="../test_data/input800k.txt"

for P in 1 2 4 8
do
    echo "Running weak scaling with P=$P"

    INPUT=${INPUTS[$P]}
    OUT_FILE="$OUTPUT_DIR/output_P${P}.txt"

    RESULT=$(mpirun --bind-to none -n $P ./stencil $INPUT $OUT_FILE $STEPS)

    echo "$RESULT"

    echo "$RESULT" >> $OUTPUT_DIR/weak_results.csv

done

echo "Weak scaling done. Results saved in $OUTPUT_DIR/weak_results.csv"