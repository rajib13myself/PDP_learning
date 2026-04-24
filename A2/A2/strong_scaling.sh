#!/bin/bash

# =========================
# Strong Scaling Experiment
# =========================

INPUT="../test_data/input1M.txt"   # change this
STEPS=50

OUTPUT_DIR="./strong_results"
mkdir -p $OUTPUT_DIR

echo "Processes,N,Steps,Time" > $OUTPUT_DIR/strong_results.csv

for P in 1 2 4 8
do
    echo "Running strong scaling with P=$P"

    OUT_FILE="$OUTPUT_DIR/output_P${P}.txt"

    RESULT=$(mpirun --bind-to none -n $P ./stencil $INPUT $OUT_FILE $STEPS)

    echo "$RESULT"

    # append to CSV
    echo "$RESULT" >> $OUTPUT_DIR/strong_results.csv

done

echo "Strong scaling done. Results saved in $OUTPUT_DIR/strong_results.csv"