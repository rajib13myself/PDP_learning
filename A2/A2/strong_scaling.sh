#!/bin/bash

# Strong scaling experiment
# Fixed input size, increasing MPI processes

OUTPUT_FILE=strong_results.txt
INPUT_FILE="/proj/uppmax2026-1-92/A2/input120.txt"
STEPS=100

echo "Strong Scaling Results" > $OUTPUT_FILE
echo "======================" >> $OUTPUT_FILE

for p in 1 2 4 8
do
  echo "Processes: $p"
  echo "Processes: $p" >> $OUTPUT_FILE

  mpirun --bind-to none -np $p ./stencil $INPUT_FILE output.txt $STEPS >> $OUTPUT_FILE

done