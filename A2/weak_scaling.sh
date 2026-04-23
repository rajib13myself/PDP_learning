#!/bin/bash

# Weak scaling experiment
# Input size grows with number of processes

OUTPUT_FILE=weak_results.txt
STEPS=50

echo "Weak Scaling Results" > $OUTPUT_FILE
echo "====================" >> $OUTPUT_FILE

for p in 1 2 4 8
do
  #INPUT_SIZE=$((120 * p))
  INPUT_FILE="/proj/uppmax2026-1-92/A2/input120.txt"

  echo "Processes: $p Input: $INPUT_FILE"
  echo "Processes: $p Input: $INPUT_FILE" >> $OUTPUT_FILE

  mpirun --bind-to none -np $p ./stencil $INPUT_FILE output.txt $STEPS >> $OUTPUT_FILE

done