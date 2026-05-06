#!/bin/bash
#SBATCH -A uppmax2026-1-92
#SBATCH -J quicksort_benchmark
#SBATCH -t 00:15:00
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Load MPI module (adjust if needed on your cluster)
module load openmpi

# Paths
BASE=/proj/uppmax2026-1-92/A3/inputs
BASE_BACK=/proj/uppmax2026-1-92/A3/inputs/backwards
OUT=/proj/uppmax2026-1-92/rajib/PDP_Course/PDP_learning/A3/A3_output

mkdir -p "$OUT"

# Parameters
PROCS=(1 2 4 8 16 32)
PIVOTS=(0 1 2)

########################################
# DESCENDING INPUT
########################################
INPUT_BACK=input_backwards1000000000.txt

echo "Running descending input..."

for p in "${PROCS[@]}"; do
  for pivot in "${PIVOTS[@]}"; do

    echo "Backwards: p=$p pivot=$pivot"

    srun -n "$p" ./quicksort \
      "$BASE_BACK/$INPUT_BACK" dummy.txt "$pivot" \
      > "$OUT/back_p${p}_pivot${pivot}.txt"

  done
done

echo "All run for Backward."