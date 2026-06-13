#!/bin/bash
#SBATCH -A uppmax2026-1-92
#SBATCH -J quicksort_benchmark
#SBATCH -t 01:15:00
#SBATCH -n 32
#SBATCH --mem=64G
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Load MPI module (adjust if needed on your cluster)
module load openmpi

# Paths
BASE=/proj/uppmax2026-1-92/A3/inputs
//BASE_BACK=/proj/uppmax2026-1-92/A3/inputs/backwards
OUT=/proj/uppmax2026-1-92/uppmax2026-1-92/rajib/PDP_Course/PDP_learning/A3/A3_output

mkdir -p "$OUT"

# Parameters
PROCS=(1 2 4 8 16 32)
PIVOTS=(0 1 2)

########################################
# STRONG SCALING
########################################
INPUT_STRONG=input1000000000.txt

echo "Running strong scaling..."

for p in "${PROCS[@]}"; do
  for pivot in "${PIVOTS[@]}"; do

    echo "Strong: p=$p pivot=$pivot"

    srun -n "$p" ./quicksort \
      "$BASE/$INPUT_STRONG" dummy.txt "$pivot" \
      > "$OUT/strong_p${p}_pivot${pivot}.txt"

  done
done

echo "All run for Strong Scaling."
