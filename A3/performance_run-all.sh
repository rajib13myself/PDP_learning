#!/bin/bash
#SBATCH -A uppmax2026-1-92        # Project account (FIXES your error)
#SBATCH -J quicksort_benchmark    # Job name
#SBATCH -t 02:00:00               # Time limit (adjust if needed)
#SBATCH -n 32                     # Max number of tasks you'll use
#SBATCH --output=slurm_%j.out     # Stdout log
#SBATCH --error=slurm_%j.err      # Stderr log

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

########################################
# WEAK SCALING
########################################
PROCS_WEAK=(1 2 4)
FILES_WEAK=(input125000000.txt input250000000.txt input1000000000.txt)

echo "Running weak scaling..."

for i in "${!PROCS_WEAK[@]}"; do
  p=${PROCS_WEAK[$i]}
  file=${FILES_WEAK[$i]}

  for pivot in "${PIVOTS[@]}"; do

    echo "Weak: p=$p file=$file pivot=$pivot"

    srun -n "$p" ./quicksort \
      "$BASE/$file" dummy.txt "$pivot" \
      > "$OUT/weak_p${p}_pivot${pivot}.txt"

  done
done

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

echo "All runs completed."