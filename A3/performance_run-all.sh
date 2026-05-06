BASE=/proj/uppmax2026-1-92/A3/inputs
OUT=//proj/uppmax2026-1-92/rajib/PDP_Course/PDP_learning/A3/A3_output

mkdir -p $OUT

PROCS=(1 2 4 8 16 32)
PIVOTS=(0 1 2)

########################################
# STRONG SCALING
########################################
INPUT_STRONG=input1000000000.txt

for p in "${PROCS[@]}"; do
  for pivot in "${PIVOTS[@]}"; do

    srun -n $p ./quicksort \
    $BASE/$INPUT_STRONG dummy.txt $pivot \
    > $OUT/strong_p${p}_pivot${pivot}.txt

  done
done

########################################
# WEAK SCALING (adjust to available files)
########################################
PROCS_WEAK=(1 2 4)
FILES_WEAK=(input125000000.txt input250000000.txt input1000000000.txt)

for i in ${!PROCS_WEAK[@]}; do
  p=${PROCS_WEAK[$i]}
  file=${FILES_WEAK[$i]}

  for pivot in "${PIVOTS[@]}"; do

    srun -n $p ./quicksort \
    $BASE/$file dummy.txt $pivot \
    > $OUT/weak_p${p}_pivot${pivot}.txt

  done
done

########################################
# DESCENDING INPUT
########################################
BASE_Back=/proj/uppmax2026-1-92/A3/inputs/backwards
INPUT_BACK=input_backwards1000000000.txt

for p in "${PROCS[@]}"; do
  for pivot in "${PIVOTS[@]}"; do

    srun -n $p ./quicksort \
    $BASE_Back/$INPUT_BACK dummy.txt $pivot \
    > $OUT/back_p${p}_pivot${pivot}.txt

  done
done