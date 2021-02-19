#!/usr/bin/env bash
set -ex
export CUDA_VISIBLE_DEVICES=

DATETIME=$(date "+%Y%m%d-%H%M%S")
LOG_DIR=log/${DATETIME}
ITERS=20000
mkdir -p $LOG_DIR
submit(){
    sbatch \
        --partition=short \
        --nodes=1 \
        --time=5:00:00 \
        --job-name=annealing_gmm\
        --mem=8Gb \
        --cpus-per-task 1 \
        --output="$LOG_DIR/$2" \
        --wrap="$1"
}

run_methods(){
    # AVO
    submit "python main.py \
      --seed=$1\
      --num_targets=$2\
      --objective=nvo_avo\
      --iterations=$ITERS"
    # NVI
    submit "python main.py \
      --seed=$1\
      --num_targets=$2\
      --objective=nvo_rkl\
      --iterations=$ITERS"
    # NVIR
    submit "python main.py \
      --seed=$1\
      --num_targets=$2\
      --objective=nvo_rkl\
      --resample=True\
      --iterations=$ITERS"
    # NVI*
    submit "python main.py \
      --seed=$1\
      --num_targets=$2\
      --objective=nvo_rkl\
      --iterations=$ITERS \
      --optimize_path=True"
    # NVIR*
    submit "python main.py \
      --seed=$1\
      --num_targets=$2\
      --objective=nvo_rkl\
      --iterations=$ITERS \
      --resample=True\
      --optimize_path=True"
}

for seed in 0 1 2 3 4 5 6 7 8 9; do
  for num_targets in 2 4 6 8; do
        run_methods $seed $num_targets
  done
done
