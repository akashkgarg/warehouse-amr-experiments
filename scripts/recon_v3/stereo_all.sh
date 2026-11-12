#!/bin/bash
# Stereo depth for every v3 pair: left-right checked (_sl) and plain (_s) from one pass, 2 processes per GPU.
cd /home/builder/isaacsim
DIRS=$(ls -d /home/builder/warehouse_amr/recon/v3/data_s*)
k=0
for g in 0 2 3 4 5 6 7; do
  for p in 0 1; do
    CUDA_VISIBLE_DEVICES=$g ~/.venvs/stereo/bin/python tools/warehouse_amr/stereo_depth.py $DIRS --iters 12 \
      --lr --lr-tol 1.0 --suffix _sl --plain-suffix _s --shard $k/14 2>&1 | grep --line-buffered "\[stereo\]" &
    k=$((k + 1))
  done
done
wait
echo "[stereo-all] finished $(date +%T)"
