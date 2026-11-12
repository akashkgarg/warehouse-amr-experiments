#!/bin/bash
# Stereo depth for every v2 pair, re-scanning for new captures until the collectors exit.
cd /home/builder/isaacsim
DIRS=$(ls -d /home/builder/warehouse_amr/recon/v2/data_s*)
while true; do
  running=$(pgrep -f collect_recon.py | wc -l)
  pids=()
  k=0
  for g in 2 3 4 5 6 7; do
    for p in 0 1; do
      CUDA_VISIBLE_DEVICES=$g ~/.venvs/stereo/bin/python tools/warehouse_amr/stereo_depth.py $DIRS --iters 12 \
        --shard $k/12 2>&1 | grep --line-buffered "\[stereo\]" &
      pids+=($!)
      k=$((k + 1))
    done
  done
  wait "${pids[@]}"
  echo "[stereo-loop] pass done at $(date +%T), collectors running: $running"
  [ "$running" -eq 0 ] && break
  sleep 60
done
echo "[stereo-loop] finished"
