#!/bin/bash
cd $HOME/isaacsim
V=$HOME/warehouse_amr/recon/v3
D="$V/data_s31 $V/data_s32 $V/data_s33 $V/data_s34 $V/data_s35 $V/data_s36 $V/data_s37"
PY=~/.venvs/recon/bin/python
R=tools/warehouse_amr/recon.py
RECIPE="--depth stereo --depth-near 4 --depth-max 8 --far-voxel 0.04 --color-max 20 --lidar-fuse --free-space"
run() { name=$1; shift; mkdir -p $V/$name; $PY $R $D "$@" --out $V/$name > $V/$name/log.txt 2>&1; echo "[run_all] $name exit $? $(date +%T)" >> $V/run_all.log; }
run out_nomast --poses slam_pr --slam-cache $V/slam_pr_poses.npy $RECIPE --stereo-suffix _sl --no-mast --procs 20 &
run out_kidnap --poses slam_kidnap --slam-cache $V/slam_kidnap_poses.npy $RECIPE --stereo-suffix _sl --procs 20 &
run out_priors --poses slam --slam-cache $V/slam_poses.npy $RECIPE --stereo-suffix _sl --procs 20 &
wait
