#!/bin/bash
# v3 reconstructions + ablations, once stereo depth and the SLAM caches exist.
cd $HOME/isaacsim
V=$HOME/warehouse_amr/recon/v3
D="$V/data_s31 $V/data_s32 $V/data_s33 $V/data_s34 $V/data_s35 $V/data_s36 $V/data_s37"
PY=~/.venvs/recon/bin/python
R=tools/warehouse_amr/recon.py
RECIPE="--depth stereo --depth-near 4 --depth-max 8 --far-voxel 0.04 --color-max 20 --lidar-fuse --free-space"
while pgrep -f stereo_all.sh > /dev/null || pgrep -f v3_slam.py > /dev/null; do sleep 60; done
echo "[run_all] inputs ready $(date +%T)"
run() { name=$1; shift; mkdir -p $V/$name; $PY $R $D "$@" --out $V/$name > $V/$name/log.txt 2>&1; echo "[run_all] $name exit $? $(date +%T)"; }
run out_final --poses slam_pr --slam-cache $V/slam_pr_poses.npy $RECIPE --stereo-suffix _sl --procs 40 &
run out_final_gt --poses gt $RECIPE --stereo-suffix _sl --procs 40 &
run out_plain --poses slam_pr --slam-cache $V/slam_pr_poses.npy $RECIPE --stereo-suffix _s --procs 40 &
wait
run out_nomast --poses slam_pr --slam-cache $V/slam_pr_poses.npy $RECIPE --stereo-suffix _sl --no-mast --procs 40 &
run out_kidnap --poses slam_kidnap --slam-cache $V/slam_kidnap_poses.npy $RECIPE --stereo-suffix _sl --procs 40 &
run out_priors --poses slam --slam-cache $V/slam_poses.npy $RECIPE --stereo-suffix _sl --procs 40 &
wait
echo "[run_all] all done $(date +%T)"
