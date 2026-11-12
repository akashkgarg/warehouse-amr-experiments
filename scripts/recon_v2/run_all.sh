#!/bin/bash
# v2 reconstructions + ablations, launched once stereo depth is complete.
cd $HOME/isaacsim
V2=$HOME/warehouse_amr/recon/v2
D="$V2/data_s21 $V2/data_s22 $V2/data_s23 $V2/data_s24 $V2/data_s25 $V2/data_s26 $V2/data_s27"
PY=~/.venvs/recon/bin/python
R=tools/warehouse_amr/recon.py
while pgrep -f stereo_loop.sh > /dev/null; do sleep 60; done
echo "[run_all] stereo done $(date +%T)"
STEREO="--depth stereo --depth-max 6 --color-max 20"
run() { name=$1; shift; mkdir -p $V2/$name; $PY $R $D "$@" --out $V2/$name > $V2/$name/log.txt 2>&1; echo "[run_all] $name exit $? $(date +%T)"; }
# Wave 1 (the headline run writes the SLAM cache the other SLAM runs reuse).
run out_slam_stereo_lidar --poses slam $STEREO --lidar-fuse --slam-cache $V2/slam_poses.npy --procs 36 &
run out_gt_gt_lidar --poses gt --depth gt --lidar-fuse --procs 32 &
run out_gt_gt --poses gt --depth gt --procs 32 &
run out_est_stereo_lidar --poses est $STEREO --lidar-fuse --procs 32 &
run out_slamnp_stereo_lidar --poses slam_noprior $STEREO --lidar-fuse --slam-cache $V2/slam_noprior_poses.npy --procs 32 &
wait
# Wave 2.
run out_slam_stereo --poses slam $STEREO --slam-cache $V2/slam_poses.npy --procs 48 &
run out_gt_stereo_lidar --poses gt $STEREO --lidar-fuse --procs 48 &
wait
echo "[run_all] all done $(date +%T)"
