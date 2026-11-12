# Warehouse AMR experiments in Isaac Sim

Experiments from Oct–Nov 2026 in Isaac Sim 6.1's `Simple_Warehouse` scene with walking workers, a
ForkliftC and Nova Carters:

1. **Learned navigation vs. Nav2.** A teacher/student driver (privileged scripted teacher → 256×256 MLP
   student on lidar + odometry + pose estimate, trained with DAgger then PPO) benchmarked against stock Nav2
   Jazzy, both localized by AMCL. Carter: 67% vs. 70% success, but 15.5 s vs. 39.8 s per trip. Forklift: 80%
   vs. 32%.
2. **3D reconstruction from the fleet's sensors.** Multi-robot pose-graph SLAM, FoundationStereo depth on the
   Hawk stereo pairs and lidar fused into a colored mesh, with no ground truth as input. Round three adds
   Scan Context place recognition (2.5 mm median pose error without priors), stereo left-right checks and a
   tilted mast lidar on the forklift.
3. **Forklift pallet handling** (in progress, no writeup yet): `pallet_task.py`, the `pallet_*` training runs
   and `configs/nav2_pallet/`.

## Reports

| report | what it covers |
|---|---|
| [`docs/01_navigation_learned_vs_nav2/README.md`](docs/01_navigation_learned_vs_nav2/README.md) | Sim setup, teacher/student training, Nav2 config, the four sim bugs (Carter spin-out, 0.7 m omap offset, lidar encoding, small-sample benchmarks), results and videos |
| [`docs/02_reconstruction/README.md`](docs/02_reconstruction/README.md) | Rounds one and two of the reconstruction: GT vs. estimated poses, stereo depth, SLAM, ablations |
| [`docs/03_reconstruction_v3/README.md`](docs/03_reconstruction_v3/README.md) | Round three: place recognition, kidnapped-robot SLAM, stereo L/R check, mast lidar, pole landmarks |

The reports were written against the original layout (`~/isaacsim/tools/warehouse_amr/`, `~/warehouse_amr/`);
see the path mapping below. The reconstruction report links to `warehouse_recon_v2*.glb` files, which are
not in this repo (146 MB / 16 MB).

## Layout

```
src/warehouse_amr/
  warehouse_env.py, warehouse_sim.py, warehouse_client.py   environment, socket sim server, client
  bench_utils.py, recorder.py                               benchmark episodes/metrics, video recording
  train_ppo.py, policy_np.py                                DAgger + PPO training, numpy inference
  nav2_bridge.py, ros_env.sh                                ROS 2 Jazzy bridge for Nav2 and policy+AMCL benchmarks
  hybrid_astar.py, pallet_task.py                           forklift planning and pallet task
  collect_recon.py, stereo_depth.py, slam.py, recon.py,
  recolor_glb.py, recon_views.py                            reconstruction pipeline
scripts/recon_v2/, scripts/recon_v3/   batch scripts for the reconstruction ablations
configs/nav2/, configs/nav2_pallet/    per-robot Nav2 params and AMCL maps used in the benchmarks
docs/                                  reports, figures, videos and figure-generation scripts
results/bench/                         per-episode benchmark records + summaries (final/ = report numbers)
results/recon/                         metrics.json from every reconstruction run
results/train_logs/                    metrics.jsonl per training run
checkpoints/<run>/policy*.npz          final numpy policies (sensor4 is the one in the report)
requirements/                          frozen venvs: policy (training), recon (open3d), stereo (FoundationStereo)
```

Not included (too large; still on the h200-training devspace under `~/warehouse_amr/`): the 196 GB of raw
reconstruction captures and output meshes, intermediate `.pt` checkpoints, ROS logs and the full-length
report videos.

## Setup

Requirements: Isaac Sim 6.1 (standalone), ROS 2 Jazzy with Nav2 1.3.x, and Python 3.12 venvs.

```bash
for v in policy recon stereo; do
  python3.12 -m venv ~/.venvs/$v && ~/.venvs/$v/bin/pip install -r requirements/$v.txt
done
# training/eval commands in the reports use ~/.venvs/franka-policy; the same freeze is requirements/policy.txt

# stereo depth needs FoundationStereo (tested at 6e88068) and its pretrained weights
git clone https://github.com/NVlabs/FoundationStereo ~/workspace/stereo/FoundationStereo

# commands are written to run from the Isaac Sim root
ln -s "$PWD/src/warehouse_amr" ~/isaacsim/tools/warehouse_amr
```

## Running

```bash
cd ~/isaacsim
# one sim server per GPU
./python.sh tools/warehouse_amr/warehouse_sim.py serve --gpu 5 --robots forklift,carter,carter,carter \
    --address 127.0.0.1:47800

# train (DAgger then PPO)
~/.venvs/policy/bin/python tools/warehouse_amr/train_ppo.py train \
    --address 127.0.0.1:47810,...,127.0.0.1:47815 --run ~/warehouse_amr/runs/sensor4 --total-steps 3000000

# benchmarks
~/.venvs/policy/bin/python tools/warehouse_amr/train_ppo.py eval --checkpoint expert --episodes 20 --out expert.json
tools/warehouse_amr/ros_env.sh python3.12 tools/warehouse_amr/nav2_bridge.py nav2 --episodes 20 --out nav2.json
tools/warehouse_amr/ros_env.sh python3.12 tools/warehouse_amr/nav2_bridge.py policy \
    --policy <repo>/checkpoints/sensor4/policy.npz --episodes 20 --out ppo_amcl.json
```

The reconstruction pipeline commands are at the end of `docs/02_reconstruction/README.md`. The batch scripts
under `scripts/` hardcode `/home/builder/...` paths from the original devspace, so edit those before you run them.

## Path mapping from the original workspace

| original | here |
|---|---|
| `~/isaacsim/tools/warehouse_amr/` | `src/warehouse_amr/` |
| `~/warehouse_amr/report/` | `docs/01_navigation_learned_vs_nav2/` |
| `~/warehouse_amr/recon/report/` | `docs/02_reconstruction/` |
| `~/warehouse_amr/recon/report_v3/` | `docs/03_reconstruction_v3/` |
| `~/warehouse_amr/recon/v{2,3}/*.sh` | `scripts/recon_v{2,3}/` |
| `~/warehouse_amr/ros_nav2/`, `ros_pallet/` | `configs/nav2/`, `configs/nav2_pallet/` |
| `~/warehouse_amr/bench/` | `results/bench/` |
| `~/warehouse_amr/recon/**/metrics.json` | `results/recon/<run_path>.json` |
| `~/warehouse_amr/runs/<run>/` | `results/train_logs/<run>.jsonl`, `checkpoints/<run>/` |
