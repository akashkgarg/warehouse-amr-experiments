"""Recolor a finished ``recon.py`` output from the camera images and re-export its GLBs (no re-fusion).

Usage (Open3D venv)::

    ~/.venvs/recon/bin/python tools/warehouse_amr/recolor_glb.py ~/warehouse_amr/recon/v2/out_final_slam \\
        ~/warehouse_amr/recon/v2/data_s2{1,2,3,4,5,6,7} --poses slam --slam-cache ~/warehouse_amr/recon/v2/slam_poses.npy
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

import recon  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--poses", default="slam")
    ap.add_argument("--slam-cache", default="")
    ap.add_argument("--depth", default="stereo")
    ap.add_argument("--reach", type=float, default=8.0)
    ap.add_argument("--lite-tris", type=int, default=300_000)
    ap.add_argument("--procs", type=int, default=48)
    args = ap.parse_args()

    recs = recon.load(args.dirs)
    recon.estimate_poses(recs, args.poses, cache=args.slam_cache)
    src = os.path.join(args.out, "clean_mesh_fusedcolor.ply")
    if not os.path.exists(src):
        src = os.path.join(args.out, "clean_mesh.ply")
        o3d.io.write_triangle_mesh(os.path.join(args.out, "clean_mesh_fusedcolor.ply"), o3d.io.read_triangle_mesh(src))
    mesh = o3d.io.read_triangle_mesh(src)
    mesh = recon.recolor(mesh, recs, args.depth, reach=args.reach, procs=args.procs)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(os.path.join(args.out, "clean_mesh.ply"), mesh)
    pcd = recon.color_lidar(recs, 0.04, args.depth, args.procs)
    recon.export_glb(os.path.join(args.out, "warehouse_recon.glb"), mesh, pcd)
    lite = o3d.geometry.TriangleMesh(mesh).simplify_quadric_decimation(args.lite_tris)
    lite.compute_vertex_normals()
    recon.export_glb(os.path.join(args.out, "warehouse_recon_lite.glb"), lite, pcd.voxel_down_sample(0.12))
    m = json.load(open(os.path.join(args.out, "metrics.json")))
    m["recolored"] = True
    json.dump(m, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
