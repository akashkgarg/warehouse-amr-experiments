"""Offscreen preview renders of a reconstruction (``recon.py`` output) next to the true warehouse geometry.

Usage::

    ~/.venvs/recon/bin/python tools/warehouse_amr/recon_views.py ~/warehouse_amr/recon/out_gt \\
        --gt ~/warehouse_amr/recon/data_s7/gt_mesh.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


def views(bounds: o3d.geometry.AxisAlignedBoundingBox) -> dict:
    lo, hi = bounds.get_min_bound(), bounds.get_max_bound()
    c = (lo + hi) / 2
    ext = hi - lo
    return {
        "overview": (c + np.array([-0.45 * ext[0], -0.75 * ext[1], 0.9 * max(ext[:2])]), c, 45),
        "top": (c + np.array([0, 0.001, 1.15 * max(ext[:2])]), c, 50),
        "aisle": (c + np.array([-0.2 * ext[0], -0.25 * ext[1], 1.6]), c + np.array([0.1 * ext[0], 0.15 * ext[1], 1.2]), 70),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1000)
    args = ap.parse_args()

    mesh = o3d.io.read_triangle_mesh(os.path.join(args.out, "clean_mesh.ply"))
    mesh.compute_vertex_normals()
    g = np.load(args.gt)
    gt = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(g["vertices"].astype(np.float64)),
                                   o3d.utility.Vector3iVector(g["faces"]))
    gt.compute_vertex_normals()
    # Crop the truth (and the views) to the reconstructed footprint and below the roof, which the robots never see.
    b = mesh.get_axis_aligned_bounding_box()
    lo, hi = b.get_min_bound(), b.get_max_bound()
    crop = o3d.geometry.AxisAlignedBoundingBox(lo - [0.5, 0.5, 0.5], [hi[0] + 0.5, hi[1] + 0.5, min(hi[2], 6.0)])
    gt = gt.crop(crop)
    mesh_view = mesh.crop(crop)

    r = rendering.OffscreenRenderer(args.width, args.height)
    r.scene.set_background([1, 1, 1, 1])
    r.scene.scene.set_sun_light([0.3, 0.4, -1.0], [1, 1, 1], 60000)
    r.scene.scene.enable_sun_light(True)
    lit = rendering.MaterialRecord()
    lit.shader = "defaultLit"
    unlit = rendering.MaterialRecord()
    unlit.shader = "defaultUnlit"
    gray = rendering.MaterialRecord()
    gray.shader = "defaultLit"
    gray.base_color = [0.75, 0.75, 0.78, 1.0]
    for name, (eye, target, fov) in views(crop).items():
        for tag, geom, mat in (("recon", mesh_view, unlit), ("gt", gt, gray)):
            r.scene.clear_geometry()
            r.scene.add_geometry(tag, geom, mat)
            r.setup_camera(fov, target, eye, [0, 0, 1])
            img = r.render_to_image()
            path = os.path.join(args.out, f"view_{name}_{tag}.png")
            o3d.io.write_image(path, img)
            print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
