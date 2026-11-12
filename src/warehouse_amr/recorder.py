"""MP4 recorder for WarehouseEnv: chase camera on one robot plus a live top-down map of the whole fleet."""

from __future__ import annotations

import math
import subprocess

import cv2
import numpy as np

ROBOT_COLORS = {"forklift": (255, 140, 0), "carter": (0, 120, 220)}
WORKER_COLOR = (220, 0, 0)


class Recorder:
    """Writes one video frame per rendered app frame (``env.frames_per_step`` per control step).

    Set ``caption`` to overlay text on subsequent frames.
    """

    def __init__(self, env, path: str, width: int = 1280, height: int = 720, follow: int = 0):
        import omni.replicator.core as rep
        from pxr import UsdGeom

        from warehouse_env import CONTROL_DT

        self.env = env
        self.follow = follow
        self.caption = ""
        self.width, self.height = width, height
        cam = UsdGeom.Camera.Define(env.stage, "/World/RecordCam")
        cam.CreateFocalLengthAttr(16.0)
        cam.CreateClippingRangeAttr((0.1, 500.0))
        self.cam = cam
        xformable = UsdGeom.Xformable(cam)
        xformable.ClearXformOpOrder()
        self.xform = xformable.MakeMatrixXform()
        self._eye = None
        self.render_product = rep.create.render_product("/World/RecordCam", (width, height))
        self.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annotator.attach([self.render_product])
        scale = height / env.map_image.shape[0]
        self.map_w = int(round(env.map_image.shape[1] * scale / 2)) * 2
        self.map_scale = self.map_w / env.map_image.shape[1]
        self.path = path
        self.frames = 0
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width + self.map_w}x{height}", "-r", str(env.frames_per_step / CONTROL_DT), "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", path,
            ],
            stdin=subprocess.PIPE,
        )
        self.update_camera(snap=True)
        env.set_render(True)
        env.frame_callbacks.append(self.frame)

    def update_camera(self, snap: bool = False) -> None:
        from pxr import Gf

        r = self.env.robots[self.follow]
        if not r.active:
            # Finished robots are parked off-map; hand the camera to one that is still driving.
            same = [i for i, o in enumerate(self.env.robots) if o.active and o.type_name == r.type_name]
            others = [i for i, o in enumerate(self.env.robots) if o.active]
            if same or others:
                self.follow = (same or others)[0]
                r = self.env.robots[self.follow]
                snap = True
        center, yaw, _ = r.pose
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        # High and steep so racks between the camera and the robot don't block the view.
        back, up = (6.0, 9.0) if r.type_name == "forklift" else (4.5, 7.5)
        eye = center + np.array([0, 0, up]) - back * fwd
        self._eye = eye if snap or self._eye is None else self._eye + 0.25 * (eye - self._eye)
        target = center + 1.0 * fwd
        f = target - self._eye
        f /= np.linalg.norm(f)
        right = np.cross(f, [0, 0, 1])
        right /= np.linalg.norm(right)
        cam_up = np.cross(right, f)
        e = self._eye
        self.xform.Set(
            Gf.Matrix4d(
                right[0], right[1], right[2], 0,
                cam_up[0], cam_up[1], cam_up[2], 0,
                -f[0], -f[1], -f[2], 0,
                e[0], e[1], e[2], 1,
            )
        )

    def map_panel(self) -> np.ndarray:
        env, m = self.env, self.env.map
        img = env.map_image.copy()
        px = lambda xy: tuple(int(v) for v in ((np.asarray(xy[:2]) - m.origin) / m.res))
        for w in env.worker_positions():
            cv2.circle(img, px(w), 4, WORKER_COLOR, -1)
        for i, r in enumerate(env.robots):
            color = ROBOT_COLORS[r.type_name]
            center, yaw, _ = r.pose
            look = m.lookahead(r.spec.inflation, r.field, center[:2], 12.0) if r.field is not None else r.goal
            cv2.line(img, px(center), px(look), color, 1)
            cv2.drawMarker(img, px(r.goal), color, cv2.MARKER_TILTED_CROSS, 10, 2)
            rect = (px(center), (2 * r.half[0] / m.res, 2 * r.half[1] / m.res), math.degrees(yaw))
            cv2.fillPoly(img, [cv2.boxPoints(rect).astype(np.int32)], color)
            if i == self.follow:
                cv2.circle(img, px(center), int(2.5 / m.res), (0, 180, 0), 1)
        img = cv2.resize(img[::-1], (self.map_w, self.height), interpolation=cv2.INTER_AREA)
        y = 18
        for r in env.robots:
            ev = r.events
            txt = f"{r.name}: ok {ev.get('success', 0)} col {ev.get('collision', 0) + ev.get('human_contact', 0)}"
            cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, ROBOT_COLORS[r.type_name], 1, cv2.LINE_AA)
            y += 15
        return img

    def frame(self) -> None:
        for r in self.env.robots:
            r.read_state()
        self.update_camera()
        rgb = self.annotator.get_data()
        if rgb is None or getattr(rgb, "size", 0) == 0:
            rgb = np.zeros((self.height, self.width, 4), np.uint8)
        rgb = np.ascontiguousarray(np.asarray(rgb)[..., :3])
        if self.caption:
            cv2.putText(rgb, self.caption, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        self.proc.stdin.write(np.hstack([rgb, self.map_panel()]).tobytes())
        self.frames += 1

    def close(self) -> None:
        self.env.frame_callbacks.remove(self.frame)
        self.env.set_render(False)
        self.annotator.detach([self.render_product])
        self.render_product.destroy()
        self.proc.stdin.close()
        self.proc.wait()
        print(f"[recorder] wrote {self.frames} frames to {self.path}", flush=True)
