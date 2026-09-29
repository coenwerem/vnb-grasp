"""Render saved physics states with physical timestamps and declared playback speed."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def validate_comparison(replays):
    """A two-panel deliverable must differ by method, not test conditions."""
    if len(replays) != 2:
        return
    left, right = replays
    ls, rs = left.meta["settings"], right.meta["settings"]
    if {ls["method"], rs["method"]} != {"variational", "cem"}:
        raise ValueError("Comparison requires one VNB and one CEM episode")
    for key in set(ls) | set(rs):
        if key not in {"method", "output"} and ls.get(key) != rs.get(key):
            raise ValueError(f"Comparison settings differ: {key}")
    source_key = "source_sha256" if "source_sha256" in left.meta else "release_commit"
    for key in [source_key, "model_sha256"]:
        if not left.meta.get(key) or left.meta[key] != right.meta.get(key):
            raise ValueError(f"Comparison lacks identical verified {key}")
    for key in ["qpos", "qvel"]:
        if not np.array_equal(left.states[key][0], right.states[key][0]):
            raise ValueError(f"Initial {key} differs between methods")


class Replay:
    def __init__(self, path, width, height):
        self.path = Path(path)
        self.states = np.load(self.path / "states.npz")
        self.meta = json.loads((self.path / "provenance.json").read_text())
        self.model = mj.MjModel.from_binary_path(str(self.path / "model.mjb"))
        self.data = mj.MjData(self.model)
        self.model.vis.global_.offwidth = max(width, self.model.vis.global_.offwidth)
        self.model.vis.global_.offheight = max(height, self.model.vis.global_.offheight)
        # Hide duplicate collision surfaces only in the replay renderer.
        for name in ["table_collision", "workspace_table_collision"]:
            gid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self.model.geom_rgba[gid, 3] = 0
        self.model.vis.headlight.ambient[:] = .3
        self.model.vis.headlight.diffuse[:] = .5
        self.renderer = mj.Renderer(self.model, height, width)
        self.camera = mj.MjvCamera()
        self.camera.type = mj.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [-.05, .88, .84]
        self.camera.distance = .55
        self.camera.elevation = -25
        self.camera.azimuth = 135
        self.scene_option = mj.MjvOption()
        self.scene_option.geomgroup[:3] = 1
        self.events = {v["phase"]: v for v in self.meta["events"]}
        self.reference_time = self.events.get("friction_drop", {}).get("time_s", 0)
        if "lift_and_shear" in self.events and "friction_drop" not in self.events:
            self.reference_time = self.events["lift_and_shear"]["time_s"] + 1.2
        self.reference_index = int(np.argmin(abs(self.states["time_s"] - self.reference_time)))

    def frame(self, t):
        i = int(np.argmin(abs(self.states["time_s"] - t)))
        self.data.qpos[:] = self.states["qpos"][i]
        self.data.qvel[:] = self.states["qvel"][i]
        mj.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.camera, scene_option=self.scene_option)
        return Image.fromarray(self.renderer.render()), i

    def label(self, t):
        mu = self.meta["settings"]["friction"]
        drop = self.events.get("friction_drop")
        if drop and t >= drop["time_s"]:
            return f"Friction drop: μ = {mu:.2f} → {drop['after']:.2f}"
        return f"Contact friction: μ = {mu:.2f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episodes", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--speed", type=float, default=1.)
    p.add_argument("--title", required=True)
    p.add_argument("--camera-azimuth", type=float, default=135.)
    p.add_argument("--camera-elevation", type=float, default=-25.)
    p.add_argument("--camera-distance", type=float, default=.55)
    p.add_argument("--camera-lookat", type=float, nargs=3, default=[-.05, .88, .84])
    p.add_argument("--protocol-label", default="Same scene and test")
    p.add_argument("--highlight-thumb", action="store_true")
    p.add_argument("--height-readout", action="store_true")
    p.add_argument("--align-event", choices=["friction_drop", "lift_and_shear"],
                   help="Use time relative to this event; display each panel's episode time")
    a = p.parse_args()
    if len(a.episodes) not in (1, 2) or a.end <= a.start or a.speed <= 0:
        raise ValueError("One or two episodes and a positive interval/speed are required")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    W, H, fps = 1920, 1080, 30
    n = len(a.episodes)
    rw, rh = W // n, 780
    replays = [Replay(ep, rw, rh) for ep in a.episodes]
    for r in replays:
        r.camera.azimuth = a.camera_azimuth
        r.camera.elevation = a.camera_elevation
        r.camera.distance = a.camera_distance
        r.camera.lookat[:] = a.camera_lookat
        if a.highlight_thumb:
            for gid in range(r.model.ngeom):
                if "thumb" in r.model.body(r.model.geom_bodyid[gid]).name:
                    r.model.geom_rgba[gid] = [.95, .64, .1, 1.]
                    r.model.geom_matid[gid] = -1
    validate_comparison(replays)
    offsets = [r.events[a.align_event]["time_s"] if a.align_event else 0. for r in replays]
    if any(a.end+offset > r.states["time_s"][-1] + .025 or a.start+offset < 0
           for r, offset in zip(replays, offsets)):
        raise ValueError("Requested interval exceeds executed states")
    title_font = ImageFont.truetype(BOLD, 48)
    label_font = ImageFont.truetype(BOLD, 32)
    small_font = ImageFont.truetype(FONT, 29)
    out = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.output)], stdin=subprocess.PIPE)
    times = np.arange(a.start, a.end+1e-9, a.speed/fps)
    frame_records = []
    thumbnails = []
    sample_ids = set(np.linspace(0, len(times)-1, 8).astype(int))
    try:
        for frame_id, t in enumerate(times):
            canvas = Image.new("RGB", (W, H), "#f8f9fb")
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, W, 88), fill="#081522")
            draw.text((40, 16), a.title, font=title_font, fill="#edf2f7")
            draw.rectangle((0, 88, W, 92), fill="#ffc629")
            for panel, r in enumerate(replays):
                panel_t = t + offsets[panel]
                im, index = r.frame(panel_t)
                x = panel*rw
                canvas.paste(im, (x, 154))
                method = {"variational": "VNB", "cem": "CEM"}[r.meta["settings"]["method"]]
                draw.text((x+28, 107), method + "  |  " + r.label(panel_t), font=label_font, fill="#0d2034")
                f = r.states["object_wrench"][index, :3]
                lost = r.events.get("grasp_lost")
                if lost and panel_t >= lost["time_s"]:
                    state_label = "30 mm limit · force removed"
                    color = "#b03028"
                elif np.linalg.norm(f) > 1e-8:
                    state_label = f"Lateral force: {np.linalg.norm(f):.0f} N"
                    color = "#155493"
                elif "lift_and_shear" in r.events and panel_t >= r.events["lift_and_shear"]["time_s"]:
                    state_label = "Lift and hold"
                    color = "#155493"
                else:
                    state_label = "Grasp acquisition"
                    color = "#155493"
                draw.text((x+28, 950), state_label, font=label_font, fill=color)
                if a.height_readout:
                    body = r.meta['settings']['object']
                    if body == 'sphere':
                        body = 'cube'
                    bid = r.model.body(body).id
                    adr = r.model.jnt_qposadr[r.model.body_jntadr[bid]]
                    lift_t = r.events['lift_and_shear']['time_s']
                    ref = int(np.argmin(abs(r.states['time_s']-lift_t)))
                    dz = 1000*(r.states['qpos'][index,adr+2]-r.states['qpos'][ref,adr+2])
                    draw.text((x+420, 951), f"Height change: {dz:+.1f} mm", font=small_font, fill="#46596c")
                if a.align_event:
                    draw.text((x+28, 895), f"Episode t = {panel_t:.2f} s", font=small_font, fill="#fff",
                              stroke_width=2, stroke_fill="#172332")
                frame_records.append({"frame": frame_id, "panel": panel, "state_index": index,
                    "sim_time_s": float(r.states["time_s"][index])})
            if n == 2:
                draw.line((rw, 96, rw, 992), fill="#f8f9fb", width=8)
            settings = replays[0].meta["settings"]
            obj = settings["object"].replace("graspit_", "").replace("_", " ").title()
            text = f"MuJoCo simulation  ·  {obj}  ·  Seed {settings['seed']}  ·  {a.speed:g}× playback  ·  t = {t:.2f} s"
            if a.align_event:
                event_label = "lift onset" if a.align_event == "lift_and_shear" else a.align_event.replace('_', ' ')
                text = f"{a.protocol_label}  ·  {obj}  ·  {a.speed:g}× playback  ·  {t:+.2f} s from {event_label}"
            draw.text((28, 1010), text, font=small_font, fill="#46596c")
            out.stdin.write(canvas.tobytes())
            if frame_id in sample_ids:
                thumbnails.append(canvas.resize((640, 360)))
            if frame_id == len(times)//2:
                canvas.save(a.output.with_suffix(".png"))
    finally:
        out.stdin.close()
        code = out.wait()
        for r in replays:
            r.renderer.close()
    if code:
        raise RuntimeError("ffmpeg encoding failed")
    sheet = Image.new("RGB", (1280, 360*((len(thumbnails)+1)//2)), "white")
    for i, im in enumerate(thumbnails):
        sheet.paste(im, ((i%2)*640, (i//2)*360))
    sheet.save(a.output.with_name(a.output.stem+"_contact_sheet.jpg"))
    manifest = {"title": a.title, "source_episodes": [str(x.resolve()) for x in a.episodes],
        "interval_s": [a.start, a.end], "speed": a.speed, "fps": fps,
        "frames": len(times), "size": [W, H], "sha256": hashlib.sha256(a.output.read_bytes()).hexdigest(),
        "render_changes": ["Replay only: hide overlapping table collision visuals", "Camera and headlight adjusted"] +
                          (["Thumb highlighted in gold identically in both panels"] if a.highlight_thumb else []),
        "motion": "Nearest recorded state at each timestamp; no motion interpolation or synthesized motion",
        "alignment_event": a.align_event, "episode_time_offsets_s": offsets,
        "camera": dict(azimuth=a.camera_azimuth, elevation=a.camera_elevation,
                       distance=a.camera_distance, lookat=a.camera_lookat),
        "protocol_label": a.protocol_label,
        "frame_records": frame_records}
    a.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(str(a.output))


if __name__ == "__main__":
    main()
