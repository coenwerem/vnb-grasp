"""Render synchronized VNB/CEM assembly prototype episodes from recorded states."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydrake.geometry import (RenderCameraCore, ColorRenderCamera, ClippingRange, SceneGraph,
                             RenderEngineVtkParams, MakeRenderEngineVtk, LightParameter)
from pydrake.systems.sensors import CameraInfo
from pydrake.math import RigidTransform, RotationMatrix
from bimanual_assembly.scene import build_scene


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episodes", nargs=2, type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preview-only", action="store_true")
    a = p.parse_args()
    meta = [json.loads((ep / "summary.json").read_text()) for ep in a.episodes]
    states = [np.load(ep / "states.npz") for ep in a.episodes]
    if [m["method"] for m in meta] != ["variational", "cem"]:
        raise ValueError("Expected VNB left and CEM right")
    keys = ["seed", "friction", "dt_s", "control_dt_s", "duration_s", "contact_model",
            "contact_stiffness_N_m", "clearance_m", "max_residual_rad", "residual_rate_limit_rad_s",
            "planner_period_s", "speed", "beta", "checkpoint", "drake", "actuator_names",
            "bimanual_commit", "release_commit", "source_sha256"]
    for key in keys:
        if meta[0][key] != meta[1][key]:
            raise ValueError(f"Mismatched comparison: {key}")
    for key in ["q", "v"]:
        if not np.array_equal(states[0][key][0], states[1][key][0]):
            raise ValueError(f"Initial {key} differs")
    if (a.episodes[0] / "reference.json").read_bytes() != (a.episodes[1] / "reference.json").read_bytes():
        raise ValueError("Reference trajectories differ")
    scenes, contexts = [], []
    for ep, m in zip(a.episodes, meta):
        scene = build_scene(ep.parent / ".assets", render=False, dt=m["dt_s"], friction=m["friction"],
            stiffness=m["contact_stiffness_N_m"], clearance=m["clearance_m"], contact_model=m["contact_model"])
        params = RenderEngineVtkParams()
        params.default_clear_color = np.array([.95, .96, .98])
        key = LightParameter(); key.intensity = 1.2; key.direction = np.array([.2, .3, 1.])
        fill = LightParameter(); fill.intensity = .7; fill.direction = np.array([-.6, -.2, .7])
        params.lights = [key, fill]
        scene.scene_graph.AddRenderer("renderer", MakeRenderEngineVtk(params))
        root = scene.diagram.CreateDefaultContext()
        scenes.append(scene); contexts.append((root, scene.plant.GetMyMutableContextFromRoot(root)))
    W, H, rw, rh, fps = 1920, 1080, 960, 780, 30
    eye = np.array([.79, -.60, -.085]); target = np.array([.48, -.04, -.43])
    z = target-eye; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1]); x /= np.linalg.norm(x); y = np.cross(z, x)
    pose = RigidTransform(RotationMatrix(np.column_stack([x, y, z])), eye)
    cam = ColorRenderCamera(RenderCameraCore("renderer", CameraInfo(rw, rh, np.pi/4),
        ClippingRange(.01, 5.), RigidTransform()), False)
    fonts = [ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", n) for n in [44, 32]]
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    times = np.array([0., 6., 12.]) if a.preview_only else np.arange(0., 12.5+1e-9, 1/fps)
    video = None
    if not a.preview_only:
        video = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.output)], stdin=subprocess.PIPE)
    records, thumbs = [], []
    selected = set(np.linspace(0, len(times)-1, 8).astype(int))
    try:
        for n, t in enumerate(times):
            canvas = Image.new("RGB", (W, H), "#f8f9fb"); draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, W, 88), fill="#081522")
            draw.text((30, 18), "Bimanual Assembly | VNB / CEM Grip Transfer", font=fonts[0], fill="#edf2f7")
            draw.rectangle((0, 88, W, 92), fill="#ffc629")
            for panel, (s, contexts_pair, log, m) in enumerate(zip(scenes, contexts, states, meta)):
                pc = contexts_pair[1]; i = int(np.argmin(abs(log["time_s"]-t)))
                s.plant.SetPositions(pc, log["q"][i]); s.plant.SetVelocities(pc, log["v"][i])
                query = s.plant.get_geometry_query_input_port().Eval(pc)
                frame = Image.fromarray(query.RenderColorImage(cam, SceneGraph.world_frame_id(), pose).data).convert("RGB")
                canvas.paste(frame, (panel*rw, 154))
                method = ["VNB", "CEM"][panel]
                draw.text((panel*rw+28, 107), method + " hand adjustments", font=fonts[1], fill="#0d2034")
                stage = next(e["stage"] for e in reversed(m["events"]) if e["t"] <= log["time_s"][i])
                if m["outcome"]["completed"] and t >= m["outcome"]["completion_time_s"]:
                    label = "Inserted, released, and stable"
                else:
                    label = stage.replace("_", " ").capitalize()
                draw.text((panel*rw+28, 950), label, font=fonts[1], fill="#155493")
                records.append(dict(frame=n, panel=panel, state_index=i, sim_time_s=float(log["time_s"][i])))
            draw.line((rw, 96, rw, 992), fill="#f8f9fb", width=8)
            draw.text((28, 1010), f"Prototype · shared arm controller · μ = {meta[0]['friction']:.2f} · 1× playback · t = {t:.2f} s",
                      font=small, fill="#46596c")
            if video:
                video.stdin.write(canvas.tobytes())
            if n in selected or a.preview_only:
                thumbs.append(canvas.resize((640, 360)))
            if abs(t-6.) < .01:
                canvas.save(a.output.with_suffix(".png"))
    finally:
        if video:
            video.stdin.close()
            if video.wait():
                raise RuntimeError("Video encoding failed")
    sheet = Image.new("RGB", (1280, 360*((len(thumbs)+1)//2)), "white")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i%2)*640, (i//2)*360))
    sheet.save(a.output.with_name(a.output.stem+"_contact_sheet.jpg"))
    if video:
        record = dict(source_episodes=[str(ep.resolve()) for ep in a.episodes], size=[W, H], fps=fps,
            frames=len(times), playback_speed=1., camera_eye=eye.tolist(), camera_target=target.tolist(),
            lighting="Two identical camera-frame directional lights, intensity 1.2 and 0.7; light background",
            matching_fields=keys, initial_states_identical=True, reference_identical=True,
            motion="Replay nearest executed states; no interpolated or synthesized motion",
            sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(), frame_records=records)
        a.output.with_suffix(".json").write_text(json.dumps(record, indent=2))
    print(str(a.output))


if __name__ == "__main__":
    main()
