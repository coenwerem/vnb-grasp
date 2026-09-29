"""New VNB/CEM grip-transfer prototype on the existing Drake assembly task.

Both methods share the assembly reference, object-pose arm feedback, dynamics,
and impedance controller. The only method-dependent command is a bounded hand
position residual. This is not the paper's controller or an assembly benchmark.
The source NeuralBeliefFilter is untrained; no checkpoint is loaded.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Tuple
import numpy as np


def project_hand_action(delta):
    """Least-squares projection of 11 joint deltas to six L6 actuators.

    Order: thumb roll, thumb pitch, thumb distal, then proximal/distal for
    index, middle, ring, pinky. The Drake model couples thumb distal at 2.22
    and other distal joints at 1.9 times the corresponding pitch actuator.
    """
    delta = np.asarray(delta, dtype=float)
    if delta.shape != (11,) or not np.isfinite(delta).all():
        raise ValueError("Expected 11 finite joint deltas")
    S = np.zeros((11, 6))
    S[0, 0] = S[1, 1] = 1.
    S[2, 1] = 2.22
    for i in range(4):
        S[3+2*i, 2+i] = 1.
        S[4+2*i, 2+i] = 1.9
    return np.linalg.lstsq(S, delta, rcond=None)[0]


def load_release_optimizers(release, output):
    """Reuse exact optimizer functions without importing unrelated hardware modules."""
    import torch
    import torch.nn.functional as F
    from vnb_grasp.belief.variational_belief import GaussianMixtureBelief
    source = (release / "examples/run_variational_belief_experiments.py").read_text()
    names = {"_make_contact_cost_fn", "_make_action_conditioned_cost_fn",
             "_optimize_action_with_gradients", "_score_candidate_actions"}
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    code = "\n\n".join(ast.get_source_segment(source, n) for n in nodes)
    (output / "release_optimizer_functions.py").write_text(code)
    namespace = dict(torch=torch, F=F, np=np, Tuple=Tuple,
                     GaussianMixtureBelief=GaussianMixtureBelief, MAX_CVAR_GRAD_NORM=1.)
    exec(compile(code, str(output / "release_optimizer_functions.py"), "exec"), namespace)
    return namespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=["variational", "cem"], required=True)
    p.add_argument("--friction", type=float, default=.7)
    p.add_argument("--duration", type=float, default=14.)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-residual", type=float, default=.03)
    a = p.parse_args()
    output = a.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    import torch
    from bimanual_assembly.scene import build_scene, PART_START, HOUSING_START
    from bimanual_assembly.reference import make_reference
    from bimanual_assembly.nominal import ObjectPoseNominal
    from bimanual_assembly.experiment import observe
    from bimanual_assembly.outcomes import summarize
    from vnb_grasp.belief.variational_belief import VariationalBeliefConfig, GaussianMixtureBelief, NeuralBeliefFilter
    from pydrake.math import RigidTransform
    from pydrake.systems.analysis import Simulator
    import importlib.metadata
    import bimanual_assembly.scene as scene_module
    import vnb_grasp.belief.variational_belief as belief_module
    torch.set_num_threads(1); torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    release = Path("/home/drce/ResearchProjects/MuJoCoDex_iros_release")
    assembly = Path("/home/drce/ResearchProjects/bimanual-assembly")
    seed_path = assembly / "configs/reference_seed.json"
    funcs = load_release_optimizers(release, output)
    snapshot = output / "source_snapshot"; snapshot.mkdir()
    hashes = {}
    for f in Path(scene_module.__file__).parent.glob("*.py"):
        shutil.copy2(f, snapshot / f.name); hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    shutil.copy2(belief_module.__file__, snapshot / "variational_belief.py")
    shutil.copy2(seed_path, output / "reference_seed.json")
    dt = .002
    scene = build_scene(output.parent / ".assets", dt=dt, friction=a.friction,
                        clearance=.003, contact_model="hydroelastic")
    plant, adapter = scene.plant, scene.adapter
    reference = make_reference(scene, seed_path); reference.save(output / "reference.json")
    nominal = ObjectPoseNominal(scene, reference)
    simulator = Simulator(scene.diagram); root = simulator.get_mutable_context()
    pc = plant.GetMyMutableContextFromRoot(root)
    adapter.set_robot_positions(pc, reference.samples[0])
    plant.SetFreeBodyPose(pc, scene.part, RigidTransform(PART_START))
    plant.SetFreeBodyPose(pc, scene.housing, RigidTransform(HOUSING_START))
    port = plant.get_actuation_input_port(scene.robot)
    port.FixValue(pc, np.zeros(len(adapter.names))); simulator.Initialize()
    suffixes = ["thunb_cmc_roll", "thumb_cmc_pitch", "index_mcp_pitch",
                "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch"]
    indices = [[adapter.names.index(prefix + suffix) for suffix in suffixes] for prefix in ["lh_", "rh_"]]
    # Verify the inspected coupling contract against this instantiated model.
    for follower, (leader, multiplier, offset) in scene.info["mimics"].items():
        expected = 2.22 if "thumb" in follower else 1.9
        if not np.isclose(multiplier, expected) or offset != 0:
            raise ValueError(f"Unexpected hand coupling: {follower}")
    hand = np.array([n.startswith(("lh_", "rh_")) for n in adapter.names])
    K, D = np.where(hand, 40., 180.), np.where(hand, 1.4, 28.)
    config = VariationalBeliefConfig(n_components=8 if a.method == "variational" else 1,
        obs_dim=plant.num_positions()+plant.num_velocities(), action_dim=11,
        cvar_beta=.9, risk_weight=.5)
    beliefs = [GaussianMixtureBelief(config) for _ in indices]
    filters = [NeuralBeliefFilter(config) for _ in indices]
    previous = [None, None]
    cost_fn = funcs["_make_contact_cost_fn"]()
    target = np.zeros(len(adapter.names)); residual = target.copy()
    rows, states, velocities, torques, commands, events, planner_log = [], [], [], [], [], [], []
    phase = 0.; last_stage = None; reason = "duration_limit"; tick = time.time()
    try:
        for k in range(round(a.duration / dt)+1):
            t = k * dt
            if k:
                simulator.AdvanceTo(t); phase = min(1., phase + dt * 3. / reference.duration)
            qref, slope, stage = reference.evaluate(phase)
            active = stage in {"lift", "transfer", "align", "insert"}
            if not active:
                target[:] = 0
            elif k % 250 == 0:
                obs_t = torch.tensor(plant.GetPositionsAndVelocities(pc), dtype=torch.float32)
                for side, idx in enumerate(indices):
                    if previous[side] is not None:
                        with torch.no_grad():
                            beliefs[side] = filters[side](beliefs[side], previous[side], obs_t)
                    if a.method == "variational":
                        delta, diag = funcs["_score_candidate_actions"](beliefs[side], cost_fn, .9,
                            n_actions=11, n_samples=256, rng=rng)
                        diagnostic = {key: float(diag[key]) for key in ["cvar", "grad_norm_means", "grad_norm_logits"]}
                    else:
                        mu, sigma = np.full(11, .10), np.full(11, .08)
                        for _ in range(3):
                            population = np.clip(rng.normal(mu, sigma, size=(64, 11)), 0., .30)
                            with torch.no_grad():
                                base = float(cost_fn(beliefs[side].rsample(256)).mean())
                            scores = base*(1.+.2*population.mean(axis=1))-.1*(population>.01).mean(axis=1)
                            elite = population[np.argsort(scores)[:12]]
                            mu, sigma = elite.mean(axis=0), elite.std(axis=0)+1e-4
                        delta = mu
                        diagnostic = {"mean_belief_cost": base}
                    previous[side] = torch.tensor(delta, dtype=torch.float32)
                    projected = project_hand_action(delta)
                    target[idx] = np.clip(projected / .35 * a.max_residual, 0., a.max_residual)
                    planner_log.append(dict(time_s=t, hand=["left", "right"][side],
                        action_11=delta.tolist(), projected_6=projected.tolist(),
                        residual_rad=target[idx].tolist(), **diagnostic))
            # Common rate limit in radians per second; arms always receive zero residual.
            residual += np.clip(target-residual, -.2*dt, .2*dt)
            qcommand = np.clip(nominal.command(pc, qref, stage, phase)+residual, adapter.lower, adapter.upper)
            tau = np.clip(adapter.nominal_torque(pc, qcommand, K, D, dt), -adapter.effort, adapter.effort)
            port.FixValue(pc, tau)
            if stage != last_stage:
                events.append(dict(t=t, event="reference_stage", stage=stage)); last_stage = stage
                print(json.dumps(events[-1]), flush=True)
            if k % 5 == 0:
                obs = observe(scene, pc)
                energy = adapter.energy(pc, qref, slope, np.where(hand, 2., 5.), radius=.5)
                rows.append(dict(time_s=t, phase=phase, stage=stage, V=energy.energy, solver_s=0., **obs))
                states.append(plant.GetPositions(pc).copy()); velocities.append(plant.GetVelocities(pc).copy())
                torques.append(tau.copy()); commands.append(qcommand.copy())
    except Exception as exc:
        reason = "simulation_failure: " + str(exc)
        raise
    finally:
        if rows:
            with (output / "trace.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
            np.savez_compressed(output / "states.npz", time_s=[r["time_s"] for r in rows],
                                q=states, v=velocities, tau=torques, command=commands)
        meta = dict(method=a.method, policy=a.method+"_grip_transfer_prototype", seed=a.seed,
            friction=a.friction, dt_s=dt, control_dt_s=dt, duration_s=a.duration,
            contact_stiffness_N_m=2e5, clearance_m=.003, contact_model="hydroelastic",
            max_residual_rad=a.max_residual, residual_rate_limit_rad_s=.2, planner_period_s=.5,
            speed=3., beta=.9, checkpoint=None, events=events, reason=reason, wall_s=time.time()-tick,
            drake=importlib.metadata.version("drake"), actuator_names=adapter.names,
            bimanual_commit=subprocess.check_output(["git", "-C", str(assembly), "rev-parse", "HEAD"], text=True).strip(),
            release_commit=subprocess.check_output(["git", "-C", str(release), "rev-parse", "HEAD"], text=True).strip(),
            source_sha256=hashes,
            limitations=["New bounded hand-residual transfer, not the original full controller or paper evaluation",
                "Shared reference and observed-object arm feedback perform the assembly motion",
                "Untrained neural belief filter; no checkpoint loaded",
                "No benchmark or superiority claim from these two episodes"],
            outcome=summarize(rows, reason=reason, energy_radius=.5, load_limit=30.))
        (output / "planner_log.json").write_text(json.dumps(planner_log, indent=2))
        (output / "summary.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["outcome"], indent=2))


if __name__ == "__main__":
    main()
