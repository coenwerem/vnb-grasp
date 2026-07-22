#!/usr/bin/env python3
"""
Test that VNB and CEM now have deterministic physics after the fresh-env fix.
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_variational_belief_experiments import (
    make_env,
    position_arm_and_object,
    OBJECT_CONFIGS,
    extract_contacts,
)
import mujoco as mj


def run_approach_phase(env, obj_cfg, friction, seed, method_name, steps=10):
    """Run approach phase and record physics state at each step."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Position arm
    ok = position_arm_and_object(env, obj_cfg, friction)
    if not ok:
        return None

    # Record initial state
    record = {
        "method": method_name,
        "seed": seed,
        "friction": friction,
        "qpos": [],
        "contacts": [],
    }

    # Run steps with uniform 0.15 closing (approach phase action)
    for step in range(steps):
        record["qpos"].append(env.data.qpos.copy())
        contacts = extract_contacts(
            env.model, env.data, geom_filter=env.fingertip_geoms
        )
        record["contacts"].append(len(contacts))

        # Apply uniform closing (same action for all methods)
        delta = np.ones(11) * 0.15
        ctrl = env.data.ctrl.copy()
        ctrl[6:17] += delta
        env.data.ctrl[6:17] = np.clip(ctrl[6:17], 0.0, 1.0)

        mj.mj_step(env.model, env.data)

    return record


def main():
    print("=" * 60)
    print("VNB vs CEM DETERMINISM TEST (fresh env per method)")
    print("=" * 60)

    obj_cfg = OBJECT_CONFIGS["cube"]
    friction = 0.425  # nominal
    seed = 123

    # Simulate what happens in the experiment loop:
    # Fresh env for VNB
    print("\n--- Running VNB simulation (fresh env) ---")
    env_vnb = make_env()
    r_vnb = run_approach_phase(env_vnb, obj_cfg, friction, seed, "vnb", steps=10)

    # Fresh env for CEM
    print("\n--- Running CEM simulation (fresh env) ---")
    env_cem = make_env()
    r_cem = run_approach_phase(env_cem, obj_cfg, friction, seed, "cem", steps=10)

    if r_vnb and r_cem:
        print(f"\nComparing VNB vs CEM:")
        print(f"Seed: {seed}, Friction: {friction}")

        all_match = True
        for step in range(len(r_vnb["qpos"])):
            qpos_match = np.allclose(
                r_vnb["qpos"][step], r_cem["qpos"][step], atol=1e-10
            )
            contact_match = r_vnb["contacts"][step] == r_cem["contacts"][step]

            if qpos_match and contact_match:
                print(f"  Step {step}: MATCH (contacts={r_vnb['contacts'][step]})")
            else:
                all_match = False
                print(f"  Step {step}: MISMATCH")
                if not qpos_match:
                    diff = np.abs(r_vnb["qpos"][step] - r_cem["qpos"][step])
                    print(f"    qpos max diff: {diff.max():.2e}")
                if not contact_match:
                    print(
                        f"    contacts: VNB={r_vnb['contacts'][step]} vs CEM={r_cem['contacts'][step]}"
                    )

        print(
            f"\nResult: {'DETERMINISTIC - VNB and CEM have identical physics!' if all_match else 'NON-DETERMINISTIC - Still have state leakage'}"
        )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
