# IROS comparison experiment scripts

These scripts preserve the controller experiments and recording tools used while preparing the September 29, 2026 talk. The final selected box comparison is documented in [the demo report](../media/vnb_cem_opposing_box/README.md). The bimanual code is an exploratory prototype and is not part of the conference presentation or a successful assembly benchmark.

## Script map

| Scripts | Purpose |
| --- | --- |
| `dataset_grasp_trials.py`, `opposing_grasp_setup.py` | Transfer dataset preshapes, run recovered VNB/CEM controllers, constrain shared joint targets, and record contact/state traces. |
| `record_talk_rollout.py` | Record the current or historical runner, with explicit friction, timing, collision, and grasp-initialization options. |
| `record_release_comparison.py` | Record the release comparison runner with matched physics and physical-time lift/hold durations. This is an earlier diagnostic path, not the final box experiment. |
| `stress_saved_opposing.py` | Continue saved grasps through Cartesian lift, force, torque, and friction tests. |
| `audit_dataset_trials.py` | Summarize attempted transfers, including failures. Its hold summary covers the early two-second interval; use the final-box validator for the complete six-second hold. |
| `audit_opposing_trials.py` | Audit the earlier cube/friction-drop experiments named `hold2_drop01_s*`. |
| `validate_opposing_box.py` | Check the selected box's four recorded pairs, initial states, source/model hashes, contact opposition, and external forces. It asserts the recorded selected-case outcomes; it is not a general success-rate evaluator. |
| `render_talk_rollout.py`, `test_talk_comparison.py` | Render recorded MuJoCo states and reject comparisons with mismatched methods, settings, sources, models, or initial positions/velocities. |
| `bimanual_grip_transfer.py`, `render_bimanual_transfer.py`, `test_bimanual_transfer.py` | Apply bounded hand residuals to the shared Drake assembly controller, record/render the prototype, and check the 11-joint to 6-actuator coupling map. |

## Historical dependencies

The scripts retain their original experiment paths and source snapshots. They are not a self-contained rerun of the paper evaluation; use [REPRODUCE.md](../REPRODUCE.md) for the repository's benchmark workflow.

- The final box experiments and their binary models use **MuJoCo 3.4.0**. Binary `.mjb` replay requires the matching MuJoCo version. Install the repository's Python dependencies plus Pillow for rendering; FFmpeg and the DejaVu fonts are also required.
- `dataset_grasp_trials.py` imports the recovered source tree at `outputs/iros_advantage_diagnosis_20260929/recovered_mar01_snapshot`. That local tree includes the historical `mujocodex` package, runner, arenas, assets, and configuration. It is not included in this commit. Its `recovery_manifest.json` identifies the three files recovered from editor history; this is not an untouched submission tag.
- `record_talk_rollout.py` uses the sibling `MuJoCoDex` repository at revision `cd9c77fa` for historical modes, and `MuJoCoDex_iros_release` for release assets and libraries. The release recorder also requires that checkout; its `--release` argument can select another location.
- `opposing_grasp_setup.py` references the local `realhand_o6_iron_hw_v2/cube.json` dataset. The final box runner instead receives a candidate JSON through `--bank`; pass an absolute path because the runner changes its working directory to the recovered source tree.
- The bimanual scripts additionally require Drake and the local `bimanual-assembly` repository, including its reference seed and assets. Their release and assembly paths are recorded in `bimanual_grip_transfer.py`.

The final box run used candidate 82 from `candlog_graspit_box.jsonl`, object `graspit_box`, yaw 90°, friction 0.70, maximum 80 controller steps, beta 0.9, torque limit 2.0 N·m, and seeds 42, 123, 456, and 789. Each method was run with `--lock-opposition --freeze-inactive --always-lift --hold-seconds 6 --audit-every 1`. The controller code retains its original method-specific action scoring, stopping, and VNB tightening; it loads no trained belief checkpoint.

Each recording writes source snapshots, settings, contact audits, and state traces under its output directory. Full simulation logs, datasets, binary models, and the recovered source tree remain outside version control. The final paired runs also have `model_sha256` in their provenance; compute that from each saved `model.mjb` before invoking the renderer or validator on newly recorded dataset trials.

## Validation and replay

Run the focused tests in an environment with NumPy, MuJoCo, and Pillow:

```bash
PYTHONPATH=examples python -m unittest test_talk_comparison test_bimanual_transfer -v
```

With the original eight recording directories available, audit the full selected-case hold:

```bash
python examples/validate_opposing_box.py outputs/iros_opposing_box_final_20260929
```

Render the final seed-42 comparison with native hand/thumb colors:

```bash
MUJOCO_GL=egl python examples/render_talk_rollout.py   outputs/iros_opposing_box_final_20260929/seed42_variational   outputs/iros_opposing_box_final_20260929/seed42_cem   --output outputs/iros_opposing_box_final_20260929/comparison.mp4   --start -0.7 --end 6.98 --speed 1   --title 'Opposing grasp: VNB vs CEM'   --camera-azimuth 165 --camera-elevation -5 --camera-distance 0.48   --camera-lookat -0.05 0.88 0.88   --protocol-label 'Same scene + initial state'   --height-readout --align-event lift_and_shear
```

The default renderer preserves the model's thumb color. `--highlight-thumb` is an optional visual-only override and was not used for the final published video.
