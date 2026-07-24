# Reproducing vnb-grasp's Simulation Results

This guide reproduces the simulation-based tables and summary statistics in
"Variational Neural Belief Parameterizations for Robust Dexterous Grasping
under Multimodal Uncertainty" (IROS 2026). It is simulation-only: the paper's
hardware trials are out of scope for this repository, and no figure-rendering
step is included here, only the data each figure is built from.

Install with `pip install -e ".[all]"` first (see [README.md](README.md)),
and export `MUJOCO_GL=egl` on headless machines before running anything below.

## Benchmark protocol

The sweep that produced the paper's main results table:

- **5 methods**: particle filter (PF), Gaussian, Gaussian-CVaR, CEM, VNB (ours)
- **2 objects**: cube, graspit_box
- **4 friction regimes**: nominal, adversarial, wide, bimodal
- **4 risk levels (beta)**: 0.5, 0.9, 0.95, 0.99
- **3 seeds**: 42, 123, 456
- **28-test perturbation battery** applied post-grasp, regardless of nominal success
- **Total**: 5 x 2 x 4 x 3 x 4 = **480 episodes**

Every constant above is defined once in `config/iros26_experiments.yaml`.
Change a number there rather than in the runner.

## Commands

| Result | Command |
| --- | --- |
| Sanity check: 5 methods, 1 episode each (cube, beta=0.9, seed=42, nominal) | `MUJOCO_GL=egl python examples/run_variational_belief_experiments.py --quick --tag sanity` |
| Full 480-episode sweep, 4-8 hours (particle filter dominates) | `MUJOCO_GL=egl python examples/run_variational_belief_experiments.py --tag iros26_full --episode-timeout 300` |
| Table I (per-method, per-regime), Table II tail stats, and the per-object supplementary table, as LaTeX, plus a plain-text summary | `python examples/generate_paper_tables.py outputs/variational_belief_experiments/experiment_iros26_full_*.json -o tables/` |
| Single belief-MPC grasp episode | `MUJOCO_GL=egl python examples/run_belief_mpc_grasp.py` |
| Determinism check across VNB and CEM | `MUJOCO_GL=egl python examples/test_vnb_cem_determinism.py` |

## Parallel execution

The full sweep is embarrassingly parallel across methods or regimes.

Split by method (particle filter is roughly 10x slower than the rest, so
isolate it):

```bash
python examples/run_variational_belief_experiments.py \
  --methods gauss gauss_cvar cem variational \
  --episode-timeout 300 --tag iros26_fast

python examples/run_variational_belief_experiments.py \
  --methods particle \
  --episode-timeout 300 --tag iros26_particle
```

Split by friction regime (4-way parallel, 120 episodes each):

```bash
python examples/run_variational_belief_experiments.py --regimes nominal     --tag regime_nominal
python examples/run_variational_belief_experiments.py --regimes adversarial --tag regime_adversarial
python examples/run_variational_belief_experiments.py --regimes wide        --tag regime_wide
python examples/run_variational_belief_experiments.py --regimes bimodal     --tag regime_bimodal
```

Merge the resulting JSONs by passing all of them to `generate_paper_tables.py`
at once. Narrow any run further with `--methods`, `--objects`, `--betas`,
`--seeds`, and `--regimes`.

## Output format

Results save to
`outputs/variational_belief_experiments/experiment_<tag>_<timestamp>.json`,
with an intermediate save after every episode (crash-safe).

| JSON field | Paper column |
| --- | --- |
| `success` | SR% |
| `robust_success` | Robust% |
| `perturbation_survival_rate` | PertSurv% |
| `final_epsilon` | epsilon |
| `final_contact_quality` | Quality |
| `failure_prob_empirical` | P_fail (empirical) |
| `runtime_s` | Time (s) |
| `friction_regime` | Regime grouping key |
| `perturbation_details` | Per-test breakdown |

## Out of scope here

- **Figure rendering.** Figs. 1, 4, and 5 in the paper are rendered from the
  same JSON this guide produces; the rendering scripts themselves are not
  part of this release.
- **Hardware trials.** Out of scope for this repository entirely.
