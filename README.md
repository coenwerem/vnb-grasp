# VNB-Grasp

Variational neural belief parameterizations for robust dexterous grasping under
multimodal uncertainty (IROS 2026). This repository holds two things of equal
weight. A library that represents a belief over latent contact parameters and
object pose as a differentiable Gaussian mixture, so that pathwise gradients
reach a smooth CVaR surrogate. A benchmark that scores risk-sensitive grasp
planners in MuJoCo across three friction regimes and a perturbation battery,
and regenerates every table and figure in the paper from one experiment sweep.

- Paper, `ARXIV_URL_PENDING`
- Project page, <https://www.clintonenwerem.com/vnb-grasp>

## Installation

VNB-Grasp needs Python 3.10 or newer. The core library depends on PyTorch,
NumPy, and SciPy. Simulation adds MuJoCo 3 and robosuite, the differentiable
metrics add JAX, and the figure scripts add matplotlib.

```bash
git clone https://github.com/coenwerem/vnb-grasp.git
cd vnb-grasp
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

In practice, you may want only part of the stack.

```bash
pip install -e .                # belief library only
pip install -e ".[sim]"         # add MuJoCo and robosuite
pip install -e ".[diff]"        # add the JAX differentiable metrics
pip install -e ".[figures]"     # add matplotlib and pillow
```

In addition, headless machines need an offscreen GL backend, so export
`MUJOCO_GL=egl` before any simulation command below.

## Library quickstart

The belief is a Gaussian mixture over per-contact latent parameters.
Gumbel-Softmax component selection and location-scale reparameterization keep
every sample a smooth function of the mixture logits, means, and log standard
deviations, so `torch.autograd` propagates a CVaR gradient straight back to
those parameters.

```python
import torch
from vnb_grasp.belief import GaussianMixtureBelief, VariationalBeliefConfig

# Eight mixture components over five contacts, four latent parameters each
config = VariationalBeliefConfig(n_components=8, n_contacts=5)
belief = GaussianMixtureBelief(config)

# Reparameterized samples keep gradients back to the mixture parameters
samples = belief.rsample(256)
print(samples.shape, samples.requires_grad)   # torch.Size([256, 20]) True

# Any differentiable cost over latent contact parameters plugs in here
def cost_fn(theta):
    return -theta[:, 0]        # stand-in, lower friction costs more

out = belief.cvar_gradient(cost_fn, beta=0.9, n_samples=256)
print(float(out["cvar"].detach()))      # CVaR of the worst 10 percent
print(out["means_grad"].shape)          # gradient with respect to the means
print(float(belief.entropy().detach()))  # belief spread
```

Specifically, `cvar_gradient` returns the CVaR value alongside gradients for
`mixture_logits`, `means`, and `log_stds`. A particle filter cannot supply
those, since resampling breaks the path from the risk measure to the belief.

In addition, the JAX side holds the grasp-quality metrics that the planner
differentiates.

```python
from vnb_grasp.grasping import analyze_gws, ferrari_canny_quality
from vnb_grasp.belief.differentiable_metrics import (
    soft_epsilon_metric, cvar_metric, grasp_fragility,
)
```

For example, `python examples/test_differentiable_metrics.py` walks through the
soft epsilon metric, gradient-based grasp optimization, and grasp fragility.

## Benchmark reproduction

Every number and figure in the paper comes from one experiment sweep followed
by the analysis scripts. The table below maps each paper result to its command.

| Paper result | Command |
| --- | --- |
| Sanity check, 5 methods, 1 episode each, about 6 minutes | `MUJOCO_GL=egl python examples/run_variational_belief_experiments.py --quick --tag sanity` |
| Table I, aggregate performance across friction regimes | `MUJOCO_GL=egl python examples/run_variational_belief_experiments.py --tag iros26_full --episode-timeout 300` |
| Table I plus the per-object supplementary table, as LaTeX | `python examples/generate_paper_tables.py outputs/variational_belief_experiments/experiment_iros26_full_*.json -o tables/` |
| Fig. 4, per-regime bar chart | `python examples/plot_regime_bars.py --json outputs/variational_belief_experiments/experiment_iros26_full_*.json --output regime_bars.pdf` |
| Fig. 5, grasp quality under force perturbations | `python examples/plot_eps_trajectory.py --json outputs/variational_belief_experiments/experiment_iros26_full_*.json --output eps_trajectory.pdf` |
| Fig. 1, teaser panels | `MUJOCO_GL=egl python examples/side_figure1_teaser_simplified.py --object soup_can` |
| Summary statistics quoted in the text | `python examples/compute_paper_stats.py outputs/variational_belief_experiments/experiment_iros26_full_*.json` |
| Single belief-MPC grasp episode | `MUJOCO_GL=egl python examples/run_belief_mpc_grasp.py` |
| Analytic grasp synthesis under assumed contacts | `MUJOCO_GL=egl python examples/run_grasp_optimization.py --object cube --method sqp` |
| Determinism check across VNB and CEM | `MUJOCO_GL=egl python examples/test_vnb_cem_determinism.py` |

In practice, the full sweep runs 5 methods times 2 objects times 4 risk levels
times 3 seeds times 4 friction regimes, so 480 episodes. Expect 4 to 8 hours on
one machine, with the particle-filter baseline dominating the wall clock.
Narrow the sweep with `--methods`, `--objects`, `--betas`, `--seeds`, and
`--regimes`.

In particular, `config/iros26_experiments.yaml` holds every benchmark constant
in one place, including the three friction regimes, the lift-and-shear stress
test, the 28-test perturbation battery, and the termination and success
criteria. Change a number there rather than in the runner.

## Repository layout

```
vnb_grasp/
  belief/            variational belief, particle filter, belief MPC,
                     neural belief dynamics, JAX differentiable metrics
  grasping/          grasp wrench space and Ferrari-Canny quality,
                     risk-sensitive CVaR metrics, grasp synthesis,
                     pregrasp planning, YCB object configuration
  control/           actuator bookkeeping for the arm and hand
  envs/              MuJoCo arena loading and a gym adapter
  robosuite_ext/     robosuite arenas, models, environment registration
  scripted_policies/ geometry-aware pregrasp policies
  visualization/     grasp, camera, and overlay rendering
  wrappers/          raw MuJoCo environment wrapper
examples/            experiment runners, analysis, figure scripts
config/              benchmark and control configuration
arenas/              MuJoCo scenes for the 6-DoF arm and RealHand L6
assets/              meshes and textures the arenas reference
grasp_db/            grasp candidate databases per object
```

## Roadmap

Planned work, in rough priority order.

1. Split the benchmark into a standalone package so other planners score
   against the same regimes and perturbation battery without depending on the
   VNB planner.
2. One-command Docker reproduction that regenerates the paper figures from a
   fixed image.
3. Notes on driving a physical arm and multifingered hand, covering the tactile
   grasp-quality proxy and the pose-estimation front end. The hardware
   interface itself stays out of this repository.

## Citation

```bibtex
@inproceedings{enweremVariationalNeuralBeliefParameterizations2026,
  title     = {Variational Neural Belief Parameterizations for Robust Dexterous Grasping under Multimodal Uncertainty},
  author    = {Enwerem, Clinton and Kalyanaraman, Shreya and Baras, John S. and Belta, Calin},
  booktitle = {Proceedings of the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026},
}
```

[CITE NEEDED: arXiv identifier for the VNB-Grasp preprint. Add the eprint,
eprinttype, eprintclass, doi, and url fields once the real identifier is
confirmed, and replace ARXIV_URL_PENDING above.]

## License

Apache-2.0. See [LICENSE](LICENSE).
