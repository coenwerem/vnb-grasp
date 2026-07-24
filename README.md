# vnb-grasp

Code (pip-installable library, [MuJoCo](https://mujoco.org/) simulation assets and programs, and [GraspIt](https://github.com/graspit-simulator/graspit)!-generated grasps (in JSON format)) for the paper, "Variational Neural Belief Parameterizations for Robust Dexterous Grasping under Multimodal Uncertainty," by C. Enwerem, S. Kalyanaraman, J. S. Baras, and C. Belta, to appear in the Proceedings of the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). [arXiv Preprint](https://arxiv.org/abs/2604.25897).

### Citation
If you find VNB-Grasp (either the code, simulation assets, grasp dataset, benchmark tooling, or the paper) useful in your work, please cite us using the following BibTeX entry:
```bibtex
@article{enweremVariationalNeuralParameterizations2026a,
  author     = {Enwerem, Clinton and Kalyanaraman, Shreya and Baras, John S. and Belta, Calin},
  title      = {{Variational} {Neural} {Parameterizations} {for} {Robust} {Dexterous} {Grasping} {under} {Multimodal} {Uncertainty}},
  year       = {2026},
  eprint     = {2604.25897},
  eprinttype = {arxiv},
  note       = {Preprint, arXiv:2604.25897}
}
```

## Overview
This repository holds two components: a belief-based multimodal uncertainty representation module and a grasp robustness benchmark based on MuJoCo. The uncertainty representation module mirrors the paper's modeling choices, casting uncertainty as a belief over latent contact parameters and object pose represented by a differentiable Gaussian mixture. Unlike particle filters that obstruct gradient-based risk-sensitive optimization, VNB-Grasp's differentiable belief enables the computation of pathwise gradients of a smooth CVaR surrogate. Our MuJoCo-based grasp robustness benchmark tests the friction sensitivity and perturbation survival of hand grasp planner-executors across four friction regimes and a perturbation battery.

For reproducing the paper's simulation results table by table, see [REPRODUCE.md](REPRODUCE.md).

## Installation
VNB-Grasp needs Python 3.10 or newer. The core library depends on PyTorch,
NumPy, SciPy, and MuJoCo 3. The differentiable metrics add JAX, and a few
optional plotting helpers inside the grasping module add matplotlib.

```bash
git clone https://github.com/coenwerem/vnb-grasp.git
cd vnb-grasp
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

In practice, you may want only part of the stack.
```bash
pip install -e .                # belief library, grasp metrics, and the MuJoCo benchmark
pip install -e ".[diff]"        # add the JAX differentiable metrics
pip install -e ".[figures]"     # add matplotlib and pillow
```

In addition, headless machines need an offscreen GL backend, so export
`MUJOCO_GL=egl` before any simulation command below.

## Library Quickstart
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

## Friction-Sensitivity & External Force Perturbation Benchmark
The benchmark runs a hand grasp planner-executor against a MuJoCo scene
across friction regimes and a post-grasp perturbation battery of lateral
impulses, torque impulses, and sudden friction drops, then scores each
episode for grasp success, robustness, and perturbation survival.

Try it directly:

```bash
# Single belief-MPC grasp episode
MUJOCO_GL=egl python examples/run_belief_mpc_grasp.py

# Sanity check: 5 methods, 1 episode each, about 6 minutes
MUJOCO_GL=egl python examples/run_variational_belief_experiments.py --quick --tag sanity
```

`config/iros26_experiments.yaml` holds every benchmark constant in one place,
including the friction regimes, the lift-and-shear stress test, the
perturbation battery, and the termination and success criteria. Change a
number there rather than in the runner.

For the full parameter sweep and the exact commands that regenerate every
table in the paper, see [REPRODUCE.md](REPRODUCE.md), which covers simulation
reproduction only.

## Repository Layout
```
vnb_grasp/
  belief/            variational belief, particle filter, belief MPC,
                     neural belief dynamics, JAX differentiable metrics
  grasping/          grasp wrench space and Ferrari-Canny quality,
                     risk-sensitive CVaR metrics, grasp synthesis,
                     pregrasp planning, YCB object configuration
  control/           actuator bookkeeping for the arm and hand
  envs/              MuJoCo arena loading and a gym adapter
  scripted_policies/ geometry-aware pregrasp policies
  wrappers/          raw MuJoCo environment wrapper
examples/            experiment runners, analysis, table generation
config/              benchmark and control configuration
arenas/              MuJoCo scenes for the 6-DoF arm and RealHand L6
assets/              meshes and textures the arenas reference
grasp_db/            GraspIt!-generated grasp candidate databases per object
```

## License
Apache-2.0. See [LICENSE](LICENSE).
