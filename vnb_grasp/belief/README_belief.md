# vnb_grasp.belief

Belief-space planning for dexterous manipulation under contact uncertainty.

## Modules

| Module | Description |
|--------|-------------|
| `belief_mpc.py` | Belief-space MPC planner with CVaR risk measure |
| `particle_filter.py` | Particle filter for belief propagation |
| `contact_belief.py` | Contact-specific belief representations |
| `mujoco_rollout.py` | MuJoCo rollout utilities for MPC |
| `env_wrapper.py` | Environment wrapper integrating belief MPC |
| `differentiable_metrics.py` | JAX-based differentiable grasp quality metrics |

## Core Classes

### BeliefMPCPlanner

The main belief-space MPC planner:

```python
from vnb_grasp.belief import BeliefMPCPlanner, BeliefMPCConfig

config = BeliefMPCConfig(
    horizon=10,
    n_particles=100,
    cvar_alpha=0.9,
    lambda_cvar=0.5,
)
planner = BeliefMPCPlanner(config)
action = planner.plan(belief, env_state)
```

### ParticleBelief

Particle-based belief representation:

```python
from vnb_grasp.belief import ParticleBelief

belief = ParticleBelief(particles, weights)
belief = belief.update(observation, likelihood_fn)
entropy = belief.entropy()
```

## Risk-Sensitive Planning

The planner uses CVaR (Conditional Value-at-Risk) for tail-risk management:

- `cvar_alpha`: Quantile level (0.9 = worst 10%)
- `lambda_cvar`: Weight of CVaR vs expected cost

## Usage

### Run Belief MPC

```bash
# Basic grasp planning with belief-space MPC
python examples/run_belief_mpc_grasp.py --steps 100

# With video recording (inline)
python examples/run_belief_mpc_grasp.py --steps 100 --record

# Risk-sensitive parameters
python examples/run_belief_mpc_grasp.py --beta 0.95 --lambda-cvar 0.7 --particles 200

# With lift test to verify grasp stability
python examples/run_belief_mpc_grasp.py --steps 100 --lift-test --lift-height 0.05
```

### Video Recording

```bash
# Record belief-MPC episodes with 4x slow-motion
python examples/record_belief_mpc.py --episodes 2 --slowdown 4

# With lift test and metric overlay
python examples/record_belief_mpc.py --episodes 3 --slowdown 4 --lift-test --overlay

# Custom camera and resolution
python examples/record_belief_mpc.py --camera agent-view --width 1920 --height 1080

# Vary risk parameters across recordings
python examples/record_belief_mpc.py --beta 0.95 --lambda-cvar 0.7 --particles 200
```

Videos are saved to `outputs/video_demos/`.

### Generate Expert Demonstrations

```bash
# Collect belief-MPC expert trajectories for BC training
python examples/run_belief_distillation.py --collect --num-episodes 100 --output-dir data/expert

# Train BC policy from demonstrations
python examples/run_belief_distillation.py --train --data-dir data/expert
```

## See Also

- [examples/run_belief_mpc_grasp.py](../../examples/run_belief_mpc_grasp.py) -- headless belief-MPC run (with optional `--record`)
- [examples/record_belief_mpc.py](../../examples/record_belief_mpc.py) -- slow-motion video recording wrapper
- [examples/record_autonomous_grasp.py](../../examples/record_autonomous_grasp.py) -- video recording for scripted policy
- [examples/run_belief_distillation.py](../../examples/run_belief_distillation.py)
