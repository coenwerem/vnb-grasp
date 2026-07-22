# Variational Neural Beliefs for Risk-Aware Manipulation

New neural belief representations replacing discrete particle filters with continuous, differentiable distributions.

## Xteristics

- **Continuous Belief Representations**: Gaussian mixtures and implicit neural fields
- **Exact Risk Gradients**: Direct CVaR optimization through automatic differentiation  
- **Adaptive Complexity**: Resource allocation based on epistemic uncertainty
- **End-to-End Learning**: Differentiable belief dynamics and observation models

## Usage

```python
from vnb_grasp.belief import (
    VariationalBeliefConfig, 
    GaussianMixtureBelief,
    RiskAwareNeuralPolicy
)

# Configure neural belief system
config = VariationalBeliefConfig(
    belief_latent_dim=64,
    n_components=8,
    cvar_beta=0.9,  # Risk level
    uncertainty_threshold=0.1
)

# Create continuous belief representation
belief = GaussianMixtureBelief(config)
policy = RiskAwareNeuralPolicy(config)

# Plan with exact risk gradients
action, updated_belief = policy(observation, belief)

# Compute exact CVaR (not approximated!)
risk_gradients = belief.cvar_gradient(costs, beta=0.9)
```

## Advanced Usage

### Implicit Neural Beliefs
For complex multimodal uncertainties:
```python
from vnb_grasp.belief import ImplicitNeuralBelief

# High-capacity belief networks
implicit_belief = ImplicitNeuralBelief(config)
samples = implicit_belief.sample(1000)  # Langevin MCMC
```

### Adaptive Complexity
Dynamic resource allocation:
```python
from vnb_grasp.belief import AdaptiveBeliefManager

manager = AdaptiveBeliefManager(config)
belief = manager(observation)  # Selects appropriate complexity
```

### Meta-Learning
Rapid adaptation to new objects:
```python
# Train belief dynamics across objects
meta_trainer = MetaBeliefLearner()
adapted_belief = meta_trainer.adapt_to_object(few_shot_data)
```

## Integration

```python
# Particle MPC
from vnb_grasp.belief import BeliefMPCPlanner  # nominal particle-based
# vs
from examples.variational_belief_demo import NeuralBeliefMPC  # variational

# internal continuous beliefs
action = planner.plan(observation, state)
```

## Possible Extensions

- **Higher-order risk measures**: Spectral risk, entropic risk
- **Multi-agent beliefs**: Belief interactions in collaborative manipulation  
- **Hardware deployment**: Real-time performance optimization
- **Normalizing flows**: Alternative to MCMC sampling in implicit beliefs

--- 

## Simulation Experiments (IROS 2026)

### Protocol
- **5 methods**: PF, Gauss, Gauss-CVaR, CEM, VNB (Ours)
- **2 objects**: cube, graspit_box
- **4 friction regimes**: nominal, adversarial, wide, bimodal
- **4 β values**: 0.5, 0.9, 0.95, 0.99
- **3 seeds**: 42, 123, 456
- **32-test perturbation battery**: lateral impulses (20) + torque impulses (9) + friction drops (3)
- **Finger control**: Direct torque (GRASP_TORQUE profile from pregrasp planner), ramped 0-->1 over 30 MPC steps
- **Total**: 5 x 2 x 4 x 3 x 4 = **480 episodes**

---

### Full 480-episode sweep (recommended)

```bash
cd vnb-grasp
python examples/run_variational_belief_experiments.py \
  --episode-timeout 300 --tag iros26_full
```

Runtime: ~4–8 hours depending on hardware (particle filter is the bottleneck).

### Quick sanity check (1 episode)

```bash
python examples/run_variational_belief_experiments.py --quick --tag sanity
```

Runs: 1 episode (variational, cube, β=0.9, seed=42, nominal). Takes ~10 seconds.

---

## Parallel Execution

### Option A: Split by method (best parallelism)

Particle filter is ~10 slower than the other methods. Run it separately:

**Terminal 1** --- fast methods (~30–60 min):
```bash
python examples/run_variational_belief_experiments.py \
  --methods gauss gauss_cvar cem variational \
  --episode-timeout 300 --tag iros26_fast
```

**Terminal 2** --- particle filter (~4–8 hrs):
```bash
python examples/run_variational_belief_experiments.py \
  --methods particle \
  --episode-timeout 300 --tag iros26_particle
```

### Option B: Split by friction regime (4-way parallel)

```bash
# Terminal 1
python examples/run_variational_belief_experiments.py --regimes nominal     --tag regime_nominal
# Terminal 2
python examples/run_variational_belief_experiments.py --regimes adversarial --tag regime_adversarial
# Terminal 3
python examples/run_variational_belief_experiments.py --regimes wide        --tag regime_wide
# Terminal 4
python examples/run_variational_belief_experiments.py --regimes bimodal     --tag regime_bimodal
```

Each regime = 120 episodes (5 methods x 2 objects x 4 β x 3 seeds).

### Option C: Split by method (5-way parallel)

```bash
python examples/run_variational_belief_experiments.py --methods particle    --tag method_pf
python examples/run_variational_belief_experiments.py --methods gauss       --tag method_gauss
python examples/run_variational_belief_experiments.py --methods gauss_cvar  --tag method_gcvar
python examples/run_variational_belief_experiments.py --methods cem         --tag method_cem
python examples/run_variational_belief_experiments.py --methods variational --tag method_vnb
```

Each method = 96 episodes (2 objects x 4 β x 3 seeds x 4 regimes).

### Selective reruns

```bash
# Rerun just particle filter on specific conditions
python examples/run_variational_belief_experiments.py \
  --methods particle \
  --regimes nominal adversarial \
  --objects graspit_box \
  --betas 0.9 0.95 \
  --seeds 42 123 456 \
  --tag pf_graspit_rerun
```

---

## Generating Paper Tables

After experiments complete, generate LaTeX tables automatically:

```bash
# Print tables to stdout
python examples/generate_paper_tables.py \
  outputs/variational_belief_experiments/iros26/experiment_iros26_full_*.json

# Merge multiple parallel-run JSONs
python examples/generate_paper_tables.py \
  outputs/variational_belief_experiments/iros26/experiment_iros26_fast_*.json \
  outputs/variational_belief_experiments/iros26/experiment_iros26_particle_*.json

# Write .tex files directly into the paper directory
python examples/generate_paper_tables.py \
  --output-dir docs/tex/IROS2026_Variational_Neural_Beliefs_for_Robust_Dexterous_Grasping_Under_Multimodal_Uncertainty/tables \
  outputs/variational_belief_experiments/iros26/experiment_*.json
```

Produces:
| File | Paper Reference |
|---|---|
| `tab_main_results.tex` | Table I --- aggregate per-method, per-regime |
| `tab_tail_stats.tex` | Table II --- tail statistics of contact quality |
| `tab_per_object.tex` | Supplementary --- per-object breakdown |

---

## Output Format

Results save to `outputs/variational_belief_experiments/iros26/experiment_<tag>_<timestamp>.json`.
Intermediate saves after every episode (crash-safe).

### Key fields per episode (matching paper tables)

| JSON Field | Paper Column |
|---|---|
| `success` | SR% |
| `robust_success` | Robust% |
| `perturbation_survival_rate` | PertSurv% |
| `final_epsilon` | ε |
| `final_contact_quality` | Quality |
| `failure_prob_empirical` | P̂_fail |
| `runtime_s` | Time (s) |
| `friction_regime` | Regime grouping key |
| `failure_mode` | Failure categorization |
| `perturbation_details` | Per-test breakdown (32 tests) |
| `lift_height_achieved` | Lift height (m) |
| `time_to_slip` | Time to first slip (s) |
| `peak_slip_distance` | Max slip distance (m) | 