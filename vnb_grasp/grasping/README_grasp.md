# vnb_grasp.grasping

Grasp quality metrics, sampling-based grasp synthesis, and GraspIt! integration.

## Modules

| Module | Description |
|--------|-------------|
| `grasp_sampler.py` | Sampling-based multi-finger grasp solver with collision-aware IK |
| `object_surface.py` | Object surface representation, SDF, and contact-point sampling |
| `gws_quality.py` | Grasp Wrench Space analysis (Ferrari-Canny, GWS volume) |
| `rs_quality.py` | Risk-sensitive quality metrics (CVaR-based) |
| `ycb_objects.py` | YCB object definitions and GraspIt! transforms |
| `grasp_loader.py` | GraspIt! database loading and best-grasp retrieval |

## Grasp Sampler Pipeline

The `GraspSampler` runs a seven-stage pipeline per trial:

1. **Surface sampling**: `ObjectSurface.sample()` draws N candidate
   contact points (with outward normals) on the object.
2. **Finger assignment**: `FingerAssigner.assign()` greedily matches
   fingers to surface points with three cost components:
   - *Distance*: prefer nearby targets (weight `w_distance=1.0`)
   - *Normal alignment*: mild preference for approach-aligned faces
     (`w_normal=0.1`, reduced from 0.3 to avoid same-face clustering)
   - *Opposition diversity*: bonus for targets on faces that
     **oppose** already-assigned normals (`w_opposition=0.4`,
     GraspIt!-inspired virtual contact diversity)
   - *Reachability mask*: `dot(finger−point, normal) > 0`
3. **Contact IK**: `ContactIKSolver.solve()` drives fingertips via
   damped-least-squares IK to targets offset by `contact_margin` along
   outward normals.  An in-loop barrier detects when any fingertip
   enters the object (SDF < 0) and *replaces* the error with a pure
   outward push (weight `penetration_weight=30`).
4. **SDF projection refinement** - after DLS IK
   converges, up to `sdf_refine_iters` additional DLS steps drive each
   fingertip to SDF=0 (exactly on the surface).  This enforces
5. **Post-IK validation**: four hard filters reject bad grasps:
   - **Penetration**: worst fingertip SDF ≥ −`max_penetration` (3 mm)
   - **Contact validity**: tips within `contact_tolerance` (15 mm)
   - **Finger separation**: pairwise ≥ `min_finger_separation` (5 mm)
   - **Minimum contacts**: ≥ `min_valid_contacts` (3)
6. **Distance-weighted GWS** *(GraspIt!-inspired)*: contact force
   scaled by proximity: `w = (cos(π·d/50mm)+1)/2`, decaying from 1.0
   at the surface to 0.0 at 50 mm.  Ferrari-Canny epsilon computed
   from validated contacts only.
7. **Ranking**: grasps sorted by descending epsilon quality.

### Approach Pose Generation

The solver generates diverse initial hand poses from 9 canonical
approach directions (6 cardinal + 3 diagonal) plus random spherical
sampling biased toward side approaches (θ ∼ arccos(U[0,1])).  Each
pose uses:
- **Closer approach**: hand center at 30-60 mm from object center
  (close enough for fingers to wrap around)
- **Head-on orientation**: palm normal aligned toward object, with
  small random roll perturbation (±0.3 rad)
- **Pre-shaped fingers**: 40% of joint range ± 15% Gaussian noise

### Key Parameters (`SamplerConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ik_max_iter` | 300 | DLS iterations (up from 80) |
| `ik_step_size` | 0.25 | Integration step (slightly conservative) |
| `penetration_weight` | 30.0 | IK barrier push magnitude |
| `contact_margin` | 1 mm | Target offset along outward normal |
| `max_penetration` | 3 mm | Max allowable pen depth |
| `contact_tolerance` | 15 mm | Max SDF for valid contact |
| `min_finger_separation` | 5 mm | Min pairwise finger-finger gap |
| `min_valid_contacts` | 3 | Reject grasps with fewer |
| `sdf_refine_iters` | 30 | SDF-->0 projection refinement steps |
| `sdf_refine_tol` | 0.5 mm | SDF convergence threshold |
| `friction_coef` | 0.8 | Assumed μ for GWS (rubber: 0.8) |

## GraspIt! - MuJoCo Transform
- L6 Righ Hand DOF mapping:
```text
GraspIt DOF name           --> MuJoCo qpos index (joint name)
---------------------------|-----------------------------
GraspIt d0 (index_mcp)     --> MuJoCo qpos[9]  (index_mcp_pitch)   --> hand_idx 3
GraspIt d1 (index_dip)     --> MuJoCo qpos[10] (index_dip)          --> hand_idx 4
GraspIt d2 (middle_mcp)    --> MuJoCo qpos[11] (middle_mcp_pitch)   --> hand_idx 5
GraspIt d3 (middle_dip)    --> MuJoCo qpos[12] (middle_dip)         --> hand_idx 6
GraspIt d4 (pinky_mcp)     --> MuJoCo qpos[15] (pinky_mcp_pitch)   --> hand_idx 9
GraspIt d5 (pinky_dip)     --> MuJoCo qpos[16] (pinky_dip)         --> hand_idx 10
GraspIt d6 (ring_mcp)      --> MuJoCo qpos[13] (ring_mcp_pitch)    --> hand_idx 7
GraspIt d7 (ring_dip)      --> MuJoCo qpos[14] (ring_dip)          --> hand_idx 8
GraspIt d8 (thumb_cmc_yaw) --> MuJoCo qpos[6]  (thumb_cmc_yaw)     --> hand_idx 0
GraspIt d9 (thumb_cmc_pitch)--> MuJoCo qpos[7] (thumb_cmc_pitch)   --> hand_idx 1
GraspIt d10 (thumb_ip)     --> MuJoCo qpos[8]  (thumb_ip)          --> hand_idx 2
```

## Core Functions

### Ferrari-Canny Quality

```python
from vnb_grasp.grasping import ferrari_canny_quality, GWSResult

# Compute GWS quality from contact wrenches
result: GWSResult = ferrari_canny_quality(contact_points, contact_normals, friction_coeff)
print(f"Epsilon: {result.epsilon:.3f}, Force closure: {result.is_force_closure}")
```

### GraspIt! Database

```python
from vnb_grasp.grasping import get_object_config, load_grasp_database

config = get_object_config("cube")
db = load_grasp_database(config)
best_grasp = db.best_grasp()
print(f"Best grasp epsilon: {best_grasp.epsilon_quality:.3f}")
```

### Transform to MuJoCo

```python
from vnb_grasp.grasping import GraspItToMuJoCoTransform

# Transform GraspIt! grasp to MuJoCo frame
mj_grasp = GraspItToMuJoCoTransform.grasp_to_mujoco(graspit_grasp, mesh_scale)
```

## Risk-Sensitive Quality

```python
from vnb_grasp.grasping import RiskSensitiveGraspQuality

quality = RiskSensitiveGraspQuality(alpha=0.9)  # CVaR at 90th percentile
score = quality.compute(contact_wrenches, friction_samples)
```

## See Also

- [grasp_db/](../../grasp_db/) - Pre-computed GraspIt! grasps
