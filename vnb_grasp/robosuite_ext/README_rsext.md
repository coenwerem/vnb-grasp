# vnb_grasp.robosuite_ext

Robosuite extensions: custom environments, robots, and models.

## Structure

```
robosuite_ext/
├#### environments/       # Custom robosuite environments
│   ├#### vnb_grasp_lift.py
│   └#### vnb_grasp_lift_touch.py
├#### models/             # Robot and gripper models
│   ├#### zarm.py
│   └#### realhand_l6_right.py
├#### arenas/             # Arena definitions
│   └#### zarm_table_astra_arena.py
├#### sensors/            # Custom sensors
│   └#### realhand_l6_tactile_mock.py
├#### register_all.py     # Registration utilities
└#### paths.py            # Path configuration
```

## Registration

Before using VNB-Grasp robosuite environments, register them:

```python
from vnb_grasp.robosuite_ext.register_all import register_all

register_all()

# Now environments are available
import robosuite as suite
env = suite.make("VNBGraspLift", robots="ZArm")
```

## Custom Environments

### VNBGraspLift

Grasp and lift task with tactile observations:

```python
import robosuite as suite
from vnb_grasp.robosuite_ext.register_all import register_all

register_all()

env = suite.make(
    "VNBGraspLift",
    robots="ZArm",
    has_renderer=True,
    has_offscreen_renderer=False,
)

obs = env.reset()
obs, reward, done, info = env.step(action)
```

## Tactile Observations

Available tactile observation keys:
- `tactile_l6_matrix` / `tactile_l6_matrix_ext`: Taxel array counts
- `tactile_l6_force` / `tactile_l6_force_ext`: Estimated normal forces (N)
- `tactile_l6_taxel` / `tactile_l6_taxel_ext`: Custom taxel units

## See Also

- [training/train.py](../../training/train.py) - RL training with robosuite
- [robosuite documentation](https://robosuite.ai/)
