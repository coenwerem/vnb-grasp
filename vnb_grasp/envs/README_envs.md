# vnb_grasp.envs

Arena loading and object spawning utilities.

## Modules

| Module | Description |
|--------|-------------|
| `arena_loader.py` | Load arenas with object pose randomization |

## Usage

```python
from vnb_grasp.envs import ArenaConfig, ArenaLoader, list_available_arenas

# List available arenas
print(list_available_arenas())
# ['zarm_realhand_l6_right_arena', ...]

# Configure arena loading
config = ArenaConfig(
    arena_name="zarm_realhand_l6_right_arena",
    keyframe="home",
    object_body_name="cube",
    randomize_position=True,
    randomize_yaw=True,
    x_range=(-0.05, 0.05),
    y_range=(-0.05, 0.05),
)

# Load arena
loader = ArenaLoader(config)
model, data = loader.load()

# Randomize object pose
loader.randomize_object_pose(data)
```

## Object Spawning

```python
from vnb_grasp.envs import ObjectSpawnConfig

spawn_config = ObjectSpawnConfig(
    body_name="cube",
    position_range=(
        (-0.1, 0.1),  # x range
        (-0.1, 0.1),  # y range
        (0.0, 0.0),   # z range
    ),
    yaw_range=(0.0, 360.0),
)
```
