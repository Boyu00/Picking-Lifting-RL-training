"""Evaluation configuration for Pick-and-Place inference benchmarking (0510 Delta IK version)."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class EvalCriteria:
    """Unified evaluation thresholds for 0510 environment (adapted for 50-step episodes)."""
    # Reach: EE-to-BBox surface distance
    reach_threshold: float = 0.05       # meters
    reach_consecutive: int = 5          # steps (0.5s at decimation=10, dt=0.01)

    # Lift: EE dist + bbox bottom height + grasp quality (NO velocity)
    lift_ee_dist: float = 0.12          # meters (max_distance_to_ee in 0510 is 0.15)
    lift_min_height: float = 0.06       # meters (min_height in 0510 is 0.06)
    lift_grasp_quality: float = 0.20    # 0~1 (min_grasp_quality in 0510 is 0.20)
    lift_consecutive: int = 10          # steps (1.0s)

    # Goal Track: obj-target dist + height + grasp quality
    goal_obj_target_dist: float = 0.08      # meters (coarse tracking threshold in 0510 is 0.08)
    precision_obj_target_dist: float = 0.03 # meters (fine precision threshold in 0510 is 0.03)
    goal_min_height: float = 0.10           # meters (goal_min_height in 0510 is 0.10)
    goal_grasp_quality: float = 0.20        # 0~1
    goal_consecutive: int = 10              # steps (1.0s)


# ──────────────────────────────────────────────
# Object Definitions
# ──────────────────────────────────────────────

SHAPE_CLASSES = {
    # flat_long: 扁細長物體
    "flat_long": ["25", "26", "27", "28", "31"],
    # cup_bowl: 杯碗盆狀（有邊緣中空）
    "cup_bowl": ["22", "39", "40", "41", "62", "trash_orange"],
    # cylinder: 圓/方柱狀
    "cylinder": ["29", "68", "93", "95", "96", "trash_cans_1", "trash_plastic_bottle_1"],
    # irregular: 較不規則狀、凹多面體
    "irregular": [
        "21", "70",
        "trash_apple", "trash_banana", "trash_cans_2", "trash_cans_3",
        "trash_plastic_bottle_2", "trash_plastic_bottle_3", "trash_egg_carton",
    ],
}

# Reverse lookup: object_key -> shape_class
OBJECT_SHAPE_MAP = {}
for shape, keys in SHAPE_CLASSES.items():
    for k in keys:
        OBJECT_SHAPE_MAP[k] = shape


def _of(obj_id: int, shape: str = None):
    """Helper to create an ObjectFolder object entry."""
    key = str(obj_id)
    return {
        "key": key,
        "source": "objectfolder",
        "usd_path": f"/root/ObjectFolder/{obj_id}/{obj_id}.usd",
        "scale": (0.6, 0.6, 0.6),
        "shape_class": shape or OBJECT_SHAPE_MAP.get(key, "unknown"),
    }


def _trash(filename: str, shape: str = None):
    """Helper to create a Trash object entry."""
    key = filename.replace(".usd", "")
    return {
        "key": key,
        "source": "trash",
        "usd_path": f"/workspace/test_isaaclab/trash_obj/{filename}",
        "scale": (0.005, 0.005, 0.005),
        "shape_class": shape or OBJECT_SHAPE_MAP.get(key, "unknown"),
    }


# Training objects (15 items, matching 0510 SELECTED_OBJECT_IDS)
TRAIN_OBJECTS = [
    _of(22), _of(25), _of(26), _of(27), _of(28), _of(31),      # flat_long + cup_bowl
    _of(39), _of(40), _of(41), _of(62),                # cup_bowl
    _of(68), _of(93), _of(95), _of(96),                # cylinder
    _of(70),                                             # irregular
]

# Testing objects (12 items, never seen during training)
TEST_OBJECTS = [
    # ObjectFolder subset
    _of(21),   # irregular
    _of(29),   # cylinder
    # Trash subset
    _trash("trash_apple.usd"),
    _trash("trash_banana.usd"),
    _trash("trash_cans_1.usd"),
    _trash("trash_cans_2.usd"),
    _trash("trash_cans_3.usd"),
    _trash("trash_egg_carton.usd"),
    _trash("trash_orange.usd"),
    _trash("trash_plastic_bottle_1.usd"),
    _trash("trash_plastic_bottle_2.usd"),
    _trash("trash_plastic_bottle_3.usd"),
]


def distribute_trials(objects: list, total_trials: int) -> list[int]:
    """Distribute total_trials across objects as evenly as possible."""
    n = len(objects)
    base = total_trials // n
    remainder = total_trials % n
    return [base + (1 if i < remainder else 0) for i in range(n)]


# ──────────────────────────────────────────────
# Stratification Bins
# ──────────────────────────────────────────────

# Object initial Y-position relative to robot → left/center/right
POSITION_BINS = {
    "left":   (0.07, float("inf")),
    "center": (-0.07, 0.07),
    "right":  (float("-inf"), -0.07),
}


def classify_position(rel_y: float) -> str:
    if rel_y > 0.07:
        return "left"
    elif rel_y < -0.07:
        return "right"
    return "center"


# Target Z height bins (4 groups)
HEIGHT_BINS = [
    ("very_low", 0.05, 0.15),
    ("low",      0.15, 0.30),
    ("mid",      0.30, 0.45),
    ("high",     0.45, 0.55),
]


def classify_height(z: float) -> str:
    for name, lo, hi in HEIGHT_BINS:
        if lo <= z < hi:
            return name
    if z >= 0.55:
        return "high"
    return "very_low"


# Mass bins (3 groups over 0.05~0.2 kg)
MASS_BINS = [
    ("light",  0.05, 0.10),
    ("medium", 0.10, 0.15),
    ("heavy",  0.15, 0.20),
]


def classify_mass(m: float) -> str:
    for name, lo, hi in MASS_BINS:
        if lo <= m <= hi:
            return name
    if m > 0.15:
        return "heavy"
    return "light"


# ──────────────────────────────────────────────
# Failure Mode Classification
# ──────────────────────────────────────────────

def classify_failure(record: dict) -> str:
    """Classify the failure mode of a trial based on recorded metrics."""
    if record["goal_track_success"]:
        return "success"
    if record["termination_reason"] == "out_of_bounds" and record["max_consecutive_reach"] == 0:
        return "out_of_bounds"
    if record["max_consecutive_reach"] == 0:
        return "no_approach"
    if not record["reach_success"]:
        return "approach_only"
    if record["max_consecutive_lift"] < 3:  # Adjusted for shorter consecutive check
        return "grasp_fail"
    if record["lift_success"] and not record["goal_track_success"]:
        return "transport_fail"
    if record["max_consecutive_lift"] >= 3 and not record["lift_success"]:
        return "drop_after_lift"
    return "approach_only"


# ──────────────────────────────────────────────
# Evaluation-time environment overrides
# ──────────────────────────────────────────────

EVAL_ENV_OVERRIDES = {
    "num_envs": 1,
    "seed": 42,
    "target_pos_range_z": (0.3, 0.55),
    "mass_range": (0.05, 0.2),
    "episode_max_steps": 50,
}
