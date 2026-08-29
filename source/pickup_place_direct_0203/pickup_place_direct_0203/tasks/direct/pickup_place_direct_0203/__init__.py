# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Pickup-Place-Direct-0203-Direct-v0",
    entry_point=f"{__name__}.pickup_place_direct_0203_env:PickupPlaceDirect0203Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_direct_0203_env_cfg:PickupPlaceDirect0203EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Vision version (with RGB + Depth camera)
gym.register(
    id="Template-Pickup-Place-Direct-0203-Vision-Direct-v0",
    entry_point=f"{__name__}.pickup_place_direct_0203_vision_env:PickupPlaceDirect0203VisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_direct_0203_vision_env_cfg:PickupPlaceDirect0203VisionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Contact GraspNet version
gym.register(
    id="Template-Pickup-Place-Direct-0208-GraspNet-v0",
    entry_point=f"{__name__}.pickup_place_direct_0208_graspnet_env:PickupPlaceDirect0208GraspNetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_direct_0208_graspnet_env_cfg:PickupPlaceDirect0208GraspNetEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Asymmetric Vision version YOLO
gym.register(
    id="Isaac-Pickup-Place-Direct-Vision-Asym-v0",
    entry_point=f"{__name__}.pickup_place_direct_0203_vision_asym_env:PickupPlaceDirect0203VisionAsymEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_direct_0203_vision_asym_env_cfg:PickupPlaceDirect0203VisionAsymEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_asym_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Asymmetric Vision version Point Cloud (0310)
gym.register(
    id="Pickup-Place-Direct-Vision-Asym-v1",
    entry_point=f"{__name__}.pickup_place_vision_asym_0310_env:PickupPlaceVisionAsym0310Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_vision_asym_0310_env_cfg:PickupPlaceVisionAsym0310EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_asym_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Asymmetric Vision version Point Cloud (0313 - With Contact Sensors & Fixed Physics)
gym.register(
    id="Pickup-Place-Direct-Vision-Asym-v2",
    entry_point=f"{__name__}.pickup_place_vision_asym_0313_env:PickupPlaceVisionAsym0313Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_vision_asym_0313_env_cfg:PickupPlaceVisionAsym0313EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_asym_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Differential IK (Task-Space Control) Version of 0313 Environment (4D Actions)
gym.register(
    id="Pickup-Place-Direct-Vision-Asym-IK-v0",
    entry_point=f"{__name__}.pickup_place_vision_asym_ik_env:PickupPlaceVisionAsymIKEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_vision_asym_ik_env_cfg:PickupPlaceVisionAsymIKEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_asym_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Task-Space Delta IK (0510) - Sim2Real Robust Control
gym.register(
    id="Template-Pickup-Place-Direct-0510-v0",
    entry_point=f"{__name__}.pickup_place_direct_0510_env:PickupPlaceDirect0510Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_direct_0510_env_cfg:PickupPlaceDirect0510EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_0510_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Latched Observation + IK (方案 A 驗證)
gym.register(
    id="Template-Pickup-Place-Latched-Obs-v0",
    entry_point=f"{__name__}.pickup_place_latched_obs_env:PickupPlaceLatchedObsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_latched_obs_env_cfg:PickupPlaceLatchedObsEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_latched_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Official Franka Cube Lift with Latched Obs (Direct Workflow)
gym.register(
    id="Template-Franka-Cube-Lift-Latched-Obs-v0",
    entry_point=f"{__name__}.franka_cube_lift_direct_env:FrankaCubeLiftDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_cube_lift_direct_env_cfg:FrankaCubeLiftDirectEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_franka_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)