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