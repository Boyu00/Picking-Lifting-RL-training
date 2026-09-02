#!/usr/bin/env python3
"""Verification script for JetRover 0510 refactored environment."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test JetRover 0510 Refactor")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default="Template-Pickup-Place-Direct-0510-v0", help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import pickup_place_direct_0203.tasks
from isaaclab_tasks.utils.hydra import hydra_task_config

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    print("==================================================================")
    print(" Testing JetRover 0510: 5-DOF Position-Only IK & 3-Stage Rewards")
    print("==================================================================")

    env_cfg.scene.num_envs = args_cli.num_envs
    print(f"\n[1/5] Creating environment: {args_cli.task} (num_envs={args_cli.num_envs})...")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    print("✓ Environment created successfully!")

    print("\n[2/5] Verifying Action & Observation Spaces...")
    action_shape = env.action_space.shape
    if hasattr(env.observation_space, "shape"):
        obs_shape = env.observation_space.shape
    elif "policy" in env.observation_space:
        obs_shape = env.observation_space["policy"].shape
    else:
        obs_shape = tuple(env.observation_space.spaces.values())[0].shape

    print(f"  - Action Space Shape: {action_shape}")
    print(f"  - Observation Space Shape: {obs_shape}")
    num_actions = action_shape[-1]
    num_obs = obs_shape[-1]
    print(f"  - Action dimension per env: {num_actions}")
    print(f"  - Observation dimension per env: {num_obs}")
    
    assert num_actions == 4, f"Expected 4 actions per env, got {num_actions}"
    assert num_obs == 46, f"Expected 46 observations per env, got {num_obs}"
    print("✓ Spaces match expected dimensions (Action=4, Obs=46)!")

    print("\n[3/5] Testing Reset...")
    obs, info = env.reset()
    policy_obs = obs["policy"] if isinstance(obs, dict) and "policy" in obs else obs
    print(f"  - Reset obs shape: {policy_obs.shape}")
    assert policy_obs.shape == (args_cli.num_envs, 46), f"Expected reset obs shape ({args_cli.num_envs}, 46), got {policy_obs.shape}"
    assert not torch.isnan(policy_obs).any(), "Reset observation contains NaNs!"
    print("✓ Reset completed without errors and no NaNs.")

    print("\n[4/5] Stepping environment with random 4D actions (10 steps)...")
    for step in range(10):
        action = torch.rand((args_cli.num_envs, 4), device=env.unwrapped.device) * 2.0 - 1.0
        obs, reward, terminated, truncated, info = env.step(action)
        obs_policy = obs["policy"] if isinstance(obs, dict) and "policy" in obs else obs
        assert not torch.isnan(obs_policy).any(), f"Step {step}: Observation contains NaNs!"
        assert not torch.isnan(reward).any(), f"Step {step}: Reward contains NaNs!"
        assert not torch.isinf(reward).any(), f"Step {step}: Reward contains Infs!"
    
    print(f"✓ Successfully ran 10 steps!")
    print(f"  - Mean reward: {reward.mean().item():.4f}")
    print(f"  - Observation range: [{obs_policy.min().item():.4f}, {obs_policy.max().item():.4f}]")

    print("\n[5/5] Checking Extras & Logging Keys...")
    log_episode = info.get("log", {}).get("episode", {})
    print(f"  - Logged keys: {list(log_episode.keys())}")
    
    expected_keys = [
        "reaching_success",
        "lifting_success",
        "object_goal_tracking_success",
        "object_goal_precision_success",
    ]
    for k in expected_keys:
        assert k in log_episode, f"Expected key '{k}' missing from info['log']['episode']"
        print(f"    ✓ Key '{k}': {log_episode[k]}")

    print("\n==================================================================")
    print(" ALL TESTS PASSED! 0510 Refactored Environment is fully operational!")
    print("==================================================================")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
