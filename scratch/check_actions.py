import argparse
import sys
import os
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Action checking.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")

# append RSL-RL cli arguments
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts/rsl_rl")))
import cli_args
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# default overrides if not provided
if args_cli.checkpoint is None:
    args_cli.checkpoint = "/workspace/test_isaaclab/pickup_place_direct_0203/logs/rsl_rl/franka_cube_lift_latched/2026-07-13_18-46-20_franka_cups_curricular_resume/model_12199.pt"
if args_cli.num_envs is None:
    args_cli.num_envs = 32
if args_cli.task is None:
    args_cli.task = "Template-Franka-Cube-Lift-Latched-Obs-v0"

# always set headless=True
args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import pickup_place_direct_0203.tasks  # noqa: F401

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.log_dir = os.path.dirname(args_cli.checkpoint)
    
    # Create env
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # Load runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy
    
    obs = env.get_observations()
    
    steps = 100
    all_raw_actions = []
    
    print("[INFO] Starting action saturation check...")
    for _ in range(steps):
        with torch.inference_mode():
            actions = policy(obs)
            all_raw_actions.append(actions.clone())
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
            
    all_raw_actions = torch.cat(all_raw_actions, dim=0)
    abs_actions = torch.abs(all_raw_actions)
    
    total_elements = abs_actions.numel()
    clipped_elements = torch.sum(abs_actions >= 2.0).item()
    pct_clipped = (clipped_elements / total_elements) * 100
    
    mean_val = all_raw_actions.mean().item()
    max_val = all_raw_actions.max().item()
    min_val = all_raw_actions.min().item()
    
    print("=" * 60)
    print(f"ACTION SATURATION REPORT ({os.path.basename(args_cli.checkpoint)})")
    print(f"Total action commands checked: {total_elements}")
    print(f"Clipped to boundary (>= 2.0 in abs): {clipped_elements} ({pct_clipped:.2f}%)")
    print(f"Mean raw action value: {mean_val:.4f}")
    print(f"Max raw action value:  {max_val:.4f}")
    print(f"Min raw action value:  {min_val:.4f}")
    print("=" * 60)
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
