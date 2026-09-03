#!/usr/bin/env python3
"""Diagnostic script to inspect action distribution, saturation, jitter, and task metrics of model_350."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose model actions and stability")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments.")
parser.add_argument("--task", type=str, default="Template-Pickup-Place-Direct-0510-v0")
parser.add_argument("--checkpoint", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
import pickup_place_direct_0203.tasks

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Create env
    print(f"\n[1/4] Creating Environment ({args_cli.num_envs} envs)...")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load model
    print(f"\n[2/4] Loading Policy Checkpoint...")
    ckpt_path = args_cli.checkpoint or "/workspace/test_isaaclab/pickup_place_direct_0203/logs/rsl_rl/pickup_place_direct_0510/2026-09-02_11-17-33_JetRover_0902_PosIK_3Stage_run1/model_350.pt"
    print(f"  - Checkpoint: {ckpt_path}")
    
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
    ppo_runner.load(ckpt_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    print("✓ Policy loaded successfully!")

    print(f"\n[3/4] Running 100-step Diagnostic Rollout...")
    obs, _ = env.reset()
    
    # Action history: list of shape (num_envs, 4)
    actions_history = []
    ee_pos_history = []
    obj_pos_history = []
    goal_dist_history = []
    
    dim_names = ["dx (X)", "dy (Y)", "dz (Z)", "gripper"]
    
    num_steps = 100
    with torch.no_grad():
        for step in range(num_steps):
            act = policy(obs)
            actions_history.append(act.clone().cpu())
            
            # Step env
            obs, rew, dones, extra = env.step(act)
            
            # Record EE and Object
            raw_env = env.unwrapped
            ee_pos = raw_env.ee_frame.data.target_pos_w[..., 0, :].clone().cpu()
            obj_pos = raw_env.scene["object"].data.root_pos_w.clone().cpu()
            target_pos = (raw_env.target_poses + raw_env.scene.env_origins).clone().cpu()
            goal_dist = torch.norm(obj_pos - target_pos, dim=-1).cpu()
            
            ee_pos_history.append(ee_pos)
            obj_pos_history.append(obj_pos)
            goal_dist_history.append(goal_dist)

    actions_history = torch.stack(actions_history, dim=1) # (num_envs, num_steps, 4)
    ee_pos_history = torch.stack(ee_pos_history, dim=1)   # (num_envs, num_steps, 3)
    obj_pos_history = torch.stack(obj_pos_history, dim=1) # (num_envs, num_steps, 3)
    goal_dist_history = torch.stack(goal_dist_history, dim=1) # (num_envs, num_steps)

    print("\n==================================================================")
    print(" 📊 動作指令分佈與飽和度診斷 (Action Distribution & Saturation)")
    print("==================================================================")
    
    for d in range(4):
        act_d = actions_history[:, :, d].numpy()
        mean_val = np.mean(act_d)
        std_val = np.std(act_d)
        min_val = np.min(act_d)
        max_val = np.max(act_d)
        p5 = np.percentile(act_d, 5)
        p95 = np.percentile(act_d, 95)
        
        # Saturation checks
        pct_gt_1 = np.mean(np.abs(act_d) > 1.0) * 100
        pct_gt_2 = np.mean(np.abs(act_d) > 2.0) * 100
        pct_gt_3 = np.mean(np.abs(act_d) > 3.0) * 100
        
        print(f"\n[{dim_names[d]}]:")
        print(f"  - Mean ± Std: {mean_val:+.3f} ± {std_val:.3f}")
        print(f"  - Range [Min, Max]: [{min_val:+.3f}, {max_val:+.3f}]")
        print(f"  - 90% Range [P5, P95]: [{p5:+.3f}, {p95:+.3f}]")
        print(f"  - |action| > 1.0 比例: {pct_gt_1:.1f}%")
        print(f"  - |action| > 2.0 比例: {pct_gt_2:.1f}%")
        print(f"  - |action| > 3.0 比例: {pct_gt_3:.1f}%")

    print("\n==================================================================")
    print(" 📈 動作高頻抖動與連續性診斷 (Action Jitter & Smoothness)")
    print("==================================================================")
    
    # Delta action: a_t - a_{t-1}
    delta_act = actions_history[:, 1:, :] - actions_history[:, :-1, :] # (num_envs, num_steps-1, 4)
    delta_act_np = delta_act.numpy()
    
    # Direction reversals: a_t * a_{t-1} < 0
    act_prev = actions_history[:, :-1, :].numpy()
    act_curr = actions_history[:, 1:, :].numpy()
    sign_flips = (act_curr * act_prev < 0) & (np.abs(act_curr) > 0.1) & (np.abs(act_prev) > 0.1)
    
    for d in range(4):
        mean_step_change = np.mean(np.abs(delta_act_np[:, :, d]))
        flip_rate = np.mean(sign_flips[:, :, d]) * 100
        print(f"\n[{dim_names[d]}]:")
        print(f"  - 每步動作平均變化量 (|Δa|): {mean_step_change:.3f}")
        print(f"  - 符號方向逆轉率 (Sign Flip Rate): {flip_rate:.1f}%")
        if flip_rate > 35.0:
            print("    ⚠️ 警告：存在較高頻率的符號逆轉，可能有些許顫抖！")
        else:
            print("    ✓ 連續性良好，未出現高頻激烈反向抽搐。")

    print("\n==================================================================")
    print(" 🎯 任務指標統計 (Task Progression across 8 test envs)")
    print("==================================================================")
    
    # Initial distance vs final distance
    init_dist = goal_dist_history[:, 0].mean().item()
    min_dist = goal_dist_history.min(dim=1)[0].mean().item()
    final_dist = goal_dist_history[:, -1].mean().item()
    
    # Max height achieved
    max_height = obj_pos_history[:, :, 2].max(dim=1)[0].mean().item()
    
    # Reaching & Lifting counts
    num_lifted = (obj_pos_history[:, :, 2].max(dim=1)[0] > 0.20).sum().item()
    num_close_goal = (goal_dist_history.min(dim=1)[0] < 0.15).sum().item()
    num_tracked_8cm = (goal_dist_history.min(dim=1)[0] < 0.08).sum().item()
    num_precision_3cm = (goal_dist_history.min(dim=1)[0] < 0.03).sum().item()

    print(f"  - 起始物體到目標平均距離: {init_dist:.3f} m")
    print(f"  - 軌跡中最接近目標平均距離: {min_dist:.3f} m (逼近了 {(init_dist - min_dist)*100:.1f} cm)")
    print(f"  - 終點平均距離: {final_dist:.3f} m")
    print(f"  - 最高抬升平均高度: {max_height:.3f} m")
    print(f"  - 成功抬起 (Z > 0.20m): {num_lifted}/{args_cli.num_envs} ({num_lifted/args_cli.num_envs*100:.1f}%)")
    print(f"  - 逼近至 15cm 內: {num_close_goal}/{args_cli.num_envs} ({num_close_goal/args_cli.num_envs*100:.1f}%)")
    print(f"  - 進入 8cm 目標圈: {num_tracked_8cm}/{args_cli.num_envs} ({num_tracked_8cm/args_cli.num_envs*100:.1f}%)")
    print(f"  - 進入 3cm 精準圈: {num_precision_3cm}/{args_cli.num_envs} ({num_precision_3cm/args_cli.num_envs*100:.1f}%)")
    print("==================================================================")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
