# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--reset_critic", action="store_true", default=False, help="Reset critic weights when resuming.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import logging
import os
import torch
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import pickup_place_direct_0203.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if getattr(args_cli, "reset_critic", False):
            # Read state dict manually
            loaded_dict = torch.load(resume_path, weights_only=False, map_location=agent_cfg.device)
            state_dict = loaded_dict["model_state_dict"]
            # Filter out all critic keys
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("critic")}
            # Load with strict=False (keeps Critic at default random initialization)
            runner.alg.policy.load_state_dict(filtered_state_dict, strict=False)
            print("[INFO]: Loaded ONLY Actor weights from checkpoint. Critic weights have been reset to default random initialization.")
            # Set the iteration number
            runner.current_learning_iteration = loaded_dict["iter"]
        else:
            # load previously trained model
            runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # ===== [訓練健康監測與安全自動早停機制 (Training Health Monitor)] =====
    # 說明：此機制經 Franka 歷史 14 個 Runs 校驗，可即時偵測數值爆炸與死鎖發散，防止 GPU 資源浪費。
    # 觸發時會自動儲存當前權重 (model_earlystop_<iter>.pt) 並優雅終止。
    import statistics
    _original_log = runner._log
    runner._early_stop_deadlock_counter = 0
    runner._action_warn_counter = 0

    def _monitored_log(locs, width=80, pad=35):
        # 1. 執行 RSL-RL 原生日誌輸出
        _original_log(locs, width, pad)

        it = locs["it"]
        mean_std = runner.alg.policy.action_std.mean().item()
        vf_loss = locs.get("loss_dict", {}).get("value_function", 0.0)
        recent_rew = statistics.mean(locs["rewbuffer"]) if len(locs["rewbuffer"]) > 0 else 0.0

        # -------------------------------------------------------------
        # 規則 1：致命數值爆炸判定 (Fatal Value Function / NaN Divergence)
        # 條件：Value Loss > 1e7 或出現 NaN/Inf (正常訓練通常在 1e2 ~ 1e5 區間)
        # -------------------------------------------------------------
        if vf_loss > 1.0e7 or torch.isnan(torch.tensor(mean_std)) or mean_std > 50.0:
            print(f"\n🛑 [FATAL DIVERGENCE] Iteration {it}: Value Loss ({vf_loss:.2e}) 或 Noise Std ({mean_std:.2f}) 發生數值爆炸！")
            save_path = os.path.join(runner.log_dir, f"model_fatal_divergence_{it}.pt")
            runner.save(save_path)
            print(f"💾 已緊急保存崩潰前模型至: {save_path}")
            raise SystemExit(f"Training terminated at iter {it} due to fatal gradient divergence.")

        # -------------------------------------------------------------
        # 規則 2：探索死鎖與噪聲正反饋發散 (Exploration Deadlock)
        # 條件：Noise Std > 2.5 且 Mean Reward < 100 持續超過 150 iters (代表策略完全學不會搬運且持續擴大隨機噪聲)
        # -------------------------------------------------------------
        if mean_std > 2.5 and recent_rew < 100.0:
            runner._early_stop_deadlock_counter += 1
            if runner._early_stop_deadlock_counter >= 150:
                print(f"\n🛑 [EARLY STOP] Iteration {it}: 偵測到策略死鎖發散！")
                print(f"   原因: Noise Std ({mean_std:.2f}) > 2.5 且 Mean Reward ({recent_rew:.2f}) < 100 已持續 150 iters。")
                save_path = os.path.join(runner.log_dir, f"model_deadlock_earlystop_{it}.pt")
                runner.save(save_path)
                print(f"💾 已保存當前模型至: {save_path}")
                raise SystemExit(f"Early stopped at iter {it} due to policy exploration deadlock.")
        else:
            runner._early_stop_deadlock_counter = 0

        # -------------------------------------------------------------
        # 規則 3：Action 飽和度警示 (非致命，僅提醒不中斷)
        # -------------------------------------------------------------
        if mean_std > 1.8:
            runner._action_warn_counter += 1
            if runner._action_warn_counter % 50 == 0:
                print(f"⚠️ [WARN] Iteration {it}: Noise Std 偏高 ({mean_std:.2f})，請留意 PPO 穩定性。")
        else:
            runner._action_warn_counter = 0

        # -------------------------------------------------------------
        # 定期健康彙報 (每 200 iters 印出關鍵指標摘要)
        # -------------------------------------------------------------
        if it % 200 == 0:
            print(f"📊 [HEALTH] Iter {it:5d}/{locs['tot_iter']} | Noise Std: {mean_std:.3f} | VF Loss: {vf_loss:.2e} | Reward: {recent_rew:.1f} | LR: {runner.alg.learning_rate:.6f}")

    runner._log = _monitored_log
    # ===== [End Health Monitor] =====

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
