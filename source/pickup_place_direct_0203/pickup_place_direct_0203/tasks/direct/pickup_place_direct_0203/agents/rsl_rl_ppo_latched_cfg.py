from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 4000        # 從頭訓練，先跑 4000 iterations 看收斂
    save_interval = 100
    experiment_name = "pickup_place_latched_obs"

    # ===== 日誌設定 =====
    logger = "wandb"#"tensorboard"       # 預設使用 Tensorboard，如需 WandB 可修改為 "wandb"
    wandb_project = "pickup_place_latched_obs"

    # ===== 非對稱觀測組 =====
    # "policy" → Actor 用 (25D)
    # "critic" → Critic 用 (49D)
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }

    clip_actions = 100.0
    clip_observations = 100.0

    # ===== 網絡架構 (Round 1: 降低初始噪聲以提升穩定性) =====
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,                  # Round 1: 從 1.0 降至 0.5，避免前期巨大隨機跳動
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],   # 與 0510 相同
        critic_hidden_dims=[256, 128, 64],  # 與 0510 相同
        activation="elu",
    )

    # ===== PPO 演算法參數 (Round 1: 固定 LR + 降低 Entropy 防止發散) =====
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,                 # Round 1: 從 0.003 降至 0.001，抑制噪聲膨脹
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,               # Round 1: 設為穩健的 3e-4 固定學習率
        schedule="fixed",                   # Round 1: 關閉 adaptive，防止 LR 暴衝至 0.01
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
