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

    # ===== 網絡架構 =====
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],   # 與 0510 相同
        critic_hidden_dims=[256, 128, 64],  # 與 0510 相同
        activation="elu",
    )

    # ===== PPO 演算法參數 =====
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
