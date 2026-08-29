from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 100
    experiment_name = "franka_cube_lift_latched"

    # ===== Logger Config =====
    logger = "tensorboard"       # Default to TensorBoard
    wandb_project = "franka_cube_lift_latched"

    # ===== Asymmetric AC Observation Groups =====
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }

    clip_actions = 100.0 # 2.0
    clip_observations = 100.0

    # ===== Actor & Critic Networks =====
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    # ===== PPO Algorithm Parameters =====
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0005, # 0.003
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=5.0e-5, #3.0e-5 # originally 1.0e-4, for critic 充足的時間重新熟悉新物體（杯子）的價值分佈 1.0e-5
        schedule="adaptive", #"fixed" # originally "adaptive"
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
