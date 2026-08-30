import math
from isaaclab.utils import configclass
from .pickup_place_direct_0510_env_cfg import PickupPlaceDirect0510EnvCfg, FRAME_MARKER_SMALL_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class PickupPlaceLatchedObsEnvCfg(PickupPlaceDirect0510EnvCfg):
    """Latched Observation environment configuration adapted from Franka (v2).
    
    Actor: 25D (joint pos, joint vel, latched obj pos, target, action)
    Critic: 49D (joint pos, joint vel, gt obj pos, gt obj bbox corners, target, action)
    """
    
    # ===== 觀測空間 =====
    observation_space = 25
    state_space = 49

    use_46_dim_obs = False

    # ===== 控制頻率與時長 (對齊 Franka: 50 Hz, 250 steps) =====
    decimation = 2
    episode_length_s = 5.0

    # ===== 末端夾爪精確偏移量覆寫 =====
    ee_frame_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_footprint",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ee_frame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/gripper_link",
                name="end_effector",
                offset=OffsetCfg(pos=[-0.007, 0.0, 0.075]),
            ),
        ],
    )

    # ===== 除錯視覺化設定 =====
    debug_vis = True
    debug_vis_settings = {
        "target": True,
        "com": True,
        "latch": True,
        "ee": True,
        "bbox": False,
    }

    # ===== 鎖存觀測設定 =====
    latched_obs_noise_bias = 0.0
    latched_obs_noise_std = 0.0
    max_displacement = 0.15

    # ===== 手臂初始位置（對齊原本穩定初始位置）=====
    randomize_arm_init = True
    arm_init_offset_range = {
        "joint1": (0.0, 0.0),
        "joint2": (-30 * math.pi/180, -30 * math.pi/180),
        "joint3": (45 * math.pi/180, 45 * math.pi/180),
        "joint4": (119 * math.pi/180, 119 * math.pi/180),
        "joint5": (0.0, 0.0),
    }

    # ===== Action configuration (對齊 Franka 的較大探索步長) =====
    action_cfg = {
        "ik_method": "dls",
        "position_scale": 0.05,  # 相比原本的 0.005 放大 10 倍以利探索
        "rotation_scale": 0.5,   # 對齊 Franka
        "gripper_scale": 1.0,
    }

    # ===== 獎勵尺度設定 (Round 3.5: 近距閉爪引導 + 實體夾持感知 + 連續抬起) =====
    rew_scale_hover = 0.0              # 關閉
    rew_scale_displacement = -1.0       # Round 2: 保守地面防推懲罰 (-1.0)
    rew_scale_reach = 1.0              # 對齊 Franka: 空間接近 (+1.0)
    rew_scale_close = 2.0              # Round 3.5: 近距閉爪平滑引導 (+2.0 * (1 - pos/1.57))
    rew_scale_grasp = 4.0              # Round 3.5: 實體夾持感知 (Joint Jamming) 獎勵 (+4.0)
    rew_scale_lift = 5.0               # 對齊 Franka: 連續漸進抬起 (+5.0 * height)
    rew_scale_goal = 16.0              # 對齊 Franka
    rew_scale_goal_fine = 5.0          # 對齊 Franka
    rew_scale_action = -1e-4           # 對齊 Franka: 動作平滑約束 (-1e-4)
    rew_scale_joint_vel = -1e-4        # 對齊 Franka: 關節速度約束 (-1e-4)
    rew_scale_torques = 0.0            # 關閉

    # ===== 獎勵細部參數 (對齊 Franka 標準) =====
    reward_settings = {
        "reaching_std": 0.1,                    # 對齊 Franka
        "lifting_min_height": 0.04,              # 對齊 Franka
        "close_threshold": 0.08,
        "goal_std": 0.6,                         # 對齊 Franka
        "goal_min_height": 0.04,                 # 對齊 Franka
        "goal_fine_std": 0.05,                   # 對齊 Franka
        "goal_fine_min_height": 0.04,            # 對齊 Franka
        "object_is_reached_threshold": 0.04,     # 對齊 Franka
    }

    # ===== 物體尺寸縮放設定 =====
    object_scale = [0.5, 0.5, 0.5]

    # ===== 課程學習 (對齊 Franka：全數禁用，不設起始權重) =====
    curriculum_starting_weights = {}

    def __post_init__(self):
        super().__post_init__()
        
        # 採用適應 JetRover 夾爪開口尺寸的 DexCube (scale 0.5)
        from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
        self.object_cfg.spawn.assets_cfg = [
            UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.5, 0.5, 0.5),
            ),
        ]
