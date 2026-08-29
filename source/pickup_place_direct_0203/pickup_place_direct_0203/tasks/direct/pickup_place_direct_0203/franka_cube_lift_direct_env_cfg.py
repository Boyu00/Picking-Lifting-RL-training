import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg, GroundPlaneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg


@configclass
class FrankaCubeLiftDirectEnvCfg(DirectRLEnvCfg):
    """Configuration for the Franka Cube Lift Direct RL Environment with Latched Obs."""

    # ===== Basic Settings =====
    episode_length_s = 5.0
    decimation = 2
    action_space = 7  # 6D Task Space (XYZ, RxRyRz) + 1 Gripper
    observation_space = 31  # Actor: jpos(9) + jvel(9) + latched_obj(3) + target(3) + actions(7)
    state_space = 55  # Critic: jpos(9) + jvel(9) + gt_obj(3) + gt_bbox_corners(24) + target(3) + actions(7)

    # ===== Debug Visualization =====
    debug_vis = True
    debug_vis_settings = {
        "target": True,
        "com": True,
        "latch": True,
        "ee": True,
        "bbox": False,
    }

    # ===== Simulation =====
    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=decimation,
    )

    # ===== Scene =====
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,  # Default environments
        env_spacing=2.5,
        replicate_physics=False,
    )

    # ===== Robot =====
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # ===== Table =====
    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    # ===== Object (3 Cubes + 3 Cups Curriculum) =====
    object_scale = [0.8, 0.8, 0.8]
    SELECTED_OBJECT_IDS = [39, 40, 41]
    from isaaclab.sim.spawners.wrappers.wrappers_cfg import MultiAssetSpawnerCfg
    object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.055), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=MultiAssetSpawnerCfg(
            assets_cfg=[
                # 3 Cubes (50% probability) for curriculum stability
                UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                    scale=(0.8, 0.8, 0.8),
                ),
                UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                    scale=(0.8, 0.8, 0.8),
                ),
                UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                    scale=(0.8, 0.8, 0.8),
                ),
                # 3 Cups (50% probability)
                UsdFileCfg(
                    usd_path="/root/ObjectFolder/39/39.usd",
                    scale=(0.8, 0.8, 0.8),
                ),
                UsdFileCfg(
                    usd_path="/root/ObjectFolder/40/40.usd",
                    scale=(0.8, 0.8, 0.8),
                ),
                UsdFileCfg(
                    usd_path="/root/ObjectFolder/41/41.usd",
                    scale=(0.8, 0.8, 0.8),
                ),
            ],
            random_choice=True,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )


    # ===== End Effector Frame Transformer =====
    ee_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                ),
            ),
        ],
    )

    # ===== Ground Plane =====
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    # ===== Dome Light =====
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # ===== Action configuration =====
    action_cfg = {
        "ik_method": "dls",
        "position_scale": 0.5,  # Max 50cm displacement command scale (scaled by policy actions)
        "rotation_scale": 0.5,
    }

    # ===== Command generation ranges =====
    # Destination goal position for the cube
    target_pos_range = {
        "x": (0.4, 0.6),
        "y": (-0.25, 0.25),
        "z": (0.25, 0.5),
    }

    # ===== Reward scales (Official Lift weights) =====
    rew_scale_reach = 1.0
    rew_scale_lift = 5.0 # 15.0
    rew_scale_goal = 16.0
    rew_scale_goal_fine = 5.0
    rew_scale_action = -1e-4       # Start at -0.1 for resume (set to -1e-4 for from-scratch training)
    rew_scale_joint_vel = -1e-4    # Start at -0.1 for resume (set to -1e-4 for from-scratch training)

    # Penalty curriculum targets and schedule
    rew_curriculum_start_step = 10000 # 10000
    rew_curriculum_ramp_steps = 20000 # 50000
    rew_curriculum_target_action = -0.1
    rew_curriculum_target_joint_vel = -0.1

    # ===== Reward settings =====
    reward_settings = {
        "reaching_std": 0.1,
        "lifting_min_height": 0.04,  # Relative to world frame (table surface is z=0)
        "goal_std": 0.6, # 0.3
        "goal_min_height": 0.04,
        "goal_fine_std": 0.05,
        "goal_fine_min_height": 0.04,
        "object_is_reached_threshold": 0.04,
    }

    def __post_init__(self):
        super().__post_init__()

        # Configure PhysX simulation parameters
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.viewer.eye = [3.9, 3.9, 3.2]
        self.viewer.lookat = [0.0, 0.0, 0.2]

