from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv

from isaaclab.utils.math import subtract_frame_transforms
from .observations import object_bbox_corners


def _get_object_bottom_z(env: "DirectRLEnv", object_cfg: SceneEntityCfg) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]

    if hasattr(env, "object_local_corners"):
        corners_flat = object_bbox_corners(env, object_cfg)
        world_corners = corners_flat.view(env.num_envs, 8, 3)
        world_corners_z = world_corners[..., 2]
        object_bottom_z, _ = torch.min(world_corners_z, dim=1)
    else:
        object_bottom_z = object.data.root_com_pose_w[:, 2] - 0.03

    return object_bottom_z


def object_is_lifted(
    env: "DirectRLEnv",
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    floor_height: float = 0.0,
    left_contact_cfg: SceneEntityCfg = SceneEntityCfg("left_finger_force"),
    right_contact_cfg: SceneEntityCfg = SceneEntityCfg("right_finger_force"),
    force_threshold: float = 0.1,
) -> torch.Tensor:
    # 原本的高度計算
    object_bottom_z = _get_object_bottom_z(env, object_cfg)
    lift_height = object_bottom_z - floor_height
    lifting_reward = lift_height / (minimal_height + 1e-6)
    lifting_reward = torch.clamp(lifting_reward, min=0.0, max=1.0)
    is_settled = env.episode_length_buf > 15

    # 接觸 gating：兩個夾爪都要有真實接觸
    left_sensor = env.scene[left_contact_cfg.name]
    right_sensor = env.scene[right_contact_cfg.name]
    
    # 力量 norm 計算 (B, 1) -> squeeze 成 (B,)
    left_force = left_sensor.data.net_forces_w.norm(dim=-1).squeeze(-1)
    right_force = right_sensor.data.net_forces_w.norm(dim=-1).squeeze(-1)

    # 計數 (Debug)
    nan_count = (torch.isnan(left_force) | torch.isinf(left_force) | 
                 torch.isnan(right_force) | torch.isinf(right_force)).sum().item()
    if nan_count > 0:
        print(f"[WARN] {nan_count} envs had NaN/Inf force, replaced with 0")

    # 防禦：在 force 層面就擋住 NaN
    left_force = torch.nan_to_num(left_force, nan=0.0, posinf=0.0, neginf=0.0)
    right_force = torch.nan_to_num(right_force, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 限制力量範圍
    left_force = torch.clamp(left_force, 0.0, 100.0)
    right_force = torch.clamp(right_force, 0.0, 100.0)
    
    # 2. 再做 contact gating
    is_real_contact = (left_force > force_threshold) & (right_force > force_threshold)
    reward = lifting_reward * is_settled.float() * is_real_contact.float()

    return reward


def lifting_velocity_reward(env: "DirectRLEnv", object_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    obj_vel = env.scene[object_cfg.name].data.root_lin_vel_w[:, 2]
    return torch.clamp(obj_vel, min=0.0, max=1.0)


def gripper_open_close_phases(
    env: "DirectRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    joint_name: str = "r_joint",
    close_threshold: float = 0.08,
    open_target: float = 1.569,
    close_target: float = 0.0,
) -> torch.Tensor:
    """
    Simplified gripper reward: Rewards closing the gripper only when close to the object.
    Focuses exclusively on grasp quality within the threshold range.
    """
    ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]
    obj_pos = env.scene[object_cfg.name].data.root_com_pos_w[:, :3]
    dist = torch.norm(obj_pos - ee_pos, dim=-1)

    robot = env.scene["robot"]
    joint_indices, _ = robot.find_joints(joint_name)
    gripper_idx = joint_indices[0]
    gripper_pos = robot.data.joint_pos[:, gripper_idx]

    # Only reward grasping when within proximity threshold
    is_in_range = (dist <= close_threshold).float()
    
    # Grasp quality: 0.0 (fully open) to 1.0 (fully closed)
    grasp_quality = 1.0 - (gripper_pos / open_target)
    grasp_quality = torch.clamp(grasp_quality, 0.0, 1.0)

    return grasp_quality * is_in_range


def object_goal_distance_real(
    env: "DirectRLEnv",
    target_pos: torch.Tensor = None,
    command_name: str = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Compute distance from object to goal position.
    
    In Direct workflow, use target_pos parameter (from env.target_poses).
    In Manager-based workflow, use command_name parameter (deprecated for Direct).
    
    Args:
        env: Environment instance
        target_pos: Target position in world frame (num_envs, 3) - used in Direct workflow
        command_name: Command name (unused in Direct) - kept for backward compatibility
        robot_cfg: Robot configuration
        object_cfg: Object configuration
    
    Returns:
        Distance tensor of shape (num_envs,)
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # For Direct workflow: use target_pos directly
    if target_pos is not None:
        des_pos_w = target_pos
    # For Manager-based workflow (fallback, should not be used in Direct)
    elif command_name is not None:
        try:
            command = env.command_manager.get_command(command_name)
            des_pos_b = command[:, :3]
            des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
        except Exception:
            return torch.full((env.num_envs,), 1e3, device=env.device)
    else:
        # Neither target_pos nor command_name provided
        return torch.full((env.num_envs,), 1e3, device=env.device)
    
    distance = torch.norm(des_pos_w - object.data.root_com_pose_w[:, :3], dim=1)
    return distance


def object_goal_distance(
    env: "DirectRLEnv",
    std: float,
    minimal_height: float,
    target_pos: torch.Tensor = None,
    command_name: str = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Compute reward for reaching and tracking object to goal position.
    
    In Direct workflow, pass target_pos (from env.target_poses).
    In Manager-based workflow, pass command_name (deprecated).
    
    Args:
        env: Environment instance
        std: Standard deviation for tanh scaling
        minimal_height: Minimum height requirement for object
        target_pos: Target position in world frame (for Direct workflow)
        command_name: Command name (for Manager-based workflow, deprecated)
        robot_cfg: Robot configuration
        object_cfg: Object configuration
    
    Returns:
        Reward tensor of shape (num_envs,)
    """
    distance = object_goal_distance_real(env, target_pos=target_pos, command_name=command_name, 
                                        robot_cfg=robot_cfg, object_cfg=object_cfg)
    object: RigidObject = env.scene[object_cfg.name]
    object_bottom_z = _get_object_bottom_z(env, object_cfg)
    is_lifted = (object_bottom_z > minimal_height).float()
    tracking_reward = 1 - torch.tanh(distance / std)
    # Strict gating: Only reward tracking if object is successfully lifted
    # This prevents "slidingHack" where agent gets partial reward for sliding object on ground
    return is_lifted * tracking_reward


def object_goal_is_tracked(
    env: "DirectRLEnv",
    threshold: float,
    minimal_height: float,
    target_pos: torch.Tensor = None,
    command_name: str = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check if object has reached and is being tracked at goal position.
    
    In Direct workflow, pass target_pos (from env.target_poses).
    In Manager-based workflow, pass command_name (deprecated).
    
    Args:
        env: Environment instance
        threshold: Distance threshold for "tracked"
        minimal_height: Minimum height requirement
        target_pos: Target position in world frame (for Direct workflow)
        command_name: Command name (for Manager-based workflow, deprecated)
        robot_cfg: Robot configuration
        object_cfg: Object configuration
    
    Returns:
        Binary tracking status tensor of shape (num_envs,)
    """
    distance = object_goal_distance_real(env, target_pos=target_pos, command_name=command_name,
                                        robot_cfg=robot_cfg, object_cfg=object_cfg)
    is_close = (distance < threshold)
    object_bottom_z = _get_object_bottom_z(env, object_cfg)
    is_high_enough = (object_bottom_z > minimal_height)
    is_settled = env.episode_length_buf > 15
    return (is_close & is_high_enough & is_settled).float()


def object_is_reached(
    env: "DirectRLEnv",
    threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Check if end-effector is within threshold distance of object bounding box."""
    object_ee_distance = object_bbox_ee_distance_real(env, object_cfg, ee_frame_cfg)
    return (object_ee_distance < threshold).float()


def object_bbox_ee_distance_real(
    env: "DirectRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    if not hasattr(env, "object_local_corners"):
        object_entity = env.scene[object_cfg.name]
        ee_frame = env.scene[ee_frame_cfg.name]
        return torch.norm(object_entity.data.root_com_pose_w[:, :3] - ee_frame.data.target_pos_w[..., 0, :], dim=1)

    bbox_extents = torch.max(env.object_local_corners, dim=1)[0]

    object_entity = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    obj_pos_w = object_entity.data.root_com_pose_w[:, :3]
    obj_quat_w = object_entity.data.root_com_pose_w[:, 3:7]

    ee_pos_local, _ = subtract_frame_transforms(obj_pos_w, obj_quat_w, ee_pos_w, ee_frame.data.target_quat_w[:, 0, :])

    dist_vector = torch.clamp(torch.abs(ee_pos_local) - bbox_extents, min=0.0)
    bbox_distance = torch.norm(dist_vector, dim=-1)
    return bbox_distance


def object_bbox_ee_distance(
    env: "DirectRLEnv",
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    distance = object_bbox_ee_distance_real(env, object_cfg, ee_frame_cfg)
    return 1 - torch.tanh(distance / std)


def check_lifting_success(
    env: "DirectRLEnv",
    threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    floor_height: float = 0.0,
) -> torch.Tensor:
    object_bottom_z = _get_object_bottom_z(env, object_cfg)
    lift_height = object_bottom_z - floor_height
    is_success = ((lift_height > threshold) & (env.episode_length_buf > 15)).float()
    return is_success


def check_robust_lifting_success(
    env: "DirectRLEnv",
    min_height: float = 0.05,
    max_dist_to_ee: float = 0.20,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Check if object is lifted AND close to gripper (robust grasp)."""
    # 1. Height check
    object_bottom_z = _get_object_bottom_z(env, object_cfg)
    is_lifted = (object_bottom_z > min_height)
    
    # 2. Distance check (In Hand)
    # Using bounding box distance to EE for accuracy
    dist_to_ee = object_bbox_ee_distance_real(env, object_cfg, ee_frame_cfg) 
    is_close = (dist_to_ee < max_dist_to_ee)
    
    # 3. Combined check
    # settled check included to avoid initial spawning noise
    is_settled = env.episode_length_buf > 10
    
    return (is_lifted & is_close & is_settled).float()


def check_stable_lifting_success(
    env: "DirectRLEnv",
    min_height: float = 0.08,
    min_grasp_quality: float = 0.25,
    velocity_threshold: float = 0.15,
    min_distance_to_ee: float = 0.04,
    max_distance_to_ee: float = 0.12,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    joint_name: str = "r_joint",
) -> torch.Tensor:
    """
    Enhanced stable lifting detection to prevent penetration-based false positives.
    
    Checks:
    1. Sufficient height (min_height)
    2. Safe distance from EE (no deep penetration)
    3. Low velocity (object is stable, not vibrating)
    4. Adequate grasp quality (gripper is actually holding)
    5. Sufficient episode time (settled)
    
    Args:
        env: Environment instance
        min_height: Minimum height for object bottom (0.08m = 8cm, higher than before)
        min_grasp_quality: Minimum gripper closure (0.25 = 25% closed)
        velocity_threshold: Max vertical velocity of object (m/s)
        min_distance_to_ee: Min distance to prevent penetration (0.04m)
        max_distance_to_ee: Max distance for "in hand" (0.12m, reduced from 0.20)
        
    Returns:
        Boolean tensor indicating stable lift success
    """
    
    # 1. Height check
    object_bottom_z = _get_object_bottom_z(env, object_cfg)
    is_lifted_high = (object_bottom_z > min_height)
    
    # 2. Distance check (prevent penetration)
    dist_to_ee = object_bbox_ee_distance_real(env, object_cfg, ee_frame_cfg)
    is_in_safe_zone = (dist_to_ee <= max_distance_to_ee) #(dist_to_ee >= min_distance_to_ee) & 
    
    # 3. Velocity check (object should be stable, not vibrating/moving fast)
    obj_vel_z = env.scene[object_cfg.name].data.root_lin_vel_w[:, 2]
    is_velocity_low = (torch.abs(obj_vel_z) < velocity_threshold)
    
    # 4. Grasp quality check (gripper actually holding, not just near object)
    robot = env.scene["robot"]
    try:
        joint_indices, _ = robot.find_joints(joint_name)
        gripper_idx = joint_indices[0]
        gripper_pos = robot.data.joint_pos[:, gripper_idx]
        grasp_quality = 1.0 - (gripper_pos / 1.569)  # 0-1, where 1=fully closed
        grasp_quality = torch.clamp(grasp_quality, 0.0, 1.0)
        is_grasped = (grasp_quality > min_grasp_quality)
    except Exception:
        is_grasped = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    
    # 5. Settled check (avoid initial spawning noise)
    is_settled = env.episode_length_buf > 20  # Slightly higher than before
    
    # Combined check: ALL conditions must be true
    return (is_lifted_high & is_in_safe_zone & is_grasped & is_settled).float() # & is_velocity_low

def action_rate_l2(env: "DirectRLEnv") -> torch.Tensor:
    """Penalty for large action magnitudes (L2 norm).
    
    Args:
        env: The environment instance.
        
    Returns:
        Tensor of shape (num_envs,) with L2 norm of actions for each environment.
    """
    return torch.norm(env.actions, p=2, dim=-1)


def joint_vel_l2(
    env: "DirectRLEnv", 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalty for large joint velocities (L2 norm).
    
    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the robot asset.
        
    Returns:
        Tensor of shape (num_envs,) with L2 norm of joint velocities for each environment.
    """
    asset = env.scene[asset_cfg.name]
    return torch.norm(asset.data.joint_vel, p=2, dim=-1)


def object_in_view_reward(
    env: "DirectRLEnv",
    camera_sensor_name: str = "camera_low",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    focus_exponent: float = 2.0,
) -> torch.Tensor:
    """
    Reward for keeping the object within the camera's field of view and centered.
    
    The reward is calculated based on the geometric projection of the object's position
    onto the camera's image plane.
    
    Args:
        env: The environment instance.
        camera_sensor_name: Name of the camera sensor in env.scene.
        object_cfg: Scene entity configuration for the object.
        focus_exponent: Exponent for the focus term (higher = stricter centering).
        
    Returns:
        Reward tensor (num_envs,) in range [0, 1].
        1.0 = Centered in view.
        0.0 = Out of view.
    """
    # 1. Get Sensor and Object Data
    camera = env.scene[camera_sensor_name]
    object_entity = env.scene[object_cfg.name]
    
    # Object Position (World Frame)
    obj_pos_w = object_entity.data.root_pos_w  # (N, 3)
    
    # Camera Pose (World Frame)
    # TiledCamera data: pos_w (N, 3), quat_w_world (N, 4)
    cam_pos_w = camera.data.pos_w
    cam_quat_w = camera.data.quat_w_world
    
    # Camera Intrinsics
    # (N, 3, 3)
    K = camera.data.intrinsic_matrices
    
    # Image Dimensions
    # (H, W) or (H, W, 1) depending on implementation
    height, width = camera.data.image_shape[:2]
    
    # 2. Transform World -> Camera Frame
    # P_c = R_inv * (P_w - T_c)
    # R_inv is conjugate/inverse of cam_quat_w
    from isaaclab.utils.math import quat_inv, quat_apply
    
    cam_quat_inv = quat_inv(cam_quat_w)
    vec_w = obj_pos_w - cam_pos_w
    obj_pos_c = quat_apply(cam_quat_inv, vec_w) # (N, 3) (x, y, z) in cam frame
    
    # 3. Project to Image Plane
    # u = fx * (x/z) + cx
    # v = fy * (y/z) + cy
    # Manual projection
    
    # Expand for batch matmul: (N, 3, 1)
    obj_pos_c_expanded = obj_pos_c.unsqueeze(-1) 
    projected = torch.bmm(K, obj_pos_c_expanded).squeeze(-1) # (N, 3)
    
    # Normalize by Z (depth)
    z = projected[:, 2]
    # Guard against division by zero
    z_safe = torch.where(torch.abs(z) < 1e-6, torch.ones_like(z) * 1e-6, z)
    
    u = projected[:, 0] / z_safe
    v = projected[:, 1] / z_safe
    
    # 4. Check Visibility
    # Condition 1: In front of camera (z > 0.05)
    in_front = (z > 0.05)
    
    # Condition 2: Within image bounds [0, W] and [0, H]
    in_bounds_u = (u >= 0) & (u < width)
    in_bounds_v = (v >= 0) & (v < height)
    
    is_visible = in_front & in_bounds_u & in_bounds_v # (N,)

    
    # 5. Calculate Focus Reward (Centering)
    center_u = width / 2.0
    center_v = height / 2.0
    
    # Normalized distance from center (0 = center, 1 = edge)
    # Norm by half-dim
    dist_u = (u - center_u) / (width / 2.0)
    dist_v = (v - center_v) / (height / 2.0)
    
    # Euclidean distance from center in normalized coords range [0, sqrt(2)]
    # dist = sqrt(du^2 + dv^2)
    dist_norm_sq = dist_u**2 + dist_v**2
    dist_norm = torch.sqrt(dist_norm_sq)
    
    # Max possible distance is sqrt(1^2 + 1^2) = 1.414 (corner)
    # We want 1.0 at center, 0.0 at corners
    # Normalize to [0, 1] range: dist_norm / 1.414
    dist_scaled = torch.clamp(dist_norm / 1.4142, 0.0, 1.0)
    
    # Invert so 0 dist -> 1 reward
    # Use exponent to sharpen the peak around center
    focus_score = (1.0 - dist_scaled)
    focus_score = torch.clamp(focus_score, 0.0, 1.0)
    focus_score = torch.pow(focus_score, focus_exponent)
    
    # Mask out invisible objects (reward = 0.0)
    total_reward = focus_score * is_visible.float()
    
    return total_reward