from __future__ import annotations

import torch
import itertools
from collections.abc import Sequence

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import subtract_frame_transforms, quat_apply
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.assets import RigidObject, Articulation

from .franka_cube_lift_direct_env_cfg import FrankaCubeLiftDirectEnvCfg
from .mdp import observations as mdp_obs
from .mdp import rewards as mdp_rewards


class FrankaCubeLiftDirectEnv(DirectRLEnv):
    """Franka Cube Lift Direct RL Environment with Latched Observations (v2).
    
    Actor: 31D (joint pos, joint vel, latched object pos, target, action)
    Critic: 55D (joint pos, joint vel, gt object pos, gt object bbox corners, target, action)
    """

    cfg: FrankaCubeLiftDirectEnvCfg

    def _setup_scene(self):
        # 1. Spawn ground plane and dome light
        self.cfg.plane.spawn.func("/World/ground", self.cfg.plane.spawn, translation=self.cfg.plane.init_state.pos)
        self.cfg.light.spawn.func("/World/light", self.cfg.light.spawn)

        # 2. Add robot
        self.robot = Articulation(self.cfg.robot)

        # 3. Add end-effector frame transformer
        from isaaclab.sensors import FrameTransformer
        self.ee_frame = FrameTransformer(self.cfg.ee_frame)

        # 4. Spawn table at env_0
        self.cfg.table.spawn.func(
            "/World/envs/env_0/Table",
            self.cfg.table.spawn,
            translation=self.cfg.table.init_state.pos,
            orientation=self.cfg.table.init_state.rot,
        )


        # 5. Clone environments to replicate them across num_envs (use copy_from_source=True as requested by user)
        self.scene.clone_environments(copy_from_source=True)

        # 6. Spawn object (Cup) at all envs (after cloning)
        #    Since all environments now exist on the stage, the spawner will match "/World/envs/env_.*/Object"
        #    and spawn a randomized choice for each environment independently in a single fast block!
        self.cfg.object_cfg.spawn.func(
            "/World/envs/env_.*/Object",
            self.cfg.object_cfg.spawn,
            translation=self.cfg.object_cfg.init_state.pos,
            orientation=self.cfg.object_cfg.init_state.rot,
        )

        # 7. Enable USD Instancing for spawned objects to minimize CPU stage graph overhead
        try:
            import isaaclab.sim as sim_utils_local
            stage = sim_utils_local.get_current_stage()
            for env_id in range(self.num_envs):
                prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Object")
                if prim.IsValid():
                    prim.SetInstanceable(True)
        except Exception:
            pass

        # 7. Add object wrapper matching regex
        self.object = RigidObject(self.cfg.object_cfg)

        # 8. Register to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["ee_frame"] = self.ee_frame

    def __init__(self, cfg: FrankaCubeLiftDirectEnvCfg, render_mode=None, **kwargs):
        # 1. Initialize Scene objects via super()
        super().__init__(cfg, render_mode, **kwargs)

        # 2. Extract asset references
        self._robot: Articulation = self.robot
        self._object: RigidObject = self.object
        self.ee_frame = self.ee_frame

        # 3. Configure Arm and Gripper Joint indices
        self._arm_joint_indices, _ = self._robot.find_joints("panda_joint.*")
        self._gripper_joint_idx, _ = self._robot.find_joints("panda_finger.*")
        
        # Default target joints for initialization
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)

        # 4. Initialize Differential IK Controller for Franka
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method=self.cfg.action_cfg.get("ik_method", "dls")
        )
        self.ik_controller = DifferentialIKController(ik_cfg, self.num_envs, self.device)
        self.ee_body_idx = self._robot.find_bodies("panda_hand")[0][0]

        # 5. Initialize Latched Obs Buffers
        self.latched_object_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.object_initial_pos_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.target_poses = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)

        # 6. Initialize object local corners dynamically (for the three cups)
        self.cfg.object_scale = getattr(self.cfg, "object_scale", [0.8, 0.8, 0.8])
        self._initialize_object_local_corners()

        # Initialize debug visualization markers
        if getattr(self.cfg, "debug_vis", False):
            self._setup_markers()
            self._setup_bbox_visualizer()


        # 7. Accumulators for metrics & logging
        self.episode_sums = {
            "reward_reaching": torch.zeros(self.num_envs, device=self.device),
            "reward_lifting": torch.zeros(self.num_envs, device=self.device),
            "reward_goal": torch.zeros(self.num_envs, device=self.device),
            "reward_goal_fine": torch.zeros(self.num_envs, device=self.device),
            "reward_action_rate": torch.zeros(self.num_envs, device=self.device),
            "reward_joint_vel": torch.zeros(self.num_envs, device=self.device),
        }

        # 8. Success rate buffers (for population-average metrics)
        self.env_reaching_success = torch.zeros(self.num_envs, device=self.device)
        self.env_lifting_success = torch.zeros(self.num_envs, device=self.device)
        self.env_goal_success = torch.zeros(self.num_envs, device=self.device)
        self._pending_reward_logs = {}

        print(f"\n[FrankaCubeLiftDirectEnv] Initialized with Latched Obs v2")
        print(f"  Actor Obs space:  {self.cfg.observation_space} dims")
        print(f"  Critic State space: {self.cfg.state_space} dims")

    def _latch_object_positions(self, env_ids: torch.Tensor):
        """Latch the object position in the robot root frame."""
        gt_pos = mdp_obs.object_position_in_robot_root_frame(
            self, object_cfg=SceneEntityCfg("object")
        )
        self.latched_object_pos[env_ids] = gt_pos[env_ids].clone()

    def _sample_target_poses(self, env_ids: torch.Tensor):
        """Sample a target destination position for the cube."""
        cfg = self.cfg
        x_range = cfg.target_pos_range["x"]
        y_range = cfg.target_pos_range["y"]
        z_range = cfg.target_pos_range["z"]

        self.target_poses[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * (x_range[1] - x_range[0]) + x_range[0]
        self.target_poses[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * (y_range[1] - y_range[0]) + y_range[0]
        self.target_poses[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * (z_range[1] - z_range[0]) + z_range[0]

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset internal buffers and teleport articulation/objects."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # 0. Compute success rate and reward logging before we overwrite/reset anything
        if len(env_ids) > 0:
            cube_pos_w = self._object.data.root_com_pose_w[:, :3]
            ee_w = self.ee_frame.data.target_pos_w[..., 0, :]
            distance_ee_obj = torch.norm(cube_pos_w - ee_w, dim=1)
            target_pos_w = self.target_poses + self.scene.env_origins
            distance_obj_goal = torch.norm(target_pos_w - cube_pos_w, dim=1)
            
            is_reached = distance_ee_obj < self.cfg.reward_settings["object_is_reached_threshold"]
            lift_height = cube_pos_w[:, 2] - self.object_initial_pos_w[:, 2]
            is_lifted = lift_height > self.cfg.reward_settings["lifting_min_height"]
            is_tracked = distance_obj_goal < 0.08
            
            # Store in population buffers
            self.env_reaching_success[env_ids] = is_reached[env_ids].float()
            self.env_lifting_success[env_ids] = is_lifted[env_ids].float()
            self.env_goal_success[env_ids] = is_tracked[env_ids].float()
            
            # Calculate overall population success rates
            reach_rate = torch.mean(self.env_reaching_success)
            lift_rate = torch.mean(self.env_lifting_success)
            track_rate = torch.mean(self.env_goal_success)
            
            # Store in pending logs for TensorBoard logging in step()
            self._pending_reward_logs = {
                "reaching_success": reach_rate,
                "lifting_success": lift_rate,
                "object_goal_tracking_success": track_rate,
            }
            
            # Log Episode Reward Sums (Total Return) and Mean Step Rewards
            for key, tensor_sum in self.episode_sums.items():
                finished_sums = tensor_sum[env_ids].clone()
                tensor_sum[env_ids] = 0.0  # Clear sums for resetting envs
                
                if len(finished_sums) > 0:
                    reward_sum_mean = torch.mean(finished_sums)
                    self._pending_reward_logs[key] = reward_sum_mean
                    
                    lengths = self.episode_length_buf[env_ids].float()
                    lengths = torch.clamp(lengths, min=1.0)
                    mean_rewards = finished_sums / lengths
                    reward_step_mean = torch.mean(mean_rewards)
                    
                    self._pending_reward_logs[f"{key}_mean"] = reward_step_mean

        super()._reset_idx(env_ids)

        # 1. Reset Robot Joints to defaults + small noise
        joint_pos = self._robot.data.default_joint_pos[env_ids] + (torch.rand(
            (len(env_ids), self._robot.num_joints), device=self.device
        ) * 2 - 1) * 0.1
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 2. Reset Object to spawn pos (pos_x=0.5, pos_y=0, pos_z=0.055) + small XY randomization
        env_origins = self.scene.env_origins[env_ids]
        object_pos_local = torch.tensor([0.5, 0.0, 0.055], device=self.device).repeat(len(env_ids), 1)
        # Randomize XY offset slightly (within table range)
        object_pos_local[:, 0] += (torch.rand(len(env_ids), device=self.device) * 2 - 1) * 0.08
        object_pos_local[:, 1] += (torch.rand(len(env_ids), device=self.device) * 2 - 1) * 0.15
        
        default_root_state = self._object.data.default_root_state[env_ids].clone()

        # Apply per-prototype offsets if configured, otherwise use default local pos
        if hasattr(self, "selected_object_id_per_env") and self.selected_object_id_per_env is not None:
            try:
                proto_idxs = self.selected_object_id_per_env[env_ids]
                proto_offsets = getattr(self.cfg, "object_proto_offsets", None)
                if proto_offsets is not None:
                    offsets = torch.zeros_like(object_pos_local)
                    for i_e, env_id in enumerate(env_ids.cpu().tolist()):
                        sel = int(proto_idxs[i_e].item())
                        # safe indexing into proto_offsets list-like
                        try:
                            off = torch.tensor(proto_offsets[sel], device=self.device, dtype=torch.float32)
                        except Exception:
                            off = torch.zeros(3, device=self.device)
                        offsets[i_e] = off
                    default_root_state[:, :3] = env_origins + object_pos_local + offsets
                else:
                    default_root_state[:, :3] = env_origins + object_pos_local
            except Exception:
                default_root_state[:, :3] = env_origins + object_pos_local
        else:
            default_root_state[:, :3] = env_origins + object_pos_local

        self._object.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        self._object.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

        # Best-effort: apply per-prototype physical property overrides if provided
        # cfg.object_physics can be a list/dict of per-prototype property dicts, e.g. [{"mass":0.1, "friction":0.5}, ...]
        object_physics = getattr(self.cfg, "object_physics", None)
        if object_physics is not None and hasattr(self, "selected_object_id_per_env") and self.selected_object_id_per_env is not None:
            try:
                import isaaclab.sim as sim_utils_local
                from pxr import Usd
                stage = sim_utils_local.get_current_stage()
                proto_idxs = self.selected_object_id_per_env[env_ids]
                for i_e, env_id in enumerate(env_ids.cpu().tolist()):
                    sel = int(proto_idxs[i_e].item())
                    # select props from list-like or dict-like
                    props = None
                    try:
                        if isinstance(object_physics, (list, tuple)):
                            props = object_physics[sel]
                        elif isinstance(object_physics, dict):
                            # dict keyed by int or str
                            props = object_physics.get(sel, object_physics.get(str(sel), None))
                    except Exception:
                        props = None

                    if not props:
                        continue

                    prim_path = f"/World/envs/env_{env_id}/Object"
                    prim = stage.GetPrimAtPath(prim_path)
                    if not prim.IsValid():
                        continue

                    # Try to set common physics attributes (best-effort, may be schema-dependent)
                    if "mass" in props:
                        try:
                            attr = prim.GetAttribute("physics:mass")
                            if attr:
                                attr.Set(float(props["mass"]))
                        except Exception:
                            pass
                    if "friction" in props:
                        try:
                            # dynamic friction attribute names vary; try a few common ones
                            for name in ("physics:dynamicFriction", "physics:friction"):
                                attr = prim.GetAttribute(name)
                                if attr:
                                    attr.Set(float(props["friction"]))
                                    break
                        except Exception:
                            pass
            except Exception:
                # ignore failures to set physics attributes
                pass

        # 3. Sample new target position for the cube
        self._sample_target_poses(env_ids)

        # Force write teleported states to sim and refresh data buffers
        self.scene.write_data_to_sim()
        self.scene.update(self.cfg.sim.dt)

        if getattr(self.cfg, "debug_vis", False):
            self._update_debug_vis()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """Process policy action command outputs through Inverse Kinematics."""

        # Scale actions
        position_scale = self.cfg.action_cfg.get("position_scale", 0.005)
        rotation_scale = self.cfg.action_cfg.get("rotation_scale", 0.05)
        
        # # Clip actions to [-2.0, 2.0] for physical robot safety,
        # # but keep self.actions unclipped for PPO optimization and reward penalty computation.
        # self.clipped_actions = torch.clamp(self.actions, -2.0, 2.0)
        self.clipped_actions = self.actions
        
        # actions shape: (num_envs, 7) -> [dx, dy, dz, drx, dry, drz, gripper]
        ik_commands = torch.zeros((self.num_envs, 6), device=self.device)
        ik_commands[:, :3] = self.clipped_actions[:, :3] * position_scale
        ik_commands[:, 3:6] = self.clipped_actions[:, 3:6] * rotation_scale
        
        ee_pos_w = self.ee_frame.data.target_pos_w[..., 0, :]
        ee_quat_w = self.ee_frame.data.target_quat_w[..., 0, :]
        
        self.ik_controller.set_command(ik_commands, ee_pos=ee_pos_w, ee_quat=ee_quat_w)
        
        # Compute IK Jacobian
        ee_jacobians = self._robot.root_physx_view.get_jacobians()[:, self.ee_body_idx, :, :]
        arm_jacobians = ee_jacobians[:, :, self._arm_joint_indices]
        
        # Correct Jacobian translational part due to offset (0.1034m along local Z of end-effector)
        offset_local = torch.tensor([0.0, 0.0, 0.1034], device=self.device).repeat(self.num_envs, 1)
        offset_w = quat_apply(ee_quat_w, offset_local)
        
        # Construct the skew-symmetric matrix of offset_w
        x, y, z = offset_w[:, 0], offset_w[:, 1], offset_w[:, 2]
        zero = torch.zeros_like(x)
        skew = torch.stack([
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1)
        ], dim=-2)  # Shape: (num_envs, 3, 3)
        
        # J_v_E = J_v_B - skew(offset_w) @ J_w_B
        arm_jacobians_corrected = arm_jacobians.clone()
        arm_jacobians_corrected[:, 0:3, :] += torch.bmm(-skew, arm_jacobians[:, 3:6, :])
        
        arm_joint_pos = self._robot.data.joint_pos_target[:, self._arm_joint_indices]
        
        # Calculate target position for arm joints
        joint_pos_desired = self.ik_controller.compute(
            ee_pos_w,
            ee_quat_w,
            arm_jacobians_corrected,
            arm_joint_pos
        )
        # Clamp to joint limits
        joint_pos_desired = torch.clamp(joint_pos_desired, -3.0, 3.0)
        
        # Apply binary command to gripper (Finger positions: Open: 0.04m, Close: 0.0m)
        gripper_action = self.clipped_actions[:, 6]
        gripper_target = torch.where(gripper_action > 0.0, torch.full_like(gripper_action, 0.04), torch.zeros_like(gripper_action))
        gripper_targets = torch.stack([gripper_target, gripper_target], dim=1)
        
        # Set target joints
        self._robot.set_joint_position_target(joint_pos_desired, joint_ids=self._arm_joint_indices)
        self._robot.set_joint_position_target(gripper_targets, joint_ids=self._gripper_joint_idx)

    def _get_observations(self) -> dict:
        """Asymmetric Observations: Actor uses latched pose; Critic uses real-time pose and BBox."""
        # Sync reset coordinates after PhysX settles (lazy latching)
        # Latches the object position when the object has landed on the table surface (e.g. at step 3)
        # For the first 3 steps, we dynamically track the real-time position to capture the resting pose.
        settling_env_ids = (self.episode_length_buf <= 3).nonzero(as_tuple=False).squeeze(-1)
        if len(settling_env_ids) > 0:
            self.object_initial_pos_w[settling_env_ids] = self._object.data.root_com_pose_w[settling_env_ids, :3].clone()
            self._latch_object_positions(settling_env_ids)

        # 1. Joint positions/velocities relative to default
        joint_indices = list(self._arm_joint_indices) + list(self._gripper_joint_idx)
        jpos = self._robot.data.joint_pos[:, joint_indices] - self._robot.data.default_joint_pos[:, joint_indices]
        jvel = self._robot.data.joint_vel[:, joint_indices] - self._robot.data.default_joint_vel[:, joint_indices]

        # 2. Target position (Relative to robot frame)
        target_pos_w = self.target_poses + self.scene.env_origins
        target_in_robot, _ = subtract_frame_transforms(
            self._robot.data.root_com_pose_w[:, :3],
            self._robot.data.root_quat_w,
            target_pos_w
        )

        # Observation Noise
        obs_noise_scale = getattr(self.cfg, "observation_noise_scale", 0.0)
        if obs_noise_scale > 0.0:
            ns = obs_noise_scale
            jpos += torch.randn_like(jpos) * 0.0001 * ns
            jvel += torch.randn_like(jvel) * 0.001 * ns

        # ===== Actor Observation (31D) =====
        actor_obs = torch.cat([
            jpos.view(self.num_envs, -1),      # 9
            jvel.view(self.num_envs, -1),      # 9
            self.latched_object_pos,           # 3
            target_in_robot,                   # 3
            self.actions,                      # 7
        ], dim=1)

        # ===== Critic State/Observation (55D) =====
        gt_obj_pos = mdp_obs.object_position_in_robot_root_frame(
            self, object_cfg=SceneEntityCfg("object")
        )
        gt_bbox = mdp_obs.object_bbox_corners_relative(
            self, object_cfg=SceneEntityCfg("object")
        )

        if obs_noise_scale > 0.0:
            ns = obs_noise_scale
            gt_obj_pos += torch.randn_like(gt_obj_pos) * 0.001 * ns
            gt_bbox += torch.randn_like(gt_bbox) * 0.001 * ns

        critic_obs = torch.cat([
            jpos.view(self.num_envs, -1),      # 9
            jvel.view(self.num_envs, -1),      # 9
            gt_obj_pos,                        # 3
            gt_bbox,                           # 24
            target_in_robot,                   # 3
            self.actions,                      # 7
        ], dim=1)

        return {
            "policy": actor_obs,
            "critic": critic_obs,
        }

    def _get_rewards(self) -> torch.Tensor:
        """Compute the official Lift environment reward terms."""
        rewards_dict = {}

        # 1. Reaching Reward
        cube_pos_w = self._object.data.root_com_pose_w[:, :3]
        ee_w = self.ee_frame.data.target_pos_w[..., 0, :]
        distance_ee_obj = torch.norm(cube_pos_w - ee_w, dim=1)
        rewards_dict["reaching"] = 1.0 - torch.tanh(distance_ee_obj / self.cfg.reward_settings["reaching_std"])

        # 2. Lifting Reward (minimal_height = 0.04m above initial height)
        lift_height = cube_pos_w[:, 2] - self.object_initial_pos_w[:, 2]
        rewards_dict["lifting"] = torch.where(
            lift_height > self.cfg.reward_settings["lifting_min_height"], 1.0, 0.0
        )

        # 3. Object Goal Tracking (std = 0.3)
        target_pos_w = self.target_poses + self.scene.env_origins
        distance_obj_goal = torch.norm(target_pos_w - cube_pos_w, dim=1)
        rewards_dict["goal"] = (
            lift_height > self.cfg.reward_settings["goal_min_height"]
        ).float() * (1.0 - torch.tanh(distance_obj_goal / self.cfg.reward_settings["goal_std"]))

        # 4. Object Goal Tracking Fine-Grained (std = 0.05)
        rewards_dict["goal_fine"] = (
            lift_height > self.cfg.reward_settings["goal_fine_min_height"]
        ).float() * (1.0 - torch.tanh(distance_obj_goal / self.cfg.reward_settings["goal_fine_std"]))

        # 5. Actions L2 rate penalty
        rewards_dict["action_rate"] = torch.sum(torch.square(self.actions), dim=1)

        # 6. Joint velocity penalty
        rewards_dict["joint_vel"] = torch.sum(torch.square(self._robot.data.joint_vel), dim=1)

        # Get weights
        w_reach = self.cfg.rew_scale_reach
        w_lift = self.cfg.rew_scale_lift
        w_goal = self.cfg.rew_scale_goal
        w_goal_fine = self.cfg.rew_scale_goal_fine
        
        # Step-wise Curriculum for action and joint velocity penalties using config parameters
        start_step = getattr(self.cfg, "rew_curriculum_start_step", 10000)
        ramp_steps = getattr(self.cfg, "rew_curriculum_ramp_steps", 10000)
        target_action = getattr(self.cfg, "rew_curriculum_target_action", -0.1)
        target_joint = getattr(self.cfg, "rew_curriculum_target_joint_vel", -0.1)
        
        if self.common_step_counter > start_step + ramp_steps:
            w_action = target_action
            w_joint = target_joint
        elif self.common_step_counter > start_step:
            progress = min(1.0, float(self.common_step_counter - start_step) / float(max(1, ramp_steps)))
            w_action = self.cfg.rew_scale_action + progress * (target_action - self.cfg.rew_scale_action)
            w_joint = self.cfg.rew_scale_joint_vel + progress * (target_joint - self.cfg.rew_scale_joint_vel)
        else:
            w_action = self.cfg.rew_scale_action
            w_joint = self.cfg.rew_scale_joint_vel

        # Compute Total Reward
        total_reward = (
            w_reach * rewards_dict["reaching"]
            + w_lift * rewards_dict["lifting"]
            + w_goal * rewards_dict["goal"]
            + w_goal_fine * rewards_dict["goal_fine"]
            + w_action * rewards_dict["action_rate"]
            + w_joint * rewards_dict["joint_vel"]
        )

        # Logging sums
        self.episode_sums["reward_reaching"] += w_reach * rewards_dict["reaching"]
        self.episode_sums["reward_lifting"] += w_lift * rewards_dict["lifting"]
        self.episode_sums["reward_goal"] += w_goal * rewards_dict["goal"]
        self.episode_sums["reward_goal_fine"] += w_goal_fine * rewards_dict["goal_fine"]
        self.episode_sums["reward_action_rate"] += w_action * rewards_dict["action_rate"]
        self.episode_sums["reward_joint_vel"] += w_joint * rewards_dict["joint_vel"]

        # Calculate metrics for direct logging
        is_reached = distance_ee_obj < self.cfg.reward_settings["object_is_reached_threshold"]
        is_lifted = cube_pos_w[:, 2] > self.cfg.reward_settings["lifting_min_height"]
        is_tracked = distance_obj_goal < 0.08

        if not hasattr(self, "extras"):
            self.extras = {}
        if "episode" not in self.extras:
            self.extras["episode"] = {}

        self.extras["episode"]["reaching_success"] = torch.mean(is_reached.float())
        self.extras["episode"]["lifting_success"] = torch.mean(is_lifted.float())
        self.extras["episode"]["object_goal_tracking_success"] = torch.mean(is_tracked.float())
        self.extras["episode"]["mean_lift_height"] = torch.mean(cube_pos_w[:, 2]).item()

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminations: Timeout or Cube dropped below minimum height."""
        # Timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Cube dropped (e.g. falls off the table)
        cube_dropped = self._object.data.root_com_pose_w[:, 2] < -0.05
        
        return cube_dropped, time_out

    def step(self, actions: torch.Tensor) -> tuple[dict, torch.Tensor, bool, bool, dict]:
        """Step the environment and populate TensorBoard logs."""
        obs, reward, terminated, truncated, extras = super().step(actions)
        
        # Initialize logging structure
        if "log" not in extras:
            extras["log"] = {}
        if "episode" not in extras["log"]:
            extras["log"]["episode"] = {}
            
        # Log action stats
        extras["log"]["episode"]["action_mean"] = actions.mean().item()
        extras["log"]["episode"]["action_max"] = actions.abs().max().item()
        
        # Merge metrics calculated in _get_rewards()
        if hasattr(self, "extras") and "episode" in self.extras:
            for key, value in self.extras["episode"].items():
                extras["log"]["episode"][key] = value
                
        # Merge pending metrics calculated in _reset_idx()
        if self._pending_reward_logs:
            for key, value in self._pending_reward_logs.items():
                extras["log"]["episode"][key] = value
            self._pending_reward_logs.clear()
            
        # Compatibility backup
        if "log" in extras and "episode" in extras["log"]:
            extras["episode"] = extras["log"]["episode"]
            
        self.extras = extras

        # Update visualization markers at every step
        if getattr(self.cfg, "debug_vis", False):
            self._update_debug_vis()

        return obs, reward, terminated, truncated, extras

    def _update_object_local_corners(self, env_ids: torch.Tensor | None = None):
        """Dynamically compute bounding box corners for specified environments."""
        import isaaclab.sim as sim_utils_local
        from pxr import UsdGeom, Usd

        try:
            stage = sim_utils_local.get_current_stage()
        except Exception:
            return

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        env_ids_list = env_ids.cpu().tolist()
        
        # Initialize tensor if not exist yet
        if not hasattr(self, 'object_local_corners') or self.object_local_corners.shape[0] != self.num_envs:
            base_corners = torch.tensor(
                list(itertools.product([1.0, -1.0], repeat=3)), dtype=torch.float32, device=self.device
            )
            default_half_extents = torch.ones((self.num_envs, 3), device=self.device) * 0.05
            local_corners = base_corners.unsqueeze(0) * default_half_extents.unsqueeze(1) * 0.6
            self.object_local_corners = local_corners

        # Update specified environments
        for env_id in env_ids_list:
            prim_path = f"/World/envs/env_{env_id}/Object"
            prim = stage.GetPrimAtPath(prim_path)
            
            half_extents = torch.ones(3, device=self.device) * 0.05
            centers = torch.zeros(3, device=self.device)
            
            if prim.IsValid():
                bound = bbox_cache.ComputeUntransformedBound(prim)
                box_range = bound.GetRange()
                min_pt = box_range.GetMin()
                max_pt = box_range.GetMax()

                hx = (max_pt[0] - min_pt[0]) / 2.0
                hy = (max_pt[1] - min_pt[1]) / 2.0
                hz = (max_pt[2] - min_pt[2]) / 2.0
                half_extents = torch.tensor([hx, hy, hz], dtype=torch.float32, device=self.device)
                centers = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

            config_scale = torch.tensor(self.cfg.object_scale, device=self.device)
            half_extents = half_extents * config_scale

            base_corners = torch.tensor(
                list(itertools.product([1.0, -1.0], repeat=3)), dtype=torch.float32, device=self.device
            )
            local_corners = base_corners * half_extents + centers
            self.object_local_corners[env_id] = local_corners
    
    def _initialize_object_local_corners(self):
        """Initialize object local corners for all environments."""
        self._update_object_local_corners()

    def _setup_markers(self):
        """Setup visualization markers for debugging."""
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.sim.spawners.shapes import SphereCfg
        from isaaclab.sim.spawners.materials import PreviewSurfaceCfg

        # Red marker for object center of mass
        com_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/CoM_Marker",
            markers={
                "default": SphereCfg(
                    radius=0.015,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                )
            },
        )
        self.com_marker = VisualizationMarkers(com_marker_cfg)

        # Target marker (Green)
        target_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/TargetMarker",
            markers={
                "default": SphereCfg(
                    radius=0.015,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                )
            },
        )
        self.target_marker = VisualizationMarkers(target_marker_cfg)

        # Latch marker (Cyan - 0.0, 1.0, 1.0)
        latch_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/LatchMarker",
            markers={
                "default": SphereCfg(
                    radius=0.015,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                )
            },
        )
        self.latch_marker = VisualizationMarkers(latch_marker_cfg)

        # End Effector marker (Yellow - 1.0, 1.0, 0.0)
        ee_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/EEMarker",
            markers={
                "default": SphereCfg(
                    radius=0.015,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                )
            },
        )
        self.ee_marker = VisualizationMarkers(ee_marker_cfg)

    def _setup_bbox_visualizer(self):
        """Setup bounding box corner visualization."""
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.sim.spawners.shapes import SphereCfg
        from isaaclab.sim.spawners.materials import PreviewSurfaceCfg

        bbox_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/BBoxCorners",
            markers={
                "corner": SphereCfg(
                    radius=0.008,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),  # Orange
                )
            },
        )
        self.bbox_visualizer = VisualizationMarkers(bbox_marker_cfg)

    def _update_debug_vis(self):
        """Update visualization markers for debugging."""
        if self.sim.render_mode < self.sim.RenderMode.PARTIAL_RENDERING:
            return

        settings = self.cfg.debug_vis_settings

        # Target position marker (green sphere)
        if settings.get("target", True) and hasattr(self, "target_marker"):
            try:
                self.target_marker.visualize(self.target_poses + self.scene.env_origins)
            except Exception:
                pass

        # Center of Mass marker (red sphere)
        if settings.get("com", True) and hasattr(self, "com_marker"):
            try:
                object_entity = self.scene["object"]
                com_pos_w = object_entity.data.root_com_pose_w[:, :3]
                self.com_marker.visualize(translations=com_pos_w)
            except Exception:
                pass

        # Latch position marker (cyan sphere)
        if settings.get("latch", True) and hasattr(self, "latch_marker"):
            try:
                robot = self.scene["robot"]
                latch_pos_w = robot.data.root_pos_w + quat_apply(robot.data.root_quat_w, self.latched_object_pos)
                self.latch_marker.visualize(translations=latch_pos_w)
            except Exception:
                pass

        # End Effector position marker (yellow sphere)
        if settings.get("ee", True) and hasattr(self, "ee_marker"):
            try:
                ee_pos_w = self.ee_frame.data.target_pos_w[..., 0, :]
                self.ee_marker.visualize(translations=ee_pos_w)
            except Exception:
                pass

        # Bounding box corners (orange spheres)
        if settings.get("bbox", True) and hasattr(self, "bbox_visualizer") and hasattr(self, "object_local_corners"):
            try:
                object_entity = self.scene["object"]
                obj_pos_w = object_entity.data.root_com_pose_w[:, :3]
                obj_rot_w = object_entity.data.root_quat_w[:, :]
                
                local_corners = self.object_local_corners
                rot_expanded = obj_rot_w.unsqueeze(1).expand(-1, 8, -1)
                
                corners_rot = quat_apply(rot_expanded, local_corners)
                corners_world = corners_rot + obj_pos_w.unsqueeze(1)
                world_corners_all = corners_world.view(-1, 3)
                
                self.bbox_visualizer.visualize(translations=world_corners_all)
            except Exception:
                pass


