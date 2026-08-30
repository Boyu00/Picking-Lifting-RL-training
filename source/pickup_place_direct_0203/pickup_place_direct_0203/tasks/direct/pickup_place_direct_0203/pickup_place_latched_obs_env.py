from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_apply

from .pickup_place_direct_0510_env import PickupPlaceDirect0510Env
from .pickup_place_latched_obs_env_cfg import PickupPlaceLatchedObsEnvCfg
from .mdp import observations as mdp_obs
from .mdp import rewards as mdp_rewards
from .mdp import terminations as mdp_term


class PickupPlaceLatchedObsEnv(PickupPlaceDirect0510Env):
    """Latched Observation environment (v2).
    
    Actor: 25D (凍結物體座標)
    Critic: 49D (即時特權物體座標 + BBox corners)
    新功能：
    - 初始物體位置記錄與防撞位移懲罰/終止。
    - 懸停接近引導獎勵。
    """
    
    cfg: PickupPlaceLatchedObsEnvCfg

    def __init__(self, cfg: PickupPlaceLatchedObsEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # ===== 鎖存觀測 Buffer =====
        self.latched_object_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )
        
        # ===== 初始物體位置記錄 Buffer =====
        self.object_initial_pos_w = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )
        
        # ===== 重置延遲與原因追蹤 Buffer =====
        self.displaced_too_far_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.delay_reset_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
        # 註冊額外的 episode_sums 鍵以供日誌紀錄
        self.episode_sums["reward_hover"] = torch.zeros(self.num_envs, device=self.device)
        self.episode_sums["reward_displacement"] = torch.zeros(self.num_envs, device=self.device)
        self.episode_sums["reward_grasp"] = torch.zeros(self.num_envs, device=self.device)
        
        # 同步物理與數據緩衝區，確保 Step 0 數據完全一致
        self.scene.write_data_to_sim()
        self.scene["object"].update(dt=0.0)
        
        # 初始鎖存（用當前真值）
        self._latch_object_positions(
            torch.arange(self.num_envs, device=self.device)
        )
        # 初始記錄物體世界位置
        self.object_initial_pos_w.copy_(self.scene["object"].data.root_com_pose_w[:, :3])
        
        print(f"\n[LatchedObsEnv] Initialized with latched observation v2")
        print(f"  Actor obs dim: {self.cfg.observation_space}")
        print(f"  Critic obs dim: {self.cfg.state_space}")
        print(f"  Noise bias: {self.cfg.latched_obs_noise_bias} m")
        print(f"  Noise std: {self.cfg.latched_obs_noise_std} m")
    
    def _latch_object_positions(self, env_ids: torch.Tensor):
        """在 episode reset 時凍結物體座標。"""
        gt_pos = mdp_obs.object_position_in_robot_root_frame(
            self, object_cfg=SceneEntityCfg("object")
        )
        
        latched = gt_pos[env_ids].clone()
        bias = self.cfg.latched_obs_noise_bias
        std = self.cfg.latched_obs_noise_std
        
        if bias > 0.0:
            latched += (torch.rand_like(latched) * 2 - 1) * bias
        if std > 0.0:
            latched += torch.randn_like(latched) * std
            
        self.latched_object_pos[env_ids] = latched
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset selected environments."""
        super()._reset_idx(env_ids)
            
    def _get_observations(self) -> dict:
        """非對稱觀測：Critic 擴展回 49D 帶 BBox。"""
        # === 延遲鎖存機制 ===
        # 1. 在 step 0 (剛 reset 完時)，進行初步鎖存，提供第 0 步的初始觀測值
        reset_env_ids = (self.episode_length_buf == 0).nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.object_initial_pos_w[reset_env_ids] = self.scene["object"].data.root_com_pose_w[reset_env_ids, :3].clone()
            self._latch_object_positions(reset_env_ids)
            
        # 2. 在 step 1 (經過一次物理步進，物體在重力下完全沉降穩定後)，
        # 重新更新並鎖定真實沉降後的初始世界位置，以防止初始彈跳造成位移終止誤判。
        settled_env_ids = (self.episode_length_buf == 1).nonzero(as_tuple=False).squeeze(-1)
        if len(settled_env_ids) > 0:
            self.object_initial_pos_w[settled_env_ids] = self.scene["object"].data.root_com_pose_w[settled_env_ids, :3].clone()
            self._latch_object_positions(settled_env_ids)

        # === 共用部分 ===
        joint_indices = list(self._arm_joint_indices) + list(self._gripper_joint_idx)
        jpos = self.joint_pos[:, joint_indices] - self.robot.data.default_joint_pos[:, joint_indices]
        jvel = self.joint_vel[:, joint_indices] - self.robot.data.default_joint_vel[:, joint_indices]
        
        # Target position (Env Frame → World Frame → Robot Frame)
        target_pos_w = self.target_poses + self.scene.env_origins
        target_in_robot, _ = subtract_frame_transforms(
            self.scene["robot"].data.root_pos_w,
            self.scene["robot"].data.root_quat_w,
            target_pos_w
        )
        
        # 觀測雜訊
        if self.cfg.observation_noise_scale > 0.0:
            ns = self.cfg.observation_noise_scale
            jpos = jpos + torch.randn_like(jpos) * 0.0001 * ns
            jvel = jvel + torch.randn_like(jvel) * 0.001 * ns
        
        # === Actor 觀測：用凍結座標 (25D) ===
        actor_obs = torch.cat([
            jpos.view(self.num_envs, -1),     # 6
            jvel.view(self.num_envs, -1),     # 6
            self.latched_object_pos,          # 3
            target_in_robot,                  # 3
            self.actions,                     # 7
        ], dim=1)  # → (B, 25)
        
        # === Critic 觀測：用即時真值 + BBox (49D) ===
        gt_obj_pos = mdp_obs.object_position_in_robot_root_frame(
            self, object_cfg=SceneEntityCfg("object")
        )
        gt_bbox = mdp_obs.object_bbox_corners_relative(
            self, object_cfg=SceneEntityCfg("object")
        )
        
        if self.cfg.observation_noise_scale > 0.0:
            ns = self.cfg.observation_noise_scale
            gt_obj_pos = gt_obj_pos + torch.randn_like(gt_obj_pos) * 0.001 * ns
            gt_bbox = gt_bbox + torch.randn_like(gt_bbox) * 0.001 * ns
            
        critic_obs = torch.cat([
            jpos.view(self.num_envs, -1),     # 6
            jvel.view(self.num_envs, -1),     # 6
            gt_obj_pos,                       # 3
            gt_bbox,                          # 24
            target_in_robot,                  # 3
            self.actions,                     # 7
        ], dim=1)  # → (B, 49)
        
        return {
            "policy": actor_obs,
            "critic": critic_obs,
        }

    def _get_rewards(self) -> torch.Tensor:
        """計算對齊 Franka 的直觀密集獎勵值 (繞過課程系統)。"""
        rewards_dict = {}

        # 1. Decoupled Reaching Reward (Round 5.5: 水平對中 + 垂直深度下沉解耦引導，消除 3.7cm 懸停盲區)
        cube_pos_w = self.scene["object"].data.root_com_pose_w[:, :3]
        ee_w = self.ee_frame.data.target_pos_w[..., 0, :]
        dist_xy = torch.norm(cube_pos_w[:, :2] - ee_w[:, :2], dim=1)
        dist_z = torch.abs(cube_pos_w[:, 2] - ee_w[:, 2])
        distance_ee_obj = torch.norm(cube_pos_w - ee_w, dim=1)
        reach_xy = 1.0 - torch.tanh(dist_xy / self.cfg.reward_settings.get("reaching_std", 0.08))
        reach_z = 1.0 - torch.tanh(dist_z / self.cfg.reward_settings.get("reaching_z_std", 0.035))
        rewards_dict["reaching"] = 0.5 * reach_xy + 0.5 * reach_z

        # 2. Close on Proximity (Round 5.5: 動作意圖即時閉爪引導，消除 0.66s 物理時延斷層)
        close_std = self.cfg.reward_settings.get("close_std", 0.08)
        close_proximity = torch.clamp(1.0 - (distance_ee_obj / close_std), 0.0, 1.0)
        close_intent = torch.clamp(-self.actions[:, 6], 0.0, 1.0)
        rewards_dict["close"] = close_proximity * close_intent

        # 3. Physical Grasp / Motor Stall (Round 5: 物理意志 + 速度停滯 雙核真夾持判定)
        #    條件: (1) 距離接近方塊 (d < 0.06m) (2) 策略發力閉爪 (action <= -0.2)
        #          (3) 馬達接觸停滯 (|vel| < 0.30) (4) 關節卡入阻擋角度 (0.25 < pos < 0.90)
        grasp_thresh = self.cfg.reward_settings.get("close_threshold", 0.06)
        is_in_grasp_range = (distance_ee_obj < grasp_thresh)
        is_firm_close = (self.actions[:, 6] <= -0.2)
        gripper_vel = self.robot.data.joint_vel[:, self._gripper_joint_idx[0]]
        is_motor_stalled = (gripper_vel.abs() < 0.30)
        is_in_jam_range = (gripper_pos > 0.25) & (gripper_pos < 0.90)
        is_grasped = is_in_grasp_range & is_firm_close & is_motor_stalled & is_in_jam_range
        rewards_dict["grasp"] = is_grasped.float()

        # 4. Progressive Continuous Lifting Reward (Round 4: 沉降遮罩 + 連續漸進抬起大獎)
        #    在開局前 5 步物體沉降期間屏蔽 Lifting 獎勵，消除掉落假訊號
        is_settled = (self.episode_length_buf >= 5).float()
        lift_height = cube_pos_w[:, 2] - self.object_initial_pos_w[:, 2]
        lift_prog = torch.clamp(lift_height / self.cfg.reward_settings["lifting_min_height"], 0.0, 1.0)
        rewards_dict["lifting"] = is_settled * torch.where(
            lift_height > self.cfg.reward_settings["lifting_min_height"],
            torch.ones_like(lift_prog),
            lift_prog * (lift_height > 0.005).float()
        )

        # 5. Object Goal Tracking (std = 0.6)
        target_pos_w = self.target_poses + self.scene.env_origins
        distance_obj_goal = torch.norm(target_pos_w - cube_pos_w, dim=1)
        rewards_dict["goal"] = (
            lift_height > self.cfg.reward_settings["goal_min_height"]
        ).float() * (1.0 - torch.tanh(distance_obj_goal / self.cfg.reward_settings["goal_std"]))

        # 6. Object Goal Tracking Fine-Grained (std = 0.05)
        rewards_dict["goal_fine"] = (
            lift_height > self.cfg.reward_settings["goal_fine_min_height"]
        ).float() * (1.0 - torch.tanh(distance_obj_goal / self.cfg.reward_settings["goal_fine_std"]))

        # 7. Actions L2 rate penalty (作用在未裁剪的 actions 上以拉回 Actor 均值)
        rewards_dict["action_rate"] = torch.sum(torch.square(self.actions), dim=1)

        # 8. Joint velocity penalty (僅計算 5 軸手臂 + 1 軸夾爪，排除底盤 4 輪與未受控關節之高頻微震)
        controlled_joint_ids = list(self._arm_joint_indices) + list(self._gripper_joint_idx)
        rewards_dict["joint_vel"] = torch.sum(torch.square(self.robot.data.joint_vel[:, controlled_joint_ids]), dim=1)

        # 9. Ground Displacement / Anti-Push Penalty (Round 4: 8mm 對中滑移寬容死區)
        cube_xy = cube_pos_w[:, :2]
        initial_xy = self.object_initial_pos_w[:, :2]
        dist_xy = torch.norm(cube_xy - initial_xy, dim=1)
        is_on_table = (lift_height < 0.015).float()  # 1.5cm 以內視為在桌面上
        deadzone = self.cfg.reward_settings.get("displacement_deadzone", 0.008)
        excess_push = torch.clamp(dist_xy - deadzone, min=0.0)
        rewards_dict["displacement"] = is_on_table * torch.tanh(excess_push / 0.015)

        # ===== 課程權重讀取 (已按要求改為註解/禁用以防止死鎖，改為直接使用 config 固定權重) =====
        # w_lift = self._curriculum_term_weights.get("lifting_object", self.cfg.rew_scale_lift)
        # w_lift_vel = self._curriculum_term_weights.get("lifting_object_velocity", self.cfg.rew_scale_lift_vel)
        # w_close = self._curriculum_term_weights.get("close_reward", self.cfg.rew_scale_close)
        # w_goal = self._curriculum_term_weights.get("object_goal_tracking", self.cfg.rew_scale_goal)
        # w_goal_fine = self._curriculum_term_weights.get("object_goal_tracking_fine_grained", self.cfg.rew_scale_goal_fine)
        # w_action = self._curriculum_term_weights.get("action_rate", self.cfg.rew_scale_action)
        # w_joint = self._curriculum_term_weights.get("joint_vel", self.cfg.rew_scale_joint_vel)
        w_reach = self.cfg.rew_scale_reach
        w_close = getattr(self.cfg, "rew_scale_close", 1.0)
        w_grasp = getattr(self.cfg, "rew_scale_grasp", 2.0)
        w_lift = self.cfg.rew_scale_lift
        w_goal = self.cfg.rew_scale_goal
        w_goal_fine = self.cfg.rew_scale_goal_fine
        w_action = self.cfg.rew_scale_action
        w_joint = self.cfg.rew_scale_joint_vel
        w_disp = self.cfg.rew_scale_displacement

        # 計算總獎勵 (Round 4 嚴謹物理防作弊階梯)
        total_reward = (
            w_reach * rewards_dict["reaching"]
            + w_close * rewards_dict["close"]
            + w_grasp * rewards_dict["grasp"]
            + w_lift * rewards_dict["lifting"]
            + w_goal * rewards_dict["goal"]
            + w_goal_fine * rewards_dict["goal_fine"]
            + w_action * rewards_dict["action_rate"]
            + w_joint * rewards_dict["joint_vel"]
            + w_disp * rewards_dict["displacement"]
        )

        # 累加 Episode 總和日誌 (獨立分開紀錄，完全透明)
        self.episode_sums["reward_displacement"] += w_disp * rewards_dict["displacement"]
        self.episode_sums["reward_reaching"] += w_reach * rewards_dict["reaching"]
        self.episode_sums["reward_close"] += w_close * rewards_dict["close"]
        self.episode_sums["reward_grasp"] += w_grasp * rewards_dict["grasp"]
        self.episode_sums["reward_lifting"] += w_lift * rewards_dict["lifting"]
        self.episode_sums["reward_goal"] += w_goal * rewards_dict["goal"]
        self.episode_sums["reward_goal_fine"] += w_goal_fine * rewards_dict["goal_fine"]
        self.episode_sums["reward_action_rate"] += w_action * rewards_dict["action_rate"]
        self.episode_sums["reward_joint_vel"] += w_joint * rewards_dict["joint_vel"]

        # 指標紀錄 (對應 Franka 標準)
        is_reached = distance_ee_obj < self.cfg.reward_settings["object_is_reached_threshold"]
        is_lifted = (lift_height > self.cfg.reward_settings["lifting_min_height"]) & (is_settled > 0.5)
        is_tracked = distance_obj_goal < 0.08

        # 寫入日誌系統
        if not hasattr(self, "extras"):
            self.extras = {}
        if "episode" not in self.extras:
            self.extras["episode"] = {}

        self.extras["episode"]["reaching_success"] = torch.mean(is_reached.float())
        self.extras["episode"]["grasp_success"] = torch.mean(is_grasped.float())
        self.extras["episode"]["lifting_success"] = torch.mean(is_lifted.float())
        self.extras["episode"]["object_goal_tracking_success"] = torch.mean(is_tracked.float())
        self.extras["episode"]["mean_push_dist_xy"] = torch.mean(dist_xy).item()
        
        object_bottom_z = mdp_rewards._get_object_bottom_z(self, SceneEntityCfg("object"))
        self.extras["episode"]["mean_lift_height"] = torch.mean(object_bottom_z).item()

        total_reward = torch.nan_to_num(total_reward, nan=0.0, posinf=0.0, neginf=0.0)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """計算終止條件（整合位移過大與延遲重置功能）。"""
        out_of_bounds, time_out = super()._get_dones()
        
        # 新增終止條件：物體被推開（自定義閾值，預設放寬到 15cm）
        # 僅在 step > 1 (即完全沉降後) 才進行位移終止判定，防止初始彈跳誤判
        self.displaced_too_far_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        active_mask = self.episode_length_buf > 1
        
        if active_mask.any():
            max_disp = getattr(self.cfg, "max_displacement", 0.15)
            all_displaced = mdp_term.object_displaced_too_far(
                self, max_displacement=max_disp, object_cfg=SceneEntityCfg("object")
            )
            self.displaced_too_far_buf = active_mask & all_displaced
            
        # 整合真實終止條件
        raw_terminated = out_of_bounds | self.displaced_too_far_buf
        
        # ===== 延遲重置機制 (以利畫面渲染與除錯) =====
        # 當任一環境觸發 raw_terminated，我們延遲 5 步 (0.5秒) 才真正回報 True 觸發 reset
        # 這能讓播放影片或實時渲染時，能清楚看到物體被撞飛或飛出界外的物理動畫，隨後才被傳送重置。
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        for env_id in range(self.num_envs):
            if raw_terminated[env_id]:
                # 如果是剛觸發，且計數器為 0，設定延遲步數為 5 步
                if self.delay_reset_counter[env_id] == 0:
                    self.delay_reset_counter[env_id] = 5
            
            # 如果計數器 > 0
            if self.delay_reset_counter[env_id] > 0:
                if self.delay_reset_counter[env_id] == 1:
                    # 當計數器倒數到 1 時，這一步回報 True，觸發真實的 _reset_idx
                    terminated[env_id] = True
                    self.delay_reset_counter[env_id] = 0
                else:
                    # 大於 1 時，繼續倒數，但這一步欺騙系統回報 False（不進行 reset）
                    self.delay_reset_counter[env_id] -= 1
                    terminated[env_id] = False
            else:
                terminated[env_id] = False
                
        return terminated, time_out

    def _apply_action(self) -> None:
        """套用動作（加入 Jacobian 偏置修正以解決運動學不對齊問題）。"""
        # actions shape: (num_envs, 7) -> [dx, dy, dz, drx, dry, drz, gripper]
        position_scale = self.cfg.action_cfg.get("position_scale", 0.05)
        rotation_scale = self.cfg.action_cfg.get("rotation_scale", 0.5)

        # 對 actions 進行 [-2.0, 2.0] 物理限幅以保護機器人
        self.clipped_actions = torch.clamp(self.actions, -2.0, 2.0)
        
        ik_commands = torch.zeros((self.num_envs, 6), device=self.device)
        ik_commands[:, :3] = self.clipped_actions[:, :3] * position_scale
        ik_commands[:, 3:] = self.clipped_actions[:, 3:6] * rotation_scale
        
        if hasattr(self, 'ee_frame'):
            ee_pos_w = self.ee_frame.data.target_pos_w[..., 0, :]
            ee_quat_w = self.ee_frame.data.target_quat_w[..., 0, :]
        else:
            ee_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
            ee_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
            ee_quat_w[:, 0] = 1.0
            
        self.ik_controller.set_command(ik_commands, ee_pos=ee_pos_w, ee_quat=ee_quat_w)
        
        # 1. 計算原始 Jacobian 矩陣 (基於 gripper_link 原點)
        ee_jacobians = self.scene["robot"].root_physx_view.get_jacobians()[:, self.ee_body_idx, :, :]
        arm_jacobians = ee_jacobians[:, :, self._arm_joint_indices]
        
        # 2. 進行 Jacobian 偏置修正 (根據 USD 精確量測，末端感測器相對於 gripper_link 有 X=-7mm, Z=7.5cm 的偏置)
        offset_local = torch.tensor([-0.007, 0.0, 0.075], device=self.device).repeat(self.num_envs, 1)
        offset_w = quat_apply(ee_quat_w, offset_local)
        
        # 構造偏置向量的反對稱矩陣 (skew-symmetric matrix)
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
        
        # 3. 求解 IK 關節目標 (使用修正後的 Jacobian 達到精確運動學控制)
        arm_joint_pos = self.scene["robot"].data.joint_pos_target[:, self._arm_joint_indices]
        joint_pos_desired = self.ik_controller.compute(
            ee_pos_w,
            ee_quat_w,
            arm_jacobians_corrected,
            arm_joint_pos
        )
        
        # 4. 二進位夾爪控制 (對齊 Franka 語意：action > 0 為張開，action <= 0 為閉合)
        #    JetRover 物理定義：1.57 rad = 全開 (Open)，0.0 rad = 全閉 (Close)
        gripper_action = self.clipped_actions[:, 6]
        gripper_target = torch.where(
            gripper_action > 0.0,
            torch.full_like(gripper_action, 1.57),  # > 0 -> Open (1.57 rad)
            torch.zeros_like(gripper_action)        # <= 0 -> Close (0.0 rad)
        )
        
        # 5. 限制關節極值並輸出
        joint_pos_desired = torch.clamp(joint_pos_desired, -2.09, 2.09)
        gripper_target = torch.clamp(gripper_target, 0.0, 1.57)
        
        targets = torch.cat([joint_pos_desired, gripper_target.unsqueeze(1)], dim=1)
        self.robot.set_joint_position_target(targets, joint_ids=list(self._arm_joint_indices) + list(self._gripper_joint_idx))

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """重置特定環境 (覆寫物體重置位置，使其直接貼在桌面，消除初始碰撞彈跳，並印出重置原因日誌)。"""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
            
        # 1. 重設延遲計數器
        self.delay_reset_counter[env_ids] = 0
        
        # 2. 印出重置詳細原因，便於 play / train 日誌除錯與追蹤 (已註解)
        # if len(env_ids) > 0:
        #     env_origins = self.scene.env_origins[env_ids]
        #     dist_to_origin = torch.norm(env_origins[:, :2], dim=-1)
        #     closest_idx = torch.argmin(dist_to_origin).item()
        #     closest_env_id = env_ids[closest_idx].item()
        #     
        #     out_of_bounds_triggered = self.reset_terminated[closest_env_id].item() if hasattr(self, 'reset_terminated') else False
        #     displacement_triggered = self.displaced_too_far_buf[closest_env_id].item() if hasattr(self, 'displaced_too_far_buf') else False
        #     timeout_triggered = self.reset_time_outs[closest_env_id].item() if hasattr(self, 'reset_time_outs') else False
        #     
        #     print(f"[RESET LOG] (Closest to 0,0 among resetting) Env {closest_env_id} reset triggered! "
        #           f"| Details: Out_of_Bounds={out_of_bounds_triggered}, "
        #           f"Displaced_Too_Far={displacement_triggered}, "
        #           f"Time_Out={timeout_triggered}")
                  
        # 3. 呼叫父級 _reset_idx
        super()._reset_idx(env_ids)
        
        try:
            # 覆寫 Z 座標：直接重置到桌面表面 (Z = 0.022m 以避免穿透與大落差彈跳)
            # 這能保證物體一重置就處於穩定靜止狀態，消除重力下墜彈跳誤判
            object_body = self.scene["object"]
            env_origins = self.scene.env_origins[env_ids]
            current_pose = object_body.data.root_pose_w[env_ids].clone()
            
            # 0.022m 為物體 COM 貼齊桌面的理想物理高度
            current_pose[:, 2] = env_origins[:, 2] + 0.022
            
            # 套用至模擬器 (Teleport)
            object_body.write_root_pose_to_sim(current_pose, env_ids)
            
            # 清除所有殘餘速度，保證靜止
            zeros_vel = torch.zeros((len(env_ids), 6), device=self.device)
            object_body.write_root_velocity_to_sim(zeros_vel, env_ids)
            object_body.update(dt=0.0)
        except Exception as e:
            print(f"[Error] Failed to teleport object directly on table: {e}")
