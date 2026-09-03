#!/usr/bin/env python3
"""Inject boosted exploration noise standard deviation into model_1650.pt."""

import os
import torch

src_path = "/workspace/test_isaaclab/pickup_place_direct_0203/logs/rsl_rl/pickup_place_direct_0510/2026-09-02_11-17-33_JetRover_0902_PosIK_3Stage_run1/model_1650.pt"
dst_path = "/workspace/test_isaaclab/pickup_place_direct_0203/logs/rsl_rl/pickup_place_direct_0510/2026-09-02_11-17-33_JetRover_0902_PosIK_3Stage_run1/model_1650_boosted.pt"

print(f"Loading checkpoint: {src_path}")
data = torch.load(src_path, map_location="cpu")

old_std = data["model_state_dict"]["std"].clone()
print(f"Old noise std [dx, dy, dz, grip]: {old_std.tolist()}")

# Reset std to [0.45, 0.45, 0.45, 0.50]
# 0.45 gives ~2.25mm random step exploration in 3D translation
new_std = torch.tensor([0.45, 0.45, 0.45, 0.50], dtype=torch.float32)
data["model_state_dict"]["std"] = new_std
print(f"New noise std [dx, dy, dz, grip]: {new_std.tolist()}")

# Reset Adam optimizer momentum buffer for std
if "optimizer_state_dict" in data and "param_groups" in data["optimizer_state_dict"]:
    opt_sd = data["optimizer_state_dict"]
    param_ids = opt_sd["param_groups"][0]["params"]
    # param_ids[0] corresponds to std
    std_id = param_ids[0]
    if "state" in opt_sd and std_id in opt_sd["state"]:
        print(f"Resetting optimizer momentum for std param id: {std_id}")
        del opt_sd["state"][std_id]

torch.save(data, dst_path)
print(f"✓ Saved boosted checkpoint to: {dst_path}")
