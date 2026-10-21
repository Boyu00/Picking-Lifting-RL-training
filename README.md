# Picking-Lifting RL Training

以 [Isaac Lab](https://github.com/isaac-sim/IsaacLab)（Direct workflow）建立的強化學習環境，訓練 JetRover 機械手臂抓取並舉起桌上的物體，再移動到隨機目標點。使用 RSL-RL 的 PPO 訓練。

> 2026-06-30 之前的版本（7D DeltaIK 動作與當時的評估結果）保存在 [`before-0630`](../../tree/before-0630) branch。

## 專案架構

```
.
├── scripts/rsl_rl/
│   ├── train_0510.py, play_0510.py  # 0510 環境訓練 / 推論
│   ├── eval_config_0510.py          # 評估設定：物體分組、成功判定
│   ├── eval_analysis_0510.py        # 評估結果統計（依物體、形狀、質量、位置、失敗原因）
│   ├── train.py, play.py            # 通用版，含 early stopping 與訓練健康監控
│   └── cli_args.py                  # RSL-RL 共用參數
└── source/pickup_place_direct_0203/       # Isaac Lab extension（pip install -e）
    └── pickup_place_direct_0203/tasks/direct/pickup_place_direct_0203/
        ├── __init__.py                          # gym 環境註冊
        ├── jetrover.py                          # JetRover ArticulationCfg
        ├── pickup_place_direct_0510_env_cfg.py  # 0510 環境設定：場景、物體、獎勵、課程、隨機化
        ├── pickup_place_direct_0510_env.py      # 0510 環境本體：IK 動作、觀測、獎勵、重置
        ├── pickup_place_latched_obs_env*.py     # 鎖存觀測版本（actor 只看到初始物體位置）
        ├── franka_cube_lift_direct_env*.py      # Franka 對照環境
        ├── mdp/                                 # rewards / observations / terminations / curriculum
        └── agents/rsl_rl_ppo_*_cfg.py           # 各環境的 PPO 超參數
```

## 環境：`Template-Pickup-Place-Direct-0510-v0`

| 項目 | 內容 |
|---|---|
| 機器人 | JetRover，5 軸手臂 + 二值夾爪 |
| 動作（4 維） | 末端執行器位置增量（dx, dy, dz），經 Differential IK（DLS）轉成關節目標；第 4 維控制夾爪開/關 |
| 觀測（46 維） | 關節位置與速度、物體位置、物體 bbox 8 個角點、物體到目標的相對向量、上一步動作 |
| 物體 | 從 ObjectFolder 隨機挑選 15 種（杯子、餐具、瓶罐、盤子等） |
| 隨機化 | 物體種類、質量、yaw、依材質設定的靜/動摩擦力、目標位置 |
| 獎勵 | reaching → close gripper → lifting，舉起後再以 coarse / mid / fine 三層 goal tracking 引導到目標點 |
| 終止條件 | 物體離開工作區，或達到 episode 時長（10 s） |

## 實驗結果

評估模型為 2026-09-03 訓練的 `model_3750.pt`，於 2026-09-04 評估：單一環境、seed 42、關閉觀測雜訊，每種物體 20 次。

- **Train**：訓練時使用過的 12 種物體，共 240 次
- **Test**：訓練時沒看過的 12 種物體（ObjectFolder 21、29，以及 10 種垃圾類物體，例如蘋果、香蕉、鋁罐、寶特瓶、蛋盒），共 240 次

成功判定（皆需連續維持）：

| 階段 | 條件 |
|---|---|
| Reach | 夾爪與物體 bbox 距離 < 5 cm，連續 5 步 |
| Lift | 物體離地 ≥ 6 cm、grasp quality ≥ 0.2、與夾爪距離 ≤ 12 cm，連續 10 步 |
| Goal | 物體與目標點距離 < 8 cm、高度 ≥ 10 cm，連續 10 步 |
| Precision | 同 Goal，但距離 < 3 cm |

### 整體成功率

| 組別 | 次數 | Reach | Lift | Goal | Precision |
|---|---|---|---|---|---|
| Train | 240 | 100.0% | 87.9% | 79.2% | 60.0% |
| Test | 240 | 99.6% | 70.4% | 37.1% | 19.6% |
| 合計 | 480 | 99.8% | 79.2% | 58.1% | 39.8% |

### 依物體形狀

| 形狀 | 次數 | Reach | Lift | Goal | Precision |
|---|---|---|---|---|---|
| 杯碗類 | 80 | 100.0% | 98.8% | 97.5% | 97.5% |
| 扁長類（餐具、筆） | 100 | 100.0% | 76.0% | 74.0% | 65.0% |
| 圓柱類（瓶罐） | 120 | 99.2% | 75.8% | 42.5% | 12.5% |
| 不規則 | 180 | 100.0% | 74.4% | 42.2% | 18.3% |

### 失敗原因

| 類型 | Train | Test | 合計 |
|---|---|---|---|
| 成功 | 190 (79.2%) | 89 (37.1%) | 279 (58.1%) |
| 舉起後沒送到目標（transport_fail） | 21 (8.8%) | 80 (33.3%) | 101 (21.0%) |
| 沒抓起來（grasp_fail） | 23 (9.6%) | 63 (26.2%) | 86 (17.9%) |
| 舉起後掉落（drop_after_lift） | 6 (2.5%) | 7 (2.9%) | 13 (2.7%) |
| 只接近未抓取（approach_only） | 0 (0.0%) | 1 (0.4%) | 1 (0.2%) |

**小結**

- 訓練物體上，Lift 87.9%、Goal 79.2%，大多數物體都能完成抓取並送到目標點。
- 未見過的物體（與 6/30 版評估相同的 12 種）：Lift 從 45.8% 提升到 70.4%、Goal 從 18.3% 提升到 37.1%；搬運仍是主要瓶頸（transport_fail 33.3%）。
- 杯碗類幾乎全部成功；寶特瓶與 ObjectFolder 21 的 Lift 只有 20–70%，是最難的物體。

## 安裝

1. 依 [官方文件](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安裝 Isaac Lab。
2. 安裝本專案的 extension：

   ```bash
   python -m pip install -e source/pickup_place_direct_0203
   ```

3. 準備資產。以下路徑寫死在設定檔中，需依自己的環境修改：
   - JetRover USD：`jetrover.py` 中的 `usd_path`
   - ObjectFolder 物體 USD：`pickup_place_direct_0510_env_cfg.py` 中的 `usd_path`（預設 `/root/ObjectFolder/{id}/{id}.usd`）

## 使用方式

```bash
# 列出已註冊的環境
python scripts/list_envs.py

# 訓練
python scripts/rsl_rl/train_0510.py --task Template-Pickup-Place-Direct-0510-v0 --num_envs 256 --headless

# 從 checkpoint 繼續訓練
python scripts/rsl_rl/train_0510.py --task Template-Pickup-Place-Direct-0510-v0 --headless --resume --load_run <run_dir>

# 推論 / 錄影
python scripts/rsl_rl/play_0510.py --task Template-Pickup-Place-Direct-0510-v0 --num_envs 1 --load_run <run_dir> --video
```

若 Isaac Lab 未裝在 venv / conda 中，請將 `python` 換成 `<IsaacLab>/isaaclab.sh -p`。訓練紀錄輸出至 `logs/rsl_rl/pickup_place_direct_0510/`。
