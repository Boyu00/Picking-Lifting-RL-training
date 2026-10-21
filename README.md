# Picking-Lifting RL Training

以 [Isaac Lab](https://github.com/isaac-sim/IsaacLab)（Direct workflow）建立的強化學習環境，訓練 JetRover 機械手臂抓取並舉起桌上的物體，再移動到隨機目標點。使用 RSL-RL 的 PPO 訓練。

## 專案架構

```
.
├── scripts/
│   ├── rsl_rl/
│   │   ├── train_0510.py        # 0510 DeltaIK 環境訓練
│   │   ├── play_0510.py         # 0510 DeltaIK 環境推論 / 錄影
│   │   ├── train.py, play.py    # Isaac Lab 模板通用版
│   │   └── cli_args.py          # RSL-RL 共用參數
│   ├── rl_games/, skrl/         # 其他 RL 框架（模板保留）
│   └── list_envs.py, zero_agent.py, random_agent.py
└── source/pickup_place_direct_0203/       # Isaac Lab extension（pip install -e）
    └── pickup_place_direct_0203/tasks/direct/pickup_place_direct_0203/
        ├── __init__.py                         # gym 環境註冊
        ├── jetrover.py                         # JetRover ArticulationCfg（關節、致動器、初始姿態）
        ├── pickup_place_direct_0510_env_cfg.py # 0510 環境設定：場景、物體、獎勵、課程、隨機化
        ├── pickup_place_direct_0510_env.py     # 0510 環境本體：動作、觀測、獎勵、重置
        ├── pickup_place_direct_0203_env*.py    # 模板起始環境（佔位用）
        ├── mdp/
        │   ├── rewards.py       # reaching / lifting / gripper / goal tracking 等獎勵與成功判定
        │   ├── observations.py  # 物體位置、bbox 角點
        │   ├── terminations.py  # 物體離開工作區
        │   └── curriculum.py    # 依成功率逐步提高獎勵權重
        └── agents/
            └── rsl_rl_ppo_0510_cfg.py          # PPO 超參數
```

## 環境：`Template-Pickup-Place-Direct-0510-v0`

| 項目 | 內容 |
|---|---|
| 機器人 | JetRover，5 軸手臂 + 二值夾爪 |
| 動作（7 維） | 末端執行器的位置與姿態增量（dx, dy, dz, drx, dry, drz），經 Differential IK（DLS）轉成關節目標；第 7 維控制夾爪開/關 |
| 觀測（49 維） | 關節位置與速度、物體位置、物體 bbox 8 個角點、目標點、上一步動作 |
| 物體 | 從 ObjectFolder 隨機挑選 15 種（杯子、餐具、瓶罐、盤子等） |
| 隨機化 | 物體種類、質量、yaw、依材質設定的靜/動摩擦力、目標位置 |
| 獎勵 | reaching → close gripper → lifting → goal tracking（粗 + 細），由 curriculum 依成功率解鎖並提高權重 |
| 終止條件 | 物體離開工作區，或達到 episode 時長（10 s） |

## 實驗結果

最終模型為 2026-05-28 訓練的 `model_21050.pt`（約 21k iterations），於 2026-06-05 用兩種觀測條件評估：

1. **標準模擬**：每一步都給策略真實的物體位置與 bbox
2. **Sim-to-Real 模擬**：模擬真實相機只能在開始時定位一次物體的情況（見[下方](#sim-to-real-模擬凍結物體位置)）

兩者皆為單一環境、seed 42、關閉觀測雜訊，每種物體 10 次。

- **Train**：訓練時使用的 15 種物體，共 150 次
- **Test**：訓練時沒看過的 12 種物體（ObjectFolder 21、29，以及 10 種垃圾類物體，例如蘋果、香蕉、鋁罐、寶特瓶、蛋盒），共 120 次

成功判定（皆需連續維持）：

| 階段 | 條件 |
|---|---|
| Reach | 夾爪與物體 bbox 距離 < 5 cm，連續 5 步 |
| Lift | 物體離地 ≥ 6 cm、grasp quality ≥ 0.2、與夾爪距離 ≤ 12 cm，連續 10 步 |
| Goal | 物體與目標點距離 < 8 cm、高度 ≥ 10 cm，連續 10 步 |

### 標準模擬：整體成功率

| 組別 | 次數 | Reach | Lift | Goal |
|---|---|---|---|---|
| Train | 150 | 100.0% | 92.7% | 58.7% |
| Test | 120 | 100.0% | 45.8% | 18.3% |
| 合計 | 270 | 100.0% | 71.9% | 40.7% |

### 標準模擬：依物體形狀

| 形狀 | 次數 | Reach | Lift | Goal |
|---|---|---|---|---|
| 杯碗類 | 60 | 100.0% | 93.3% | 58.3% |
| 扁長類（餐具、筆） | 50 | 100.0% | 80.0% | 60.0% |
| 圓柱類（瓶罐） | 70 | 100.0% | 64.3% | 35.7% |
| 不規則 | 90 | 100.0% | 58.9% | 22.2% |

### 標準模擬：失敗原因

| 類型 | Train | Test | 合計 |
|---|---|---|---|
| 成功 | 88 (58.7%) | 22 (18.3%) | 110 (40.7%) |
| 舉起後沒送到目標（transport_fail） | 51 (34.0%) | 33 (27.5%) | 84 (31.1%) |
| 沒抓起來（grasp_fail） | 9 (6.0%) | 55 (45.8%) | 64 (23.7%) |
| 舉起後掉落（drop_after_lift） | 2 (1.3%) | 10 (8.3%) | 12 (4.4%) |

**小結**

- Reach 在所有物體上都已解決。
- 訓練物體能穩定抓起（Lift 92.7%），主要瓶頸是舉起後送到目標點。
- 未見過的物體泛化有限，主要卡在抓取（grasp_fail 45.8%），寶特瓶與 ObjectFolder 21、29 的 Lift 只有 0–40%。

### Sim-to-Real 模擬（凍結物體位置）

真實機器人上，物體位置只能在開始時由相機定位一次，夾取過程中會被夾爪遮擋。為模擬這點，評估時改寫策略的觀測：

- 物體位置與 bbox 固定為第一幀的值，不再更新
- 直到判定已夾住物體（夾爪閉合、夾爪低於物體初始高度 + 10 cm、與物體距離 < 10 cm，連續 5 步），之後物體位置改為「初始位置 + 夾爪位移」推算
- 成功判定仍使用模擬器的真實狀態

| 組別 | 次數 | Reach | Lift | Goal |
|---|---|---|---|---|
| Train | 150 | 92.7% | 2.7% | 0.7% |
| Test | 120 | 97.5% | 1.7% | 0.0% |
| 合計 | 270 | 94.8% | 2.2% | 0.4% |

失敗原因（合計 270 次）：沒抓起來 238 (88.1%)、只接近未抓取 14 (5.2%)、舉起後掉落 12 (4.4%)、舉起後沒送到目標 5 (1.9%)、成功 1 (0.4%)。

**小結**：策略仍能接近物體，但幾乎無法完成抓取。這表示策略在抓取階段依賴持續更新的物體位置做閉迴路修正，直接部署到只能定位一次的真實系統上會失敗。

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
