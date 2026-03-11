# personality-dungeon

（專案骨架建立中）

## Docs

- SDD（Spec-Driven Development）研究規格：見 `SDD.md`
- 研究流程 / SOP / 實驗筆記：見 `研發日誌.md`

## Quick start

```bash
python3 -m simulation.run_simulation --out outputs/timeseries.csv
python3 -m simulation.run_simulation --payoff-mode matrix_ab --a 1.0 --b 1.2 --out outputs/matrix_ab.csv
```

## Research workflow (cycle_level + env_gamma)

### 1) 多 seed 穩健性報表（含 `env_gamma`）

用同一組參數跑多個 seed，輸出 `cycle_level` 分佈，並附加回報「包絡衰減率」：

```bash
python3 -m simulation.seed_stability \
	--payoff-mode matrix_ab --a 0.385 --b 0.105 \
	--players 300 --rounds 6000 --selection-strength 0.02 \
	--seeds 0:9 \
	--series w \
	--burn-in-frac 0.3 --tail 2000 \
	--out outputs/seed_stability.csv
```

`env_gamma` 欄位解讀（經驗指標，不改變 `cycle_level` 判定規則）：

- `env_gamma < 0`：振幅包絡衰減（damped spiral / stable focus）
- `env_gamma ≈ 0`：近中性（更可能維持旋轉，使 Level 3 比例上升）
- `env_gamma > 0`：包絡增長（通常會被非線性飽和成有限振盪）
- 可靠度欄位：`env_gamma_r2`、`env_gamma_n_peaks`（峰值太少或 r2 太低時，不做強結論）

命名注意：payoff 也有 `--gamma`（`count_cycle` 懲罰強度）；報表中的包絡衰減率固定用 `env_gamma` / `mean_env_gamma`。

### 2) 多玩家數 sweep（每個 players 仍是多 seed）

```bash
python3 -m simulation.seed_stability \
	--payoff-mode matrix_ab --a 0.385 --b 0.105 \
	--players-grid 100,300,1000 \
	--rounds 6000 --selection-strength 0.02 \
	--seeds 0:9 \
	--series w --burn-in-frac 0.3 --tail 2000
```

### 3) 可選：先用 Jacobian 粗掃 near-neutral，再回接 simulation 驗證

```bash
python3 -m simulation.seed_stability \
	--payoff-mode matrix_ab --a 0.385 --b 0.105 \
	--hopf-select --hopf-mode lagged --hopf-scan b --hopf-min 0.05 --hopf-max 0.25 --hopf-n 41 \
	--players 300 --rounds 6000 --selection-strength 0.02 \
	--seeds 0:9 \
	--series w --burn-in-frac 0.3 --tail 2000
```

### 3.1) 研究用：確定性 mean-field（產生真正相位旋轉，免抽樣噪音）

當你想確認「模型本身」能不能產生乾淨旋轉（而不是 sampled 噪音），可用：

- `--evolution-mode mean_field`：用期望 payoff $u=Ax$ 直接做 replicator 映射
- `--payoff-lag 0|1`：控制 payoff 使用 $x(t)$ 或 $x(t-1)$（lag=1 對應 lagged Jacobian / ρ 分析）
- `--init-bias`：打破均勻對稱（否則可能卡在均勻固定點）

例：在 `payoff_lag=1` 下，用 lagged Hopf scan 找 ρ≈1 的邊界，再驗證 Level3：

```bash
python3 -m analysis.hopf_scan \
	--mode lagged --scan b --a 0.4 --b 0.4 --b-min 0.01 --b-max 2.0 \
	--selection-strength 0.5 --n 41 --refine

python3 -m simulation.seed_stability \
	--payoff-mode matrix_ab --a 0.4 --b 0.2425406997 \
	--evolution-mode mean_field --payoff-lag 1 --init-bias 0.05 \
	--players 1 --rounds 4000 --selection-strength 0.5 \
	--seeds 0:0 --series p --burn-in-frac 0.25 --tail 2500
```

### 4) 從單條 `timeseries.csv` 獨立重算 `env_gamma`

```bash
python3 -m analysis.decay_rate --csv outputs/timeseries.csv --series w --metric linf
```
