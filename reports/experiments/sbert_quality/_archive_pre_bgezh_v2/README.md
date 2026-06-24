# 封存：pre-bge-zh-v2 的舊模型 eval 產物（2026-06-25）

這裡的檔案描述的是 **舊 SBERT 模型 `sbert-mlp-v7`（`paraphrase-multilingual-MiniLM-L12-v2` + `outputs/mlp_v7_mlp.joblib`，qwen 標籤蒸餾）**，
在 2026-06-24 換成 `sbert-mlp-opus-bgezh-v2`（`BAAI/bge-base-zh-v1.5` + `outputs/mlp_opus_bgezh_v2.joblib`）後即 **過時、誤導**，故封存。

| 檔案 | 是什麼 | 為何過時 |
|---|---|---|
| `eval_vs_claude_ref.json` | 舊模型對 35 筆 Opus-ref 的 per-trait r（mean ~0.45、`risk_aversion`=0.19） | 由 `sbert_quality_eval_claude_ref.py` 跑**當時的 live 模型**（仍是 MiniLM+v7）產生。現行 bge-zh held-out mean **0.74**、`risk_aversion`=**0.73**。 |
| `trait_reliability.json` | 從上者導出的「tier 信任表」+ 處方（**HIDE risk_aversion / tier3 drop / 別把 risk_aversion 併入合成軸**） | 處方是針對**舊模型**的破洞。現行模型**沒有壞 trait**（全 ≥0.57），此處方**作廢**，雷達/顯示決策**不要**照它。 |

**現行真相來源**：`研發日誌.md` 的「SBERT 人格推斷大改版」段（生產模型表 + per-trait r）。

**⚠ 重要陷阱**：`scripts/experiments/sbert_quality_eval_claude_ref.py` 評的是 **live 模型**。由於**出貨模型（v1/v2）是 fit 全量 176 筆（含這 35 筆 frozen held-out）**，
直接重跑它去評「已出貨」模型會得到 **r≈1.0 的 in-sample 假象**（2026-06-25 實證）。
→ **誠實 held-out 只能看「飛輪重訓報告」**（train on 141、eval on frozen 35），見 `研發日誌.md` 的「模型替換與封存規範」。

封存原則：只封「描述舊模型現況」的 eval 產物；**不封** live 資料（`opus_train_labels.json`、`claude_reference_labels.json`、`held_out_split.json`）與作為裁決證據的實驗 log（`step1_results_*`、`finetune_results`、`synthetic_labels`）。
