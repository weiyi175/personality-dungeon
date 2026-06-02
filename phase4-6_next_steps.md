# Phase 04–06 — 下一步工作清單

此文件為 Phase 04（Demo Harden）、Phase 05（Research Validation）與 Phase 06（Regression & CI Signoff）的可執行工作項目清單，包含驗證步驟、驗收證據與建議 owner（空白供填寫）。

> **封卷狀態：Phase 04 / 05 / 06 均已完成並簽核 ✅（2026-06-01）**  
> 詳見 [signoff_checklist.md](docs/progress/04_phase_review/signoff_checklist.md)

## Phase 04 — Demo Harden & UX Acceptance ✅ Completed (2026-06-01)

- 04-01: 完成 Demo Checklist 修補項目（Owner: n1166）✅
  - 驗證步驟：執行 `scripts/phase2_smoke_test.py` 並確認 7/7 pass；手動播放 demo 流程並錄影 1 分鐘
  - 證據：`logs/smoke_phase2_summary.json`、demo video（`docs/progress/04_phase_review/demo_video.mp4`）

- 04-02: 修正 UX 問題與教學提示（Owner: n1166）✅
  - 驗證步驟：UI reviewer 逐項檢查 `src/ui/PlayableLoopScene.tscn` 與 `SessionStatusPanel.gd`，回填 comment
  - 證據：PR 與 reviewer comments

- 04-03: Demo 截圖與錄影（Owner: n1166）✅
  - 驗證步驟：使用 headless Godot 命令錄製輸出（或捕捉截圖），上傳到 `docs/progress/04_phase_review/media/`
  - 命令示例（Windows PowerShell）：
    ```powershell
    & "C:\Program Files (x86)\Godot_v4.6.2-stable_win64\Godot_v4.6.2-stable_win64.exe" --headless --path "C:\Users\n1166\personality-dungeon" "res://src/ui/PlayableLoopScene.tscn"
    ```

- 04-04: Signoff Table 填寫並收齊簽署（PM/UX/QA）✅
  - 驗證步驟：在 `docs/progress/04_phase_review/signoff_checklist.md` 回填日期與姓名

## Phase 05 — Research Validation ✅ Completed (2026-06-01)

- 05-01: 建立可重現實驗腳本（seeded）✅
  - 產物：`scripts/experiments/run_experiment.py`（seed=42, N=10，REPRO_PASS）
  - 證據：`reports/experiments/run_seed42_r10.json`

- 05-02: 執行實驗矩陣並收集結果 ✅
  - 產物：smoke 3 rows + full 12 rows，COUNT_MATCH=True
  - 證據：`analysis/experiments_summary.csv`、`analysis/experiments_summary_full.csv`

- 05-03: 產出研究驗證報告 ✅
  - 證據：`reports/experiments/validation_report.md`（Signoff v1.0）

## Phase 06 — Regression Suite & CI Signoff Automation ✅ Completed (2026-06-01)

- 06-01: 設計 regression + smoke CI jobs ✅
  - 產物：`.github/workflows/ci-smoke.yml`（PR 觸發）、`.github/workflows/ci-regression.yml`（nightly + 手動）

- 06-02: 自動產生 signoff artifact ✅
  - 產物：`scripts/experiments/generate_signoff_summary.py` → `artifacts/signoff_summary.json`（gate_pass=true）

- 06-03: 排程夜間回歸與監控 ✅
  - 產物：`ci-regression.yml` schedule cron `0 2 * * *`、`docs/phase06_sla_monitoring.md`

---

更新紀錄：
- 2026-05-31 — 建立初版（Copilot）
- 2026-06-01 — Phase 04 / 05 / 06 全數完成並簽核（n1166）
