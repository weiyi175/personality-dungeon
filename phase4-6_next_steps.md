# Phase 04–06 — 下一步工作清單

此文件為 Phase 04（Demo Harden）、Phase 05（Research Validation）與 Phase 06（Regression & CI Signoff）的可執行工作項目清單，包含驗證步驟、驗收證據與建議 owner（空白供填寫）。

## Phase 04 — Demo Harden & UX Acceptance

- 04-01: 完成 Demo Checklist 修補項目（Owner: ）
  - 驗證步驟：執行 `scripts/phase2_smoke_test.py` 並確認 7/7 pass；手動播放 demo 流程並錄影 1 分鐘
  - 證據：`logs/smoke_phase2_summary.json`、demo video（`docs/progress/04_phase_review/demo_video.mp4`）

- 04-02: 修正 UX 問題與教學提示（Owner: ）
  - 驗證步驟：UI reviewer 逐項檢查 `src/ui/PlayableLoopScene.tscn` 與 `SessionStatusPanel.gd`，回填 comment
  - 證據：PR 與 reviewer comments

- 04-03: Demo 截圖與錄影（Owner: ）
  - 驗證步驟：使用 headless Godot 命令錄製輸出（或捕捉截圖），上傳到 `docs/progress/04_phase_review/media/`
  - 命令示例（Windows PowerShell）：
    ```powershell
    & "C:\Program Files (x86)\Godot_v4.6.2-stable_win64\Godot_v4.6.2-stable_win64.exe" --headless --path "C:\Users\n1166\personality-dungeon" "res://src/ui/PlayableLoopScene.tscn"
    ```

- 04-04: Signoff Table 填寫並收齊簽署（PM/UX/QA）
  - 驗證步驟：在 `docs/progress/04_phase_review/signoff_checklist.md` 回填日期與姓名

Estimated effort: 2–4 days

## Phase 05 — Research Validation

- 05-01: 建立可重現實驗腳本（seeded）
  - 產物：`scripts/experiments/` 下的可執行腳本，含固定 seed 與 output JSON
  - 驗證步驟：在 CI 或本機執行 N=10 runs，確認相同 seed 下結果可重現

- 05-02: 執行實驗矩陣並收集結果
  - 驗證步驟：運行預定參數組並將輸出寫入 `reports/experiments/`，收集 summary CSV

- 05-03: 產出研究驗證報告
  - 證據：`reports/experiments/validation_report.pdf` 或 `analysis/experiments_summary.csv`

Estimated effort: 3–7 days (依矩陣大小)

## Phase 06 — Regression Suite & CI Signoff Automation

- 06-01: 設計 regression + smoke CI jobs
  - 產物：`.github/workflows/ci-regression.yml` 與 `ci-smoke.yml`
  - 驗證步驟：將 `scripts/run_phase2_batch.py --runs 1` 加入 smoke job，確認 PR 時可執行

- 06-02: 自動產生 signoff artifact
  - 產物：CI job 成功時輸出 `artifacts/signoff_summary.json` 並註記 PR
  - 驗證步驟：建立 sample PR 並觀察 artifact 產出

- 06-03: 排程夜間回歸與監控
  - 驗證步驟：設定 GitHub Actions schedule（或類似 runner），並在 `docs/` 中記錄 SLA 與通知設定

Estimated effort: 4–10 days

## 優先順序建議

1. Phase 04（Demo）先完成，以利展示與 stakeholder 驗收。
2. Phase 05（Research）並行小規模實驗（N=10），確保方法可重現再擴大。
3. Phase 06（CI）在 Phase 04/05 輸出穩定後導入自動化。

---

更新紀錄：
- 2026-05-31 — 建立初版（Copilot）
