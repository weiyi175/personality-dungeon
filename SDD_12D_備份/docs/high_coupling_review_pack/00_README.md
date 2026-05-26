# 高耦合診斷審閱包

這個資料夾把本次高耦合失敗診斷相關內容集中到同一處，方便依序審閱，不必在多個檔案之間來回找。

## 建議閱讀順序

1. [提問框架](PROMPT_TEMPLATE_high_coupling_analysis.md)
2. [正式規格](high_coupling_failure_diagnosis_spec.md)
3. [研發日誌對應章節](../../研發日誌.md#phase-99)

## 內容對照

- [PROMPT_TEMPLATE_high_coupling_analysis.md](PROMPT_TEMPLATE_high_coupling_analysis.md)：定義分析問題、五大研究重點、輸出格式要求。
- [high_coupling_failure_diagnosis_spec.md](high_coupling_failure_diagnosis_spec.md)：完整規格版，已對齊提問框架，適合做正式審閱。
- [研發日誌.md § Phase 99](../../研發日誌.md#phase-99)：把相同分析放回研發時間線，適合追溯研究脈絡。

## 審閱重點

- 先看提問框架，確認這份分析要回答什麼。
- 再看正式規格，確認比較表、三層均化、Phase-based coupling 是否完整。
- 最後看研發日誌章節，確認內容是否和既有研究紀錄一致。

## 你可以直接關注的三個核心區塊

- 虛擬錨點 vs 真實交互 vs 新增橋接
- 三層均化的失效機制
- Phase 0–3 的逐步耦合驗證方案

## 備註

這裡不複製原始內容，只提供集中入口，避免你在多份檔案之間分散閱讀。
