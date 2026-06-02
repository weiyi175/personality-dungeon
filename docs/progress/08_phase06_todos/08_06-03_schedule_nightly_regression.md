# 08_06-03 — 排程夜間回歸與監控

**Owner:** n1166  
**狀態:** 封卷 ✅  
**建立日期:** 2026-06-01

---

## 目標

在 ci-regression.yml 加入 `schedule` 觸發，每夜自動執行 full matrix regression，
並在 `docs/` 中建立 SLA 與監控規格文件。

## 行為契約

| ID | 規格 |
|---|---|
| C6-08 | schedule 觸發設定為每日 UTC 02:00（台灣時間 10:00）|
| C6-09 | 夜間 job 失敗時必須產生 workflow failure 通知（GitHub 原生 email）|
| C6-10 | SLA 文件記載：最大容許失敗連續天數 = 2，超過則升級告警 |

## 產物

- `ci-regression.yml` 加入 `schedule:` 觸發（或獨立 `ci-nightly.yml`）
- `docs/phase06_sla_monitoring.md`

## 執行步驟

- [x] N6-01 在 `ci-regression.yml` 中加入 `schedule: cron: '0 2 * * *'`
- [x] N6-02 建立 `docs/phase06_sla_monitoring.md`，記錄 SLA 定義、通知方式、升級流程
- [x] N6-03 確認 schedule syntax 有效（GitHub Actions cron 格式）
- [x] N6-04 確認 docs 文件已記載 monitoring 設定

## 驗證標準（Gate）

| Check | 標準 |
|---|---|
| N6-01 | ci-regression.yml 含 schedule cron 設定 | **PASS (2026-06-01)** |
| N6-02 | SLA 文件建立 | **PASS (2026-06-01)** |
| N6-03 | YAML cron 語法正確 | **PASS (2026-06-01)** |
| N6-04 | SLA 文件含 SLA 表格與告警流程 | **PASS (2026-06-01)** |

## 證據索引

- E6-06: `ci-regression.yml`（含 schedule trigger）
- E6-07: `docs/phase06_sla_monitoring.md`

## Estimated Effort

1–3 days（已排入 2026-06-01）
