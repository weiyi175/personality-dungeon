# 07_05-01 — 建立可重現實驗腳本（seeded）

Owner: 

產物:
- scripts/experiments/ 下的可執行腳本，含固定 seed 與 output JSON

驗證步驟:
- 在 CI 或本機執行 N=10 runs，確認相同 seed 下結果可重現

證據:
- eports/experiments/ 下的 run JSON 與 nalysis/experiments_summary.csv

Estimated effort: 3–7 days
