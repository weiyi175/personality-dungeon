# 08_06-01 — 設計 regression + smoke CI jobs

Owner: 

產物:
- .github/workflows/ci-regression.yml 與 ci-smoke.yml

驗證步驟:
- 將 scripts/run_phase2_batch.py --runs 1 加入 smoke job，確認 PR 時可執行

證據:
- 成功執行的 workflow 與 artifacts

Estimated effort: 3–7 days
