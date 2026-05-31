# Ollama 連線測試打包

本資料夾提供一套可重現流程，用來在 WSL venv 環境下檢查 Ollama 服務是否可連線。

## 打包位置

- docs/ollama_connection_package

## 內容

- run_ollama_check.sh：在 WSL 內執行（含 venv 啟用）
- run_ollama_check.ps1：在 Windows PowerShell 執行，會呼叫 WSL
- scripts/ollama_smoke_test.py：實際健康檢查程式（位於專案 scripts）

## 先決條件

1. 已安裝 Ollama（Windows）。
2. Ollama 有監聽外部介面（Windows 只需做一次）：

```powershell
[System.Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0:11434", "User")
```

3. 重新啟動 Ollama（系統匣右鍵 Quit 後再開）。

## 使用方式

### 方式 A：你指定的 WSL 逐步流程

```bash
wsl -d Ubuntu-22.04
cd /home/user/personality-dungeon
source venv/bin/activate
python scripts/ollama_smoke_test.py --wait-sec 15 --timeout-sec 3
```

### 方式 B：WSL 一鍵執行

```bash
bash docs/ollama_connection_package/run_ollama_check.sh
```

### 方式 C：PowerShell 一鍵執行

```powershell
.\docs\ollama_connection_package\run_ollama_check.ps1
```

## 成功判定

出現以下訊息代表成功：

- [ok] Ollama server is running at ...
- [ok] models: ...

## 已驗證結果（2026-05-29）

- 連線成功主機：http://172.31.128.1:11434
- 偵測模型：qwen3.5:9b
