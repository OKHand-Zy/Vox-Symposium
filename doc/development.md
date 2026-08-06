# 開發與維護

專案最低支援 Python 3.11。單元測試只使用標準函式庫 `unittest`；Ruff 與 Pyright 放在 `dev` optional dependencies。

## 安裝開發環境

```bash
conda create -n vox-symposium python=3.11
conda activate vox-symposium
pip install -e '.[dev]'
```

若要執行實際 provider smoke test，再依需求安裝：

```bash
pip install -e '.[moshi]'
pip install -e '.[freeze-omni]'
```

## 必跑檢查

```bash
python -m unittest discover -s tests -v
ruff check src tests
ruff format --check src tests
pyright
python -m compileall -q src tests
```

自動格式化與可安全修正的 lint：

```bash
ruff check --fix src tests
ruff format src tests
```

`tests/` 已納入版控。修正 bug 時，先加入能重現問題的測試；重構時則保留既有行為測試，並為新抽出的邊界補測。

## 測試範圍

目前單元測試涵蓋：

- PCM16 channel conversion、frame 切分與格式一致性。
- 環境變數解析、Gemini model generation 差異與 Moshi URL。
- Scenario normalization、prompt/history 與答案抽取。
- Evaluation 分段、續跑、artifacts 清理、summary 與 CLI 驗證。

單元測試不會連外，也不需要 API key。OpenAI、Gemini 與 self-hosted provider 的網路協定仍應在對應環境做 smoke test。

## 變更原則

- Provider-specific 預設值與環境變數只放在 `config.py`，不要在 runner 與 adapter 各自解析。
- Provider 名稱與 alias 只放在 `providers.py`。
- Scenario 角色規則與歷史格式只放在 `scenario.py`。
- Adapter 只處理模型協定；dataset、artifact 路徑與 CLI 行為不應進入 adapter。
- JSON result 與 log 使用 `json_io.write_json()`，避免程序中斷時留下半份檔案。
- 新增 background task 時，關閉流程必須 cancel 並 await task，避免 event loop 結束時殘留工作。
- `.env`、API key、service-account JSON 與實際評測資料不可加入版控。

## 文件同步清單

修改使用者可見行為時，至少確認：

- CLI 參數：README 與 `doc/evaluation-cli.md`。
- 環境變數或預設值：`.env.example`、README、provider 專屬文件。
- 模組責任或擴充方式：`doc/architecture.md`。
- 開發指令或品質規則：本文件與 `pyproject.toml`。
