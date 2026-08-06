# Evaluation CLI

`src/vox_symposium/evaluation.py` 用來跑自動化 scenario evaluation。基本格式：

```bash
python3 -m vox_symposium SCENARIO RESULT [options]
```

也可以直接指定 evaluation module：

```bash
python3 -m vox_symposium.evaluation SCENARIO RESULT [options]
```

也可以使用 `pyproject.toml` 註冊的 console script：

```bash
vox-symposium-evaluate SCENARIO RESULT [options]
```

## 位置參數

| 參數 | 說明 |
| --- | --- |
| `scenario` | Normalized scenario JSON 路徑，或包含多筆 records 的 source dataset。 |
| `result` | evaluation result JSON 輸出路徑。單筆時輸出單一 JSON object；多筆 batch 時輸出 JSON array。 |

## Scenario 選擇與續跑

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `--id SCENARIO_ID` | 無 | 從 dataset 中選指定 `id` 的單一 scenario。 |
| `--index SCENARIO_INDEX` | 無 | 從 dataset 中選指定 zero-based index 的單一 scenario。例：第 7 筆是 `--index 6`。 |
| `--start-index START_INDEX` | `0` | batch 從指定 zero-based index 開始跑。例：log 停在 `Running scenario 7/185`，重跑可用 `--start-index 6`。 |
| `--limit LIMIT` | 無 | batch 只跑選取範圍的前 N 筆。搭配 `--start-index 6 --limit 10` 會跑原 dataset 第 7 到第 16 筆。 |

使用限制：

- `--id` 和 `--index` 都是單筆選擇，不要同時使用。
- `--limit` 不能和 `--id` / `--index` 混用。
- `--start-index` 不能和 `--id` / `--index` 混用。
- `--start-index` 與 `--limit` 可一起使用，用來跑 dataset 的一段區間。
- `--start-index` 是 zero-based；最後一筆 185 筆 dataset 的 index 是 `184`，不是 `185`。

## 對話與音訊

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `--audio-dir AUDIO_DIR` | 無 | 原始 speech wav 檔所在資料夾。用 source dataset 轉 normalized scenario 時會用到。 |
| `--dialogue-turns DIALOGUE_TURNS` | `5` | evaluation question 前要收集的 robot 回合數。會覆蓋 scenario 裡的 `run.dialogue_turns` / `evaluation.ask_after_turns`。 |
| `--question-audio QUESTION_AUDIO` | scenario 的 `evaluation.question_audio` | 指定 evaluation question 音檔，會覆蓋 scenario JSON 內設定。支援 `.wav`；`.mp3` 會先轉成 WAV。 |
| `--answer-audio ANSWER_AUDIO` | artifacts 內的 `robot-answer.wav` | 指定 robot evaluation answer 的輸出 WAV 路徑。只能在單筆 evaluation 使用。 |
| `--frame-ms FRAME_MS` | `20` | 串流音訊時每個 audio frame 的毫秒數。 |
| `--audio-speed AUDIO_SPEED` | `EVALUATION_AUDIO_SPEED` 或 `1.0` | 音訊注入速度。`1.0` 是 real-time；更大的值會更快送音訊，可能降低長批次 timeout 風險，但也可能影響 VAD / turn detection；`0` 表示不 sleep。 |
| `--no-tts` | `false` | 要求必須有 question audio；如果 scenario 沒有 `evaluation.question_audio` 且沒有傳 `--question-audio`，就直接失敗，不嘗試用 macOS `say` 產生題目音訊。 |

當角色使用 `provider=minicpm` 或 `provider=freeze_omni` 時，scenario prompt
會在 `Dialogue behavior` 自動追加短回覆規則：每次最多 2 句、最多問 1 個問題，且不要反覆總結。
這是為了降低長回覆造成的 session 時間與 turn-taking 風險；不會套用到 OpenAI、Gemini、Moshi
或 PersonaPlex。

## Timeout 與輸出

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `--idle-timeout IDLE_TIMEOUT` | `1.5` | 收到模型音訊後，連續多少秒沒有新音訊就視為該 utterance 結束。 |
| `--max-utterance-seconds MAX_UTTERANCE_SECONDS` | `30.0` | 等待單次 utterance 的最長秒數。若模型常超時，可調大，例如 `--max-utterance-seconds 300`。 |
| `--text-idle-timeout TEXT_IDLE_TIMEOUT` | `0.7` | 收到模型音訊後，文字 delta 連續多少秒沒有新內容就視為該文字輸出結束。 |
| `--text-max-wait TEXT_MAX_WAIT` | `5.0` | 收到模型音訊後，最多補等多少秒以收集延遲到達的文字 delta。MiniCPM text/audio delta 不同步時可調大。 |
| `--case-retries CASE_RETRIES` | `3` | 每筆 scenario 最多嘗試次數。單筆 case timeout 或其他 exception 時，會刪除該 case artifact 子資料夾後重試；達到次數仍失敗才讓整次 evaluation 失敗退出。 |
| `--case-delay CASE_DELAY` | `0.0` | 成功完成一筆 scenario 並寫入 result / summary 後，下一筆 scenario 開始前等待秒數。最後一筆不會等待。 |
| `--case-retry-delay CASE_RETRY_DELAY` | `30.0` | 單筆 case 失敗後，下一次重試前等待秒數。 |
| `--overnight` | `false` | Overnight batch 模式。單筆 case retry 到上限仍失敗時，記錄該 case 為 failed，寫入 result / summary，然後繼續下一筆，不讓整次 evaluation 中斷。 |
| `--artifact-dir ARTIFACT_DIR` | `RESULT` 同資料夾的 `<run-id>-artifacts` | artifacts 輸出資料夾，包含 console log、summary、run env snapshot、每筆 dialogue log 和 WAV。 |
| `--run-id RUN_ID` | `result` 檔名 stem | 本次 evaluation 的穩定 run id。batch 模式會自動加上 case 編號和 scenario row id，例如 `two_Gemini-Gemini-0007-00000006`。 |

每次執行會建立或更新：

| 檔案 | 說明 |
| --- | --- |
| `RESULT` | 主要 evaluation result。batch 模式每完成一筆就會更新。 |
| `<artifact-dir>/summary.json` | batch 摘要。每完成一筆 case 就會立刻更新；使用 `--start-index` 從中段繼續時，`index` 仍保留原 dataset 的 zero-based index。 |
| `<artifact-dir>/console-log.txt` | CLI stdout / stderr log。重複使用同一個 artifact dir 時會用 `###################################` 分隔並繼續追加，不會清除舊 log。 |
| `<artifact-dir>/run-env.txt` | 本次 provider、model、backend、voice、run id、參數等非敏感環境快照。API key 不會寫入。 |
| `<artifact-dir>/<row-id>/dialogue-log.json` | 單筆 scenario 的 opening、dialogue turns、evaluation question / answer event log。 |
| `<artifact-dir>/<row-id>/robot-answer.wav` | robot 對 evaluation question 的回答音訊。 |

如果單筆 case 因 timeout、connection error 或其他 exception 失敗，runner 會刪除該 case 的 `<artifact-dir>/<row-id>/` 子資料夾，等待 `--case-retry-delay` 秒後重試。預設最多嘗試 3 次；第 3 次仍失敗時，整次 evaluation 會失敗退出。

`--case-delay` 與 `--case-retry-delay` 是不同用途：前者是成功 case 到下一個 case 中間等待；後者是同一 case 失敗後 retry 前等待。

CLI 會在建立 provider 連線前驗證數值：frame duration、utterance timeout 必須大於 0；index、turn count、audio speed、文字 timeout 與 delay 不可小於 0；`--case-retries` 與 `--limit` 至少為 1。錯誤會直接指出對應參數。

使用 `--overnight` 時，第 3 次仍失敗不會退出。runner 會先完成該 attempt 的斷線、刪除 case artifact 子資料夾等清理流程，再把該 case 以 `status: "failed"` 和 error message 寫入 result / summary，然後繼續下一個 case。

## 常用範例

跑單筆 smoke test：

```bash
python3 -m vox_symposium.evaluation \
  data/scenarios/00000000.json \
  data/results/00000000-smoke.json \
  --run-id 00000000-smoke \
  --dialogue-turns 2
```

跑 dataset 第一筆：

```bash
python3 -m vox_symposium.evaluation \
  data/scenarios/two_test_dataset.json \
  data/results/two_test_Gemini-smoke.json \
  --run-id two_test_Gemini-smoke \
  --dialogue-turns 10 \
  --limit 1
```

跑 dataset 指定單筆：

```bash
python3 -m vox_symposium.evaluation \
  data/scenarios/two_test_dataset.json \
  data/results/two_Gemini-Gemini-0007-00000006.json \
  --run-id two_Gemini-Gemini-0007-00000006 \
  --dialogue-turns 10 \
  --index 6
```

從第 7 筆繼續跑到結束：

```bash
python3 -m vox_symposium.evaluation \
  data/scenarios/two_test_dataset.json \
  data/results/two_Gemini-Gemini.json \
  --run-id two_Gemini-Gemini \
  --dialogue-turns 10 \
  --start-index 6
```

從第 7 筆開始只跑 10 筆：

```bash
python3 -m vox_symposium.evaluation \
  data/scenarios/two_test_dataset.json \
  data/results/two_Gemini-Gemini-0007-to-0016.json \
  --run-id two_Gemini-Gemini-0007-to-0016 \
  --dialogue-turns 10 \
  --start-index 6 \
  --limit 10
```

模型回覆較慢時加長等待：

```bash
python3 -m vox_symposium.evaluation \
  data/scenarios/two_test_dataset.json \
  data/results/two_Gemini-Gemini.json \
  --run-id two_Gemini-Gemini \
  --dialogue-turns 10 \
  --start-index 6 \
  --max-utterance-seconds 300
```

加快音訊注入速度：

```bash
python3 -m vox_symposium.evaluation \
  data/scenarios/two_test_dataset.json \
  data/results/two_Gemini-Gemini.json \
  --run-id two_Gemini-Gemini \
  --dialogue-turns 10 \
  --start-index 6 \
  --audio-speed 8
```

調整 case retry 次數與等待時間：

```bash
python3 -m vox_symposium.evaluation \
  data/scenarios/two_test_dataset.json \
  data/results/two_Gemini-Gemini.json \
  --run-id two_Gemini-Gemini \
  --dialogue-turns 10 \
  --start-index 6 \
  --case-retries 3 \
  --case-retry-delay 30
```

每個成功 case 之間等待 10 秒：

```bash
python3 -m vox_symposium.evaluation \
  data/scenarios/two_test_dataset.json \
  data/results/two_Gemini-Gemini.json \
  --run-id two_Gemini-Gemini \
  --dialogue-turns 10 \
  --start-index 6 \
  --case-delay 10
```

Overnight 模式，失敗 case 記錄後繼續跑完整批：

```bash
python3 -m vox_symposium.evaluation \
  data/scenarios/two_test_dataset.json \
  data/results/two_Gemini-Gemini.json \
  --run-id two_Gemini-Gemini \
  --dialogue-turns 10 \
  --start-index 6 \
  --case-retries 3 \
  --case-retry-delay 30 \
  --overnight
```

## Provider 設定提醒

evaluation runner 依 `.env` 決定兩個 agent 使用哪個 provider：

```env
AGENT_HUMAN_PROVIDER=gemini
AGENT_ROBOT_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-api-key
```

若沒有設定，預設是 `human=openai`、`robot=gemini`，不會是兩個 Gemini 對講。
OpenAI Realtime 可用 `OPENAI_REALTIME_REASONING_EFFORT=low` 設定 Realtime 2 的 reasoning effort。
如果遇到 `ConnectionClosedError: sent 1011 (internal error) keepalive ping timeout`，
可先設定 `OPENAI_REALTIME_PING_INTERVAL=20` 與 `OPENAI_REALTIME_PING_TIMEOUT=120`。

當某個角色使用 `provider=minicpm` 時，runner 會只在那位角色的 `Dialogue behavior`
自動追加短回覆規則：每次最多 2 句、最多問 1 個問題，且不要反覆總結。這不會套用到
Gemini、OpenAI 或其他 provider 的角色。
