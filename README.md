# Vox Symposium

Vox Symposium 是一個 Python runtime，用 LiveKit Room 當 WebRTC audio router，讓兩個可程式化的 realtime audio participant 在同一個房間中互相通話。

```text
LiveKit Room
  agent-citizen:
    訂閱 agent-scholar audio track
    將音訊送進自己的 realtime model input
    將 model audio output 發布成 LiveKit audio track

  agent-scholar:
    訂閱 agent-citizen audio track
    將音訊送進自己的 realtime model input
    將 model audio output 發布成 LiveKit audio track
```

資料流：

```text
Agent-Citizen model output -> Agent-Citizen LiveKit audio track -> Agent-Scholar model input
Agent-Scholar model output -> Agent-Scholar LiveKit audio track -> Agent-Citizen model input
```

## 文件導覽

- [Evaluation CLI 完整參數](doc/evaluation-cli.md)
- [專案架構與模組責任](doc/architecture.md)
- [開發、測試與維護規則](doc/development.md)
- [Self-hosted full-duplex provider 串接](doc/self-hosted-full-duplex-gateways.md)
- [MiniCPM-o 4.5 部署](doc/minicpm-o-4_5-deployment.md)
- [Freeze-Omni 文字事件與 VAD patch](doc/freeze-omni-text-events.md)
- [PersonaPlex live server](doc/personaplex-live-server.md)

## 角色定義

- Agent-Citizen：代表人類使用者，用來模擬一般人對 Voice Agent 的提問、追問與互動。
- Agent-Scholar：代表被測試的 Voice Agent，也就是你要觀察、驗證與調整的目標代理。

Agent-Citizen 和 Agent-Scholar 都可以自行設定使用 OpenAI Realtime、Gemini Live、MiniCPM-o 4.5、Freeze-Omni、Moshi 或 PersonaPlex。你可以在 `.env` 裡分別調整兩個角色的 provider、model 和 instructions；但 Kyutai 官方 Moshi server 不會套用 per-session instructions，細節見下方 Moshi 限制說明。

## 重要資料位置

從 [OmniCharacter-plus](https://huggingface.co/datasets/haonanzhang/OmniCharacter-plus) 下載的資料放在專案根目錄的 `data/` 下。建議使用以下位置：

| 資料 | 位置 | 說明 |
| --- | --- | --- |
| 資料集 JSON | `data/<dataset>.json` | 例如 `data/two_test.json` |
| 劇情對話音檔 | `data/<audio_dir>/` | 例如 `data/test/instruct_00000000_9.wav`；轉換時用 `--audio-dir data/test` 指定 |
| 測驗題音檔 | `data/question_audio/<dataset>/` | 例如 `data/question_audio/two_test/question_00000000.wav`；`<dataset>` 必須和 JSON 檔名一致 |
| 轉換後的 scenario | `data/scenarios/` | 存放評測用的 scenario JSON |
| 評測結果與錄音 | `data/results/` | 存放 result JSON 與 `*-artifacts/` |

```text
data/
├── two_test.json
├── test/
│   └── instruct_00000000_9.wav
├── question_audio/
│   └── two_test/
│       └── question_00000000.wav
├── scenarios/
└── results/
```

## OmniCharacter-plus 快速使用

`OmniCharacter-plus` 分支新增資料集驅動的雙語音代理評測流程。它會把 `data/two_test.json` 轉成 scenario，讓兩個 realtime audio model 延續劇情對話，完成指定回合數後播放測驗題音訊給被測模型，最後保存回答、音檔與是否答對。

資料集欄位固定映射：

- `human` -> Agent-Citizen，模擬對話對象
- `gpt` / `system` -> Agent-Scholar，被測語音模型
- `conversations[:-1]` -> 歷史對話，放進 prompt
- `conversations[-1]` -> 開場白，評測開始時自動播放
- `type` / `subtype` / `topic` / `goal` -> 場景設定
- `question` / `multichoice` / `correct_answer` -> 測驗階段使用，不會放進角色 prompt

**1. 安裝與 provider 設定**

```bash
conda activate vox-symposium
pip install -r requirements.txt
```

`.env` 範例，兩邊都使用 Gemini：

```env
AGENT_CITIZEN_PROVIDER=gemini
AGENT_SCHOLAR_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-api-key
GEMINI_LIVE_THINKING_LEVEL=minimal
```

如果 `.env` 同時有 `GOOGLE_API_KEY` 和 `GEMINI_API_KEY`，Google SDK 會優先使用 `GOOGLE_API_KEY`，執行時會看到提示。要明確使用 `GEMINI_API_KEY`，請移除或 unset `GOOGLE_API_KEY`。

也可以改用 Vertex AI 與 service account JSON 金鑰：

```env
AGENT_CITIZEN_PROVIDER=gemini
AGENT_SCHOLAR_PROVIDER=gemini
GEMINI_BACKEND=vertex
GOOGLE_CLOUD_PROJECT=your-google-cloud-project
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
GEMINI_LIVE_MODEL=gemini-live-2.5-flash-native-audio
```

Vertex 模式不需要 `GEMINI_API_KEY`。JSON 路徑建議使用絕對路徑，service account 必須具備呼叫 Vertex AI 的權限，且專案需啟用 Vertex AI API。未設定 location 時預設為 `us-central1`，未設定 model 時預設為 `gemini-live-2.5-flash-native-audio`。此 Live model 不支援 `global` endpoint。

Gemini 3.1 Flash Live 使用 `GEMINI_LIVE_THINKING_LEVEL` 設定思考強度：`minimal`（預設、最低延遲）、`low`、`medium` 或 `high`；不再支援 `thinkingBudget`。如需在開始即時對話前植入既有對話，可設定 `GEMINI_LIVE_INITIAL_HISTORY_JSON` 為 `Content[]` JSON，例如：

```env
GEMINI_LIVE_INITIAL_HISTORY_JSON='[{"role":"user","parts":[{"text":"We already discussed a travel plan."}]},{"role":"model","parts":[{"text":"Yes, we selected Taipei."}]}]'
```

程式會先開啟 `initial_history_in_client_content`，再以 `send_client_content(..., turn_complete=True)` 傳送這段初始歷史；之後的音訊與即時文字一律透過 `send_realtime_input`，不會混用兩種訊息流程。設定 `SCENARIO_FILE` 時，不必將每筆資料序列化到環境變數：程式會將每個 scenario 的完整 `history` 轉成 agent-relative `Content[]`（對方為 `user`、自身為 `model`），並取代這個環境變數的 fallback。固定角色規則、人格、場景與目標則保留在 system instruction。opening 仍會以即時音訊送給接收者，因此不會被誤當成已完成歷史。Scenario history 不會被摘要、截斷或改寫；若超過模型 context 上限，Gemini API 會直接回報限制錯誤。

`gemini-live-2.5-flash-native-audio` 仍完整保留原本流程：不會送出 Gemini 3 的 `thinkingLevel` 或 `history_config`，手動 VAD 時也維持原本的 `TURN_INCLUDES_ALL_INPUT`。若要設定 2.5 的思考，使用 `GEMINI_LIVE_THINKING_BUDGET`（`-1` 動態思考、`0` 關閉、`1` 到 `24576` 指定 token 預算）；未設定時維持 API 的動態思考預設值。2.5 的 `send_client_content` 仍是一般逐輪訊息機制，因此本程式的「初始歷史」JSON 選項只適用於 3.1。

Gemini 2.5 另可啟用情緒感知對話：設定 `GEMINI_LIVE_ENABLE_AFFECTIVE_DIALOG=true`。預設為關閉；啟用時程式會選用必要的 `v1alpha` API 版本，並在工作階段設定傳送 `enable_affective_dialog=true`。此功能不支援 Gemini 3.1 Flash Live；若在 3.1 啟用，程式會在連線前明確報錯。

**2. 轉換 scenario**

轉換完整資料集：

```bash
vox-symposium-scenario data/two_test.json data/scenarios/two_test.normalized.json --audio-dir data/test
```

轉換器會依輸入檔名自動尋找 `data/question_audio/two_test/question_{id}.wav`，並寫入
`evaluation.question_audio`。如果題目音檔放在其他位置，使用
`--question-audio-dir /path/to/question_audio` 指定資料夾。

只轉單一筆：

```bash
vox-symposium-scenario data/two_test.json data/scenarios/00000000.json --id 00000000 --audio-dir data/test
```

已轉好的 scenario 主要包含：

```json
{
  "id": "00000000",
  "history": [],
  "opening": {
    "agent": "scholar",
    "audio": "data/test/instruct_00000000_9.wav"
  },
  "run": {
    "dialogue_turns": 5
  },
  "evaluation": {
    "question": "Based on the dialogue...",
    "choices": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "correct_answer": "C",
    "question_audio": "data/question_audio/two_test/question_00000000.wav"
  }
}
```

**3. 準備測驗題音訊**

`evaluation.question_audio` 可以是 `.wav` 或 `.mp3`。相對路徑會依序嘗試：

- 目前工作目錄
- scenario JSON 所在資料夾
- `data/question/`

因此這種設定是有效的：

```json
"question_audio": "question_00000000.mp3"
```

只要實際檔案存在：

```text
data/question/question_00000000.mp3
```

runner 會自動用 `ffmpeg` 把 MP3 轉成 24 kHz mono PCM WAV 後送給模型。如果沒有設定 `question_audio`，runner 會嘗試用 macOS `say` 產生題目音訊；這在某些 sandbox 或 headless 環境可能產生失敗，所以建議正式評測直接提供題目音檔。

**4. 跑 smoke test**

先跑 2 次 scholar 回覆，確認整條流程能完成：

```bash
python3 -m vox_symposium.evaluation \
  data/scenarios/00000000.json \
  data/results/00000000-smoke.json \
  --run-id 00000000-smoke \
  --dialogue-turns 2
```

成功後會輸出類似：

```text
Playing opening from scholar into citizen: data/test/instruct_00000000_9.wav
Captured citizen turn 1: ...
Captured scholar turns 1: ...
Captured citizen turn 2: ...
Captured scholar turns 2: ...
Playing evaluation question into scholar: data/question_audio/two_test/question_00000000.wav
Captured scholar answer evaluation question: data/results/00000000-smoke-artifacts/00000000/scholar-answer.wav
Saved evaluation result: data/results/00000000-smoke.json (...)
```

**5. 跑完整評測**

預設是 5 次 scholar 回覆後播放題目：

```bash
python3 -m vox_symposium.evaluation \
  data/scenarios/00000000.json \
  data/results/00000000-auto001.json \
  --run-id 00000000-auto001
```

如果模型回覆太長或 websocket keepalive timeout，可以顯式加快音訊注入速度：

```bash
python3 -m vox_symposium.evaluation \
  data/scenarios/00000000.json \
  data/results/00000000-auto002.json \
  --run-id 00000000-auto002 \
  --audio-speed 8
```

`--audio-speed 1` 是 real-time 速度，也是預設值。數字越大，runner 越快把音訊送給另一個模型，較不容易因整體評測時間太長而斷線；但高於 real-time 可能影響 streaming VAD / turn detection，因此正式比較建議固定並記錄這個參數。

如果整批 dataset 中途斷掉，可以用 `--start-index` 從指定的 zero-based case index 繼續跑。例：log 顯示停在 `Running scenario 7/185` 時，下一次從第 7 筆重跑要用 `--start-index 6`：

```bash
python3 -m vox_symposium.evaluation \
  data/scenarios/two_test_dataset.json \
  data/results/two_Gemini-Gemini-resume.json \
  --run-id two_Gemini-Gemini \
  --dialogue-turns 10 \
  --start-index 6
```

每完成一筆 case，runner 會立刻更新 artifacts 目錄下的 `summary.json`；從中段開始跑時，summary 裡的 `index` 仍會保留原始 dataset 的 zero-based index。

**6. 評測輸出**

主要 result：

```text
data/results/00000000-auto001.json
```

附檔 artifacts：

```text
data/results/00000000-auto001-artifacts/
  run-env.txt
  00000000/
    dialogue-01-citizen.wav
    dialogue-02-scholar.wav
    ...
    question.wav
    scholar-answer.wav
    dialogue-log.json
```

每筆測試的 log 和音檔會放在該次 run artifacts 目錄底下的 `<row_id>/` 子資料夾；資料沒有 `row_id` 時會使用 scenario `id`。`run-env.txt` 只會在該次 run artifacts 根目錄保存一份。

`dialogue-log.json` 會在每個 `dialogue_turn` / `evaluation_answer` event 中同時保存 `text` 和 `audio`，讓一段回合文字能直接對應到同一筆紀錄的 WAV 檔。
`run-env.txt` 會保存本次 evaluation 實際使用的 agent provider、backend、model/deployment、voice 和非敏感 provider 參數；API key 與 secret 不會寫入 artifacts。

result 會保存：

```json
{
  "scenario_id": "00000000",
  "run_id": "00000000-auto001",
  "evaluation": {
    "question": "Based on the dialogue...",
    "choices": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "correct_answer": "C"
  },
  "response": {
    "text": "",
    "audio": "data/results/00000000-auto001-artifacts/00000000/scholar-answer.wav",
    "choice": null,
    "is_correct": null
  },
  "artifacts": {
    "dialogue_log": "data/results/00000000-auto001-artifacts/00000000/dialogue-log.json",
    "env_snapshot": "data/results/00000000-auto001-artifacts/run-env.txt"
  }
}
```

如果 provider 回傳 output transcript，runner 會自動從最後回答抽取 `A/B/C/D` 並填入 `choice` / `is_correct`。如果沒有 transcript，仍會保存 `scholar-answer.wav`；你可以轉寫後用 `save-result` 補文字答案。

**7. 手動保存或覆蓋最後答案**

直接保存文字答案：

```bash
vox-symposium-scenario save-result \
  data/scenarios/00000000.json \
  data/results/00000000-run001.json \
  --run-id 00000000-run001 \
  --response "C. Skeptical but willing to listen"
```

從文字檔讀取答案：

```bash
vox-symposium-scenario save-result \
  data/scenarios/00000000.json \
  data/results/00000000-run001.json \
  --response-file data/results/00000000-response.txt
```

`save-result` 會自動抽取 `A/B/C/D`，也可以從完整選項文字反推答案。例如只回答 `Skeptical but willing to listen` 也會比對成 `C`。

**常見問題**

- `Missing required environment variable: OPENAI_API_KEY`：evaluation runner 沒讀到 `.env` 裡的 provider/backend 設定。若使用 Azure，確認已設定 `OPENAI_BACKEND=azure`；若不使用 OpenAI，確認角色 provider 已改為 `gemini`。確認 `.env` 在專案根目錄，並重新執行。
- `Both GOOGLE_API_KEY and GEMINI_API_KEY are set`：Google SDK 提示會使用 `GOOGLE_API_KEY`。這不是錯誤；若不想使用它，請 unset `GOOGLE_API_KEY`。
- `Question audio does not exist`：確認 `evaluation.question_audio` 指向的檔案存在。若 JSON 寫 `"question_00000000.mp3"`，檔案可放在 `data/question/question_00000000.mp3`。
- `ConnectionClosedError` 或 keepalive ping timeout：OpenAI Realtime 先在 `.env` 設 `OPENAI_REALTIME_PING_TIMEOUT=120`；若是大量 batch，再用 `--dialogue-turns 2` 做 smoke test，完整評測可顯式加 `--audio-speed 8` 或 `--audio-speed 16`，並確保角色回覆不要太長。
- `choice` 是 `null`：provider 沒回傳 transcript。先聽 `scholar-answer.wav` 或用 STT 轉寫，再用 `save-result` 保存文字答案。

## 安裝

```bash
conda create -n vox-symposium python=3.11
conda activate vox-symposium
pip install -r requirements.txt
```

## 設定

可以先複製 `.env.example` 成 `.env`，再依實際使用的 provider 填入必要 key。程式只會要求實際使用到的 provider 設定。

```bash
cp .env.example .env
```

編輯 `.env`，常見必要 key：

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- OpenAI 官方 API：`OPENAI_API_KEY`
- Azure OpenAI：`AZURE_OPENAI_API_KEY`、`AZURE_OPENAI_ENDPOINT`、`AZURE_OPENAI_DEPLOYMENT_NAME`
- Gemini 使用 AI Studio 時：`GEMINI_API_KEY`
- Gemini 使用 Vertex AI 時：`GOOGLE_CLOUD_PROJECT`、`GOOGLE_APPLICATION_CREDENTIALS`
- MiniCPM-o 4.5：`MINICPM_REALTIME_URL`
- Freeze-Omni：`FREEZE_OMNI_REALTIME_URL`
- Moshi：`MOSHI_REALTIME_URL`
- PersonaPlex：`PERSONAPLEX_REALTIME_URL`

Gemini 預設使用 AI Studio，因此需要 `GEMINI_API_KEY`。若要使用 Vertex AI，改設 `GEMINI_BACKEND=vertex`，並提供 `GOOGLE_CLOUD_PROJECT`、`GOOGLE_CLOUD_LOCATION` 與指向 service account JSON 的 `GOOGLE_APPLICATION_CREDENTIALS`；此時不需要 `GEMINI_API_KEY`。

主要角色設定：

```env
AGENT_CITIZEN_IDENTITY=agent-citizen
AGENT_CITIZEN_PROVIDER=openai
AGENT_CITIZEN_INSTRUCTIONS=You are Agent-Citizen, representing a human user. Keep replies concise and conversational.

AGENT_SCHOLAR_IDENTITY=agent-scholar
AGENT_SCHOLAR_PROVIDER=gemini
AGENT_SCHOLAR_INSTRUCTIONS=You are Agent-Scholar, the voice agent under test. Keep replies concise and conversational.
```

`AGENT_CITIZEN_PROVIDER` 和 `AGENT_SCHOLAR_PROVIDER` 都接受：

- `openai`：使用 OpenAI Realtime。
- `gemini`：使用 Gemini Live。
- `minicpm`：使用自架 MiniCPM-o 4.5 Audio Full-Duplex Gateway。
- `freeze_omni`：使用自架 VITA-MLLM Freeze-Omni Flask-SocketIO demo server。
- `moshi`：使用 Kyutai Moshi server `/api/chat` WebSocket。
- `personaplex`：使用 PersonaPlex live server，也就是 Moshi `/api/chat` WebSocket protocol。

例如兩邊都使用 OpenAI：

```env
AGENT_CITIZEN_PROVIDER=openai
AGENT_SCHOLAR_PROVIDER=openai
```

例如兩邊都使用 Gemini：

```env
AGENT_CITIZEN_PROVIDER=gemini
AGENT_SCHOLAR_PROVIDER=gemini
```

程式只會要求實際使用到的 provider 認證。OpenAI 官方 backend 使用
`OPENAI_API_KEY`；Azure backend 使用 Azure endpoint、deployment 與 API key。Gemini
則依 `GEMINI_BACKEND` 要求 AI Studio API key 或 Vertex service account JSON。

OpenAI 預設使用官方 API。若要改用 Azure OpenAI Realtime，角色仍設為
`provider=openai`，並設定共用 backend：

```env
AGENT_CITIZEN_PROVIDER=openai
AGENT_SCHOLAR_PROVIDER=openai
OPENAI_BACKEND=azure
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=your-gpt-realtime-deployment
OPENAI_REALTIME_VOICE=marin
OPENAI_REALTIME_REASONING_EFFORT=low
OPENAI_REALTIME_PING_INTERVAL=20
OPENAI_REALTIME_PING_TIMEOUT=120
```

預設使用 Azure GA endpoint（`/openai/v1/realtime`）。只有 deployment 使用 preview
API 時才加上 `AZURE_OPENAI_API_VERSION=2025-04-01-preview`；程式會改用 preview
endpoint（`/openai/realtime`）。Azure 模式不需要 `OPENAI_API_KEY`，且 deployment
名稱取代 `OPENAI_REALTIME_MODEL`。

MiniCPM-o 4.5 必須先部署官方 Gateway、Worker 與 Backend，Vox Symposium 只連公開
Gateway，不直接連 Worker 或 Backend。以下範例只將被測的 Scholar 換成 MiniCPM：

```env
AGENT_CITIZEN_PROVIDER=gemini
AGENT_SCHOLAR_PROVIDER=minicpm
MINICPM_REALTIME_URL=ws://127.0.0.1:8006/v1/realtime?mode=audio
MINICPM_LENGTH_PENALTY=1.1
MINICPM_INPUT_CHUNK_MS=1000
MINICPM_QUEUE_TIMEOUT=300
```

正式環境應使用 `wss://`。若 reverse proxy 驗證 Bearer token，再設定
`MINICPM_API_KEY`。adapter 會持續轉送 LiveKit 音訊，不使用 client-side VAD
切斷靜音，讓模型保留完整的 full-duplex listen/speak 判斷。
自動 evaluation 需要固定交替回合，因此只在 evaluation runner 中額外要求 MiniCPM
於對方說完後輸出語音，並追加靜音 input chunk 讓模型繼續進行 listen/speak
決策。模型說話期間會持續以即時速度送入靜音，直到模型回到 `listen`，避免
full-duplex 生成因沒有後續 input 而中途停止；一般 LiveKit participant 不會加入這項限制。

Freeze-Omni 需要先啟動官方 [VITA-MLLM/Freeze-Omni](https://github.com/VITA-MLLM/Freeze-Omni)
Flask-SocketIO demo server。Vox Symposium 只連 server，不直接在主流程載入
Freeze-Omni 權重。使用前先安裝可選依賴：

```bash
pip install -e '.[freeze-omni]'
```

只將 Scholar 換成 Freeze-Omni：

```env
AGENT_CITIZEN_PROVIDER=gemini
AGENT_SCHOLAR_PROVIDER=freeze_omni
FREEZE_OMNI_REALTIME_URL=https://127.0.0.1:8081
FREEZE_OMNI_SSL_VERIFY=false
FREEZE_OMNI_INPUT_CHUNK_MS=20
FREEZE_OMNI_TURN_START_DELAY=3
FREEZE_OMNI_TURN_PREROLL_SILENCE_MS=1200
FREEZE_OMNI_MAX_INPUT_SILENCE_MS=200
FREEZE_OMNI_INPUT_SILENCE_RMS_THRESHOLD=800
FREEZE_OMNI_CONNECT_TIMEOUT=60
FREEZE_OMNI_PROMPT_TIMEOUT=60
FREEZE_OMNI_CONNECT_RETRIES=10
FREEZE_OMNI_CONNECT_RETRY_DELAY=5
FREEZE_OMNI_POST_TURN_POLL_SECONDS=60
FREEZE_OMNI_POST_TURN_IDLE_SECONDS=3
FREEZE_OMNI_POST_TURN_POLL_CHUNK_MS=160
FREEZE_OMNI_STOP_RECORDING_AFTER_TURN=true
```

官方 server 預設使用自簽憑證，因此本 adapter 預設 `FREEZE_OMNI_SSL_VERIFY=false`。
正式環境如果換成可信任憑證，可以設為 `true`。一般 LiveKit participant 會維持連續音訊流；
evaluation runner 則會啟用固定回合控制，包含 `recording-started` / `recording-stopped`、
turn 前靜音、輸入靜音壓縮，以及回合結束後用短靜音輪詢 queued TTS 音訊。

官方 `bin/server.py` 預設只 emit 音訊，不會把生成文字送回 client。若要讓 Vox 同時保存
Freeze-Omni 的文字 transcript，或要套用多回合 VAD/reset 的 server patch，請看
[doc/freeze-omni-text-events.md](doc/freeze-omni-text-events.md)。

Moshi 和 PersonaPlex 走 Moshi 二進位 WebSocket protocol：Vox 送入/接收 24 kHz mono Opus pages，內部轉回 PCM16。使用前先安裝可選依賴：

```bash
pip install -e '.[moshi]'
```

只將 Scholar 換成 Moshi：

```env
AGENT_CITIZEN_PROVIDER=gemini
AGENT_SCHOLAR_PROVIDER=moshi
MOSHI_REALTIME_URL=ws://127.0.0.1:8998/api/chat
```

若 `MOSHI_REALTIME_URL` 只填 `ws://127.0.0.1:8998`，程式會自動補上 `/api/chat`。

注意：Kyutai 官方 Moshi server 目前不支援 OpenAI/Gemini 那種 per-session
system prompt / instructions。Vox 端會把 instructions 放進 `text_prompt` query，
但官方 `/api/chat` server 不會讀取這個參數，因此角色資料、場景與歷史對話不會真的
被 Moshi 使用；模型行為主要由 server 啟動時載入的 weights/model 決定。

PersonaPlex live server 也使用同一個 adapter。固定的 voice prompt 放在 URL query；每個 scenario/case 的 Scholar instructions 會由 Vox 自動 URL encode 後寫入 `text_prompt`，不用在 `.env` 寫死角色 prompt。

```env
AGENT_CITIZEN_PROVIDER=gemini
AGENT_SCHOLAR_PROVIDER=personaplex
PERSONAPLEX_REALTIME_URL=ws://127.0.0.1:8998/api/chat?voice_prompt=NATF2.pt
```

PersonaPlex 部署、voice prompt 與 server patch 請看 [doc/personaplex-live-server.md](doc/personaplex-live-server.md)。

## 啟動

在同一個 process 裡啟動兩個 participant：

```bash
vox-symposium
```

預設會保存對話過程中模型輸出的語音與 transcript 文字：

```text
data/recordings/<run-id>/
  conversation-log.json
  agent-citizen-0001.wav
  agent-scholar-0001.wav
  ...
```

`conversation-log.json` 的每個 `model_output_turn` event 都會把文字和音檔放在同一筆紀錄中，方便後續用 JSON index 對應：

```json
{
  "type": "model_output_turn",
  "agent": "agent-scholar",
  "text": "Transcript text from the provider.",
  "audio": "data/recordings/20260618T120000Z-vox-symposium-both/agent-scholar-0001.wav",
  "sample_rate": 24000,
  "channels": 1,
  "duration_seconds": 2.42
}
```

可以用 `--record-dir` 或 `VOX_RECORD_DIR` 改變輸出位置；若只想跑即時轉發、不保存紀錄，可加 `--no-record`。

也可以分開啟動：

```bash
vox-symposium --participant agent-citizen
vox-symposium --participant agent-scholar
```

## 開發檢查

安裝開發工具並執行不需 API key 的單元測試與靜態檢查：

```bash
pip install -e '.[dev]'
python -m unittest discover -s tests -v
ruff check src tests
ruff format --check src tests
pyright
```

完整規範見 [doc/development.md](doc/development.md)。

## LiveKit 整合測試

先確認 conda 環境已啟用，並且已安裝依賴：

```bash
conda activate vox-symposium
pip install -r requirements.txt
```

建立 `.env`，至少填入 LiveKit 設定：

```env
LIVEKIT_URL=wss://your-livekit-url
LIVEKIT_API_KEY=your-livekit-api-key
LIVEKIT_API_SECRET=your-livekit-api-secret
```

如果兩個角色都使用 OpenAI，加入：

```env
AGENT_CITIZEN_PROVIDER=openai
AGENT_SCHOLAR_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key
OPENAI_REALTIME_MODEL=gpt-realtime-2
OPENAI_REALTIME_VOICE=marin
OPENAI_REALTIME_REASONING_EFFORT=low
OPENAI_REALTIME_PING_INTERVAL=20
OPENAI_REALTIME_PING_TIMEOUT=120
```

改用 Azure OpenAI 時，將上段的 `OPENAI_API_KEY` 換成：

```env
OPENAI_BACKEND=azure
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=your-gpt-realtime-deployment
```

如果其中一個角色使用 Gemini，才需要加入：

```env
GEMINI_API_KEY=your-gemini-api-key
```

或使用 Vertex AI：

```env
GEMINI_BACKEND=vertex
GOOGLE_CLOUD_PROJECT=your-google-cloud-project
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
```

如果其中一個角色使用 MiniCPM-o 4.5，加入：

```env
AGENT_SCHOLAR_PROVIDER=minicpm
MINICPM_REALTIME_URL=ws://127.0.0.1:8006/v1/realtime?mode=audio
```

如果其中一個角色使用 Moshi，加入：

```env
AGENT_SCHOLAR_PROVIDER=moshi
MOSHI_REALTIME_URL=ws://127.0.0.1:8998/api/chat
```

Kyutai 官方 Moshi server 不會讀取 Vox 寫入 URL 的 `text_prompt` query，所以
Moshi 不會套用角色 prompt、場景 prompt 或歷史對話 prompt。若需要 prompt
conditioning，請使用支援 `text_prompt` 的 Moshi-compatible server，例如
PersonaPlex live server，或自行 patch server。

如果其中一個角色使用 PersonaPlex live server，加入：

```env
AGENT_SCHOLAR_PROVIDER=personaplex
PERSONAPLEX_REALTIME_URL=ws://127.0.0.1:8998/api/chat?voice_prompt=NATF2.pt
```

Vox 會把每個 scenario/case 的 Scholar instructions 動態寫入 PersonaPlex 的 `text_prompt` query。
部署與已知問題請看 [doc/personaplex-live-server.md](doc/personaplex-live-server.md)。

啟動測試：

```bash
vox-symposium
```

也可以分開兩個 terminal 測試：

```bash
vox-symposium --participant agent-citizen
vox-symposium --participant agent-scholar
```

預設房間名稱是 `vox-symposium`。如需改房間名，在 `.env` 加上：

```env
LIVEKIT_ROOM=test-room
```

## 音訊格式

- LiveKit 發布音訊時使用 mono 48 kHz PCM frame。
- OpenAI Realtime input 會被 resample 成 mono 24 kHz PCM。
- Gemini Live input 會被 resample 成 mono 16 kHz PCM。
- MiniCPM input 會轉成 mono 16 kHz float32 PCM；output 24 kHz float32 PCM 會轉回 PCM16。
- Freeze-Omni input 會轉成 mono 16 kHz PCM16；output 24 kHz PCM16 會直接發布。
- Moshi / PersonaPlex adapter 會把 PCM16 轉成 24 kHz mono Opus pages，並將 output Opus pages 解回 PCM16。
- Model output 預期為 mono 24 kHz PCM，發布回 LiveKit 前會 resample 成 LiveKit publish sample rate。

## 擴充其他模型

Provider adapter 放在 `src/vox_symposium/models/`，provider 名稱與 factory 集中在 `src/vox_symposium/providers.py` 與 `src/vox_symposium/models/factory.py`。之後如果要改接 self-hosted full-duplex model，新增專用 adapter，再在 factory 中註冊。

目前的模組邊界、設定生命週期與新增 provider 清單見 [doc/architecture.md](doc/architecture.md)。

如果要使用自己的本地 Hugging Face 即時語音模型，請看 [doc/local-hf-realtime-model.md](doc/local-hf-realtime-model.md)。

Moshi protocol 串接建議請看 [doc/self-hosted-full-duplex-gateways.md](doc/self-hosted-full-duplex-gateways.md)。

MiniCPM-o 4.5 的完整非 Docker 部署、TorchCodec 安裝、三程序啟動與故障排除請看
[doc/minicpm-o-4_5-deployment.md](doc/minicpm-o-4_5-deployment.md)。
