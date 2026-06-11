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

## 角色定義

- Agent-Citizen：代表人類使用者，用來模擬一般人對 Voice Agent 的提問、追問與互動。
- Agent-Scholar：代表被測試的 Voice Agent，也就是你要觀察、驗證與調整的目標代理。

Agent-Citizen 和 Agent-Scholar 都可以自行設定使用 OpenAI Realtime 或 Gemini Live。你可以在 `.env` 裡分別調整兩個角色的 provider、model 和 instructions。

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
pip install -e .
```

`.env` 範例，兩邊都使用 Gemini：

```env
AGENT_CITIZEN_PROVIDER=gemini
AGENT_SCHOLAR_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-api-key
```

如果 `.env` 同時有 `GOOGLE_API_KEY` 和 `GEMINI_API_KEY`，Google SDK 會優先使用 `GOOGLE_API_KEY`，執行時會看到提示。要明確使用 `GEMINI_API_KEY`，請移除或 unset `GOOGLE_API_KEY`。

**2. 轉換 scenario**

轉換完整資料集：

```bash
vox-symposium-scenario data/two_test.json data/scenarios/two_test.normalized.json --audio-dir data/test
```

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
    "question_audio": "question_00000000.mp3"
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
Captured citizen turn 1; scholar_turns=0
Captured scholar turn 2; scholar_turns=1
Captured citizen turn 3; scholar_turns=1
Captured scholar turn 4; scholar_turns=2
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

**6. 評測輸出**

主要 result：

```text
data/results/00000000-auto001.json
```

附檔 artifacts：

```text
data/results/00000000-auto001-artifacts/
  dialogue-01-citizen.wav
  dialogue-02-scholar.wav
  ...
  question.wav
  scholar-answer.wav
  dialogue-log.json
```

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
    "audio": "data/results/00000000-auto001-artifacts/scholar-answer.wav",
    "choice": null,
    "is_correct": null
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

- `Missing required environment variable: OPENAI_API_KEY`：evaluation runner 沒讀到 `.env` 裡的 provider 設定，或沒有設定 `AGENT_CITIZEN_PROVIDER=gemini` / `AGENT_SCHOLAR_PROVIDER=gemini`。確認 `.env` 在專案根目錄，並重新執行。
- `Both GOOGLE_API_KEY and GEMINI_API_KEY are set`：Google SDK 提示會使用 `GOOGLE_API_KEY`。這不是錯誤；若不想使用它，請 unset `GOOGLE_API_KEY`。
- `Question audio does not exist`：確認 `evaluation.question_audio` 指向的檔案存在。若 JSON 寫 `"question_00000000.mp3"`，檔案可放在 `data/question/question_00000000.mp3`。
- `ConnectionClosedError` 或 keepalive timeout：先用 `--dialogue-turns 2` 做 smoke test；完整評測可顯式加 `--audio-speed 8` 或 `--audio-speed 16`，並確保角色回覆不要太長。
- `choice` 是 `null`：provider 沒回傳 transcript。先聽 `scholar-answer.wav` 或用 STT 轉寫，再用 `save-result` 保存文字答案。

## 安裝

```bash
conda create -n vox-symposium python=3.11
conda activate vox-symposium
pip install -r requirements.txt
pip install -e .
```

## 設定

編輯 `.env`，填入必要 key：

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`

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

程式只會要求實際使用到的 provider API key。如果兩個角色都使用 OpenAI，就只需要 `OPENAI_API_KEY`；如果兩個角色都使用 Gemini，就只需要 `GEMINI_API_KEY`。

## 啟動

在同一個 process 裡啟動兩個 participant：

```bash
vox-symposium
```

也可以分開啟動：

```bash
vox-symposium --participant agent-citizen
vox-symposium --participant agent-scholar
```

## 測試執行

先確認 conda 環境已啟用，並且已安裝依賴：

```bash
conda activate vox-symposium
pip install -r requirements.txt
pip install -e .
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
```

如果其中一個角色使用 Gemini，才需要加入：

```env
GEMINI_API_KEY=your-gemini-api-key
```

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
- Model output 預期為 mono 24 kHz PCM，發布回 LiveKit 前會 resample 成 LiveKit publish sample rate。

## 擴充其他模型

Provider adapter 放在 `src/vox_symposium/models/`。之後如果要改接 self-hosted full-duplex model，只要實作 `RealtimeAudioModel` 介面，並在 `build_model()` 中註冊新的 provider。
