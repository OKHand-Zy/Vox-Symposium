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

## 劇情資料集

可以用 `data/two_test.json` 這類資料集產生雙代理對話劇情。系統採用固定映射：

- `human` -> Agent-Citizen
- `gpt` / `system` -> Agent-Scholar，也就是被測語音模型
- `conversations[:-1]` -> 歷史對話
- `conversations[-1]` -> 開場白
- `type` / `subtype` / `topic` / `goal` -> 場景設定
- `question` / `multichoice` / `correct_answer` -> 測驗階段使用，不放進角色 prompt

先把原始資料轉成 normalized scenario：

```bash
vox-symposium-scenario data/two_test.json data/scenarios/two_test.normalized.json --audio-dir data/test
```

只轉單一筆：

```bash
vox-symposium-scenario data/two_test.json data/scenarios/00000000.json --id 00000000 --audio-dir data/test
```

normalized scenario 的主要結構：

```json
{
  "id": "00000000",
  "agents": {
    "citizen": {
      "name": "Stephen V",
      "source_role": "human",
      "profile": "..."
    },
    "scholar": {
      "name": "Hollis Lomax",
      "source_role": "gpt",
      "profile": "..."
    }
  },
  "scene": {
    "type": "Persuasion",
    "subtype": "Service recommendation",
    "topic": "Explaining the advantages of hiring a professional organizer.",
    "goal": "Persuade a person to consider hiring a professional organizer to improve home efficiency."
  },
  "history": [],
  "opening": {
    "agent": "scholar",
    "speaker": "Hollis Lomax",
    "text": "(hesitantly) Well, I guess it would be nice to not have to search for my tools every time I need them...",
    "audio": "data/test/instruct_00000000_9.wav"
  },
  "run": {
    "dialogue_turns": 5
  },
  "evaluation": {
    "ask_after_turns": 5,
    "target_agent": "scholar",
    "question": "Based on the dialogue...",
    "choices": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "correct_answer": "C",
    "question_audio": null
  }
}
```

執行時指定 scenario：

```env
SCENARIO_FILE=data/scenarios/00000000.json
SCENARIO_DIALOGUE_TURNS=5
```

或直接從原始資料集選一筆：

```env
SCENARIO_FILE=data/two_test.json
SCENARIO_ID=00000000
SCENARIO_AUDIO_DIR=data/test
SCENARIO_DIALOGUE_TURNS=5
```

目前 scenario 會自動產生兩個 agent 的 instructions。開場白播放、5 回合計數、以及播放 evaluation 題目音訊屬於下一層 runtime 控制；`evaluation.question_audio` 預留給你之後產生題目語音後填入。

保存被測模型最後回答：

```bash
vox-symposium-scenario save-result data/scenarios/00000000.json data/results/00000000-run001.json --response "C. Skeptical but willing to listen"
```

如果最後回答是先存成文字檔：

```bash
vox-symposium-scenario save-result data/scenarios/00000000.json data/results/00000000-run001.json --response-file data/results/00000000-response.txt
```

輸出的 result 會保存題目、選項、正解、模型原始回答、抽取出的 `A/B/C/D`，以及 `is_correct`：

```json
{
  "scenario_id": "00000000",
  "run_id": "00000000-run001",
  "evaluation": {
    "question": "Based on the dialogue...",
    "choices": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "correct_answer": "C"
  },
  "response": {
    "text": "C. Skeptical but willing to listen",
    "audio": null,
    "choice": "C",
    "is_correct": true
  }
}
```

自動跑一遍評測：

```bash
vox-symposium-evaluate data/scenarios/00000000.json data/results/00000000-auto001.json --run-id 00000000-auto001
```

也可以直接從原始資料集選一筆：

```bash
vox-symposium-evaluate data/two_test.json data/results/00000000-auto001.json --id 00000000 --audio-dir data/test --run-id 00000000-auto001
```

自動評測流程：

- 播放 `opening.audio` 給下一位 agent
- 兩個模型自動互相傳遞語音
- 以 scholar 音訊輸出分段計算 5 次 scholar 回覆
- 產生或讀取 evaluation question audio
- 將 question audio 播給 scholar
- 錄下 scholar 最後回答音訊
- 保存 result 與 dialogue log

如果 `evaluation.question_audio` 是空的，runner 會嘗試用 macOS `say` 產生題目語音。若系統語音服務不可用，請先準備題目 wav，然後執行：

```bash
vox-symposium-evaluate data/scenarios/00000000.json data/results/00000000-auto001.json --question-audio data/questions/00000000.wav
```

若 provider 有輸出文字 transcript，runner 會自動從 scholar 最後回答抽取 `A/B/C/D`。若沒有 transcript，result 仍會保存 `response.audio`，但 `response.choice` 會是 `null`；此時可以先轉寫音檔，再用 `save-result` 保存文字版答案。

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
