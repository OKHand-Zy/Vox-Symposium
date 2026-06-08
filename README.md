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
