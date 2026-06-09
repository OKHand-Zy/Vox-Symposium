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

Agent-Citizen 和 Agent-Scholar 都可以自行設定使用 OpenAI Realtime、Gemini Live 或本機 Hugging Face realtime 語音模型。你可以在 `.env` 裡分別調整兩個角色的 provider、model 和 instructions。

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
- `local` 或 `hf`：使用本機 Hugging Face realtime 語音模型 adapter。

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

例如 Agent-Scholar 使用從 Hugging Face 下載或快取的 MiniCPM-o 4.5：

```env
AGENT_SCHOLAR_PROVIDER=local
LOCAL_MODEL_PROVIDER=minicpmo
LOCAL_MODEL=openbmb/MiniCPM-o-4_5
LOCAL_MODEL_REF_AUDIO=/absolute/path/to/ref_voice.wav
LOCAL_MODEL_LANGUAGE=zh
```

例如使用 Qwen3-Omni Instruct：

```env
AGENT_SCHOLAR_PROVIDER=local
LOCAL_MODEL_PROVIDER=qwen3_omni
LOCAL_MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
LOCAL_MODEL_ATTN_IMPLEMENTATION=flash_attention_2
LOCAL_MODEL_QWEN_SPEAKER=Ethan
LOCAL_MODEL_TURN_DETECTION=false
```

`Qwen/Qwen3-Omni-30B-A3B-Thinking` 只有 audio/video/text input 與 text output，不符合此專案目前的 audio-in/audio-out `RealtimeAudioModel` 介面；若要使用 Thinking checkpoint，需要再接一個 TTS backend。

`LOCAL_MODEL` 可以是 Hugging Face repo id，也可以是本機模型目錄。`LOCAL_MODEL_REF_AUDIO` 是 16 kHz 參考聲音來源，MiniCPM-o 的語音輸出建議設定；沒有設定時 adapter 仍會啟動，但模型是否能產生穩定聲音取決於該模型的預設 TTS 行為。

本機 HF adapter 的可調參數：

```env
LOCAL_MODEL_PROVIDER=auto
LOCAL_MODEL_DEVICE=auto
LOCAL_MODEL_DTYPE=auto
LOCAL_MODEL_ATTN_IMPLEMENTATION=sdpa
LOCAL_MODEL_TURN_DETECTION=true
LOCAL_MODEL_SILERO_THRESHOLD=0.5
LOCAL_MODEL_TURN_SILENCE_MS=700
LOCAL_MODEL_MIN_TURN_MS=400
LOCAL_MODEL_CHUNK_MS=1000
LOCAL_MODEL_MAX_NEW_TOKENS=512
LOCAL_MODEL_QWEN_SPEAKER=Ethan
```

`LOCAL_MODEL_PROVIDER` 可選：

- `auto`：依 `LOCAL_MODEL` 名稱推斷 backend。
- `minicpmo`：MiniCPM-o backend，支援目前的 realtime speech input/output API。
- `qwen3_omni`：Qwen3-Omni backend，使用 Qwen 官方 Transformers + `qwen-omni-utils` 介面。要輸出 audio 請使用 Instruct checkpoint；Thinking checkpoint 只有 text output，需另外接 TTS backend。

`LOCAL_MODEL_TURN_DETECTION=true` 會優先使用模型本身的 VAD/end-of-turn 能力，adapter 會在模型方法支援時傳入 `turn_detection=True`，由模型決定何時回覆。如果模型不支援這條 realtime/duplex API，adapter 會自動退回 Silero VAD。

`LOCAL_MODEL_TURN_DETECTION=false` 會直接使用 Silero VAD。`LOCAL_MODEL_SILERO_THRESHOLD`、`LOCAL_MODEL_TURN_SILENCE_MS` 和 `LOCAL_MODEL_MIN_TURN_MS` 用來調整 Silero fallback 的 end-of-turn 行為。

使用 `local`/`hf` provider 前，需另外安裝本機模型推論依賴：

```bash
pip install -e ".[local]"
```

Qwen3-Omni 目前還需要 Qwen 官方支援的 Transformers 版本；若 PyPI 版本尚未包含 Qwen3-Omni，請依 Qwen model card 指示安裝相容的 Transformers build。

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
- Hugging Face local realtime voice input 會被 resample 成 mono 16 kHz PCM；若模型支援 `turn_detection=True` 會由模型決定何時回覆，否則 adapter 會用 Silero VAD 判斷 end-of-turn 後產生 24 kHz PCM。
- Model output 預期為 mono 24 kHz PCM，發布回 LiveKit 前會 resample 成 LiveKit publish sample rate。

## 擴充其他模型

Provider adapter 放在 `src/vox_symposium/models/`。本機模型共用 wrapper 在 `src/vox_symposium/models/local_voice.py`，模型專用 backend 放在 `src/vox_symposium/models/local_backends/`：

- `base.py`：`LocalVoiceBackend` 介面。
- `minicpmo.py`：MiniCPM-o backend。
- `qwen3_omni.py`：Qwen3-Omni backend。

之後如果要改接其他 self-hosted full-duplex model，優先新增一個 `LocalVoiceBackend`，只有需要完全不同 transport/runtime 時才直接實作 `RealtimeAudioModel` 並在 `build_model()` 中註冊新的 provider。
