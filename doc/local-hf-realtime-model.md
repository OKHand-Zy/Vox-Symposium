# 使用自己的本地 Hugging Face 即時語音模型

這份文件說明如果要把 Vox Symposium 從 OpenAI Realtime / Gemini Live 改接自己的本地 Hugging Face 語音模型，需要把模型放在哪、程式要新增哪個 adapter，以及 `.env` 要怎麼設定。

目前專案內建 provider 包含：

- `openai`：OpenAI Realtime
- `gemini`：Gemini Live
- `minicpm`：MiniCPM-o 4.5 Audio Full-Duplex Gateway
- `moshi`：Kyutai Moshi `/api/chat`，官方 server 不支援 per-session prompt / instructions
- `personaplex`：PersonaPlex live server 的 Moshi `/api/chat`，支援 `text_prompt`

如果你的本地 Hugging Face 模型無法使用現有 provider protocol，才需要新增 provider adapter，實作 `RealtimeAudioModel` 介面，再把它註冊到 `src/vox_symposium/models/factory.py`。

## 放置位置

建議使用下面的位置：

```text
Vox-Symposium/
  models/
    hf/
      your-model/
        config.json
        model.safetensors
        tokenizer.json
        ...
  src/
    vox_symposium/
      models/
        local_hf_realtime.py
  doc/
    local-hf-realtime-model.md
```

- 模型權重：建議放在 `models/hf/your-model/`，或使用 Hugging Face cache 裡的本地路徑。
- adapter 程式：放在 `src/vox_symposium/models/local_hf_realtime.py`。
- provider 設定：放在專案根目錄 `.env`。

如果模型權重很大，不建議 commit 到 git。可以把 `models/` 加到 `.gitignore`，只在文件或 `.env.example` 記錄路徑。

## 模型需要提供的能力

Vox Symposium 的 participant 會做兩件事：

1. 從 LiveKit 收到對方音訊，呼叫 `send_audio()` 持續送進模型。
2. 從模型拿到輸出音訊，透過 `receive_audio()` 持續發布回 LiveKit。

因此本地 Hugging Face 模型最好能支援 streaming audio input / streaming audio output。若你的模型只能「整段音訊輸入，整段音訊輸出」，也可以接，但 adapter 需要自己處理：

- audio buffer
- VAD 或 turn detection
- 何時開始推論
- 何時把模型輸出切成小段 PCM 丟給 `receive_audio()`

專案目前內部音訊格式以 PCM16 mono 為主。adapter 要宣告模型吃的 sample rate 與輸出的 sample rate，例如：

```python
input_sample_rate = 16_000
output_sample_rate = 24_000
```

LiveKit 送進來的音訊會依照 `input_sample_rate` 轉成 mono PCM16；模型輸出的 `PcmAudio` 也會在發布回 LiveKit 前自動轉成 LiveKit publish sample rate。

## 新增 adapter

新增檔案：

```text
src/vox_symposium/models/local_hf_realtime.py
```

範本：

```python
from __future__ import annotations

import asyncio

from vox_symposium.audio import PcmAudio, normalize_audio
from vox_symposium.models.base import QueueBackedRealtimeAudioModel, cancel_task


class LocalHFRealtimeModel(QueueBackedRealtimeAudioModel):
    input_sample_rate = 16_000
    output_sample_rate = 24_000

    def __init__(
        self,
        *,
        model_path: str,
        instructions: str,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.instructions = instructions
        self.device = device
        self._audio_in: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=100)
        self._worker_task: asyncio.Task[None] | None = None
        self._model = None

    async def connect(self) -> None:
        # 在這裡載入你的 Hugging Face 模型。
        # 例如 transformers / torch / 自己封裝的 realtime engine。
        #
        # self._model = load_your_model(self.model_path, device=self.device)
        self._worker_task = asyncio.create_task(self._run_model_loop(), name="local-hf-realtime")

    async def send_audio(self, audio: PcmAudio) -> None:
        pcm = normalize_audio(
            audio.data,
            from_rate=audio.sample_rate,
            to_rate=self.input_sample_rate,
            channels=audio.channels,
        )
        if pcm:
            await self._audio_in.put(pcm)

    async def close(self) -> None:
        worker_task = self._worker_task
        self._worker_task = None
        await cancel_task(worker_task)
        self.close_output_streams()

    async def _run_model_loop(self) -> None:
        # 這裡要換成你的模型實際 streaming 推論邏輯。
        # 重點是：從 self._audio_in 讀 PCM16 bytes，並把輸出的 PCM16 bytes
        # 包成 PcmAudio 放進 self._audio_out。
        try:
            while True:
                chunk = await self._audio_in.get()
                if chunk is None:
                    return

                # pseudo code:
                # output_chunks = self._model.stream_audio(
                #     chunk,
                #     sample_rate=self.input_sample_rate,
                #     instructions=self.instructions,
                # )
                # for output_pcm in output_chunks:
                #     await self._audio_out.put(
                #         PcmAudio(
                #             data=output_pcm,
                #             sample_rate=self.output_sample_rate,
                #             channels=1,
                #         )
                #     )
        finally:
            self.close_output_streams()
```

這個範本只定義接線方式，不能直接產生模型音訊。你需要把 `connect()` 和 `_run_model_loop()` 裡的 pseudo code 換成自己的 Hugging Face 模型呼叫方式。

## 註冊 provider

修改 `src/vox_symposium/config.py`。

新增型別化設定與 loader：

```python
@dataclass(frozen=True)
class LocalHFSettings:
    model_path: str
    device: str


def load_local_hf_settings() -> LocalHFSettings:
    return LocalHFSettings(
        model_path=required_env("LOCAL_HF_MODEL_PATH"),
        device=os.getenv("LOCAL_HF_DEVICE", "cuda"),
    )
```

在 `Settings` 新增 `local_hf: LocalHFSettings | None`，並在 `load_settings()` 建立它：

```python
local_hf=(
    load_local_hf_settings()
    if _uses_provider("local_hf", *agents)
    else None
)
```

`src/vox_symposium/providers.py` 的 provider set、env prefix 與 label 都要註冊：

```python
SUPPORTED_PROVIDERS = frozenset({... , "local_hf"})
_PROVIDER_ENV_PREFIXES = {..., "local_hf": "LOCAL_HF"}
_PROVIDER_LABELS = {..., "local_hf": "Local Hugging Face"}
```

接著在 `models/factory.py` 新增共用建構 helper，並讓 `build_model_from_settings()` 與 `build_model_from_env()` 都呼叫它：

```python
def _build_local_hf_model(
    config: LocalHFSettings,
    instructions: str,
) -> RealtimeAudioModel:
    return LocalHFRealtimeModel(
        model_path=config.model_path,
        device=config.device,
        instructions=instructions,
    )
```

## `.env` 範例

只把 Agent-Scholar 換成本地 HF 模型，Agent-Citizen 仍使用 Gemini：

```env
LIVEKIT_URL=wss://your-livekit-url
LIVEKIT_API_KEY=your-livekit-api-key
LIVEKIT_API_SECRET=your-livekit-api-secret

AGENT_CITIZEN_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-api-key

AGENT_SCHOLAR_PROVIDER=local_hf
LOCAL_HF_MODEL_PATH=models/hf/your-model
LOCAL_HF_DEVICE=cuda
```

兩邊都使用本地 HF 模型：

```env
LIVEKIT_URL=wss://your-livekit-url
LIVEKIT_API_KEY=your-livekit-api-key
LIVEKIT_API_SECRET=your-livekit-api-secret

AGENT_CITIZEN_PROVIDER=local_hf
AGENT_SCHOLAR_PROVIDER=local_hf
LOCAL_HF_MODEL_PATH=models/hf/your-model
LOCAL_HF_DEVICE=cuda
```

如果兩個角色要用不同本地模型，建議把設定拆成：

```env
AGENT_CITIZEN_PROVIDER=local_hf
AGENT_CITIZEN_HF_MODEL_PATH=models/hf/citizen-model

AGENT_SCHOLAR_PROVIDER=local_hf
AGENT_SCHOLAR_HF_MODEL_PATH=models/hf/scholar-model
```

這種寫法需要把 `LocalHFSettings` 改成 per-agent 設定，或讓 loader 接受 agent role；兩個 factory 入口仍應共用同一個 `_build_local_hf_model()`。

## 依賴安裝

依照你的模型需要安裝 Hugging Face / PyTorch 相關套件，例如：

```bash
pip install torch transformers accelerate safetensors
```

如果模型需要 GPU，請確認 PyTorch 版本與 CUDA 版本相容。macOS 可以先用：

```env
LOCAL_HF_DEVICE=mps
```

或 CPU：

```env
LOCAL_HF_DEVICE=cpu
```

## 測試

安裝本專案：

```bash
pip install -r requirements.txt
```

啟動：

```bash
vox-symposium
```

如果只想先測本地 HF 那一邊：

```bash
vox-symposium --participant agent-scholar
```

常見問題：

- `Unsupported provider`：確認已在 `providers.py` 註冊 canonical name、env prefix 與 label，並在 `models/factory.py` 的 LiveKit / evaluation 入口加入 dispatch。
- `configuration is required`：確認 `config.py` 已定義 `local_hf` settings dataclass 與 loader，而且完整 `Settings` 會在角色使用該 provider 時載入它。
- 沒有聲音輸出：確認 `_run_model_loop()` 有把 PCM16 mono bytes 放進 `_audio_out`，且 `sample_rate` 設成模型實際輸出音訊的 sample rate。
- 延遲太高：避免在 async event loop 裡直接跑長時間 blocking 推論；可以用背景 thread/process 或本地 websocket server 包裝模型。
- 音高或語速異常：檢查 `input_sample_rate` / `output_sample_rate` 是否和模型實際格式一致。

## 建議架構

如果 Hugging Face 模型載入很慢、推論很重，建議把模型做成獨立本地服務：

```text
Vox Symposium adapter
  -> ws://127.0.0.1:9000/realtime
      -> Hugging Face model process
```

這樣 Vox Symposium 只負責 LiveKit 音訊路由與 provider adapter，模型服務可以獨立管理 GPU、batching、重啟和 logging。

共用 factory、設定生命週期與完整註冊步驟見 [專案架構](architecture.md)。
