# MiniCPM-o 4.5 Audio Full-Duplex 安裝、部署與 Vox Symposium 串接

本文說明如何在 Linux/NVIDIA GPU 主機以非 Docker 方式部署 MiniCPM-o 4.5 Audio
Full-Duplex API，並從 Vox Symposium 透過 Gateway 串接。內容包含實際部署時遇到的
TorchCodec、服務位址、全雙工音訊持續輸入與診斷注意事項。

## 1. 架構與連線邊界

正式資料流如下：

```text
Vox Symposium
    │ WebSocket: /v1/realtime?mode=audio
    ▼
Gateway :8006
    │ internal WebSocket
    ▼
Worker :22400
    │ internal HTTP/WebSocket runtime protocol
    ▼
PyTorch Backend :22500
    │
    ▼
MiniCPM-o-4_5 on NVIDIA GPU
```

只有 Gateway 是 client API。Vox Symposium 不應直接連 Worker 或 Backend。

- 開發／SSH tunnel：`ws://127.0.0.1:8006/v1/realtime?mode=audio`
- 正式環境：`wss://API_DOMAIN/v1/realtime?mode=audio`
- `22400`、`22500` 只能綁定 loopback 或內部網路，不應公開到 Internet。

## 2. 系統需求

- Linux；以下命令以 Ubuntu 22.04 為例。
- NVIDIA GPU 與可用的 NVIDIA driver。
- 官方建議至少 28 GB VRAM。
- Python 3.10 或更新版本；MiniCPM 服務與 Vox Symposium 可使用不同環境。
- 約 19 GB 模型權重空間，另需 Python/CUDA dependencies 空間。
- 一個全雙工 session 會占用一個 Worker；一般是一個 Worker 對應一張 GPU。

先驗證：

```bash
nvidia-smi
python --version
```

安裝 OS dependencies：

```bash
apt-get update
apt-get install -y git curl ffmpeg libsndfile1
```

## 3. 下載並固定 MiniCPM-o-Demo 版本

```bash
git clone https://github.com/OpenBMB/MiniCPM-o-Demo.git
cd MiniCPM-o-Demo
git rev-parse HEAD
```

正式部署必須記錄 commit hash。Gateway、Worker 和 Backend 必須由同一份 checkout、同一個
Python environment 啟動，避免 runtime protocol 不一致。

官方來源：

- 模型：<https://huggingface.co/openbmb/MiniCPM-o-4_5>
- Demo：<https://github.com/OpenBMB/MiniCPM-o-Demo>
- Audio protocol：<https://github.com/OpenBMB/MiniCPM-o-Demo/blob/main/docs/audio-duplex-protocol.md>

## 4. 建立 Python environment

優先使用 repository 提供的安裝腳本：

```bash
cd /path/to/MiniCPM-o-Demo
conda create -n <name> python=3.10 or bash install.sh
conda activate <name>/
pip install -r requirements.txt
```

若使用既有 Conda environment，後續所有安裝與三個服務程序都必須在該 environment 內執行。

驗證 PyTorch：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

`torch.cuda.is_available()` 必須是 `True`。

## 5. 安裝 TorchCodec

Audio Full-Duplex 初始化會透過 Torchaudio/TorchCodec 讀取 reference audio。缺少
TorchCodec 時，Backend 會在 `session.init` 後立即關閉 session，並記錄：

```text
TorchCodec is required for load_with_torchcodec
```

TorchCodec 必須和 PyTorch 版本相容。可先檢查：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

例如已驗證的 PyTorch 2.12 / CUDA 13.0 環境：

```bash
python -m pip install --no-deps "torchcodec==0.14.*" \
  --index-url=https://download.pytorch.org/whl/cu130
```

TorchCodec 0.12～0.14 支援 PyTorch 2.11 以上。其他 PyTorch 版本應依官方相容矩陣選擇，
不要直接套用上述版本：

<https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions>

FFmpeg 4～8 均在目前 TorchCodec 支援範圍。驗證：

```bash
ffmpeg -version
python -c "import torchcodec; print(torchcodec.__version__)"
```

必須從 Demo repository 根目錄驗證 reference audio，否則相對路徑會找不到：

```bash
cd /path/to/MiniCPM-o-Demo
python -c "import torchaudio; x, sr = torchaudio.load('assets/ref_audio/ref_minicpm_signature.wav'); print(x.shape, sr)"
```

## 6. 下載模型

```bash
python -m pip install -U huggingface_hub
hf download openbmb/MiniCPM-o-4_5 \
  --local-dir /opt/models/MiniCPM-o-4_5
```

確認模型與 Token2Wav assets 存在：

```bash
ls -lah /opt/models/MiniCPM-o-4_5
ls -lah /opt/models/MiniCPM-o-4_5/assets/token2wav
```

## 7. 準備服務設定

所有啟動命令都必須從 `MiniCPM-o-Demo` repository 根目錄執行：

```bash
cd /path/to/MiniCPM-o-Demo
test -e config.json || cp config.example.json config.json
mkdir -p data tmp certs torch_compile_cache
```

初次部署保留完整 `config.example.json` 欄位，並先設定：

```json
{
  "service": {
    "compile": false
  }
}
```

不要把 `config.json` 縮減成只有上述片段；應保留官方完整設定。

## 8. 啟動三個程序

啟動順序固定為 Backend → Worker → Gateway。每個程序使用獨立 terminal、systemd unit 或
Supervisor process。

### 8.1 PyTorch Backend

```bash
cd /path/to/MiniCPM-o-Demo
conda activate <name>/

python -m py_backend.server \
  --host 127.0.0.1 \
  --port 22500 \
  --gpu-id 0 \
  --model-path /opt/models/MiniCPM-o-4_5
```

等待 log 出現：

```text
Backend server ready
```

驗證：

```bash
curl --fail http://127.0.0.1:22500/health
```

### 8.2 Worker

```bash
cd /path/to/MiniCPM-o-Demo
conda activate <name>/

python worker.py \
  --host 127.0.0.1 \
  --port 22400 \
  --gpu-id 0 \
  --backend-server-url http://127.0.0.1:22500
```

驗證：

```bash
curl --fail http://127.0.0.1:22400/health
```

### 8.3 Gateway

開發環境或前方已有 TLS/SSH tunnel 時：

```bash
cd /path/to/MiniCPM-o-Demo
conda activate <name>/

gateway.py \
  --host 0.0.0.0 \
  --port 8006 \
  --http \
  --workers 127.0.0.1:22400
```

驗證：

```bash
curl --fail http://127.0.0.1:8006/health
curl --fail http://127.0.0.1:8006/status
curl --fail http://127.0.0.1:8006/workers
```

預期至少看到：

```json
{
  "gateway_healthy": true,
  "idle_workers": 1,
  "error_workers": 0,
  "offline_workers": 0
}
```

### 8.4 位址規則

`0.0.0.0` 只代表「服務監聽所有介面」，不能作為 client 或服務間連線目的地。

正確：

```text
Gateway bind:              0.0.0.0:8006
Gateway → Worker:          127.0.0.1:22400
Worker → Backend:          127.0.0.1:22500
Vox → SSH tunnel/Gateway:  127.0.0.1:8006
```

錯誤：

```text
ws://0.0.0.0:8006/...
--workers 0.0.0.0:22400
--backend-server-url http://0.0.0.0:22500
```

## 9. 從遠端開發機連線

若 MiniCPM 部署在遠端 GPU 主機，建議用 SSH tunnel，不要直接公開 HTTP Gateway：

```bash
ssh -L 127.0.0.1:8006:127.0.0.1:8006 USER@GPU_HOST -p SSH_PORT
```

保持該 terminal 運行，然後在 Vox 主機確認：

```bash
curl --fail http://127.0.0.1:8006/health
curl --fail http://127.0.0.1:8006/status
```

若 HTTP 與 HTTPS 都立即 reset，通常代表 SSH tunnel 存在，但遠端 `127.0.0.1:8006`
沒有正常監聽。

## 10. 設定 Vox Symposium

在 Vox Symposium environment：

```bash
cd /path/to/Vox-Symposium
python -m pip install -r requirements.txt
```

`.env` 範例，先只讓 Scholar 使用 MiniCPM；Citizen 使用 Gemini：

```env
AGENT_CITIZEN_PROVIDER=gemini
AGENT_SCHOLAR_PROVIDER=minicpm

GEMINI_API_KEY=your_gemini_api_key

MINICPM_REALTIME_URL=ws://127.0.0.1:8006/v1/realtime?mode=audio
MINICPM_LENGTH_PENALTY=1.1
MINICPM_INPUT_CHUNK_MS=1000
MINICPM_QUEUE_TIMEOUT=300
# Long evaluation runs can disable Vox-side client keepalive ping after the
# MiniCPM-o-Demo Gateway/Worker ping settings below are also disabled.
MINICPM_PING_INTERVAL=none
MINICPM_PING_TIMEOUT=none
```

若 Gateway 前方的 reverse proxy 驗證 Bearer token：

```env
MINICPM_API_KEY=your_minicpm_api_key
```

第一輪不建議 Citizen、Scholar 同時使用 MiniCPM；兩個同時存在的 full-duplex session
通常需要兩個 Worker。

## 11. Smoke test

使用既有 scenario 跑一回合：

```bash
python -m vox_symposium.evaluation \
  data/scenarios/00000000.json \
  data/results/00000000-minicpm-smoke.json \
  --run-id 00000000-minicpm-smoke \
  --dialogue-turns 1 \
  --audio-speed 1 \
  --idle-timeout 2.5 \
  --text-max-wait 8 \
  --max-utterance-seconds 45
```

成功時會產生：

```text
data/results/00000000-minicpm-smoke.json
data/results/00000000-minicpm-smoke-artifacts/
  dialogue-01-citizen.wav
  dialogue-02-scholar.wav
  scholar-answer.wav
  dialogue-log.json
```

macOS 可播放結果：

```bash
afplay data/results/00000000-minicpm-smoke-artifacts/dialogue-02-scholar.wav
afplay data/results/00000000-minicpm-smoke-artifacts/scholar-answer.wav
```

## 12. Full-Duplex 行為注意事項

- Client 應持續送入 16 kHz mono float32 PCM，包括靜音；不要用 client-side VAD
  刪除所有靜音區段。
- MiniCPM 約在每次輸入 chunk 時計算 listen/speak。停止送 input 也會停止生成後續語音。
- MiniCPM Gateway/Worker 推理期間若無法及時回 WebSocket ping，可能出現
  `keepalive ping timeout`；長批次 evaluation 建議同時關閉 MiniCPM-o-Demo
  Gateway/Worker 的 WebSocket keepalive，並把 Vox 端 `MINICPM_PING_INTERVAL` /
  `MINICPM_PING_TIMEOUT` 設為 `none`。
- Vox Symposium evaluation 在模型說話期間會持續送入即時靜音，直到模型回到 `listen`，
  避免 WAV 說到一半被中斷。
- Vox 會在使用 `provider=minicpm` 或 `provider=freeze_omni` 的那位角色的 `Dialogue behavior` 自動追加短回覆規則：
  `Keep each reply under 2 sentences. Ask at most one question. Do not summarize repeatedly.`
  這是為了降低長回覆造成的 session 時間、turn-taking 和 downstream realtime model
  連線風險；不會套用到 OpenAI、Gemini 或其他 provider 的角色。
- `text` 和 `audio` delta 不保證一一對應。
- 如果 console 或 `dialogue-log.json` 只看到句首，例如 `嗯，这`，但 WAV 播放正常，
  通常是文字 delta 比音訊晚到；先調大 `--text-max-wait` 或 `--text-idle-timeout`。
- Audio Full-Duplex 不使用 `response.done` 作為每一個口語回合的結束事件。

## 13. 常見錯誤

### `InvalidMessage: did not receive a valid HTTP response`

檢查：

- `ws://` 與 `wss://` 是否和 Gateway 的 HTTP/TLS 模式一致。
- URL 是否錯用 `0.0.0.0`。
- SSH tunnel 後方的遠端 8006 是否真的有 Gateway listener。

### `keepalive ping timeout`

常見錯誤：

```text
ConnectionClosedError: sent 1011 (internal error) keepalive ping timeout; no close frame received
ConnectionClosedError: received 1011 (internal error) keepalive ping timeout; then sent 1011 (internal error) keepalive ping timeout
```

如果只調 Vox Symposium 的 `MINICPM_PING_INTERVAL` / `MINICPM_PING_TIMEOUT`，仍可能
因 MiniCPM-o-Demo 內部 Gateway/Worker 連線使用預設 keepalive 而斷線。長批次測試可先把
MiniCPM-o-Demo 三處 WebSocket keepalive 都關掉：

`gateway.py` 對 Vox client 的 `uvicorn.run(...)`：

```python
uvicorn.run(
    app,
    host=args.host,
    port=port,
    ws_max_size=128 * 1024 * 1024,
    ws_ping_interval=None,
    ws_ping_timeout=None,
    **ssl_kwargs,
)
```

`gateway.py` 連到 Worker 的 `websockets.connect(...)`：

```python
worker_ws = await websockets.connect(
    ws_url,
    open_timeout=5,
    max_size=128 * 1024 * 1024,
    ping_interval=None,
    ping_timeout=None,
)
```

`worker.py` 對 Gateway 的 `uvicorn.run(...)`：

```python
uvicorn.run(
    app,
    host=args.host,
    port=port,
    ws_max_size=128 * 1024 * 1024,
    ws_ping_interval=None,
    ws_ping_timeout=None,
)
```

Vox Symposium `.env` 也設為：

```env
MINICPM_PING_INTERVAL=none
MINICPM_PING_TIMEOUT=none
```

改完後必須重啟 Gateway 和 Worker。若仍出現 `keepalive ping timeout`，再檢查 reverse proxy
或 SSH tunnel 是否有自己的 WebSocket idle timeout。

### `session.closed: reason=backend_error`

代表 Gateway/Worker 已連通，但 Backend 的 `duplex_prepare()` 失敗。查看 Backend terminal 的：

```text
fatal backend session termination ... message=...
```

Worker log 通常只會顯示 `backend init returned unexpected event: session.closed`，不包含真正
Backend exception。

### `TorchCodec is required for load_with_torchcodec`

依第 5 節安裝和目前 PyTorch 相容的 TorchCodec，並重啟 Backend。

### `Could not open input file: assets/ref_audio/...`

確認目前工作目錄是 `MiniCPM-o-Demo` repository 根目錄，或改用絕對路徑。

### 模型一直回 `listen`

- 確認 input 是 16 kHz mono float32 raw PCM，而非 WAV/PCM16。
- 確認仍持續送入音訊或靜音 chunk。
- Evaluation adapter 會加入明確 turn-taking policy；請確認使用目前 MiniCPM branch 的程式。

### WAV 說話到一半被截斷

Full-duplex 模型說話期間仍需要持續 input。請勿使用只追加固定一兩個靜音 chunk 的舊版
adapter；目前 evaluation adapter 會持續送靜音直到收到模型的 `listen`。

## 14. Production 要求

正式環境至少需要：

- TLS/WSS。
- API key 或 JWT 驗證。
- WebSocket connection 與 concurrent session limit。
- 最大 message size、idle timeout 與 session duration。
- Gateway/Worker/Backend 的獨立 health check 與 log retention。
- 固定 Demo repository commit、模型版本、PyTorch、TorchCodec 與 CUDA wheel 版本。
- 防火牆只公開 reverse proxy；不要公開 22400、22500。
- 不要無限制保存原始語音或把 API key 寫入 log。

建議 production topology：

```text
Internet
  → HTTPS/WSS reverse proxy :443
  → Gateway 127.0.0.1:8006
  → Worker 127.0.0.1:22400
  → Backend 127.0.0.1:22500
```
