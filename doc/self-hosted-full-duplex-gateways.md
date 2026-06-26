# Realtime Speech Provider 串接

本文說明 Vox Symposium 對自架或本地 realtime speech provider 的共通串接邊界。特定模型的部署、已知問題與 patch 請看各 provider 文件。

## Provider 與 protocol

| Provider | Wire protocol | 適用情境 |
| --- | --- | --- |
| `moshi` | Kyutai 官方 `/api/chat` 二進位 WebSocket；24 kHz mono Ogg Opus pages | 直接連 Moshi server |
| `personaplex` | PersonaPlex live server 的 Moshi `/api/chat` 二進位 WebSocket；24 kHz mono Ogg Opus pages | 直接連 PersonaPlex server；部署與 patch 見 [PersonaPlex live server](personaplex-live-server.md) |
| `covo_audio_chat_fd` | Vox PCM JSON gateway；base64 PCM16 mono | Covo-Audio-Chat-FD 沒有穩定公開 WebSocket protocol 時，用自架 gateway 包裝模型 |

Moshi adapter 需要額外安裝：

```bash
pip install -e '.[moshi]'
```

## Moshi protocol providers

`moshi` 和 `personaplex` 都使用 Moshi `/api/chat` 二進位 WebSocket protocol。Vox 會送入/接收 24 kHz mono Opus pages，內部轉回 PCM16。

啟動 Moshi-compatible server 後，設定：

```env
AGENT_SCHOLAR_PROVIDER=moshi
MOSHI_REALTIME_URL=ws://127.0.0.1:8998/api/chat
```

`MOSHI_REALTIME_URL` 可以填 `ws://127.0.0.1:8998`，Vox 會自動補 `/api/chat`。Moshi 模型實際選擇在 Moshi server 啟動參數中完成；Vox 會把每個 scenario/case 的 instructions 動態寫入 `text_prompt` query，但不改 server 端載入的 model。

PersonaPlex 使用同一個 Vox adapter，但需要固定 voice prompt、server patch 與部署注意事項；請看 [PersonaPlex live server](personaplex-live-server.md)。

Vox 的 Moshi adapter 會等待 server handshake bytes 後才讓 evaluation 開始送音訊，避免在 Moshi-compatible server 還在 prefill 長 `text_prompt` 時送入 audio frame。

## Provider-specific docs

- [PersonaPlex live server](personaplex-live-server.md)：部署、voice prompt、`text_prompt`、handshake、server patch 與已知錯誤。
- [MiniCPM-o 4.5 deployment](minicpm-o-4_5-deployment.md)：MiniCPM-o 4.5 Audio Full-Duplex Gateway 非 Docker 部署。
- [Local Hugging Face realtime model](local-hf-realtime-model.md)：新增其他本地 Hugging Face realtime model adapter 的建議。

## Scenario prompt behavior

Vox 對所有 provider 使用同一套 scenario prompt builder。prompt 包含角色資料、對話對象、場景與 prior conversation history，不包含 evaluation/runtime meta prompt。

opening 的處理依 agent 分開：

- opening speaker 的 prompt 會包含完整 history 到最後 1 句，因為這句是該 agent 已經說過的內容。
- opening receiver 的 prompt 只到 history 最後 -1 句，因為最後一句 opening 會以音訊送進 receiver。

這個規則適用 OpenAI、Gemini、MiniCPM、Moshi、PersonaPlex 和 PCM gateway providers。

## PCM JSON Gateway

Covo-Audio-Chat-FD 的 provider adapter 預期你在模型旁邊提供一個 WebSocket gateway。這個 gateway 負責：

- 載入模型權重與 tokenizer / codec。
- 把 Vox 傳來的 PCM16 bytes 轉成模型需要的 streaming feature/token。
- 把模型輸出的音訊轉回 PCM16 bytes。
- 做 GPU 排隊、session lifecycle、重啟與模型端 logging。

Vox 只負責即時音訊路由、錄音、evaluation orchestration 和結果保存。

### Client -> Gateway

連線後 Vox 先送：

```json
{
  "type": "session.init",
  "session": {
    "model": "covo_audio_chat_fd",
    "instructions": "role/system prompt",
    "output_modalities": ["audio"],
    "audio": {
      "input": {"format": "pcm16", "sample_rate": 24000, "channels": 1},
      "output": {"format": "pcm16", "sample_rate": 24000, "channels": 1}
    },
    "turn_detection": {"type": "server_vad"}
  }
}
```

每個 input chunk：

```json
{
  "type": "input_audio_buffer.append",
  "audio": "<base64 little-endian PCM16 mono>",
  "format": "pcm16",
  "sample_rate": 24000,
  "channels": 1
}
```

自動 evaluation 會額外送：

```json
{"type": "input_audio_buffer.commit"}
{"type": "response.create"}
```

如果你的模型是完全 full-duplex autonomous loop，可以忽略這兩個事件；如果它需要明確 turn boundary，就用它們觸發生成。

### Gateway -> Client

音訊 delta：

```json
{
  "type": "response.audio.delta",
  "audio": "<base64 little-endian PCM16 mono>",
  "sample_rate": 24000,
  "channels": 1
}
```

可選文字 transcript：

```json
{
  "type": "response.text.delta",
  "text": "A"
}
```

錯誤：

```json
{
  "type": "error",
  "message": "reason"
}
```

Vox 也會容忍幾種常見欄位名稱：`audio`、`delta`、`data`，以及 `payload.audio` / `response.audio` / `output.audio`。但正式 gateway 建議固定使用上面的 schema，方便錄製與除錯。

## `.env` 範例

Covo-Audio-Chat-FD：

```env
AGENT_CITIZEN_PROVIDER=gemini
AGENT_SCHOLAR_PROVIDER=covo_audio_chat_fd
COVO_AUDIO_CHAT_FD_REALTIME_URL=ws://127.0.0.1:8020/v1/realtime
COVO_AUDIO_CHAT_FD_MODEL=covo_audio_chat_fd
COVO_AUDIO_CHAT_FD_INPUT_SAMPLE_RATE=24000
COVO_AUDIO_CHAT_FD_OUTPUT_SAMPLE_RATE=24000
```

`covo` 也可作為 `covo_audio_chat_fd` 的 provider alias；環境變數也支援較短的 `COVO_REALTIME_URL`、`COVO_MODEL`、`COVO_INPUT_SAMPLE_RATE`、`COVO_OUTPUT_SAMPLE_RATE`。

若 gateway 前方有 reverse proxy Bearer token 驗證，設定：

```env
COVO_AUDIO_CHAT_FD_API_KEY=your-token
```

## 設計注意事項

- 正式比較時固定 sample rate、chunk size、audio injection speed 和 VAD policy，否則 latency / interruption 指標不可比。
- Gateway 應在 response event 中保留 sample rate，避免未來支援 16 kHz / 48 kHz model 時混淆。
- 如果模型輸出不是 PCM16，請在 gateway 內轉成 PCM16；不要讓 Vox adapter 直接承擔模型私有 codec。
- Full-duplex 模型常需要持續輸入靜音才能推進狀態，這件事應由 model gateway 或專用 adapter 明確處理，不能隱含在測試 runner 的 timeout 裡。
