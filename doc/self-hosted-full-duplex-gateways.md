# 自架 Full-Duplex Gateway 串接

本文說明 Vox Symposium 目前對 Moshi、PersonaPlex、Covo-Audio-Chat-FD 的串接邊界。

## Provider 與 protocol

| Provider | Wire protocol | 適用情境 |
| --- | --- | --- |
| `moshi` | Kyutai 官方 `/api/chat` 二進位 WebSocket；24 kHz mono Ogg Opus pages | 直接連 Moshi server |
| `personaplex` | Vox PCM JSON gateway；base64 PCM16 mono | PersonaPlex 沒有穩定公開 WebSocket protocol 時，用自架 gateway 包裝模型 |
| `covo_audio_chat_fd` | Vox PCM JSON gateway；base64 PCM16 mono | Covo-Audio-Chat-FD 沒有穩定公開 WebSocket protocol 時，用自架 gateway 包裝模型 |

Moshi adapter 需要額外安裝：

```bash
pip install -e '.[moshi]'
```

## Moshi

啟動 Kyutai Moshi server 後，設定：

```env
AGENT_SCHOLAR_PROVIDER=moshi
MOSHI_REALTIME_URL=ws://127.0.0.1:8998/api/chat
```

`MOSHI_REALTIME_URL` 可以填 `ws://127.0.0.1:8998`，Vox 會自動補 `/api/chat`。Moshi 模型實際選擇在 Moshi server 啟動參數中完成，Vox 端不傳 system prompt，也不改 model。

## PCM JSON Gateway

PersonaPlex 和 Covo-Audio-Chat-FD 的 provider adapter 預期你在模型旁邊提供一個 WebSocket gateway。這個 gateway 負責：

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
    "model": "personaplex",
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

PersonaPlex：

```env
AGENT_CITIZEN_PROVIDER=gemini
AGENT_SCHOLAR_PROVIDER=personaplex
PERSONAPLEX_REALTIME_URL=ws://127.0.0.1:8010/v1/realtime
PERSONAPLEX_MODEL=personaplex
PERSONAPLEX_INPUT_SAMPLE_RATE=24000
PERSONAPLEX_OUTPUT_SAMPLE_RATE=24000
```

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
PERSONAPLEX_API_KEY=your-token
COVO_AUDIO_CHAT_FD_API_KEY=your-token
```

## 設計注意事項

- 正式比較時固定 sample rate、chunk size、audio injection speed 和 VAD policy，否則 latency / interruption 指標不可比。
- Gateway 應在 response event 中保留 sample rate，避免未來支援 16 kHz / 48 kHz model 時混淆。
- 如果模型輸出不是 PCM16，請在 gateway 內轉成 PCM16；不要讓 Vox adapter 直接承擔模型私有 codec。
- Full-duplex 模型常需要持續輸入靜音才能推進狀態，這件事應由 model gateway 或專用 adapter 明確處理，不能隱含在測試 runner 的 timeout 裡。
