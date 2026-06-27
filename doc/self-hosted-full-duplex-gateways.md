# Realtime Speech Provider 串接

本文說明 Vox Symposium 對自架或本地 realtime speech provider 的共通串接邊界。特定模型的部署、已知問題與 patch 請看各 provider 文件。

## Provider 與 protocol

| Provider | Wire protocol | 適用情境 |
| --- | --- | --- |
| `moshi` | Kyutai 官方 `/api/chat` 二進位 WebSocket；24 kHz mono Ogg Opus pages | 直接連 Moshi server |
| `personaplex` | PersonaPlex live server 的 Moshi `/api/chat` 二進位 WebSocket；24 kHz mono Ogg Opus pages | 直接連 PersonaPlex server；部署與 patch 見 [PersonaPlex live server](personaplex-live-server.md) |

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

`MOSHI_REALTIME_URL` 可以填 `ws://127.0.0.1:8998`，Vox 會自動補 `/api/chat`。Moshi 模型實際選擇在 Moshi server 啟動參數中完成；Vox 端會把每個 scenario/case 的 instructions 動態寫入 `text_prompt` query，但不改 server 端載入的 model。

重要限制：Kyutai 官方 Moshi server 目前不支援 OpenAI/Gemini 那種 per-session system prompt / instructions，也不會讀取 `text_prompt` query。因此使用官方 Moshi server 時，角色資料、場景與 prior conversation history 不會真的進入模型 context；這些 prompt 只會在支援 `text_prompt` 的 Moshi-compatible server 上生效，例如 PersonaPlex live server 或自行 patch 的 server。

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

這個規則適用 OpenAI、Gemini、MiniCPM、Moshi 和 PersonaPlex。

## 設計注意事項

- 正式比較時固定 sample rate、chunk size、audio injection speed 和 VAD policy，否則 latency / interruption 指標不可比。
- 如果模型輸出不是 Vox adapter 可直接消費的格式，請在專用 adapter 內明確轉換；不要把模型私有 codec 隱含在 evaluation runner 裡。
- Full-duplex 模型常需要持續輸入靜音才能推進狀態，這件事應由專用 adapter 明確處理，不能隱含在測試 runner 的 timeout 裡。
