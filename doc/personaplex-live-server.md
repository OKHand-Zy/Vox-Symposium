# PersonaPlex live server

本文記錄 Vox Symposium 串接 NVIDIA PersonaPlex live server 的部署方式、`.env` 設定、prompt 行為，以及目前已確認的 server patch。

## Protocol

PersonaPlex live server 使用 Moshi `/api/chat` 二進位 WebSocket protocol，不使用 Vox PCM JSON gateway。

Vox 端設定：

```env
AGENT_ROBOT_PROVIDER=personaplex
PERSONAPLEX_REALTIME_URL=ws://127.0.0.1:8998/api/chat?voice_prompt=NATF2.pt
```

`PERSONAPLEX_REALTIME_URL` 可以只填 `ws://127.0.0.1:8998?voice_prompt=NATF2.pt`，Vox 會自動補 `/api/chat`。

固定 voice prompt 放在 URL query；每個 scenario/case 的 Robot instructions 會由 Vox 自動 URL encode 後寫入 `text_prompt`，不要在 `.env` 寫死角色 prompt。

## Server deployment

若模型、tokenizer、Mimi checkpoint 和 voices 已經下載到 `./model/`：

```bash
python -m moshi.server \
  --host 127.0.0.1 \
  --port 8998 \
  --static none \
  --moshi-weight ./model/model.safetensors \
  --mimi-weight ./model/tokenizer-e351c8d8-checkpoint125.safetensors \
  --tokenizer ./model/tokenizer_spm_32k_3.model \
  --voice-prompt-dir ./model/voices
```

`voices.tgz` 需要先解壓縮，因為 `--voice-prompt-dir` 要指向包含 `NATF2.pt` 等 voice prompt 檔案的目錄。

```bash
tar -xzf voices.tgz
```

`dist.tgz` 是 Web UI 靜態檔。若 Vox 只連 `/api/chat`，可以不用解壓縮，server 啟動時保留 `--static none`。

## Vox prompt behavior

Vox 對所有 provider 都使用同一套 scenario prompt builder。PersonaPlex 使用 Moshi `/api/chat`，所以這份 instructions 會寫入 `text_prompt`；OpenAI、Gemini、MiniCPM 和 PCM gateway 則會放到各自的 system/session instructions 欄位。

目前 prompt 包含：

- self role/name/profile
- conversation partner profile
- scene type/subtype/topic/goal
- prior conversation history

opening 的處理依 agent 分開，這是所有 provider 的共通規則：

- `agent-human` prompt 只到 history 最後 -1 句，因為最後一句 opening 會以音訊送進 human。
- `agent-robot` prompt 包含完整 history 到最後 1 句，因為那句是 robot 已經說過的 opening，robot 不會再透過音訊收到自己的話。

Vox 不會把 evaluation/runtime meta prompt 寫進 provider instructions，例如「你是被評估模型」、「opening reserved for playback」、「evaluation question」等。

## Handshake

PersonaPlex 會先處理 voice prompt、`text_prompt` 和 audio silence，完成後送出 Moshi handshake bytes。server log 會類似：

```text
Done loading audio silence.
Done loading text prompt.
Done loading audio silence.
[xxxx] done with system prompts
[xxxx] sent handshake bytes
```

Vox 的 Moshi adapter 會等收到 handshake 後才讓 `connect()` 完成，避免在 server 還在 prefill 長 `text_prompt` 時送入 audio frame。

## Server patch

目前確認 NVIDIA PersonaPlex 所帶的 `moshi.server` 在 idle 或還沒有可讀 Opus audio frame 時，`opus_reader.read_pcm()` 可能回傳 `None`。原版 `server.py` 會直接讀 `pcm.shape`，因此可能出現：

```text
AttributeError: 'NoneType' object has no attribute 'shape'
```

請 patch PersonaPlex 環境中的 `moshi/server.py`。常見位置如下，實際路徑依 conda/env 而定：

```text
/root/miniconda3/envs/personaplex/lib/python3.10/site-packages/moshi/server.py
```

找到 `handle_chat()` 內 `opus_loop()` 的這段原始碼：

```python
while True:
    if close:
        return
    await asyncio.sleep(0.001)
    pcm = opus_reader.read_pcm()
    if pcm.shape[-1] == 0:
        continue
    if all_pcm_data is None:
        all_pcm_data = pcm
    else:
        all_pcm_data = np.concatenate((all_pcm_data, pcm))
```

改成先處理 `None`：

```python
while True:
    if close:
        return
    await asyncio.sleep(0.001)
    pcm = opus_reader.read_pcm()
    if pcm is None or pcm.shape[-1] == 0:
        continue
    if all_pcm_data is None:
        all_pcm_data = pcm
    else:
        all_pcm_data = np.concatenate((all_pcm_data, pcm))
```

改完後重啟 PersonaPlex server。這是 server-side 防呆，不會改變正常 audio frame 的處理，只是讓 server 在暫時沒有 input audio 時繼續等待。

## Troubleshooting

### `KeyError: 'voice_prompt'`

`PERSONAPLEX_REALTIME_URL` 少了 `voice_prompt` query。請指定 voices 目錄中實際存在的檔名：

```env
PERSONAPLEX_REALTIME_URL=ws://127.0.0.1:8998/api/chat?voice_prompt=NATF2.pt
```

### `KeyError: 'text_prompt'`

Vox 會自動把 scenario instructions 寫入 `text_prompt`。如果你直接用其他 client 連 PersonaPlex，URL 也需要提供 `text_prompt`，即使是空字串。

### `ValueError: sending on a closed channel`

這通常出現在 server 還沒完成 long `text_prompt` prefill，client 就送入 audio frame，或 server 端 channel 已因前一個錯誤關閉。Vox 端已等待 Moshi handshake 後才開始 evaluation 音訊流程；若仍發生，請確認 server 已套用上方 `pcm is None` patch 並完整重啟。

### 長 prompt 導致 prefill 很久

PersonaPlex 對長 `text_prompt` 的 prefill 可能明顯慢於短 prompt。若要排查穩定性，可以先用短 prompt 或縮短 role profile，只保留身份、scene 和完整 conversation history。
