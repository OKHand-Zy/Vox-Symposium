# Freeze-Omni 文字事件與多回合 VAD patch

Vox Symposium 可以串接官方 VITA-MLLM Freeze-Omni Real-Time Interactive Demo
server。上游 `bin/server.py` 會 emit 合成語音音訊，但不會把生成文字 emit 給
Socket.IO client。此外，上游 `web/parms.py` 的 session reset 只會把 VAD 的
`in_dialog` 狀態設回 `False`，沒有重置 Silero VAD iterator；連續多回合自動評測時，
後續回合可能只看到 `Received PCM data`，但不再觸發 `Vad start`。

這個 patch 會保留官方音訊 protocol 不變，並新增兩個可選的 Socket.IO events：

- `text_delta`：增量生成文字片段。
- `text_done`：完整的最終生成文字，供不想處理 delta 的 client 使用。

Vox Symposium 會透過 `RealtimeAudioModel.receive_text()` 消費 `text_delta`。
如果 patched server 只 emit `text_done`，Vox 也會改用這份最終文字。

## 修改 `web/parms.py`

先讓每次 `recording-started` / `recording-stopped` 後的 `reset()` 都真正重置 VAD。
在 `GlobalParams.reset()` 裡找到：

```python
self.wakeup_and_vad.in_dialog = False
```

替換成：

```python
self.wakeup_and_vad.reset_vad()
```

這會清掉 Silero VAD iterator 的內部狀態，避免第一輪正常、第二輪開始只收到 PCM
但沒有 `Vad start` 的情況。

## 修改 `bin/server.py`

在 `generate(outputs, sid)` 裡，找到 `outputs['stat'] == 'cs'` 區段中的這段：

```python
if "�" in outputs['text'][len(last_text):]:
    continue
connected_users[sid][1].whole_text += outputs['text'][len(last_text):]
cur_text += outputs['text'][len(last_text):]
```

替換成：

```python
delta_text = outputs['text'][len(last_text):]
if "�" in delta_text:
    continue

connected_users[sid][1].whole_text += delta_text
cur_text += delta_text
if delta_text:
    socketio.emit('text_delta', {'text': delta_text}, to=sid)
```

接著在 `generate(outputs, sid)` 接近結尾處、以下這行之前：

```python
connected_users[sid][1].is_generate = False
```

加入：

```python
if connected_users[sid][1].whole_text:
    socketio.emit('text_done', {'text': connected_users[sid][1].whole_text}, to=sid)
```

patch 後的尾段應該會長這樣：

```python
    if not connected_users[sid][1].tts_over:
        if len(cur_hidden_state) != 0:
            generate_num = decoder(cur_hidden_state,
                                   cur_text, outputs,
                                   connected_users,
                                   sid,
                                   generate_num,
                                   last_text,
                                   is_last_chunk=True)
            cur_text = ""
    if connected_users[sid][1].whole_text:
        socketio.emit('text_done', {'text': connected_users[sid][1].whole_text}, to=sid)
    connected_users[sid][1].is_generate = False
```

## Client 行為

音訊 event 維持不變：

```python
emit('audio', output_data.astype(np.int16).tobytes())
```

Vox 仍會把這個 event 視為 24 kHz mono PCM16 audio 接收。套用上面的 patch 後，
Vox 也會從同一個 Socket.IO session 收到生成文字，並把它存進既有的 transcript
路徑。

## 為什麼需要這個 patch

Freeze-Omni 其實已經在 `outputs['text']` 和
`connected_users[sid][1].whole_text` 裡組出生成文字，但官方 demo 只會在 server
端印出文字，並且只把合成音訊 emit 給瀏覽器 client。這個 patch 不改模型 pipeline，
只把既有的文字狀態透過 Socket.IO 暴露出來。
