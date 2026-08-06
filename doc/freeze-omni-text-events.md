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

## Vox `.env` 參數

以下是 Vox Symposium 端支援的 `FREEZE_OMNI_*` 設定。表格中的值是目前程式預設值，
也是在 `two_test_dataset` 前 3 筆、每段最多 90 秒的 smoke test 中可完整跑完的建議值。
turn 前靜音、輸入靜音壓縮與 post-turn silence polling 會在 evaluation runner 的
固定回合模式中啟用。

| 參數 | 預設/建議值 | 說明 |
| --- | --- | --- |
| `FREEZE_OMNI_REALTIME_URL` | 無 | Freeze-Omni Real-Time Interactive Flask-SocketIO server 的網址。使用 `provider=freeze_omni` 時必填。若 server 跑在遠端機器，請填 tunnel 或可連線的 host。 |
| `FREEZE_OMNI_SSL_VERIFY` | `false` | 是否驗證 HTTPS 憑證。官方 demo 常用自簽憑證，所以本專案預設關閉驗證。若你改用正式憑證，才設為 `true`。 |
| `FREEZE_OMNI_INPUT_CHUNK_MS` | `20` | Vox 送入 Freeze-Omni 的 audio packet 長度。20 ms 比較貼近 realtime demo 的互動節奏，不建議大幅調高。 |
| `FREEZE_OMNI_TURN_START_DELAY` | `3` | evaluation 每輪送 `recording-started` 後，等待幾秒才開始送音訊。遠端 server 或 GPU reset 慢時可調高，避免音訊太早進 server。 |
| `FREEZE_OMNI_TURN_PREROLL_SILENCE_MS` | `1200` | evaluation 每輪正式語音前先送的靜音長度。用來讓官方 server 的錄音/VAD 狀態進入穩定狀態。若第二輪只看到 PCM 但沒有 `Vad start`，可調高。 |
| `FREEZE_OMNI_MAX_INPUT_SILENCE_MS` | `200` | evaluation 在使用者語音中段最多保留多少低能量靜音。設為 `0` 代表不壓縮靜音。太低可能讓 Freeze-Omni 判成 `Detect invalid break`；太高則可能讓 turn endpoint 變慢。 |
| `FREEZE_OMNI_INPUT_SILENCE_RMS_THRESHOLD` | `800` | evaluation 判斷 input chunk 是否為靜音的 PCM16 RMS 門檻。低於此值會被視為低能量靜音，並受 `FREEZE_OMNI_MAX_INPUT_SILENCE_MS` 限制。門檻太高可能切掉正常語音尾音。 |
| `FREEZE_OMNI_CONNECT_TIMEOUT` | `60` | Socket.IO 連線 timeout 秒數。Freeze-Omni 載入模型或遠端 tunnel 較慢時可調高。 |
| `FREEZE_OMNI_PROMPT_TIMEOUT` | `60` | 送出 prompt 後等待 server 回 `prompt_success` 的 timeout 秒數。server 忙碌或網路慢時可調高。 |
| `FREEZE_OMNI_CONNECT_RETRIES` | `10` | 若 server 回 `too_many_users`，Vox 會重試連線的次數。`--max_users 1` 且連續跑多個 case 時很有用。 |
| `FREEZE_OMNI_CONNECT_RETRY_DELAY` | `5` | 每次連線重試之間等待幾秒。若 server 釋放 session 很慢，可調高。 |
| `FREEZE_OMNI_POST_TURN_POLL_SECONDS` | `60` | evaluation 結束送入使用者語音後，最多繼續送靜音 polling 幾秒。官方 server 需要 client 持續送 audio event 才會 flush TTS audio。 |
| `FREEZE_OMNI_POST_TURN_IDLE_SECONDS` | `3` | evaluation 已收到 Freeze-Omni 輸出音訊後，若連續幾秒沒有新音訊，就視為該輪輸出結束。模型輸出間隔長時可調高。 |
| `FREEZE_OMNI_POST_TURN_POLL_CHUNK_MS` | `160` | evaluation post-turn silence polling 的每個靜音 chunk 長度。太大會降低 flush 反應速度，太小會增加 Socket.IO event 數量。 |
| `FREEZE_OMNI_STOP_RECORDING_AFTER_TURN` | `true` | evaluation post-turn polling 結束後是否送 `recording-stopped`。一般評測建議維持 `true`，讓官方 server 每輪重置狀態。 |

目前建議 `.env` 範例如下：

```env
FREEZE_OMNI_REALTIME_URL=https://127.0.0.1:8081
FREEZE_OMNI_SSL_VERIFY=false
FREEZE_OMNI_INPUT_CHUNK_MS=20
FREEZE_OMNI_TURN_START_DELAY=3
FREEZE_OMNI_TURN_PREROLL_SILENCE_MS=1200
FREEZE_OMNI_MAX_INPUT_SILENCE_MS=200
FREEZE_OMNI_INPUT_SILENCE_RMS_THRESHOLD=800

FREEZE_OMNI_CONNECT_TIMEOUT=60
FREEZE_OMNI_PROMPT_TIMEOUT=60
FREEZE_OMNI_CONNECT_RETRIES=10
FREEZE_OMNI_CONNECT_RETRY_DELAY=5

FREEZE_OMNI_POST_TURN_POLL_SECONDS=60
FREEZE_OMNI_POST_TURN_IDLE_SECONDS=3
FREEZE_OMNI_POST_TURN_POLL_CHUNK_MS=160
FREEZE_OMNI_STOP_RECORDING_AFTER_TURN=true
```

若 server log 長時間只有 `Received PCM data`、沒有 `Vad start`，優先調高
`FREEZE_OMNI_TURN_START_DELAY` 或 `FREEZE_OMNI_TURN_PREROLL_SILENCE_MS`。若看到
`Detect invalid break` 後沒有 `Detect break`，優先放寬靜音壓縮，例如提高
`FREEZE_OMNI_MAX_INPUT_SILENCE_MS` 或降低 `FREEZE_OMNI_INPUT_SILENCE_RMS_THRESHOLD`。
若已經看到 `Detect break` 和 `Synthesis`，但 Vox 等不到音訊，則檢查
`FREEZE_OMNI_POST_TURN_POLL_SECONDS` 是否太短，以及 server 端 `handle_audio()` 是否仍會
在 generation 期間先 flush `tts_data`。

## Server patch 總覽

官方 Freeze-Omni demo server 需要三類小改動，才能穩定支援 Vox Symposium 的多回合
自動評測：

1. 每輪 reset 時真正清掉 VAD iterator。
2. endpoint 偵測後清掉殘留 PCM，並在生成期間停止接收新的 input audio。
3. 把 server 已經生成的文字透過 Socket.IO emit 給 client。

### 1. 修改 `web/parms.py`

在 `GlobalParams.reset()` 裡找到：

```python
self.wakeup_and_vad.in_dialog = False
```

替換成：

```python
self.wakeup_and_vad.reset_vad()
```

這會清掉 Silero VAD iterator 的內部狀態，避免第一輪正常、第二輪開始只收到
`Received PCM data`，但不再觸發 `Vad start`。

### 2. 修改 `bin/server.py`：turn endpoint

在處理 `outputs['stat']` 的區段，把 `el` 和 `ss` 分支整理成下面這樣：

```python
if outputs['stat'] == 'el':
    connected_users[sid][1].wakeup_and_vad.reset_vad()
    connected_users[sid][1].pcm_fifo_queue.clear()
    print("Sid: ", sid, " Detect invalid break")

if outputs['stat'] == 'ss':
    connected_users[sid][1].interrupt()
    print("Sid: ", sid, " Detect break")
    connected_users[sid][1].wakeup_and_vad.reset_vad()
    connected_users[sid][1].pcm_fifo_queue.clear()
    connected_users[sid][1].is_generate = True
    socketio.emit('input_audio_stopped', to=sid)
    generate_thread = threading.Thread(target=generate, args=(deepcopy(outputs), sid))
    generate_thread.start()
```

重點：

- `reset_vad()`：取代原本只改 `in_dialog = False` 的做法。
- `pcm_fifo_queue.clear()`：清掉 endpoint 前後殘留的 client audio，避免同一輪尾音被當成下一輪。
- `is_generate = True` 必須放在 `generate_thread.start()` 前，避免 race condition。
- `input_audio_stopped` 只在 `ss` 有效 endpoint emit；`el` invalid break 不要 emit，否則 client 會太早停止送原始輸入。

### 3. 修改 `bin/server.py`：`handle_audio()`

`handle_audio()` 要保留原本 flush TTS 的邏輯，但在解析新 input audio 前擋掉 generation
期間的輸入。完整結構如下：

```python
def handle_audio(data):
    sid = request.sid
    if sid in connected_users:
        if not connected_users[sid][1].tts_data.is_empty():
            connected_users[sid][0].cancel()
            connected_users[sid][0] = Timer(TIMEOUT, disconnect_user, [sid])
            connected_users[sid][0].start()
            output_data = connected_users[sid][1].tts_data.get()
            if output_data is not None:
                print("Sid: ", sid, "Send TTS data")
                emit('audio', output_data.astype(np.int16).tobytes())

        if connected_users[sid][1].tts_over_time > 0:
            socketio.emit('stop_tts', to=sid)
            connected_users[sid][1].tts_over_time = 0

        if connected_users[sid][1].is_generate:
            return

        data = json.loads(data)
        audio_data = np.frombuffer(bytes(data['audio']), dtype=np.int16)
        sample_rate = data['sample_rate']

        connected_users[sid][1].pcm_fifo_queue.put(audio_data.astype(np.float32) / 32768.0)

    else:
        disconnect()
```

`is_generate` 檢查要放在 TTS flush 之後、`json.loads(data)` 之前。這樣 generation 期間
server 仍可透過 `audio` event 把 TTS 音訊送回 Vox，但不會把 client 後續 audio packet
塞進 `pcm_fifo_queue`。

### 4. 修改 `bin/server.py`：文字事件

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
