# Freeze-Omni text event patch

Vox Symposium can connect to the official VITA-MLLM Freeze-Omni Real-Time
Interactive Demo server. The upstream `bin/server.py` emits synthesized speech
audio, but it does not emit generated text to the Socket.IO client.

This patch keeps the official audio protocol unchanged and adds two optional
Socket.IO events:

- `text_delta`: incremental generated text chunks.
- `text_done`: final full generated text for clients that do not want deltas.

Vox Symposium consumes `text_delta` through `RealtimeAudioModel.receive_text()`.
If a patched server only emits `text_done`, Vox will use that final text instead.

## Patch `bin/server.py`

Inside `generate(outputs, sid)`, find this block in the `outputs['stat'] == 'cs'`
section:

```python
if "�" in outputs['text'][len(last_text):]:
    continue
connected_users[sid][1].whole_text += outputs['text'][len(last_text):]
cur_text += outputs['text'][len(last_text):]
```

Replace it with:

```python
delta_text = outputs['text'][len(last_text):]
if "�" in delta_text:
    continue

connected_users[sid][1].whole_text += delta_text
cur_text += delta_text
if delta_text:
    socketio.emit('text_delta', {'text': delta_text}, to=sid)
```

Then near the end of `generate(outputs, sid)`, just before:

```python
connected_users[sid][1].is_generate = False
```

add:

```python
if connected_users[sid][1].whole_text:
    socketio.emit('text_done', {'text': connected_users[sid][1].whole_text}, to=sid)
```

The patched tail should look like:

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

## Client behavior

The audio event remains unchanged:

```python
emit('audio', output_data.astype(np.int16).tobytes())
```

Vox still receives this as 24 kHz mono PCM16 audio. With the patch above, Vox
also receives generated text from the same Socket.IO session and stores it in
the existing transcript path.

## Why this patch is needed

Freeze-Omni already builds generated text in `outputs['text']` and
`connected_users[sid][1].whole_text`, but the official demo only prints text on
the server side and only emits synthesized audio to the browser client. The
patch exposes that existing text state over Socket.IO without changing the model
pipeline.
