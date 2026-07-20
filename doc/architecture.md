# 專案架構

Vox Symposium 把「即時模型連線」、「LiveKit 音訊路由」與「資料集評測」分成三條可獨立維護的路徑。provider adapter 只處理各家協定；上層流程統一使用 PCM16 音訊與 `RealtimeAudioModel` 介面。

## 執行路徑

一般 LiveKit 對話：

```text
main.py
  -> config.py 載入 LiveKit、角色與 provider 設定
  -> models/factory.py 建立 provider adapter
  -> livekit_participant.py 雙向轉送 LiveKit track 與 model audio
  -> recording.py 保存模型輸出與 conversation log
```

自動評測：

```text
evaluation.py
  -> scenario.py 載入／正規化資料並建立角色 prompt
  -> models/factory.py 建立 evaluation-mode adapter
  -> evaluation_audio.py 串流 opening、對話與題目音訊
  -> evaluation_artifacts.py 管理 retry、summary、console log 與環境快照
  -> json_io.py 原子寫入 result、summary 與 dialogue log
```

## 模組責任

| 模組 | 責任 |
| --- | --- |
| `audio.py` | PCM16 channel conversion、resample、frame 切分、音訊片段合併。 |
| `config.py` | `.env` 載入、值驗證，以及各 provider 的型別化設定。 |
| `providers.py` | provider 名稱、alias、顯示名稱與環境變數 prefix。 |
| `scenario.py` | dataset normalization、角色 prompt、structured history、答案抽取。 |
| `models/base.py` | 所有 adapter 共用的 async 介面、輸出 queue 與 task 清理。 |
| `models/factory.py` | 依 provider 選擇設定與 adapter；LiveKit 與 evaluation 共用建構邏輯。 |
| `livekit_participant.py` | LiveKit 訂閱、發布、音訊轉送與一般對話錄音。 |
| `evaluation.py` | 評測案例的高階編排、回合控制與 retry 流程。 |
| `evaluation_audio.py` | WAV/MP3/TTS 處理、音訊注入與 response 收集。 |
| `evaluation_artifacts.py` | artifacts 路徑、續跑結果、summary、console log、非敏感環境快照。 |
| `recording.py` | WAV 與 conversation event 寫入。 |
| `json_io.py` | 共用 UTF-8 JSON 讀取與原子寫入。 |

## 設定生命週期

`load_settings()` 只載入兩個角色實際使用到的 provider 設定。`Settings` 內的 `openai`、`gemini`、`minicpm`、`freeze_omni` 因此可能是 `None`；factory 在建立該 provider 時會再次確認設定存在。

每組 provider 設定都由自己的 loader 建立，例如 `load_openai_settings()` 與 `load_gemini_settings()`。evaluation 不需要 LiveKit credential，因此直接載入指定 provider 的設定；一般 participant 則使用完整 `Settings`。兩條路徑最後都會進入相同的 adapter 建構函式。

Scenario prompt 由 `LoadedScenario.build_prompt()` 統一建立：

- Gemini 3 Live 使用 structured initial history，歷史不再重複塞進 system instruction。
- 其他 provider 把歷史保留在 instruction 文字中。
- MiniCPM 與 Freeze-Omni 的短回覆限制也在這一層加入，LiveKit 與 evaluation 不會各自複製規則。

## 音訊契約

adapter 對上層一律收送 `PcmAudio`。`PcmAudio` 必須帶有實際的 sample rate 與 channel count；串接多個 chunk 時格式必須一致。`rechunk_pcm16()` 會把 channel count 納入 frame byte 數，因此 mono 與 stereo 都維持正確的 frame 時長。

各 adapter 負責把輸入正規化成模型協定要求的格式。LiveKit 發布前則統一轉成設定的 publish sample rate 與 mono。

## 新增 provider

新增 provider 時應依序修改：

1. 在 `providers.py` 註冊 canonical name、必要 alias、label 與 env prefix。
2. 在 `config.py` 新增 provider settings dataclass 與 loader，集中所有預設值與驗證。
3. 在 `models/` 新增 `RealtimeAudioModel` adapter。
4. 在 `models/factory.py` 的兩個入口加入 dispatch，並共用同一個 adapter 建構 helper。
5. 在 `.env.example`、README 與 provider 專屬文件加入設定。
6. 為 URL、設定解析、protocol payload 與關閉流程增加單元測試。

更完整的本地模型範例見 [使用自己的本地 Hugging Face 即時語音模型](local-hf-realtime-model.md)。
