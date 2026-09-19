# VRCRealtime セットアップ手順書

VRCRealtime は OpenAI の音声モデル（GPT-Live / Realtime）と VRChat をつなぐノンブロッキングな音声ブリッジパッケージです。VRChat 画面のキャプチャ・OSC 操作・音声合成などを外部 Python プロセスに委譲し、Wolfram 側は制御と状態監視に専念します。本書ではインストールから動作確認までの手順をまとめます。

macOS/Linux ではパス区切りやシェルコマンドを適宜読み替えてください（本書は Windows 11 を前提とします）。

## 1. 前提環境

- **Wolfram Language / Mathematica**: 13.x 以降を推奨（Association・`RawJSON` インポート・`ProcessObject` 系の非同期実行を多用するため、比較的新しいバージョンが必要です）。
- **OS**: Windows 11（`taskkill` によるプロセス終了、`%TEMP%` 上のロックファイルなど Windows 前提の実装を含みます）。
- **VRChat**: デスクトップクライアントで起動しておく必要があります。
- **VRC Pilot**: VRChat へのマウス/キーボード/OSC 入力を担当する外部プロジェクト。既定パスは `C:\Users\<ユーザー名>\gitwork\vrcpilot`（オプション `"VRCpilotDirectory"`）。
- **uv**: Python の実行・依存解決に使用します（`VRCRealtime_worker.py` / `VRCRealtime_worldjoin.py` を `uv run` で起動）。`uv.exe` は `PATH`、`%USERPROFILE%\.local\bin`、`%LOCALAPPDATA%\Programs\uv` の順に自動探索されます。見つからない場合は `"UVExecutable"` オプションで明示してください。
- **仮想オーディオデバイス**: VRChat のマイク入力へ合成音声を流し込むため、VB-Audio Virtual Cable などの仮想デバイスを導入し、出力デバイス名を `"CABLE-A Input"`（既定値）に合わせて VRChat のマイク入力に設定してください。
- **VRChat OSC**: VRChat 側で OSC を有効化してください（既定 `"OSCHost" -> "127.0.0.1"`, `"OSCPort" -> 9000`）。

## 2. 依存パッケージ

- **NBAccess.wl（必須）**: OpenAI の API キー取得・アクセス許可判定に使用します。`$packageDirectory` 直下に配置してください。無いと `VRCRealtimeStart[]` は `"NBAccessUnavailable"` で失敗します。
- **SourceVault_voice.wl（任意）**: ローカル音声合成（Piper / AivisSpeech）ランタイムと声モデルを管轄します。解決した声が AivisSpeech のものであれば、エンジンの応答確認が取れた場合に限り AivisSpeech エンジンが自動採用され、応答しなければ Piper の声へ自動的に切り替わります（どちらかが動けば読み上げは継続します）。無い場合は `VRCRealtime` が旧配置（`SourceVault_voice/` または `VRCRealtime_tts/` 配下）を自前で探索するフォールバックが働きますが、資産が見つからなければローカル読み上げ機能（非公開資料の音声応答）が無効になります。
- **SourceVault.wl（任意）**: 発表登録簿（スライド URL・発表原稿）と資料検索（`sourcevault_lookup`）を MCP サービス越しに提供します。無くても `VRCRealtimeStartPresentation` に URL を直接渡せば動作しますが、タイトル指定や原稿付き発表はできません。SourceVault MCP サービスが停止している場合は `VRCRealtimeStartPresentation` が起動を試み、保守 tick の実行中で command queue が応答しない場合は数分待ってから再試行が必要です（詳細は「8. トラブルシューティング」参照）。

## 3. インストール手順

1. 本リポジトリをクローンし、`.wl` ファイル一式を `$packageDirectory` 直下に配置します（サブディレクトリに分けないこと）。
   `git clone https://github.com/transreal/VRCRealtime.git`
2. 外部実装（Python）は配布形では `VRCRealtime_info/resources/python/` にまとまっています。旧配置（`$packageDirectory` 直下に `VRCRealtime_worker.py` 等を直接置く形）もそのまま動作しますが、新しい配置が優先されます。
3. VRC Pilot を別途クローンし、`"VRCpilotDirectory"` オプションのパス（既定 `C:\Users\<ユーザー名>\gitwork\vrcpilot`）に配置します。
4. `uv` をインストールします（`winget install astral-sh.uv` など）。ワーカースクリプトの Python 依存関係は `uv run` 実行時に自動解決されます。
5. VB-Audio Virtual Cable を導入し、Windows のサウンド設定と VRChat のマイク入力デバイスを `"CABLE-A Input"` に合わせます。異なる名前を使う場合は `"OutputDevice"` オプションで指定してください。
6. VRChat の設定で OSC を有効化します。
7. （音声コマンドを使う場合）`VRCRealtimeInstallVoiceCommandModel[]` を実行し、Vosk 日本語小モデル（約 50 MB）を事前ダウンロードします。導入先は既定で `SourceVault_voice/asr`。`VRCRealtimeStart[]` が未導入時に自動取得するため必須ではありません。

## 4. APIキー設定

VRCRealtime は API キーを直接扱わず、`NBAccess` 経由で OpenAI キーを取得します。

```wolfram
Block[{$CharacterEncoding = "UTF-8"},
  Needs["NBAccess`", "NBAccess.wl"]];

(* OpenAI API キーをシステムクレデンシャルとして登録 *)
NBAccess`NBSetAPIKey["openai", "sk-..."];
```

有料 API 呼び出しをノートブック単位で許可制にしたい場合（`"RequirePaidAPIApproval" -> True` を指定したとき）は、事前に対象ノートブックで以下を評価しておく必要があります。

```wolfram
NBAccess`NBSetNotebookPaidAPIAllowed[EvaluationNotebook[], True];
```

キーが見つからない、またはプロバイダポリシーで OpenAI アクセスが許可されていない場合、`VRCRealtimeStart[]` はそれぞれ `"APIKeyMissing"` / `"OpenAIDisabled"` の `Failure` を返します。

## 5. パッケージ読込

```wolfram
Block[{$CharacterEncoding = "UTF-8"},
  Needs["VRCRealtime`", "VRCRealtime.wl"]];
```

`$packageDirectory` が `$Path` に含まれていれば、ファイル名だけの相対指定（`"VRCRealtime.wl"`）で読み込めます。claudecode を利用している場合は `$Path` が自動設定済みです。手動設定する場合は次のようにパッケージディレクトリそのものを追加してください（サブディレクトリではなくディレクトリ本体）。

```wolfram
AppendTo[$Path, $packageDirectory];
```

## 6. 動作確認

最小構成での起動例です（VRC Pilot・VRChat・仮想オーディオデバイスが起動済みであること）。

```wolfram
VRCRealtimeStart[
  "AvatarMode" -> "Standalone",
  "Model" -> "gpt-live-1",   (* 既定。従来の Realtime なら "gpt-realtime-2.1" *)
  "Voice" -> "marin"
]
```

既定のモデルは GPT-Live（`"Model" -> "gpt-live-1"`）です。SourceVault 検索・発表操作・アバター操作・画面を見るなどの道具は Responses 委譲のバックエンドモデル（`"LiveBackendModel"`、既定 Automatic = gpt-5.6-luna）が実行し、GPT-Live がその結果を話します。従来の Realtime 経路に戻すには `"Model" -> "gpt-realtime-2.1"` を指定するか、`"API" -> "Realtime"`（または `"Live"`）で明示できます（既定 `"API" -> Automatic` はモデル名から自動判定）。GPT-Live は既定で全二重（`"FullDuplex" -> True`：発表中以外は話しながら部屋の声も聞く）で動作し、`False` にすると Realtime と同じ半二重（話している間と直後 1.2 秒は聞かない）になります。

起動後、以下で状態を確認します。

```wolfram
VRCRealtimeStatus[]      (* ProcessStatus, セッション情報などを Association で返す *)
VRCRealtimeMonitor[]     (* 1秒ごとに更新される軽量ダッシュボード *)
```

`"ProcessStatus" -> "Running"` であれば正常に起動しています。ローカル音声資産（合成ランタイム・声モデル・音声認識モデル）の導入状態は次で確認できます。

```wolfram
VRCRealtimeVoiceStatus[]
```

`"Status"` が `"OK"` でなければ `"Hint"` に導入手順が入ります。採用されたエンジン（Piper / AivisSpeech）は `"SelectedEngine"` で確認できます。

確認後は正常終了させます。

```wolfram
VRCRealtimeStop[]                    (* 既定で最大8秒待って正常終了 *)
VRCRealtimeKill[]                    (* 応答しない場合の強制終了 *)
```

オーナー追尾モードを試す場合は次のように起動します。

```wolfram
VRCRealtimeStart[
  "AvatarMode" -> "OwnerFollow",
  "OwnerAccount" -> "your-vrchat-account",
  "OwnerFollowEngine" -> "Formation",
  "OwnerFollowDistanceMeters" -> 1.8
]
```

発表機能を使う場合（SourceVault の発表登録簿が必要）は次を確認します。

```wolfram
VRCRealtimeListPresentations[]
VRCRealtimeStartPresentation["https://example.com/slides.mp4"]
```

発表中は `VRCRealtimePresentationControl["Pause"|"Resume"|"Stop"|"ResumeHere"|"ResumeBack"]` で制御できます。質疑応答中にスライドを動かした場合、発表側が「表示中のスライドから続けるか、元の位置へ戻るか」を尋ねてくるので、`"ResumeHere"`（表示中のスライドから続行）または `"ResumeBack"`（発表が進んでいた位置へ戻る）で応答してください。スライドの区切りで開く質疑応答の待ち時間は `VRCRealtimeStartPresentation` の `"QuestionSeconds"`（既定 5 秒）、全スライド終了後の質疑応答の上限は `"FinalQuestionSeconds"`（既定 120 秒）で調整できます。

発表中の質問の受け方は `"QuestionMode"` で切り替えられます（既定 `"QuestionTime"`）。

- `"QuestionTime"`: スライドの区切りでだけ質問を受け付けます。
- `"Keywords"`: それに加えて、発表中でも「質問」「スライド」「プレゼン」「発表」など（`"InterruptWords"`、既定 Automatic）を含む発話で中断して質問を受け、終われば中断した文から再開します（「発表を中断して」で一時停止も可能）。ローカル Vosk 認識が必要です。
- `"Detect"`: 区切りに加えて、発表中に声が続いたら一旦止め、モデルが自分への質問かどうかを判断して応じます（違えば発表を続けます）。

実行中に切り替えるには次を使うか、音声で「質問モードを切り替えたい」と言えばモデルが選択肢を説明して会話で設定します。

```wolfram
VRCRealtimeSetQuestionMode["QuestionTime"|"Keywords"|"Detect"]
VRCRealtimeQuestionMode[]   (* 現在のモードを確認 *)
```

`VRCRealtimeStartPresentation` の `"QuestionMode"` オプション（既定 Automatic = ワーカーの現在値のまま）で、この発表だけモードを変えることもできます。

## 7. 主な設定オプション（抜粋）

- `"UVExecutable"`, `"VRCpilotDirectory"`: 外部ツールの所在。
- `"Model"`, `"API"`, `"LiveBackendModel"`: 使用するモデルと API 経路。`"API"` は Automatic（モデル名で判定）/ `"Realtime"` / `"Live"`。`"LiveBackendModel"` は GPT-Live 経由時に道具を実行するバックエンドモデル。
- `"FullDuplex"`: GPT-Live のみ有効。既定 True（発表中以外は話しながら聞く全二重）、False で Realtime と同じ半二重。
- `"QuestionMode"`, `"InterruptWords"`: 発表中の質問の受け方（`"QuestionTime"` / `"Keywords"` / `"Detect"`）と、`"Keywords"` モードで割り込みに使う語。
- `"InputSource"`, `"InputDevice"`, `"OutputDevice"`: 音声入出力デバイス。
- `"OSCHost"`, `"OSCPort"`: VRChat OSC 接続先。
- `"VisualAttentionGate"`, `"VisualGateMinFaceFraction"` など: 視覚ゲート（正面から話しかけられたときのみ応答）の調整。
- `"EnableSourceVault"`, `"SourceVaultServiceDirectory"`, `"SourceVaultKinds"`: 資料検索連携。
- `"PrivateTTS"`, `"SpeechVoice"`, `"PiperExecutable"`: ローカル読み上げ（非公開資料用）。声の解決結果に応じて Piper / AivisSpeech のいずれのエンジンを使うかは自動選択されます。
- `"VoiceCommands"`, `"VoiceCommandModelDirectory"`: 音声コマンド認識（Vosk）。

## 8. トラブルシューティング

- `"UVNotFound"`: `uv.exe` の場所を `"UVExecutable"` で明示するか、`PATH` に追加してください。
- `"VRCpilotDirectoryMissing"`: VRC Pilot プロジェクトのパスを確認してください。
- `"NBAccessUnavailable"` / `"APIKeyMissing"`: `NBAccess.wl` の配置と API キー登録を見直してください。
- ワーカーが二重起動している疑いがある場合は、`VRCRealtimeStatus[]` の `AdoptedWorkerPID` を確認し、`VRCRealtimeKill[]` で停止してください。
- SourceVault MCP サービスが停止していると発表原稿の取得が timeout するため、`VRCRealtimeStart::mcpstopped` の警告が出た場合はサービス起動を待つか `SourceVaultMCPStatus[]` を確認してください。
- SourceVault MCP サービスが動作していても、保守 tick の実行中は command queue が一時的に応答しなくなることがあります（`VRCRealtimeStart::mcpbusy` の警告）。この間は発表登録簿を引けず発表を開始できないため、数分おいてから再度お試しください。