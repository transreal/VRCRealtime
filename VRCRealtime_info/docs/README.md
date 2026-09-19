# VRCRealtime

## 設計思想と実装の概要

**VRCRealtime** は、OpenAI の音声モデル (既定 GPT-Live、従来の Realtime API も可) と VRChat をつなぐ、ノンブロッキングな音声ブリッジパッケージです。

VRChat 上のアバターに、OpenAI Realtime によるリアルタイム音声応答をまとわせ、来場者と自然に会話させる、あるいはオーナーに付き従わせる、スライドを自動で発表させる——といった常駐的なふるまいを、Wolfram Language のノートブックから軽く起動・監視できるようにすることが狙いです。

この目的を実現するうえで最も重要な設計判断は、**重い処理をすべて外部の Python プロセスへ委譲する**ことです。VRChat 画面のキャプチャ、OSC によるアバター操作、音声認識・合成、OpenAI Realtime とのソケット通信は、`uv run` で起動される常駐ワーカー（`VRCRealtime_worker.py`）が担当します。Wolfram Language 側の関数はすべて、状態ファイル・制御ファイルを介したメッセージのやり取りに徹し、**即座に返るコマンド**として設計されています。`VRCRealtimeStart[]` 自体がノートブックカーネルをブロックしないため、起動直後から他の評価を続けられますし、`VRCRealtimeStatus[]` で任意のタイミングの状態を取得できます。

この非同期設計には副作用もあります。パッケージを読み直すとカーネル内の `$iProcess` などの内部状態は失われますが、それでもワーカーを見失わないよう、起動時に固定パスのロックファイル（PID・state ファイル・control ファイルへのポインタ）を書く仕組みを備えています。これにより、別のカーネルや読み直し後のカーネルからでも、走っているワーカーを発見・観測・停止（`VRCRealtimeKill[]` など）できます。二重起動によって同じ部屋に二重応答する事故を防ぐため、`VRCRealtimeStart` は既定で走っているワーカーを検出すると安全に停止してから起動し直します（`"ReplaceRunning" -> True`）。

もう一つの柱は、**プライバシーを最優先にした情報の使い分け**です。SourceVault と連携する場合、資料には Privacy Level（PL）が付与されており、非公開度の高い資料は OpenAI へは渡さず、ローカル音声合成（Piper / AivisSpeech、`SourceVault_voice` 管轄）でのみ読み上げます。解決された声が AivisSpeech 由来のものであれば、エンジンの応答を実際に確認したうえで採用し、応答しなければ Piper の声へ自動フォールバックします。しかもそのローカル読み上げも、フォローモードで本人確認済みのオーナーが近くにいる場合に限って許可されます。ローカル音声合成の資産（実行ファイル・声モデル）が見つからない環境では、無理に起動を止めるのではなく、SourceVault 連携自体を自動的に無効化して安全側に倒します。資料検索・発表登録簿の照合は SourceVault MCP サービス越しに行われるため、そのサービスが保守処理中で応答しない場合は起動や発表開始を一時的に見合わせます。

オーナー追尾（`"AvatarMode" -> "OwnerFollow"`）は既定で幾何制御エンジン（`"Formation"`）を使い、ネームプレート OCR による発見 → 体のランドマークによる距離推定 → オーナーの左後方への回り込み → 視野内での停止、という一連の手続きを決定論的に行います。旧来の OSC ボタン操作方式（`"Legacy"`）も互換のため残されています。

自動発表中の質疑応答は、環境音や足音・咳などのサーバ VAD 反応だけでは開始せず、**ローカル音声認識（Vosk）が実際に言葉として書き起こせたか**で判定するよう設計されています。これにより、雑音による誤発火で発表が中断される事故を避けています。スライドの区切りで質疑応答を開くふるまいは質問モードの種類（`"QuestionTime"` / `"Keywords"` / `"Detect"`）に関わらず常に有効になっており、聞いた直後に質問できないという抜けを防いでいます。

## 詳細説明

### 動作環境

- **Wolfram Language / Mathematica**: 13.x 以降を推奨します。Association、`RawJSON` インポート、`ProcessObject` 系の非同期実行を多用するため、比較的新しいバージョンが必要です。
- **OS**: Windows 11 を前提としています（`taskkill` によるプロセス終了、`%TEMP%` 上のロックファイルなど Windows 固有の実装を含みます）。macOS/Linux ではパス区切りやシェルコマンドの読み替えが必要です。
- **VRChat**: デスクトップクライアントを起動しておく必要があります。ワールド入場自動化を使う場合は UI 言語を英語にしてください（OCR 辞書に仮名がありません）。
- **VRC Pilot**: VRChat へのマウス/キーボード/OSC 入力を担当する外部プロジェクトです。既定パスは `C:\Users\<ユーザー名>\gitwork\vrcpilot`（オプション `"VRCpilotDirectory"`）。
- **uv**: Python の実行・依存解決に使用します。`uv.exe` は `PATH`、`%USERPROFILE%\.local\bin`、`%LOCALAPPDATA%\Programs\uv` の順に自動探索されます。見つからない場合は `"UVExecutable"` オプションで明示してください。
- **仮想オーディオデバイス**: VB-Audio Virtual Cable などを導入し、出力デバイス名を `"CABLE-A Input"`（既定値）に合わせて VRChat のマイク入力に設定してください。
- **VRChat OSC**: 有効化しておく必要があります（既定 `"OSCHost" -> "127.0.0.1"`, `"OSCPort" -> 9000`）。

### インストール

1. 本リポジトリの `.wl` ファイル一式を `$packageDirectory` 直下に配置します（サブディレクトリに分けないでください）。

   ```
   git clone https://github.com/transreal/VRCRealtime.git
   ```

2. VRC Pilot を別途クローンし、`"VRCpilotDirectory"` オプションのパスに配置します。
3. `uv` をインストールします（`winget install astral-sh.uv` など）。
4. VB-Audio Virtual Cable を導入し、Windows のサウンド設定と VRChat のマイク入力デバイスを揃えます。
5. 依存パッケージを `$packageDirectory` 直下に配置します。
   - **NBAccess.wl（必須）**: OpenAI の API キー取得・アクセス許可判定に使用します。無いと `VRCRealtimeStart[]` は失敗します。→ [NBAccess](https://github.com/transreal/NBAccess)
   - **SourceVault_voice.wl（任意）**: ローカル音声合成（Piper / AivisSpeech）ランタイムと声モデルを管轄します。→ [SourceVault_voice](https://github.com/transreal/SourceVault_voice)
   - **SourceVault.wl（任意）**: 発表登録簿（スライド URL・発表原稿）と資料検索を MCP サービス越しに提供します。→ [SourceVault](https://github.com/transreal/SourceVault)

外部実装（Python）は配布形では `VRCRealtime_info/resources/python/` にまとまっています。旧配置（`$packageDirectory` 直下に `VRCRealtime_worker.py` 等を直接置く形）もそのまま動作しますが、新しい配置が優先されます。

$Path には $packageDirectory 自体を含める必要があります。

```wolfram
AppendTo[$Path, $packageDirectory]
```

パッケージ固有のパス（例: `"C:\\path\\to\\VRCRealtime"`）を追加してはいけません。claudecode を使っている場合、$Path は自動的に設定されます。

読み込みは次の UTF-8 パターンを使ってください。

```wolfram
Block[{$CharacterEncoding = "UTF-8"},
  Needs["VRCRealtime`", "VRCRealtime.wl"]];
```

APIキーは NBAccess 経由で登録します。

```wolfram
Block[{$CharacterEncoding = "UTF-8"},
  Needs["NBAccess`", "NBAccess.wl"]];
NBAccess`NBSetAPIKey["openai", "sk-..."];
```

`"RequirePaidAPIApproval" -> True`（既定）を使う場合は、評価用ノートブックで事前に次を実行しておく必要があります。

```wolfram
NBAccess`NBSetNotebookPaidAPIAllowed[EvaluationNotebook[], True];
```

### クイックスタート

```wolfram
Block[{$CharacterEncoding = "UTF-8"},
  Needs["VRCRealtime`", "VRCRealtime.wl"]];

VRCRealtimeStart[
  "Voice" -> "marin",
  "OutputDevice" -> "CABLE-A Input",
  "Instructions" -> "日本語で自然かつ簡潔に会話してください。"
]

VRCRealtimeStatus[]

VRCRealtimeStop["WaitSeconds" -> 8.0]
```

主な既定値: `"Model" -> "gpt-live-1"`（GPT-Live。道具はバックエンドモデルへの Responses 委譲で実行。`"gpt-realtime-2.1"` で従来の Realtime）、`"FullDuplex" -> True`（GPT-Live使用時のみ有効。発表中以外は話しながら部屋の声も聞く全二重。`False` で Realtime と同じ半二重）、`"QuestionMode" -> "QuestionTime"`（発表中の質問の受け方。`"Keywords"` / `"Detect"` も選べ、音声「質問モードを切り替えたい」でも切り替わる。スライド区切りの質疑応答はどのモードでも開く）、`"Voice" -> "marin"`、`"AvatarMode" -> "Standalone"`、`"InputSource" -> "VRChat"`、`"AllowBargeIn" -> False`、`"EnableSourceVault" -> True`、`"EnableVoiceWebSearch" -> False`（既定オフの opt-in 機能）、`"RequirePaidAPIApproval" -> True`、`"ReplaceRunning" -> True`。

### 主な機能

- **セッション制御**: `VRCRealtimeStart[]` / `VRCRealtimeStatus[]` / `VRCRealtimeStop[]` / `VRCRealtimeKill[]` — 常駐ワーカーの起動・状態取得・正常終了・強制終了。
- **会話制御**: `VRCRealtimeSetInstructions[text]`、`VRCRealtimeMute[True|False]`、`VRCRealtimeEndConversation[]`、`VRCRealtimeSetQuestionMode[...]` — 応答方針の変更、マイク送信の一時停止、会話の終了と視覚ゲートの再有効化、発表中の質問の受け方（`"QuestionTime"` / `"Keywords"` / `"Detect"`）の実行中切り替え。
- **視覚・音声**: `VRCRealtimeLook[question]` で VRChat 画面を Realtime モデルへ送って説明させ、`VRCRealtimeVoiceStatus[]` / `VRCRealtimeInstallVoiceCommandModel[]` でローカル音声資産（Piper / AivisSpeech、Vosk 音声コマンドモデル）を確認・導入します。
- **プレゼンテーション**: `VRCRealtimeStartPresentation[target]`、`VRCRealtimeListPresentations[]`、`VRCRealtimePresentationStatus[]`、`VRCRealtimePresentationControl["Pause"|"Resume"|"Stop"|"ResumeHere"|"ResumeBack"]`、`VRCRealtimeStopPresentation[]` — Sliden の MP4 を用いた自動発表の開始・一覧・状態・制御。質疑応答中にスライドを動かした場合は「表示中のスライドから続ける（`"ResumeHere"`）」か「発表が進んでいた位置へ戻る（`"ResumeBack"`）」かを選べます。質問受付はローカル音声認識による書き起こしの成否で判定するため、環境音や足音だけでは反応しません。
- **オーナー追尾**: `VRCRealtimeSetOwnerAccount[name]`、`VRCRealtimeSetFollowDistance[m]`、`VRCRealtimeSetFormation[...]`、`VRCRealtimeSetMode[...]`、`VRCRealtimeLocateOwner[]` — 追尾対象・距離・隊形・モードの実行中変更とメニュー探索の即時実行。
- **モニタリング**: `VRCRealtimeMonitor[]` — 1秒ごとに更新される軽量な `Dynamic` 状態表示。
- **ワールド入場**: `VRCRealtimeJoinNGCGalleryWorld[world]`、`VRCRealtimeJoinNGCGalleryWorldStatus[]`、`VRCRealtimeJoinNGCGalleryWorldStop[]` — VRC Pilot を使ったワールド入場の自動化と進捗監視。
- **グローバル変数**: `$VRCRealtimeVersion`、`$VRCRealtimeOwnerAccount`（既定 `"nconc"`）、`$VRCRealtimeDefaultSlideURL`。

### ドキュメント一覧

- [api.md](api.md) — オプションを含む API リファレンス
- [example.md](example.md) — 実務でよく使う操作パターン集
- [setup.md](setup.md) — セットアップ手順書
- [user_manual.md](user_manual.md) — ユーザーマニュアル

## 使用例・デモ

`$VRCRealtimeDefaultSlideURL` の既定値である `https://transreal.sakura.ne.jp/cn/31.mp4` は、`VRCRealtimeStartPresentation[]` を引数なしで呼んだときに使われる Sliden 用デモスライドです。

```wolfram
(* オーナー追尾 + 発表の組み合わせ例 *)
VRCRealtimeStart["AvatarMode" -> "OwnerFollow", "OwnerAccount" -> "nconc"]
VRCRealtimeSetFollowDistance[1.8]
VRCRealtimeSetFormation["RearAngle" -> 35, "OwnerBearing" -> 33]

VRCRealtimeJoinNGCGalleryWorld["NCG Gallery", "ApproachConsole" -> True]
VRCRealtimeStartPresentation[VRCRealtime`$VRCRealtimeDefaultSlideURL]
VRCRealtimePresentationStatus[]

VRCRealtimeStop["WaitSeconds" -> 8.0]
```

会話中は「会話に参加して」「会話から離脱して」といった音声コマンドでも参加・離脱を制御できます。詳しい操作パターンは [example.md](example.md) を参照してください。

---

## 免責事項

本ソフトウェアは "as is"（現状有姿）で提供されており、明示・黙示を問わずいかなる保証もありません。
本ソフトウェアの使用または使用不能から生じるいかなる損害についても責任を負いません。
今後の動作保証のための更新が行われるとは限りません。
本ソフトウェアとドキュメントはほぼすべてが生成AIによって生成されたものです。
Windows 11上での実行を想定しており、MacOS, LinuxのMathematicaでの動作検証は一切していません(生成AIの処理で対応可能と想定されます)。

---

## ライセンス

```
MIT License

Copyright (c) 2026 Katsunobu Imai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
