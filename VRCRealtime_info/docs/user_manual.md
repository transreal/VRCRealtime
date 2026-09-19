# VRCRealtime ユーザーマニュアル

VRCRealtime は OpenAI Realtime API と VRChat をつなぐ非同期音声ブリッジです。外部プロセス (Python ワーカー) を起動して VRChat の音声入出力・アバター制御・スライド発表を行い、Wolfram Language 側は制御コマンドの送受信だけを担当します。インストールや初期設定は setup.md を参照してください。

## 目次

- グローバル変数
- セッションの起動・停止
- 会話制御
- 視覚・音声
- プレゼンテーション
- オーナー追尾
- モニタリング
- ワールド入場

## グローバル変数

### `$VRCRealtimeVersion`

パッケージのバージョン文字列です。

```wolfram
$VRCRealtimeVersion
(* "0.11.0" *)
```

### `$VRCRealtimeOwnerAccount`

オーナー追尾モードで追跡する VRChat アカウント名です。既定値は `"nconc"`。`VRCRealtimeSetOwnerAccount` で更新すると実行中のセッションにも反映されます。

```wolfram
$VRCRealtimeOwnerAccount
(* "nconc" *)
```

### `$VRCRealtimeDefaultSlideURL`

`VRCRealtimeStartPresentation` が引数省略時に使う既定の Sliden 用 MP4 URL です。

```wolfram
$VRCRealtimeDefaultSlideURL
```

## セッションの起動・停止

### `VRCRealtimeStart[]`

OpenAI の音声モデルと VRChat の音声ブリッジを外部プロセスとして起動し、直ちに制御用の Association を返します（ブロッキングしません）。主なオプションは `"Model"`（既定 `"gpt-live-1"`。`"gpt-realtime-2.1"` で従来の Realtime）、`"API"`（既定 Automatic：モデル名から `"Realtime"`/`"Live"` を自動判定し、明示指定もできる）、`"LiveBackendModel"`（既定 Automatic = `gpt-5.6-luna`。GPT-Live 使用時に道具を実行するバックエンドモデル）、`"QuestionMode"`（発表中の質問の受け方。下の「質問モード」参照）、`"InterruptWords"`（既定 Automatic。質問モード `"Keywords"` で発表を中断させる語。下の「質問モード」参照）、`"FullDuplex"`（既定 True。GPT-Live 使用時のみ意味を持つ。下記参照）、`"Voice"`（既定 `"marin"`）、`"Instructions"`、`"AvatarMode"`（`"Standalone"` | `"OwnerFollow"` | `"Presentation"`）、`"OwnerFollowEngine"`（`"Formation"` 既定 | `"Legacy"`）、`"EnableSourceVault"`、`"EnableVoiceWebSearch"`（既定 False）、`"VisualAttentionGate"` など多数あります。

音声合成エンジンは明示オプションではなく、解決された声の種類から自動的に選ばれます。`"Voice"` が AivisSpeech 由来の声に解決された場合はエンジンが実際に応答するかを確認したうえで AivisSpeech を採用し（エンドポイントとスタイル id を worker へ渡します）、応答しなければ Piper の声へ自動的に切り替えます（Piper の声が別に無くても起動は継続します）。声そのものが解決できない場合は `VRCRealtimeStart::novoice` で警告したうえ、SourceVault 連携を切って起動します（非公開資料は取得も読み上げもされません）。

`"EnableSourceVault"` を使う場合、資料検索と発表登録簿の照合は SourceVault MCP サービス越しに行われます。このサービスが保守処理（anomaly tick 等）の実行中で command queue が数秒以内に応答しないときは `VRCRealtimeStart::mcpbusy` が出て、その間は発表登録簿を引けず発表を開始できません。数分おいてから再度試してください。サービス自体が停止していれば `VRCRealtimeStart::mcpstopped` とともに自動起動を試み、それも失敗すれば `VRCRealtimeStart::mcpfailed` になります。

```wolfram
VRCRealtimeStart["AvatarMode" -> "OwnerFollow", "Instructions" -> "簡潔に日本語で答えてください。"]
```

既定の GPT-Live は自分では道具を持たず、調べもの・発表操作・アバター操作・画面を見るといった仕事を
バックエンドのモデル（`"LiveBackendModel"`、既定 gpt-5.6-luna）に任せます（Responses 委譲）。
会話の中身や操作できることは Realtime と同じです。GPT-Live はスライド画像を見られないので、
原稿の無いスライドはバックエンドが画像から原稿を書き、それを読み上げます。

`"API"` は `"Model"` から自動判定される経路を明示的に上書きするためのオプションです。既定の
Automatic ではモデル名が `"gpt-live"` で始まれば Live、それ以外は Realtime として扱われます。
`"API" -> "Live"` / `"API" -> "Realtime"` を指定すると、モデル名の判定に関わらず経路を固定できます。
実際に使われた経路は `VRCRealtimeStatus[]` の `"API"` で確認できます。

`"FullDuplex"`（既定 True）は GPT-Live 使用時のみ意味を持ちます。True では、発表中以外はアバターが
話しながらも部屋の声を同時に聞き続けます（全二重）。False にすると、Realtime と同じ半二重（話している
間とその直後 1.2 秒は聞かない）に切り替わります。

```wolfram
VRCRealtimeStart["Model" -> "gpt-realtime-2.1"]   (* 従来の Realtime で起動 *)
VRCRealtimeStart["API" -> "Realtime"]              (* モデル名判定を上書きして強制的に Realtime *)
VRCRealtimeStart["FullDuplex" -> False]            (* GPT-Live のまま半二重に *)
```

### `VRCRealtimeStatus[]`

常駐ワーカーの現在状態（プロセス状態・セッション情報・追尾/発表の進捗など）を Association で返します。`"Model"` に加えて実際に使われている経路を示す `"API"`（`"Realtime"` | `"Live"`）も含みます。自分のカーネルが起動していないワーカーが走っていても検出します。

```wolfram
VRCRealtimeStatus[]
```

### `VRCRealtimeStop[]`

常駐ワーカーへ正常終了を要求し、既定で最大 8 秒 (`"WaitSeconds"`) まで終了を待ちます。

```wolfram
VRCRealtimeStop["WaitSeconds" -> 5.]
```

### `VRCRealtimeKill[]`

応答しない常駐ワーカーを強制終了します。自分が起動したのでなくても、走っているワーカーを検出して止められます。

```wolfram
VRCRealtimeKill[]
```

## 会話制御

### `VRCRealtimeSetInstructions[text]`

実行中の Realtime セッションの instructions（応答方針）を更新します。

```wolfram
VRCRealtimeSetInstructions["これからは英語で応答してください。"]
```

### `VRCRealtimeMute[True|False]`

VRChat 音声の OpenAI への送信を一時停止・再開します。

```wolfram
VRCRealtimeMute[True]
```

### `VRCRealtimeEndConversation[]`

現在の会話を終了し、視覚的な開始ゲートを再び有効にします。会話の開始/離脱は「会話に参加して」「会話から離脱して」という音声コマンドでも行えます。離脱後は挨拶か明示的な参加発話でのみ再開します。

```wolfram
VRCRealtimeEndConversation[]
```

## 視覚・音声

### `VRCRealtimeLook[question]`

現在の VRChat 画面を 1 枚 Realtime モデルへ送り、音声で説明させます。引数省略時は既定の質問文が使われます。`"EnableVisionCapture" -> True`（既定）のときは「これは何」のような発話でもモデル側から同じ経路が呼ばれます。

```wolfram
VRCRealtimeLook["この作品について教えて"]
```

### `VRCRealtimeVoiceStatus[]`

VRCRealtime が実際に使うローカル音声資産（合成ランタイム・声モデル・音声認識モデル）の状態を返します。`"Status"` が `"OK"` でなければ `"Hint"` に導入手順が入ります。

音声合成エンジンは Piper に加えて AivisSpeech にも対応しています。解決された声が AivisSpeech 由来のものであれば `"SelectedEngine" -> "AivisSpeech"` となり、`"AivisEndpoint"` にエンジンのエンドポイントが入ります。Piper の声であれば `"SelectedEngine" -> "Piper"` です。AivisSpeech エンジンが実際には応答しない場合は、指定された声が AivisSpeech のものであっても自動的に Piper の声へフォールバックします（導入されていれば）。特定の声を前提にしているわけではなく、導入されている声から自動的に選ばれます。

```wolfram
VRCRealtimeVoiceStatus[]
```

### `VRCRealtimeInstallVoiceCommandModel[]`

「会話に参加」「会話から離脱」の音声コマンド認識に使う Vosk 日本語小モデル（約 50 MB）を事前ダウンロードします。`VRCRealtimeStart` 側で未導入時に自動取得されるため必須ではありません。

```wolfram
VRCRealtimeInstallVoiceCommandModel[]
```

## プレゼンテーション

### `VRCRealtimeStartPresentation[target]`

Sliden の MP4 を取得・展開し、操作パネルへ URL を入力して各スライドを自動発表します。`target` には URL か登録済みの発表タイトルを渡せます。登録簿に発表原稿があればそれを話し、無ければ 1 枚ごとの画像説明になります。主なオプションは `"NavigateToConsole"`（既定 Automatic：使える位置ならそのまま開始、使えなければ探索）、`"StartSlide"` / `"EndSlide"`、`"Talk"`（None で原稿を使わない）、`"QuestionSeconds"`（既定 5 秒。各スライドの切れ目で待つ質疑応答の秒数）、`"FinalQuestionSeconds"`（既定 120 秒。全スライド終了後の質疑応答で、無音のままこの秒数が経過すると発表を終了する上限）。

```wolfram
VRCRealtimeStartPresentation["計算と自然と31"]
```

発表タイトル・スライド URL・発表原稿は SourceVault の発表登録簿へ MCP サービス越しに照合します。開始前にこのサービスの command queue が応答するかを確認しており、保守処理中などで数秒以内に応答しない場合は登録簿を引けず、原稿だけが静かに落ちて画像説明に縮退することを防ぐため発表を開始しません。その場合は少し時間をおいてから再度試してください。

スライドの切れ目では `"QuestionSeconds"`(既定 5 秒)だけ待ちます。この間アバターは
**何も話しません**。「質問はありますか」と口頭で促すことはせず、チャットボックスに
「質問があればどうぞ」と出すだけで、誰も話さなければ黙って次のスライドへ進みます。

全スライドを話し終えた後も同様に質疑応答を受け付けますが、こちらは `"FinalQuestionSeconds"`
（既定 120 秒）が上限です。無音のままこの秒数が経過すると発表を終了します。

待っている間に実際に声がかかったときだけ応答します。判定は**ローカルの音声認識**
(join/離脱キーワードに使っている Vosk)が言葉として書き起こせたかどうかで行い、
サーバ VAD の反応だけでは質問と見なしません。VAD は音量に反応するだけなので、
ワールドの環境音・足音・咳が質問として扱われ、アバターが勝手に喋り出して発表が
止まる原因になっていました(2026-09-05 実機)。現在どちらの判定が効いているかは
`VRCRealtimePresentationStatus[]` の `"presentationQuestionGate"`(`local_speech` / `vad`)で分かります。
音声認識モデルが読み込まれていない場合のみ `vad` に落ちます。

### `VRCRealtimeListPresentations[]`

SourceVault の発表登録簿に登録済みの発表（タイトル・スライド URL・枚数）を返します。

```wolfram
VRCRealtimeListPresentations[]
```

### `VRCRealtimePresentationStatus[]`

自動 Sliden 発表の進捗状態を返します。

```wolfram
VRCRealtimePresentationStatus[]
```

### `VRCRealtimePresentationControl["Pause"|"Resume"|"Stop"|"ResumeHere"|"ResumeBack"]`

実行中の発表を一時停止・再開・打ち切ります。音声の「一時停止して」等と同じ経路です。

```wolfram
VRCRealtimePresentationControl["Pause"]
```

質疑応答は各スライドの切れ目で開き、その間に `show_slide`(音声の「一つ前を見せて」等)で
デッキを動かせます。動かしたまま発表を再開すると原稿と画面がずれるため、発表側から
「いま N 枚目を表示しています。このまま N 枚目から続けますか、それとも M 枚目に戻って
続けますか」と尋ねます。返事は音声(「このまま」「戻って」)でも、この関数でも返せます。

- `"ResumeHere"` — いま表示しているスライドから発表を続ける(そのスライドの原稿から再開)
- `"ResumeBack"` — 発表が進んでいた位置までデッキを戻して続きを話す

25 秒返事がなければ `"ResumeBack"` として扱います(原稿と画面が一致する側に倒す)。

### `VRCRealtimeStopPresentation[]`

実行中の自動発表を停止します。

```wolfram
VRCRealtimeStopPresentation[]
```

### 質問モード — `VRCRealtimeSetQuestionMode[mode]` / `VRCRealtimeQuestionMode[]`

発表中の質問の受け方を 3 つから選べます。

- `"QuestionTime"`（質問タイム・既定）— スライドの切れ目でだけ質問を受けます（上の説明のとおり）。
- `"Keywords"`（特定ワード）— 区切りの質問待ちに加えて、発表中でも「質問」「スライド」「プレゼン」
  「発表」を含む発話（「質問いいですか」「スライドを戻してください」「発表を中断して」など）を
  ローカルの音声認識が聞き取ると、読み上げを止めて質問を受けます。「発表を中断して」「プレゼンを
  止めて」なら一時停止し、「再開して」で中断した文から続けます。割り込んだ発話はもう聞き取った後なので、
  返事は「はい、どうぞ」ではなく、依頼なら「了解です」、聞き取れなければ「すみません、もう一度
  お願いします」になります（「はい、どうぞ」は「質問いいですか」とだけ言われたときに限ります）。
  「停止して」「止めて」は一時停止で、発表を終わりにするときは「発表を終了して」と言います。
  発表の一時停止・終了の依頼（「発表を停止して」「プレゼンを終了して」など）は、質問モードに
  関係なく読み上げ中でも受け付けます（手元の音声認識が聞き取り、モデルの判断を待たずに実行します）。
  発表中に別の発表の開始を頼むと、今の発表を終了してから切り替えます。質問が終わって静かになると、中断した文から「それでは、説明に戻ります」で再開します。
  語は `VRCRealtimeStart` の `"InterruptWords"` で変えられます。
- `"Detect"`（二段構え）— 区切りの質問待ちに加えて、発表中に部屋の声がしばらく続くと一旦止め、その発話をモデルに聞かせて
  自分への質問かどうかを判断させます。質問なら答え、独り言や雑談なら何も言わずに続きを読みます。

どのモードでも、スライドの切れ目ではチャットボックスに「質問があればどうぞ」と出して
`"QuestionSeconds"`（既定 5 秒）待ちます。説明を聞き終えてから質問できるようにするためです。VRChat の部屋の音にはアバター自身の声は入らないため、
発表中に聞き耳を立ててもエコーの心配はありません。

```wolfram
VRCRealtimeSetQuestionMode["Keywords"]
VRCRealtimeQuestionMode[]
VRCRealtimeStartPresentation["計算と自然31", "QuestionMode" -> "Detect"]
```

音声で「質問モードを切り替えたい」と話しかけても切り替えられます。アバターが 3 つのモードを
短く説明してどれにするか尋ね、答えたモードに設定します（「特定ワードにして」のように最初から
名前を言えば、尋ね直さずに切り替えます）。

## オーナー追尾

### `VRCRealtimeSetMode["Standalone"|"OwnerFollow"|"Presentation"]`

単独動作・オーナー追尾・プレゼンモードを実行中に切り替えます。

```wolfram
VRCRealtimeSetMode["OwnerFollow"]
```

### `VRCRealtimeSetOwnerAccount[name]`

追尾対象アカウントを更新します。`$VRCRealtimeOwnerAccount` にも反映されます。

```wolfram
VRCRealtimeSetOwnerAccount["nconc"]
```

### `VRCRealtimeSetFollowDistance[m]`

追尾距離をメートルで調整します（既定 1.8、0.8〜6）。`"OwnerFollowEngine" -> "Legacy"` のときはネームプレートの画面高比（0.005〜0.15）として解釈されます。

```wolfram
VRCRealtimeSetFollowDistance[2.2]
```

### `VRCRealtimeSetFormation["RearAngle" -> deg, "OwnerBearing" -> deg]`

追尾隊形を実行中に変更します。`RearAngle`（既定 35、0〜90）はオーナーの真横からどれだけ後方に控えるか、`OwnerBearing`（既定 33、10〜42）は停止時にオーナーを視野内のどこに置くかを指定します。

```wolfram
VRCRealtimeSetFormation["RearAngle" -> 40, "OwnerBearing" -> 30]
```

### `VRCRealtimeLocateOwner[]`

フォローモードのメニュー探索（Location ページ・Friends 一覧からの Join）を即時に実行させます。通常は全周探索が既定回数空振りすると自動実行されます。

```wolfram
VRCRealtimeLocateOwner[]
```

## モニタリング

### `VRCRealtimeMonitor[]`

1 秒ごとに `VRCRealtimeStatus[]` を更新して表示する `Dynamic` 式を返します。ノートブックに直接評価して常時監視に使います。

```wolfram
VRCRealtimeMonitor[]
```

## ワールド入場

### `VRCRealtimeJoinNGCGalleryWorld[world]`

起動直後（Home ワールド）の VRChat を操作し、Worlds → Favorite Worlds 1 → NCG Gallery（既定）→ New Instance → Group の順にグループインスタンスへ入場し、Sliden スクリーンのある部屋まで歩きます。既定では完了まで待って結果または `Failure` を返し、`"Wait" -> False` で即座に戻れます。`"ApproachConsole" -> True` で入室後に操作パネル前まで進みます。VRChat の UI 言語は英語である必要があります。

```wolfram
VRCRealtimeJoinNGCGalleryWorld["NCG Gallery", "ApproachConsole" -> True]
```

### `VRCRealtimeJoinNGCGalleryWorldStatus[]`

`VRCRealtimeJoinNGCGalleryWorld` の進捗（`worldJoinPhase` / `worldJoinSteps` / `worldJoinLastTexts` など）を返します。`"Wait" -> False` で開始した場合の進捗確認に使います。

```wolfram
VRCRealtimeJoinNGCGalleryWorldStatus[]
```

### `VRCRealtimeJoinNGCGalleryWorldStop[]`

実行中の `VRCRealtimeJoinNGCGalleryWorld` を中止します。

```wolfram
VRCRealtimeJoinNGCGalleryWorldStop[]
```

## 補足

- 追尾/発表/ワールド入場は VRChat の入力（マウス・OSC）を占有するため、同時に複数走らせることはできません。
- 非公開資料（SourceVault の Privacy Level が高いもの）は、フォローモードでオーナーが近くにいると確認できた場合のみ回答されます（`"OwnerPresencePrivacyThreshold"`、既定 0.25）。
- 各関数の詳細なオプション一覧は `Options[VRCRealtimeStart]` 等で確認できます。