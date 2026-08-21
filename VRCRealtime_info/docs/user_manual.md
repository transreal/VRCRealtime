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
(* "0.10.0" *)
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

OpenAI Realtime と VRChat の音声ブリッジを外部プロセスとして起動し、直ちに制御用の Association を返します（ブロッキングしません）。主なオプションは `"Model"`（既定 `"gpt-realtime-2.1"`）、`"Voice"`（既定 `"marin"`）、`"Instructions"`、`"AvatarMode"`（`"Standalone"` | `"OwnerFollow"` | `"Presentation"`）、`"OwnerFollowEngine"`（`"Formation"` 既定 | `"Legacy"`）、`"EnableSourceVault"`、`"EnableVoiceWebSearch"`（既定 False）、`"VisualAttentionGate"` など多数あります。

```wolfram
VRCRealtimeStart["AvatarMode" -> "OwnerFollow", "Instructions" -> "簡潔に日本語で答えてください。"]
```

### `VRCRealtimeStatus[]`

常駐ワーカーの現在状態（プロセス状態・セッション情報・追尾/発表の進捗など）を Association で返します。自分のカーネルが起動していないワーカーが走っていても検出します。

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

Sliden の MP4 を取得・展開し、操作パネルへ URL を入力して各スライドを自動発表します。`target` には URL か登録済みの発表タイトルを渡せます。登録簿に発表原稿があればそれを話し、無ければ 1 枚ごとの画像説明になります。主なオプションは `"NavigateToConsole"`（既定 Automatic：使える位置ならそのまま開始、使えなければ探索）、`"StartSlide"` / `"EndSlide"`、`"Talk"`（None で原稿を使わない）、`"QuestionSeconds"`。

```wolfram
VRCRealtimeStartPresentation["計算と自然と31"]
```

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

### `VRCRealtimePresentationControl["Pause"|"Resume"|"Stop"]`

実行中の発表を一時停止・再開・打ち切ります。音声の「一時停止して」等と同じ経路です。

```wolfram
VRCRealtimePresentationControl["Pause"]
```

### `VRCRealtimeStopPresentation[]`

実行中の自動発表を停止します。

```wolfram
VRCRealtimeStopPresentation[]
```

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