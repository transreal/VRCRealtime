# VRCRealtime 使用例集

VRCChat と OpenAI Realtime を橋渡しする常駐ワーカーの、実務でよく使う操作パターンをまとめます。各例はそのまま評価できる最小構成です。

## 1. 起動と状態確認

```mathematica
VRCRealtimeStart["Voice" -> "marin", "OutputDevice" -> "CABLE-A Input"]
VRCRealtimeStatus[]
```
`<|"ProcessStatus" -> "Running", "Model" -> "gpt-realtime-2.1", ...|>` が返ります。

## 2. 軽量モニタで常時監視

```mathematica
VRCRealtimeMonitor[]
```
1秒ごとに更新される `Dynamic` 表示のセルが得られます。

## 3. 会話中の instructions 変更とミュート

```mathematica
VRCRealtimeSetInstructions["日本語で簡潔に、語尾は丁寧に話してください。"]
VRCRealtimeMute[True]
VRCRealtimeMute[False]
VRCRealtimeEndConversation[]
```
各コマンドは `<|"Status" -> "Sent", "Command" -> "instructions", ...|>` のような Association を返します。

## 4. VRChat画面を見せて説明させる

```mathematica
VRCRealtimeLook["いま画面には何が映っていますか。"]
VRCRealtimeLook[]
```
`<|"Status" -> "Sent", "Command" -> "look"|>` が返り、応答は音声で返ります。

## 5. オーナー追尾モードの調整

```mathematica
VRCRealtimeSetOwnerAccount["nconc"]
VRCRealtimeSetFollowDistance[1.8]
VRCRealtimeSetFormation["RearAngle" -> 35, "OwnerBearing" -> 33]
VRCRealtimeSetMode["OwnerFollow"]
VRCRealtimeLocateOwner[]
```
`VRCRealtimeSetFollowDistance` はメートル指定（0.8〜6）、`VRCRealtimeSetFormation` は角度指定（RearAngle 0〜90、OwnerBearing 10〜42）です。`$VRCRealtimeOwnerAccount` の既定値は `"nconc"` です。

## 6. 発表 (プレゼンテーション) の開始と制御

```mathematica
VRCRealtimeListPresentations[]
VRCRealtimeStartPresentation["計算と自然と31"]
VRCRealtimeStartPresentation[VRCRealtime`$VRCRealtimeDefaultSlideURL]
VRCRealtimePresentationStatus[]
VRCRealtimePresentationControl["Pause"]
VRCRealtimePresentationControl["Resume"]
VRCRealtimeStopPresentation[]
```
`VRCRealtimeStartPresentation` は登録済みタイトル・直接のMP4 URLのどちらでも受け付けます。`VRCRealtimePresentationStatus[]` は `presentationStatus` などの `"presentation"` 接頭辞キーだけを返します。

## 7. ワールド入場の自動化

```mathematica
VRCRealtimeJoinNGCGalleryWorld["NCG Gallery", "ApproachConsole" -> True]
VRCRealtimeJoinNGCGalleryWorldStatus[]
VRCRealtimeJoinNGCGalleryWorldStop[]
```
既定 (`"Wait" -> True`) では完了まで待って `<|"Status" -> "Joined", "World" -> ..., "Room" -> ..., "Console" -> ...|>` または `Failure[...]` を返します。`"Wait" -> False` を指定すると即座に戻り、`VRCRealtimeJoinNGCGalleryWorldStatus[]` で進捗を追えます。

## 8. 音声資産の確認と導入、終了処理

```mathematica
VRCRealtimeVoiceStatus[]
VRCRealtimeInstallVoiceCommandModel[]
VRCRealtimeStop["WaitSeconds" -> 8.0]
VRCRealtimeKill[]
```
`VRCRealtimeVoiceStatus[]` は `"Status" -> "OK"` でなければ `"Hint"` に導入手順を含みます。`VRCRealtimeStop` は正常終了を要求して待機し、応答が無いワーカーには `VRCRealtimeKill[]` を使います。

---

各コマンドの詳細な意味・オプションの完全な一覧はマニュアル（`?VRCRealtimeStart` など）を参照してください。バージョンは `VRCRealtime`$VRCRealtimeVersion` で確認できます。