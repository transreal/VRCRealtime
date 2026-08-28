# VRCRealtime API リファレンス

VRChat上のアバターとOpenAI Realtime APIを結ぶノンブロッキング音声ブリッジ。外部プロセス (Python worker) を起動して制御する。すべての制御関数は即座に返り、実際の処理は非同期に進む。状態は `VRCRealtimeStatus[]` 等で随時取得する。

## 定数

### $VRCRealtimeVersion
型: String, 初期値: "0.10.0"
パッケージのバージョン。

### $VRCRealtimeOwnerAccount
型: String, 初期値: "nconc"
オーナー追尾モード ("AvatarMode" -> "OwnerFollow") で追跡するVRChatアカウント名の既定値。`VRCRealtimeSetOwnerAccount` で実行中に変更すると本変数も更新される。

### $VRCRealtimeDefaultSlideURL
型: String, 初期値: "https://transreal.sakura.ne.jp/cn/31.mp4"
`VRCRealtimeStartPresentation[]` を引数なしで呼んだときに使われる既定のSliden MP4 URL。

## ワーカー制御

### VRCRealtimeStart[opts]
OpenAI Realtime と VRChat の音声ブリッジを外部プロセス (uv run python) で起動し、直ちに制御用Associationを返す。既に起動中なら Failure["AlreadyRunning", ...] を返す。内部でNBAccess経由のOpenAI APIキー取得・uv実行ファイルの解決・VRC Pilotディレクトリの存在確認を行い、いずれか失敗すると対応するタグ (WorkerMissing, NBAccessUnavailable, NotebookRequired, PaidAPINotAllowed, OpenAIDisabled, APIKeyMissing, UVNotFound, VRCpilotDirectoryMissing 等) の Failure を返す。
→ <|"Status" -> ..., ...|> または Failure[tag, ...]
オーナー追尾は既定で幾何制御エンジン ("OwnerFollowEngine" -> "Formation") を使う: ネームプレートOCRで発見→体のランドマークで距離まで接近→向きを推定してオーナーの左後方へ回り込み→オーナーを視野右端に置いて停止。"Legacy" で旧OSCボタン制御に戻せる。
Options: "Model" -> "gpt-realtime-2.1", "Voice" -> "marin" (Realtime音声の声名。ローカル読み上げの声とは別物), "Instructions" -> "日本語で自然かつ簡潔に会話してください。", "VRCpilotDirectory" -> "C:\\Users\\imai_\\gitwork\\vrcpilot", "UVExecutable" -> Automatic, "Notebook" -> Automatic (承認確認に使うフロントエンドノートブック), "RequirePaidAPIApproval" -> False相当 (NBAccessでの有料API許可確認を必須にする), "InputSource" -> "VRChat", "InputDevice" -> Automatic, "OutputDevice" -> "CABLE-A Input", "VRChatPID" -> Automatic, "ChunkMilliseconds" -> 20, "VADThreshold" -> 0.5, "PrefixPaddingMilliseconds" -> 200, "SilenceDurationMilliseconds" -> 350, "ReconnectSeconds" -> 3300, "OutputCooldownMilliseconds" -> 1200, "AllowBargeIn" -> False。
視覚アテンションゲート系: "VisualAttentionGate" -> False, "VisualGateFramesPerSecond" -> 8., "VisualGateCenterTolerance" -> 0.18, "VisualGateMinFaceFraction" -> 0.05 (顔の見かけの大きさの下限、16:9換算の画面幅比。低すぎると通行人で誤発火するため0.05未満は非推奨), "VisualGateMouthMotionThreshold" -> 0.025, "VisualGateAudioRMSThreshold" -> 0.008, "VisualGateMaxStereoImbalance" -> 0.35, "VisualGateHangoverMilliseconds" -> 650, "VisualGatePrerollMilliseconds" -> 600, "VisualGateRequireNameplatePulse" -> True, "VisualGateNameplateBlueThreshold" -> 0.002, "VisualGateConversationLatch" -> True, "VisualGateConversationIdleSeconds" -> 30.。
アバター/オーナー追尾系: "AvatarMode" -> "Standalone" ("Standalone"|"OwnerFollow"|"Presentation"), "OwnerAccount" -> Automatic ($VRCRealtimeOwnerAccountを使用), "OwnerFollowEngine" -> "Formation" ("Formation"|"Legacy"), "OwnerFollowDistanceMeters" -> 1.8, "OwnerFollowRearAngleDegrees" -> 35., "OwnerFollowOwnerBearingDegrees" -> 33., "OwnerFollowMenuLocate" -> True, "OwnerFollowSearchPatrol" -> True, "OwnerFollowIdleAfterSeconds" -> 360., "OwnerFollowIdleRecheckSeconds" -> 300., "OwnerFollowTrackingFPS" -> 1.5, "OwnerFollowTargetScreenX" -> 0.66, "OwnerFollowTargetNameplateHeight" -> 0.055, "OwnerFollowLostSeconds" -> 1.25, "OwnerFollowFormationSeconds" -> 4.5, "OwnerFollowOrbitSpeed" -> 0.32, "OwnerFollowSearchTurnSpeed" -> 0.42, "OwnerFollowAssumeSingleAvatar" -> True, "OwnerFollowOCRMinSide" -> 1080, "OwnerFollowOCRBoxThreshold" -> 0.35, "OwnerFollowOCRTextScore" -> 0.35。
OSC/視覚/音声認識系: "OSCHost" -> "127.0.0.1", "OSCPort" -> 9000, "EnableVisionCapture" -> True, "VoiceCommands" -> True (「会話に参加/離脱」のローカルVosk認識), "VoiceCommandModelDirectory" -> Automatic, "VoiceCommandMinConfidence" -> 0.45, "VisionImageMaxWidth" -> 960, "VisionJPEGQuality" -> 78。
SourceVault連携系: "EnableSourceVault" -> True, "EnableVoiceWebSearch" -> False (既定オフ、opt-in), "SourceVaultServiceDirectory" -> Automatic, "SourceVaultKinds" -> {"search","packageapi","llmlog","mail","eagle","snapshot"}, "SourceVaultKB" -> Automatic, "SourceVaultGroup" -> None, "SourceVaultReleaseContext" -> None, "SourceVaultLimit" -> 3, "SourceVaultVoiceWaitSeconds" -> 60., "LocalSpeechMaxSeconds" -> 60., "OwnerPresencePrivacyThreshold" -> 0.25 (この値以上のPrivacy.Levelの資料はオーナーが近くにいるときだけ回答)。
ローカル音声合成系: "PrivateTTSBroker" -> Automatic, "PrivateTTS" -> Automatic (Automaticは資産があれば使用、無ければ警告してSourceVault連携を切って起動; Trueは必須, Falseは不使用), "SpeechVoice" -> Automatic, "PiperExecutable" -> Automatic, "PiperModel" -> Automatic, "PiperConfig" -> Automatic, "SpeechLengthScale" -> Automatic, "SpeechNoiseScale" -> Automatic。エンジンは解決した声から決まる: SourceVault_voice の既定声 (または "SpeechVoice" で指定した声) が AivisSpeech のものならエンジンの応答を確かめたうえで AivisSpeech (VOICEVOX 互換 HTTP、loopback 限定) を使い、Piper の声はセッション内フォールバックとして渡す。SpeechLengthScale (>1 = ゆっくり) は AivisSpeech では speedScale の逆数に写る。エンジンが応答しなければ従来どおり Piper。
例: VRCRealtimeStart["AvatarMode" -> "OwnerFollow", "OwnerAccount" -> "nconc", "OwnerFollowDistanceMeters" -> 2.0, "VoiceCommands" -> True]

### VRCRealtimeStatus[] → <|"ProcessStatus" -> ..., ...|>
常駐ワーカーの状態をAssociationで返す。このカーネルが起動したのでなくても、ロックファイルから発見した稼働中ワーカーがあれば "ProcessStatus" -> "Running" と "AdoptedWorkerPID" を含めて報告する。worker側のstateファイルの内容 (voiceKeywordLeft 等) もマージされる。

### VRCRealtimeStop[opts]
常駐ワーカーへ正常終了 (control fileで"stop"コマンド) を要求し、既定で最大8秒まで終了を待つ。
→ <|"Status" -> "Stopped", "ProcessStatus" -> ...|> または Failure["StopTimeout", ...]
Options: "WaitSeconds" -> 8.0

### VRCRealtimeKill[] → <|"Status" -> "Killed", ...|>
応答しないワーカーを taskkill /T /F (Windows) または KillProcess (その他) で強制終了する。このカーネルが起動したのでなくても、ロックファイルから発見した生存中ワーカーを止められる。稼働中でなければ Failure["NotRunning", ...] を返す。

### VRCRealtimeMonitor[] → Dynamic
1秒ごとに VRCRealtimeStatus[] を再評価する軽量な動的表示 (Dynamic[Refresh[...]]) を返す。

## セッション操作

### VRCRealtimeSetInstructions[text_String] → <|"Status" -> "Sent", ...|>
実行中のRealtimeセッションのinstructionsを更新する。

### VRCRealtimeMute[flag : (True | False)] → <|"Status" -> "Sent", ...|>
True でVRChat音声のOpenAIへの送信を一時停止、False で再開する。

### VRCRealtimeEndConversation[] → <|"Status" -> "Sent", ...|>
現在の会話を終了し、視覚的な開始ゲートを再び有効にする。会話の開始/終了は音声でも指示できる: 正面から話しかける (視覚ゲート)、または方向やフォロー移動中を問わず「会話に参加(して)」「会話から離脱(して)」。ただし「会話から離脱」で離脱した後は視覚ゲートだけでは再開せず、(1)「会話に参加(して)」と言われたとき、(2) 正面から挨拶 (こんにちは/こんばんは/おはよう/もしもし/ハロー/すみません) されたとき のみ再開する。これらの語はローカルのVosk認識 (文法制約、音声は外部へ出ない) で検出し、状態は VRCRealtimeStatus[] の voiceKeywordLeft で確認できる。

### VRCRealtimeLook[question_String : "現在のVRChat画面には何が見えますか。"] → <|"Status" -> "Sent", ...|>
現在のVRChat画面を1枚Realtimeモデルへ送り、音声で説明させる。"EnableVisionCapture" -> True (既定) のとき、音声で『目の前に何がある』『これは何』『何が見える』と尋ねられた場合もモデルがlook_at_vrchatを呼んで同じ経路で答える。

## プレゼンテーション

### VRCRealtimeListPresentations[] → {<|...|>, ...}
SourceVaultの発表登録簿に登録済みの発表 (タイトル/スライドURL/枚数) の一覧を返す。SourceVaultが未ロードなら Failure["SlideDeckRegistryUnavailable", ...]。

### VRCRealtimeStartPresentation[target, opts]
SlidenのMP4を取得・展開し、操作パネルへURLを入力して各スライドを元解像度で自動発表する。targetはURL (既定 $VRCRealtimeDefaultSlideURL) または登録済み発表タイトル (表記ゆれは登録簿側が吸収)。原稿があればそれを話し、無ければ1枚ごとの画像説明になる。
→ <|"Status" -> ..., ...|> または Failure[tag, ...] (例: "SlideDeckRegistryUnavailable", "PresentationNotRegistered", "PresentationWithheld", "InvalidSlideURL")
"NavigateToConsole" は既定Automatic: まず今いる場所から操作パネルが使えるか確認し、使えるならそのまま開始、使えないときだけ部屋の中でパネルを探して前まで移動してから開始する。True は常に探索してから開始、False は探索しない (パネルの前にいなければ失敗)。
Options: "SecondsPerSlide" -> Automatic (登録簿→無ければ30、3〜300の範囲), "ConsoleTimeoutSeconds" -> 30., "NavigateToConsole" -> Automatic, "NavigationTimeoutSeconds" -> 120., "DownloadTimeoutSeconds" -> 120., "MaxDownloadMegabytes" -> 256., "StartSlide" -> Automatic, "EndSlide" -> Automatic, "NarrationInstructions" -> Automatic, "Talk" -> Automatic (登録簿の発表原稿。None で原稿を使わず画像説明のみ), "Title" -> Automatic, "QuestionSeconds" -> 5. (スライド切替時の質疑応答秒数、0で質疑応答なし), "FinalQuestionSeconds" -> 120. (全スライド終了後の質疑応答上限秒数)。
例: VRCRealtimeStartPresentation["計算と自然と31", "NavigateToConsole" -> True, "QuestionSeconds" -> 10.]

### VRCRealtimePresentationStatus[] → <|"ProcessStatus" -> ..., "presentation..." -> ...|>
自動Sliden発表の状態 (presentationで始まるキーのみ) をAssociationで返す。

### VRCRealtimePresentationControl[action : ("Pause" | "Resume" | "Stop")] → <|"Status" -> "Sent", ...|>
実行中の発表を一時停止・再開・停止する。音声の「一時停止して」「再開して」「打ち切って」と同じ経路を通る。

### VRCRealtimeStopPresentation[] → <|"Status" -> "Sent", ...|>
実行中の自動Sliden発表を停止する。

## モード・オーナー追尾

### VRCRealtimeSetMode[mode : ("Standalone" | "OwnerFollow" | "Presentation")] → <|"Status" -> "Sent", ...|>
単独動作・オーナー追尾・プレゼンモードを実行中に切り替える。OwnerFollow (Formationエンジン) の探索は、全周スキャンが空振りすると最も開けた方向へ数m歩いて再スキャンし ("OwnerFollowSearchPatrol")、"OwnerFollowMenuLocateAfterSweeps"回でメニューから所在確認/Joinし、オーナーがオフライン・参加不可・または"OwnerFollowIdleAfterSeconds"経過で「待機」(旋回せず静止、見かけたら接近、"OwnerFollowIdleRecheckSeconds"ごとにメニュー再確認) に入る。待機中・不在中・別ワールドでは非公開資料 (PL >= OwnerPresencePrivacyThreshold) は答えない。

### VRCRealtimeSetOwnerAccount[account_String] → <|"Status" -> "Sent", ...|>
追尾対象アカウントを更新する ($VRCRealtimeOwnerAccountとセッション内設定の両方に反映)。空文字は Failure["InvalidOwnerAccount", ...]。

### VRCRealtimeSetFollowDistance[m_?NumericQ] → <|"Status" -> "Sent", ...|>
追尾距離をメートルで調整する (既定1.8、0.8〜6)。"OwnerFollowEngine" -> "Legacy" のときはネームプレートの画面高比 (0.005〜0.15、大きい値ほど近く) として解釈する。範囲外は Failure["InvalidFollowDistance", ...]。

### VRCRealtimeSetFormation[opts]
追尾隊形を実行中に変更する。ボットの向き = オーナーの向き + (90 - RearAngle - OwnerBearing) 度。
→ <|"Status" -> "Sent", ...|> または Failure["InvalidFormation", ...]
Options: "RearAngle" -> None (オーナーの真横 (0) からどれだけ後ろに控えるか、既定相当35、範囲0〜90), "OwnerBearing" -> None (停止時にオーナーを視野のどこに置くか、視野中心から右へ何度か、既定相当33、範囲10〜42)
例: VRCRealtimeSetFormation["RearAngle" -> 20, "OwnerBearing" -> 25]

### VRCRealtimeLocateOwner[] → <|"Status" -> "Sent", ...|>
フォローモードのメニュー探索を即時に実行させる: メインメニュー→Location (Here) ページ→下へスクロールしてUsers一覧にオーナーがいるか確認、いなければ右Wing→ハンバーガー→Friends→矢印でOnline一覧→オーナー→Join (押せる場合) でオーナーのインスタンスへ移動する。通常は全周探索が既定2回空振りすると自動実行される。

## ワールド入場

### VRCRealtimeJoinNGCGalleryWorld[world_String : "NCG Gallery", opts]
起動直後 (Homeワールド) のVRChatをVRC Pilotで操作し、メニュー Worlds → Favorite Worlds 1 → NCG Gallery → New Instance → Group の順に進み、グループ/GroupAccess/Regionを確認してCreate Instance → Joinでグループインスタンスに入場し、右手のSlidenスクリーンのある部屋まで歩く。追尾/発表中のworkerがあれば入力の競合を避けるため拒否される。VRChatのUI言語は英語である必要がある (OCR辞書に仮名が無い)。
→ 既定は完了まで待って <|"Status" -> "Joined", "World" -> ..., "Steps" -> ..., "Settings" -> ..., "Load" -> ..., "Room" -> ..., "Console" -> ..., "Warnings" -> ..., "ElapsedSeconds" -> ...|> または Failure["JoinFailed"|"JoinCancelled"|"JoinTimeout"|"JoinProcessExited"|"JoinAlreadyRunning", ...]
Options: "FavoritesList" -> "Favorite Worlds 1", "Group" -> "計算と自然" (既定値), "GroupAccess" -> "Group+", "Region" -> "JP", "EnterSlidenRoom" -> True, "ApproachConsole" -> False (Trueで入室後にSliden操作パネル前まで進む、VRCRealtimeStartPresentationの既定開始位置), "StepTimeoutSeconds" -> 20., "LoadTimeoutSeconds" -> 180., "NavigationTimeoutSeconds" -> 120., "Wait" -> True (Falseで直ちに戻り、VRCRealtimeJoinNGCGalleryWorldStatus[]で進捗を追う), "TotalTimeoutSeconds" -> Automatic, "VRCpilotDirectory" -> "C:\\Users\\imai_\\gitwork\\vrcpilot", "UVExecutable" -> Automatic, "VRChatPID" -> Automatic, "OSCHost" -> "127.0.0.1", "OSCPort" -> 9000, "DiagnosticDirectory" -> Automatic
例: VRCRealtimeJoinNGCGalleryWorld["Wait" -> False, "ApproachConsole" -> True]

### VRCRealtimeJoinNGCGalleryWorldStatus[] → <|"ProcessStatus" -> ..., "worldJoinPhase" -> ..., "worldJoinSteps" -> ..., "worldJoinLastTexts" -> ..., ...|>
VRCRealtimeJoinNGCGalleryWorldの進捗をAssociationで返す。

### VRCRealtimeJoinNGCGalleryWorldStop[opts]
実行中のVRCRealtimeJoinNGCGalleryWorldを中止する。control fileで"stop"を送り、待っても止まらなければ taskkill /T /F で強制終了する。
→ <|"Status" -> "Stopped" | "StopRequested", "Phase" -> ...|>
Options: "WaitSeconds" -> 8.0

## 音声資産管理

### VRCRealtimeInstallVoiceCommandModel[opts]
音声コマンド (「会話に参加」「会話から離脱」) のローカル認識に使うVosk日本語小モデル (約50MB) を事前ダウンロードする。モデルはSourceVault_voiceが管理する資産で、導入先はSourceVaultSpeechModelDirectory[] (既定 SourceVault_voice/asr)。旧 %LOCALAPPDATA%\VRCRealtime\models に既にあるものはそのまま使われる。VRCRealtimeStartは未導入なら初回に自動取得するので必須ではない。
→ <|...|> または Failure["UVNotFound"|"VRCpilotDirectoryMissing"|"InstallFailed", ...]
Options: "UVExecutable" -> Automatic, "VRCpilotDirectory" -> "C:\\Users\\imai_\\gitwork\\vrcpilot", "ModelDirectory" -> Automatic

### VRCRealtimeVoiceStatus[] → <|"Status" -> ..., "Missing" -> {...}, "ResolvedBy" -> ..., "SelectedEngine" -> "Piper"|"Aivis", "AivisEndpoint" -> ..., "SelectedVoice" -> ..., "SelectedLanguage" -> ..., "Executable" -> ..., "Model" -> ..., "PrivateTTSBroker" -> ..., "PrivateTTSBrokerPresent" -> ..., "Hint" -> ...|>
VRCRealtimeが実際に使うローカル音声資産 (合成ランタイム/声モデル/音声認識モデル) の状態を返す。"Status"が"OK"でなければ"Hint"に導入手順が入る。特定の声は前提にしておらず、導入されている声から選ぶ。実体はSourceVault_voice (SourceVaultVoiceStatus[] / SourceVaultVoiceView[]) が持つ薄いラッパー。