# VRCRealtime API リファレンス

VRChat上のアバターとOpenAI Realtime APIを結ぶノンブロッキング音声ブリッジ。外部プロセス (Python worker) を起動して制御する。すべての制御関数は即座に返り、実際の処理は非同期に進む。状態は `VRCRealtimeStatus[]` 等で随時取得する。

## 定数

### $VRCRealtimeVersion
型: String, 初期値: "0.11.0"
パッケージのバージョン。

### $VRCRealtimeOwnerAccount
型: String, 初期値: "nconc"
オーナー追尾モード ("AvatarMode" -> "OwnerFollow") で追跡するVRChatアカウント名の既定値。`VRCRealtimeSetOwnerAccount` で実行中に変更すると本変数も更新される。

### $VRCRealtimeDefaultSlideURL
型: String, 初期値: "https://transreal.sakura.ne.jp/cn/31.mp4"
`VRCRealtimeStartPresentation[]` を引数なしで呼んだときに使われる既定のSliden MP4 URL。

## ワーカー制御

### VRCRealtimeStart[opts]
OpenAI Realtime と VRChat の音声ブリッジを外部プロセス (uv run python) で起動し、直ちに制御用Associationを返す。このカーネル自身が既に起動中なら Failure["AlreadyRunning", ...] を返す。ロックファイル経由で発見した他プロセスの生存中ワーカーがある場合は、既定 ("ReplaceRunning" -> True) ではそれを停止してから起動する (VRCRealtimeStart::replacing を出す)。"ReplaceRunning" -> False にすると Failure["WorkerAlreadyRunning", ...] を返して二重起動を拒否する。
内部でNBAccess経由のOpenAI APIキー取得・uv実行ファイルの解決・VRC Pilotディレクトリの存在確認・(SourceVault連携が有効なら) SourceVault MCPサービスの起動確認・ローカル音声資産の解決を行い、いずれか失敗すると対応するタグ (WorkerMissing, NBAccessUnavailable, NotebookRequired, PaidAPINotAllowed, OpenAIDisabled, APIKeyMissing, UVNotFound, VRCpilotDirectoryMissing, InvalidOptions, WorkerAlreadyRunning, PrivateTTSMissing, SourceVaultServiceMissing, StartFailed, WorkerStartupFailed 等) の Failure を返す。起動直後の3秒以内にワーカーが引数エラー等で落ちた場合も Failure["WorkerStartupFailed", ...] として報告する (非ブロックの原則を保つため待機は最大3秒)。
→ <|"Status" -> ..., ...|> または Failure[tag, ...]
オーナー追尾は既定で幾何制御エンジン ("OwnerFollowEngine" -> "Formation") を使う: ネームプレートOCRで発見→体のランドマークで距離まで接近→向きを推定してオーナーの左後方へ回り込み→オーナーを視野右端に置いて停止。"Legacy" で旧OSCボタン制御に戻せる。
モデルと API: 既定は GPT-Live ("Model" -> "gpt-live-1")。GPT-Live 自身は道具を持たないため、SourceVault 検索・発表の開始/停止・スライド表示・アバター操作・画面を見る・質問モード切替などの道具は Responses 委譲でバックエンドモデル ("LiveBackendModel"、既定 Automatic = gpt-5.6-luna) が実行し、結果を GPT-Live が話す (ワーカーは VRCRealtime_live.py の変換層で Realtime と同じ内部経路に載せる)。スライドに原稿が無いときは、バックエンドが画像から原稿を書いて GPT-Live が読む (次のスライドの分は先に書いておく)。"Model" -> "gpt-realtime-2.1" で従来の Realtime 経路。"API" -> Automatic (モデル名で判定) | "Realtime" | "Live"。GPT-Live への接続が 401/403/404 で拒否されたときは再接続を繰り返さず status "error" と理由 (handshakeStatus) を残して止まる。
質問の受け方: "QuestionMode" -> "QuestionTime" (既定。スライドの区切りでだけ質問を受ける) | "Keywords" (特定ワード。区切りに加えて、発表中でも "InterruptWords" (既定 {"質問","スライド","プレゼン","発表"}。後ろ 2 語は発表自体を指す語で、「発表を中断して」「プレゼンを止めて」で割り込み→presentation_control で一時停止できる) を含む発話をローカル Vosk が聞き取ったら読み上げを止めて質問を受け、静かになったら中断した文から「それでは、説明に戻ります」で再開。Vosk が動いていないと質問タイムに戻る) | "Detect" (二段構え。区切りに加えて、発表中に部屋の声が一定時間続いたら一旦止め、その発話をモデルに渡して自分への質問か判断させる。質問なら応じ、独り言・雑談なら黙って続きを読む)。スライド区切りの質疑 (「質問があればどうぞ」表示と QuestionSeconds の待ち) はどのモードでも開く (2026-09-19 から。以前は Keywords/Detect で省いていたため、GPT-Live では説明が終わると同時に次のスライドへ進み、聞いた直後の質問ができなかった)。実行中は VRCRealtimeSetQuestionMode か、音声で「質問モードを切り替えたい」と言えばモデルが 3 つを説明して尋ね、set_question_mode ツールで設定する。
Options: "Model" -> "gpt-live-1", "API" -> Automatic, "LiveBackendModel" -> Automatic, "QuestionMode" -> "QuestionTime", "InterruptWords" -> Automatic, "FullDuplex" -> True (GPT-Live のみ。発表中以外は話しながらも部屋の声を聞く全二重。False で Realtime と同じ半二重 = 話している間と直後 OutputCooldownMilliseconds は聞かない), "Voice" -> "marin" (音声モデルの声名。ローカル読み上げの声とは別物), "Instructions" -> "日本語で自然かつ簡潔に会話してください。", "VRCpilotDirectory" -> "C:\\Users\\imai_\\gitwork\\vrcpilot", "UVExecutable" -> Automatic, "Notebook" -> Automatic (承認確認に使うフロントエンドノートブック), "RequirePaidAPIApproval" -> True (NBAccessでの有料API許可確認を必須にする。True のときフロントエンドノートブックが解決できないと Failure["NotebookRequired", ...]、未許可なら Failure["PaidAPINotAllowed", ...]), "ReplaceRunning" -> True (既に走っているワーカーがあれば停止してから起動する。False なら二重起動を許さず Failure["WorkerAlreadyRunning", ...]), "SafetyIdentifier" -> None (OpenAIへ渡す safety_identifier。None/空文字なら送らない), "InputSource" -> "VRChat", "InputDevice" -> Automatic, "OutputDevice" -> "CABLE-A Input", "VRChatPID" -> Automatic, "ChunkMilliseconds" -> 20, "VADThreshold" -> 0.5, "PrefixPaddingMilliseconds" -> 200, "SilenceDurationMilliseconds" -> 350, "ReconnectSeconds" -> 3300, "OutputCooldownMilliseconds" -> 1200, "AllowBargeIn" -> False。
視覚アテンションゲート系: "VisualAttentionGate" -> False, "VisualGateFramesPerSecond" -> 8., "VisualGateCenterTolerance" -> 0.18, "VisualGateMinFaceFraction" -> 0.05 (顔の見かけの大きさの下限、16:9換算の画面幅比。低すぎると通行人で誤発火するため0.05未満は非推奨), "VisualGateMouthMotionThreshold" -> 0.025, "VisualGateAudioRMSThreshold" -> 0.008, "VisualGateMaxStereoImbalance" -> 0.35, "VisualGateHangoverMilliseconds" -> 650, "VisualGatePrerollMilliseconds" -> 600, "VisualGateRequireNameplatePulse" -> True, "VisualGateNameplateBlueThreshold" -> 0.002, "VisualGateConversationLatch" -> True, "VisualGateConversationIdleSeconds" -> 30.。
アバター/オーナー追尾系: "AvatarMode" -> "Standalone" ("Standalone"|"OwnerFollow"|"Presentation"), "OwnerAccount" -> Automatic ($VRCRealtimeOwnerAccountを使用), "OwnerFollowEngine" -> "Formation" ("Formation"|"Legacy"), "OwnerFollowDistanceMeters" -> 1.8, "OwnerFollowRearAngleDegrees" -> 35., "OwnerFollowOwnerBearingDegrees" -> 33., "OwnerFollowMenuLocate" -> True, "OwnerFollowSearchPatrol" -> True, "OwnerFollowIdleAfterSeconds" -> 360., "OwnerFollowIdleRecheckSeconds" -> 300., "OwnerFollowTrackingFPS" -> 1.5, "OwnerFollowTargetScreenX" -> 0.66, "OwnerFollowTargetNameplateHeight" -> 0.055, "OwnerFollowLostSeconds" -> 1.25, "OwnerFollowFormationSeconds" -> 4.5, "OwnerFollowOrbitSpeed" -> 0.32, "OwnerFollowSearchTurnSpeed" -> 0.42, "OwnerFollowAssumeSingleAvatar" -> True, "OwnerFollowOCRMinSide" -> 1080, "OwnerFollowOCRBoxThreshold" -> 0.35, "OwnerFollowOCRTextScore" -> 0.35。
OSC/視覚/音声認識系: "OSCHost" -> "127.0.0.1", "OSCPort" -> 9000, "EnableVisionCapture" -> True, "VoiceCommands" -> True (「会話に参加/離脱」のローカルVosk認識), "VoiceCommandModelDirectory" -> Automatic, "VoiceCommandMinConfidence" -> 0.45, "VisionImageMaxWidth" -> 960, "VisionJPEGQuality" -> 78。
SourceVault連携系: "EnableSourceVault" -> True (有効時、SourceVault MCPサービスが停止していれば起動を試みる。起動できない/command queueが応答しないとメッセージ mcpstopped/mcpfailed/mcpbusy を出す), "EnableVoiceWebSearch" -> False (既定オフ、opt-in), "SourceVaultServiceDirectory" -> Automatic, "SourceVaultKinds" -> {"search","packageapi","llmlog","mail","eagle","snapshot"}, "SourceVaultKB" -> Automatic, "SourceVaultGroup" -> None, "SourceVaultReleaseContext" -> None, "SourceVaultLimit" -> 3, "SourceVaultVoiceWaitSeconds" -> 60., "LocalSpeechMaxSeconds" -> 60., "OwnerPresencePrivacyThreshold" -> 0.25 (この値以上のPrivacy.Levelの資料はオーナーが近くにいるときだけ回答)。
ローカル音声合成系: "PrivateTTSBroker" -> Automatic, "PrivateTTS" -> Automatic (Automaticは資産があれば使用、無ければ警告 (VRCRealtimeStart::novoice) してSourceVault連携を切って起動; Trueは必須 (資産が無ければ Failure["PrivateTTSMissing", ...]), Falseは不使用), "SpeechVoice" -> Automatic, "PiperExecutable" -> Automatic, "PiperModel" -> Automatic, "PiperConfig" -> Automatic, "SpeechLengthScale" -> Automatic, "SpeechNoiseScale" -> Automatic。エンジンは解決した声から決まる: SourceVault_voice の既定声 (または "SpeechVoice" で指定した声) が AivisSpeech のものならエンジンの応答を確かめたうえで AivisSpeech (VOICEVOX 互換 HTTP、loopback 限定) を使い、Piper の声はセッション内フォールバックとして渡す。SpeechLengthScale (>1 = ゆっくり) は AivisSpeech では speedScale の逆数に写る。エンジンが応答しなければ従来どおり Piper。
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
SlidenのMP4を取得・展開し、操作パネルへURLを入力して各スライドを元解像度で自動発表する。targetはURL (既定 $VRCRealtimeDefaultSlideURL) または登録済み発表タイトル (表記ゆれは登録簿側が吸収する)。原稿があればそれを話し、無ければ1枚ごとの画像説明になる。発表登録簿はSourceVault MCPサービス越しに引くため、呼び出し時にサービスが停止していれば起動を試みる (URL直指定なら原稿なしで続行、タイトル指定でサービスを起動できない場合は Failure["SourceVaultServiceMissing", ...])。
→ <|"Status" -> ..., ...|> または Failure[tag, ...] (例: "SlideDeckRegistryUnavailable", "PresentationNotRegistered", "PresentationWithheld", "InvalidSlideURL", "SourceVaultServiceMissing", "InvalidPresentationOptions")
"NavigateToConsole" は既定Automatic: まず今いる場所から操作パネルが使えるか確認し、使えるならそのまま開始し、使えないときだけ部屋の中でパネルを探して前まで移動してから開始する。True は常に探索してから開始、False は探索しない (パネルの前にいなければ失敗)。
Options: "SecondsPerSlide" -> Automatic (登録簿→無ければ30、3〜300の範囲), "ConsoleTimeoutSeconds" -> 30., "NavigateToConsole" -> Automatic, "NavigationTimeoutSeconds" -> 180., "DownloadTimeoutSeconds" -> 120., "MaxDownloadMegabytes" -> 256., "StartSlide" -> Automatic, "EndSlide" -> Automatic, "NarrationInstructions" -> Automatic, "Talk" -> Automatic (登録簿の発表原稿。None で原稿を使わず画像説明のみ), "Title" -> Automatic, "QuestionSeconds" -> 5. (スライド切替時の質疑応答秒数、0で質疑応答なし), "FinalQuestionSeconds" -> 120. (全スライド終了後の質疑応答上限秒数), "QuestionMode" -> Automatic (この発表の質問の受け方。Automatic はワーカーの現在値のまま。"QuestionTime"|"Keywords"|"Detect" を渡すとワーカーの値も切り替わる。QuestionSeconds の区切り質疑はどのモードでも開く)。
例: VRCRealtimeStartPresentation["計算と自然と31", "NavigateToConsole" -> True, "QuestionSeconds" -> 10.]

### VRCRealtimePresentationStatus[] → <|"ProcessStatus" -> ..., "presentation..." -> ...|>
自動Sliden発表の状態 (presentationで始まるキーのみ) をAssociationで返す。

### VRCRealtimePresentationControl[action : ("Pause" | "Resume" | "Stop" | "ResumeHere" | "ResumeBack")] → <|"Status" -> "Sent", ...|>
実行中の発表を一時停止・再開・停止する。音声の「一時停止して」「再開して」「打ち切って」と同じ経路を通る。質疑応答中にスライドを動かすと、発表側が「表示中のスライドから続けますか、元の位置へ戻りますか」と尋ねてくる。その返事は "ResumeHere" (表示中のスライドから続ける) / "ResumeBack" (発表が進んでいた位置へ戻る) で返せる。

### VRCRealtimeStopPresentation[] → <|"Status" -> "Sent", ...|>
実行中の自動Sliden発表を停止する。

### VRCRealtimeSetQuestionMode[mode : ("QuestionTime" | "Keywords" | "Detect")] → <|"Status" -> "Sent", ..., "QuestionMode" -> mode|>
発表中の質問の受け方を実行中に切り替える (発表の途中でも次の発話から効く)。"特定ワード"/"二段構え"/"質問タイム" や "Words" も受ける。不明な名前は Failure["InvalidQuestionMode", ...]。Keywords はローカル音声認識 (Vosk) が動いていないと効かず、そのときワーカーは質問タイムで進める (状態 presentationQuestionModeEffective / presentationQuestionModeReason)。音声で「質問モードを切り替えたい」と言っても同じ結果になる。
例: VRCRealtimeSetQuestionMode["Keywords"]

### VRCRealtimeQuestionMode[] → "QuestionTime" | "Keywords" | "Detect" | Missing["NotRunning"]
実行中のワーカーの質問モード (音声で切り替えられた後の値)。VRCRealtimeStatus[] の "QuestionMode" と同じ。

## モード・オーナー追尾

### VRCRealtimeSetMode[mode : ("Standalone" | "OwnerFollow" | "Presentation")] → <|"Status" -> "Sent", ...|>
単独動作・オーナー追尾・プレゼンモードを実行中に切り替える。OwnerFollow (Formationエンジン) の探索は、全周スキャンが空振りすると最も開けた方向へ数m歩いて再スキャン ("OwnerFollowSearchPatrol")、"OwnerFollowMenuLocateAfterSweeps"回でメニューから所在確認/Joinし、オーナーがオフライン・参加不可・または"OwnerFollowIdleAfterSeconds"経過で「待機」(旋回せず静止、見かけたら接近、"OwnerFollowIdleRecheckSeconds"ごとにメニュー再確認) に入る。待機中・不在中・別ワールドでは非公開資料 (PL >= OwnerPresencePrivacyThreshold) は答えない。

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
起動直後 (Homeワールド) のVRChatをVRC Pilotで操作し、メニュー Worlds → Favorite Worlds 1 → NCG Gallery → New Instance → Group の順に進み、グループ/GroupAccess/Regionを確認してCreate Instance → Joinでグループインスタンスに入場し、右手のSlidenスクリーンのある部屋まで歩く。追尾/発表中のworkerがあれば入力の競合を避けるため拒否される。VRChatのUI言語は英語である必要がある (OCR辞書に仮名が無い)。画面遷移のクリック (Worlds / お気に入りリスト / ワールドのタイル / New Instance / Group) がVRChatに取りこぼされ、押した文字が同じ位置・同じ画素のまま次の画面が出ないときは、同じ項目を最大3回まで押し直す (StepTimeoutSecondsを回数で等分、押し直しはWarningsに記録)。Create Instance はインスタンスの二重作成を避けるため対象外 (Join は従来どおり独自に1回だけ押し直す)。
→ 既定は完了まで待って <|"Status" -> "Joined", "World" -> ..., "Steps" -> ..., "Settings" -> ..., "Load" -> ..., "Room" -> ..., "Console" -> ..., "Warnings" -> ..., "ElapsedSeconds" -> ...|> または Failure["JoinFailed"|"JoinCancelled"|"JoinTimeout"|"JoinProcessExited"|"JoinAlreadyRunning"|"WorkerAvatarActive"|"InvalidOptions", ...]
Options: "FavoritesList" -> "Favorite Worlds 1", "Group" -> "計算と自然" (既定値), "GroupAccess" -> "Group+" ("Group"|"Group+"|"Group Public"), "Region" -> "JP" ("USW"|"USE"|"EU"|"JP"), "EnterSlidenRoom" -> True, "ApproachConsole" -> False (Trueで入室後にSliden操作パネル前まで進む、VRCRealtimeStartPresentationの既定開始位置), "StepTimeoutSeconds" -> 20., "LoadTimeoutSeconds" -> 180., "NavigationTimeoutSeconds" -> 210. (10〜600の範囲。入室はガラス壁越しに画面が見える回り込み分の余裕が要る), "Wait" -> True (Falseで直ちに戻り、VRCRealtimeJoinNGCGalleryWorldStatus[]で進捗を追う), "TotalTimeoutSeconds" -> Automatic, "VRCpilotDirectory" -> "C:\\Users\\imai_\\gitwork\\vrcpilot", "UVExecutable" -> Automatic, "VRChatPID" -> Automatic, "OSCHost" -> "127.0.0.1", "OSCPort" -> 9000, "DiagnosticDirectory" -> Automatic
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