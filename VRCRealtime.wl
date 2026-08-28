(* ::Package:: *)

(* VRCRealtime.wl -- non-blocking VRChat/OpenAI Realtime voice bridge *)

(* パッケージ読込より先に Global` 側で同名シンボルが作られていると (別ノートブックや
   カーネル再起動後に VRCRealtimeKill[] などを先に評価・参照した場合)、以後この
   パッケージの定義が Global` のシンボルに隠され、読込時に shdw 警告が出る。
   読込のたびに Global` 側の同名シンボル (接頭辞 VRCRealtime / $VRCRealtime のもの
   だけ) を取り除いてから定義する。利用者の変数 (vrc, rt など) には触れない。 *)
Quiet[Remove["Global`VRCRealtime*"], {Remove::rmnsm}];
Quiet[Remove["Global`$VRCRealtime*"], {Remove::rmnsm}];

BeginPackage["VRCRealtime`"];

$VRCRealtimeVersion::usage = "$VRCRealtimeVersion はパッケージのバージョン。";

$VRCRealtimeOwnerAccount::usage =
  "$VRCRealtimeOwnerAccount はオーナー追尾モードで追跡するVRChatアカウント名。既定値は \"nconc\"。";

$VRCRealtimeDefaultSlideURL::usage =
  "$VRCRealtimeDefaultSlideURL is the default Sliden MP4 URL used by automatic presentations.";

VRCRealtimeStart::usage =
  "VRCRealtimeStart[] は OpenAI Realtime と VRChat の音声ブリッジを外部プロセスで起動し、直ちに制御情報を返す。
" <>
  "オーナー追尾 (\"AvatarMode\" -> \"OwnerFollow\") は既定で幾何制御エンジン (\"OwnerFollowEngine\" -> \"Formation\") を使う: ネームプレート OCR で発見 → 体のランドマークで距離 (\"OwnerFollowDistanceMeters\"、既定 1.8 m) まで接近 → 向きを推定してオーナーの左後方 (\"OwnerFollowRearAngleDegrees\"、既定 35) へ回り込み → オーナーを視野右端 (\"OwnerFollowOwnerBearingDegrees\"、既定 33 度) に置いて停止。ヨーは相対マウス入力 (3600 counts/360 度) で決定論的に回す。\"Legacy\" で旧 OSC ボタン制御に戻せる。";

VRCRealtimeStatus::usage =
  "VRCRealtimeStatus[] は常駐ワーカーの状態を Association で返す。";

VRCRealtimeStop::usage =
  "VRCRealtimeStop[] は常駐ワーカーへ正常終了を要求し、既定で最大8秒まで終了を待つ。";

VRCRealtimeKill::usage =
  "VRCRealtimeKill[] は応答しない常駐ワーカーを強制終了する。";

VRCRealtimeSetInstructions::usage =
  "VRCRealtimeSetInstructions[text] は実行中の Realtime セッションの instructions を更新する。";

VRCRealtimeMute::usage =
  "VRCRealtimeMute[True|False] は VRChat 音声の OpenAI への送信を一時停止または再開する。";

VRCRealtimeEndConversation::usage =
  "VRCRealtimeEndConversation[] は現在の会話を終了し、視覚的な開始ゲートを再び有効にする (この経路は従来どおり、正面から話しかけられれば再開する)。
" <>
  "会話の開始/終了は音声でも指示できる: 正面から話しかける (視覚ゲート) ほか、どの方向からでも・フォロー移動中でも「会話に参加(して)」で参加し以後の質問に答え、「会話から離脱(して)」で離脱して答えるのを止める。
" <>
  "「会話から離脱」で離脱した後は視覚ゲートだけでは再開しない (目の前で話し続けても無視する)。再開するのは (1)「会話に参加(して)」と言われたとき、(2) 正面から挨拶 (こんにちは/こんばんは/おはよう/もしもし/ハロー/すみません) されたとき の 2 つだけで、時間経過では再開しない。これらの語はローカルの Vosk 認識 (文法制約、音声は外部へ出ない) で検出する (\"VoiceCommands\" option)。状態は VRCRealtimeStatus[] の voiceKeywordLeft で確認できる。";

VRCRealtimeLook::usage =
  "VRCRealtimeLook[] は現在のVRChat画面をRealtimeモデルへ1枚送り、音声で説明させる。\"EnableVisionCapture\" -> True (既定) のとき、音声で『目の前に何がある』『これは何』『何が見える』と尋ねられた場合もモデルが look_at_vrchat を呼んで同じ経路で VRChat ウインドウのキャプチャから答える (視覚ゲートが無くても focus-free に 1 枚取得)。";

VRCRealtimeStartPresentation::usage =
  "VRCRealtimeStartPresentation[url] は Sliden の MP4 を取得・展開し、操作パネルへ URL を入力して各スライドを元解像度で自動発表する。\n" <>
  "\"NavigateToConsole\" は既定 Automatic: まず今いる場所から操作パネルが使えるか確認し、使えるならそのまま開始する。使えないときだけ部屋の中でパネルを探して前まで移動してから開始する。判定は URL 入力欄のある操作パネルが見えるか (configure_url が待つ条件と同一) で、頭の上下だけの位置ずれは「見えている」に含む。\n" <>
  "\"NavigateToConsole\" -> True は常に探索してから開始、False は探索しない (パネルの前にいなければ失敗する従来の挙動)。\n" <>
  "VRCRealtimeStartPresentation[\"\:8a08\:7b97\:3068\:81ea\:713631\"] のように登録済みの発表タイトルでも開始できる。タイトル・スライド URL・発表シナリオ (原稿) は SourceVault の発表登録簿 (SlideWorkflow パレットの「登録」) から引く。原稿があればそれを話し、無ければ従来どおり 1 枚ごとの画像説明になる。\"Talk\" -> None で原稿を使わない。";

VRCRealtimeListPresentations::usage =
  "VRCRealtimeListPresentations[] は SourceVault の発表登録簿に登録済みの発表 (タイトル / スライド URL / 枚数) を返す。";

VRCRealtimePresentationStatus::usage =
  "VRCRealtimePresentationStatus[] returns the automatic Sliden presentation state.";

VRCRealtimePresentationControl::usage =
  "VRCRealtimePresentationControl[\"Pause\"|\"Resume\"|\"Stop\"] は実行中の発表を一時停止・再開・停止する。音声の「一時停止して」「再開して」「打ち切って」と同じ経路を通る。";

VRCRealtimeStopPresentation::usage =
  "VRCRealtimeStopPresentation[] stops the current automatic Sliden presentation.";


VRCRealtimeSetOwnerAccount::usage =
  "VRCRealtimeSetOwnerAccount[name] は追尾対象アカウントを更新する。";

VRCRealtimeSetFollowDistance::usage =
  "VRCRealtimeSetFollowDistance[m] は追尾距離をメートルで調整する (既定 1.8、0.8〜6)。\"OwnerFollowEngine\" -> \"Legacy\" のときは従来どおりネームプレートの画面高比 (0.005〜0.15、大きい値ほど近く) として解釈する。";

VRCRealtimeSetFormation::usage =
  "VRCRealtimeSetFormation[\"RearAngle\" -> deg, \"OwnerBearing\" -> deg] は追尾隊形を実行中に変更する。RearAngle はオーナーの真横 (0) からどれだけ後ろに控えるか (既定 35、0〜90)。OwnerBearing は停止時にオーナーを視野のどこに置くか (視野中心から右へ何度か、既定 33、10〜42)。ボットの向き = オーナーの向き + (90 - RearAngle - OwnerBearing) 度。";

VRCRealtimeMonitor::usage =
  "VRCRealtimeMonitor[] は1秒ごとに更新される軽量な状態表示を返す。";

VRCRealtimeJoinNGCGalleryWorld::usage =
  "VRCRealtimeJoinNGCGalleryWorld[] は起動直後 (Home ワールド) の VRChat を VRC Pilot で操作し、メニュー \[Rule] Worlds \[Rule] Favorite Worlds 1 \[Rule] NCG Gallery \[Rule] New Instance \[Rule] Group の順に進み、グループ (\:8a08\:7b97\:3068\:81ea\:7136) / Group+ / JP を確認して Create Instance \[Rule] Join でグループインスタンスに入場し、右手の Sliden スクリーンのある部屋まで歩く。\n" <>
  "VRCRealtimeJoinNGCGalleryWorld[world] で別ワールド名を指定する。既定では完了まで待って結果 (Association) または Failure を返す。\"Wait\" -> False で直ちに戻り、VRCRealtimeJoinNGCGalleryWorldStatus[] で進捗を追う。\"ApproachConsole\" -> True で入室後に Sliden 操作パネル前まで進む (VRCRealtimeStartPresentation の既定開始位置)。VRChat の UI 言語は英語である必要がある (OCR 辞書に仮名が無い)。";

VRCRealtimeJoinNGCGalleryWorldStatus::usage =
  "VRCRealtimeJoinNGCGalleryWorldStatus[] は VRCRealtimeJoinNGCGalleryWorld の進捗 (worldJoinPhase / worldJoinSteps / worldJoinLastTexts など) を Association で返す。";

VRCRealtimeSetMode::usage =
  "VRCRealtimeSetMode[\"Standalone\"|\"OwnerFollow\"|\"Presentation\"] は単独動作、オーナー追尾、プレゼンモードを実行中に切り替える。\n" <>
  "OwnerFollow (Formation エンジン) の探索: 全周スキャンが空振りすると最も開けた方向へ数 m 歩いて再スキャン (\"OwnerFollowSearchPatrol\")、\"OwnerFollowMenuLocateAfterSweeps\" 回でメニューから所在確認/Join、オーナーがオフライン・参加不可・または \"OwnerFollowIdleAfterSeconds\" 経過で「待機」(旋回せず静止、見かけたら接近、\"OwnerFollowIdleRecheckSeconds\" ごとにメニュー再確認)。待機中・不在中・別ワールドでは非公開資料 (PL>=OwnerPresencePrivacyThreshold) は答えない。";

VRCRealtimeLocateOwner::usage =
  "VRCRealtimeLocateOwner[] はフォローモードのメニュー探索を即時に実行させる: メインメニュー → Location (Here) ページ → 下へスクロールして Users 一覧にオーナーがいるか確認し、いなければ右 Wing → ハンバーガー → Friends → 矢印で Online 一覧 → オーナー → Join (押せる場合) でオーナーのインスタンスへ移動する。通常は全周探索が \"OwnerFollowMenuLocateAfterSweeps\" 回 (既定 2) 空振りすると自動で実行される。";

VRCRealtimeInstallVoiceCommandModel::usage =
  "VRCRealtimeInstallVoiceCommandModel[] は音声コマンド (「会話に参加」「会話から離脱」) のローカル認識に使う Vosk 日本語小モデル (約 50 MB) を事前ダウンロードする。\n" <>
  "モデルは SourceVault_voice が管理する資産で、導入先は SourceVaultSpeechModelDirectory[] (既定 SourceVault_voice/asr)。旧 %LOCALAPPDATA%\\VRCRealtime\\models に既にあるものはそのまま使われる。\n" <>
  "VRCRealtimeStart は未導入なら初回に自動取得するので必須ではない。WL だけで完結させたい場合は SourceVaultInstallSpeechModel[] でもよい。";

VRCRealtimeVoiceStatus::usage =
  "VRCRealtimeVoiceStatus[] は VRCRealtime が実際に使うローカル音声資産 (合成ランタイム / 声モデル / 音声認識モデル) の状態を返す。\n" <>
  "\"Status\" が \"OK\" でなければ \"Hint\" に導入手順が入る。特定の声は前提にしておらず、導入されている声から選ぶ。\n" <>
  "実体は SourceVault_voice (SourceVaultVoiceStatus[] / SourceVaultVoiceView[]) が持つ。";

VRCRealtimeJoinNGCGalleryWorldStop::usage =
  "VRCRealtimeJoinNGCGalleryWorldStop[] は実行中の VRCRealtimeJoinNGCGalleryWorld を中止する。";

Begin["`Private`"];

VRCRealtime`$VRCRealtimeVersion = "0.10.1";
VRCRealtime`$VRCRealtimeOwnerAccount = "nconc";
VRCRealtime`$VRCRealtimeDefaultSlideURL =
  "https://transreal.sakura.ne.jp/cn/31.mp4";

$iPackageDirectory = DirectoryName[$InputFileName];

(* 外部実装 (Python) の置き場。配布形では VRCRealtime_info/resources/python/ に
   まとめてある。旧配置 ($packageDirectory 直下) もそのまま動かす: 手元の
   作業ツリーを一度に作り替えなくてよいようにするための後方互換で、新しい
   配置が見つかればそちらが優先される。 *)
$iResourceRoot = FileNameJoin[{$iPackageDirectory, "VRCRealtime_info", "resources"}];
$iPythonDirectory =
  With[{modern = FileNameJoin[{$iResourceRoot, "python"}]},
    If[FileExistsQ[FileNameJoin[{modern, "VRCRealtime_worker.py"}]],
      modern, $iPackageDirectory]];

$iWorkerPath = FileNameJoin[{$iPythonDirectory, "VRCRealtime_worker.py"}];
$iWorldJoinPath = FileNameJoin[{$iPythonDirectory, "VRCRealtime_worldjoin.py"}];
$iPrivateTTSBrokerPath = FileNameJoin[{$iPythonDirectory, "VRCRealtime_private_tts.py"}];
$iProcess = None;
$iStateFile = None;
$iControlFile = None;
$iSession = <||>;
$iProcessDiagnostics = <||>;

ClearAll[iFailure, iEnsureNBAccess, iProcessRunningQ, iSendCommand,
  iReadState, iNumberString, iResolveNotebook, iResolveUV,
  iReadFinishedProcessDiagnostics, iResolveSourceVaultServiceDirectory,
  iPythonPath, iEnsureVoiceLayer, iFallbackVoiceScan, iResolveVoice,
  iEnsureSourceVaultService, iReadWorkerLock, iWorkerLockLive,
  iStopWorkerFromLock];

iPythonPath[name_String] := FileNameJoin[{$iPythonDirectory, name}];

iFailure[tag_String, message_String, extra_: <||>] :=
  Failure[tag, Join[<|"MessageTemplate" -> message|>, extra]];

iEnsureNBAccess[] := Module[{path, loaded},
  If[Length[DownValues[NBAccess`NBGetAPIKey]] > 0, Return[True]];
  path = FileNameJoin[{$iPackageDirectory, "NBAccess.wl"}];
  If[!FileExistsQ[path], Return[False]];
  loaded = Quiet @ Check[
    Block[{$CharacterEncoding = "UTF-8"}, Get[path]],
    $Failed
  ];
  loaded =!= $Failed && Length[DownValues[NBAccess`NBGetAPIKey]] > 0
];

iProcessRunningQ[] :=
  MatchQ[$iProcess, _ProcessObject] &&
    TrueQ @ Quiet @ Check[ProcessStatus[$iProcess] === "Running", False];

iResolveNotebook[Automatic] := Quiet @ Check[EvaluationNotebook[], $Failed];
iResolveNotebook[nb_] := nb;

iNumberString[value_] := ToString[value, InputForm];

iResolveUV[path_String] := Module[{expanded = ExpandFileName[path]},
  If[FileExistsQ[expanded], expanded, $Failed]
];

iResolveUV[Automatic] := Module[{userProfile, localAppData, candidates},
  userProfile = Quiet @ Check[Environment["USERPROFILE"], $Failed];
  localAppData = Quiet @ Check[Environment["LOCALAPPDATA"], $Failed];
  candidates = DeleteCases[
    {
      Quiet @ Check[FindExecutable["uv"], $Failed],
      If[StringQ[userProfile],
        FileNameJoin[{userProfile, ".local", "bin", "uv.exe"}],
        $Failed
      ],
      If[StringQ[localAppData],
        FileNameJoin[{localAppData, "Programs", "uv", "uv.exe"}],
        $Failed
      ]
    },
    $Failed
  ];
  SelectFirst[candidates, StringQ[#] && FileExistsQ[#] &, $Failed]
];

iResolveUV[_] := $Failed;

iResolveSourceVaultServiceDirectory[path_String] := Module[{expanded = ExpandFileName[path]},
  If[DirectoryQ[expanded] && FileExistsQ[FileNameJoin[{expanded, "status.json"}]],
    expanded, $Failed]
];

iResolveSourceVaultServiceDirectory[Automatic] := Module[
  {fromAPI, userProfile, machine, candidates},
  fromAPI = If[
    Length[DownValues[SourceVault`SourceVaultServiceRuntimeDir]] > 0,
    Quiet @ Check[SourceVault`SourceVaultServiceRuntimeDir["sourcevault"], $Failed],
    $Failed
  ];
  If[StringQ[fromAPI] && DirectoryQ[fromAPI], Return[fromAPI]];
  userProfile = Quiet @ Check[Environment["USERPROFILE"], $Failed];
  machine = ToString[$MachineName];
  candidates = If[StringQ[userProfile],
    FileNameJoin[{#, "udb", "sourcevault", "runtime", machine,
      "services", "sourcevault"}] & /@
      {FileNameJoin[{userProfile, "Dropbox"}],
       FileNameJoin[{userProfile, "Dropbox (Personal)"}]},
    {}
  ];
  SelectFirst[candidates,
    DirectoryQ[#] && FileExistsQ[FileNameJoin[{#, "status.json"}]] &,
    $Failed]
];

iResolveSourceVaultServiceDirectory[_] := $Failed;

(* ---- SourceVault MCP サービスの確保 --------------------------------------
   資料検索も発表登録簿 (スライド URL と発表原稿) も MCP サービス越しに引く。
   止まっていると発表は「登録簿の照合が timeout → 内蔵の表へ fallback」で
   静かに縮退し、原稿だけが落ちる (2026-08-21 実機: presentationCatalogError =
   "slide deck lookup timed out"、catalogRoute = builtin)。黙って劣化させず、
   止まっていることを報告して起動を試みる。
   起動できたらサービスの runtime ディレクトリも現れるので、呼び出し側は
   この後に必ず解決し直すこと。 *)
iEnsureSourceVaultService[] := Module[{result, deadline},
  If[Length[DownValues[SourceVault`SourceVaultMCPRunningQ]] === 0,
    Return[<|"Status" -> "Unavailable", "Reason" -> "ServiceManagerAbsent",
      "Started" -> False|>]];
  If[TrueQ[Quiet @ Check[SourceVault`SourceVaultMCPRunningQ[], False]],
    Return[<|"Status" -> "Running", "Started" -> False|>]];
  Message[VRCRealtimeStart::mcpstopped];
  result = Quiet @ Check[SourceVault`SourceVaultStartMCP[], $Failed];
  (* 起動は proxy → service kernel → /health の順に立ち上がるので即座には
     healthy にならない。数十秒だけ待つ (VRCRealtimeStart 自体は非ブロックだが、
     ここで待たないと直後の runtime ディレクトリ解決が必ず空振りする)。 *)
  deadline = AbsoluteTime[] + 40.;
  While[AbsoluteTime[] < deadline,
    Pause[1.];
    If[TrueQ[Quiet @ Check[SourceVault`SourceVaultMCPRunningQ[], False]],
      Return[<|"Status" -> "Started", "Started" -> True, "Result" -> result|>]]];
  Message[VRCRealtimeStart::mcpfailed,
    If[AssociationQ[result], Lookup[result, "Status", "unknown"], ToString[result]]];
  <|"Status" -> "StartFailed", "Started" -> True, "Result" -> result|>
];

(* ---- ローカル音声資産 (SourceVault_voice 管轄) ----------------------------
   合成ランタイムと声モデルは SourceVault_voice が持つ。ここは「どの声で
   しゃべるか」を決めて worker へ渡すだけで、特定の声 (旧実装は tsukuyomi を
   直書きしていた) を前提にしない。SourceVault_voice が無い環境でも従来配置を
   自前で探して動く。 *)

iEnsureVoiceLayer[] := Module[{path},
  If[Length[DownValues[SourceVault`SourceVaultVoiceRuntime]] > 0, Return[True]];
  path = FileNameJoin[{$iPackageDirectory, "SourceVault_voice.wl"}];
  If[!FileExistsQ[path], Return[False]];
  Quiet @ Check[Block[{$CharacterEncoding = "UTF-8"}, Get[path]], $Failed];
  Length[DownValues[SourceVault`SourceVaultVoiceRuntime]] > 0
];

(* SourceVault_voice が無いときの最小限の探索。配置規約は同じ。 *)
iFallbackVoiceScan[] := Module[{roots, exe, models, dirs, onnx, cfg},
  roots = Select[{
      FileNameJoin[{$iPackageDirectory, "SourceVault_voice"}],
      FileNameJoin[{$iPackageDirectory, "VRCRealtime_tts"}]},
    DirectoryQ];
  If[roots === {}, Return[<|"Executable" -> None, "Voice" -> None|>]];
  exe = SelectFirst[
    FileNameJoin[{#, "tts", "runtime", "piper", "bin", "piper.exe"}] & /@ roots,
    FileExistsQ, None];
  (* 旧 VRCRealtime_tts は tts/ 階層を持たない *)
  If[exe === None,
    exe = SelectFirst[
      FileNameJoin[{#, "runtime", "piper", "bin", "piper.exe"}] & /@ roots,
      FileExistsQ, None]];
  models = Select[
    Join[FileNameJoin[{#, "tts", "models"}] & /@ roots,
         FileNameJoin[{#, "models"}] & /@ roots],
    DirectoryQ];
  dirs = Sort @ Flatten[Select[FileNames["*", #], DirectoryQ] & /@ models];
  onnx = SelectFirst[Flatten[Sort @ FileNames["*.onnx", #] & /@ dirs], FileExistsQ, None];
  If[onnx === None, Return[<|"Executable" -> exe, "Voice" -> None|>]];
  cfg = SelectFirst[
    {onnx <> ".json", FileNameJoin[{DirectoryName[onnx], "config.json"}]},
    FileExistsQ, None];
  <|"Executable" -> exe,
    "Voice" -> <|
      "Name" -> FileNameTake[DirectoryName[onnx]],
      "Model" -> onnx, "Config" -> cfg,
      "Language" -> "", "Multilingual" -> False|>|>
];

(* 明示指定 > SourceVault_voice > 素の探索。返り値は必ず Association。
   解決した声が AivisSpeech のものなら "Engine" -> "Aivis" になり、broker には
   エンドポイントとスタイル id を渡す (piper の exe/model はフォールバック)。 *)
iResolveVoice[exeOpt_, modelOpt_, configOpt_, voiceOpt_] :=
  Module[{layer, runtime, voice, exe, model, config, language, scan, missing = {}, hint = None,
          engineChoice = "Piper", aivisEndpoint = "", aivisStyle = 0, aivisName = None},
    layer = iEnsureVoiceLayer[];
    If[layer,
      (* Check は使わない: SourceVaultVoice[] は「指定された声が消えていたので
         先頭に落とした」ときに警告を出すが、それは正常に解決できた場合であり、
         Check だと $Failed に化けて声が無いことにされてしまう。どちらの関数も
         例外を投げず Association か Failure を返すので、型で判定する。 *)
      runtime = Quiet @ SourceVault`SourceVaultVoiceRuntime[];
      exe = If[AssociationQ[runtime], Lookup[runtime, "Executable", None], None];
      hint = If[AssociationQ[runtime], Lookup[runtime, "Hint", None], None];
      voice = Quiet @ Replace[voiceOpt, {
        Automatic | None :> SourceVault`SourceVaultVoice[],
        s_String :> SourceVault`SourceVaultVoice[s]}];
      If[!AssociationQ[voice],
        (* Failure 自身が持つ理由を優先する。ここで一律に InstallHint を使うと、
           「名前で選んだ声が見つからないだけ」なのに「ローカル音声資産はそろって
           います」という矛盾したメッセージになる (実際に出た)。 *)
        With[{own = If[FailureQ[voice], Lookup[Last[voice], "Hint", None], None],
              avail = If[FailureQ[voice], Lookup[Last[voice], "Available", None], None]},
          hint = Which[
            StringQ[own] && ListQ[avail] && avail =!= {},
              own <> "\n導入済みの声: " <> StringRiffle[avail, ", "],
            StringQ[own], own,
            True, Replace[Quiet @ SourceVault`SourceVaultVoiceInstallHint[],
              Except[_String] -> hint]]];
        voice = None];
      (* 解決した声が AivisSpeech のもの: エンジンの応答を確かめてから採用する。
         採用時は piper の声をフォールバックとして選び直し、応答しなければ
         そのまま piper の声へ落とす (どちらも broker が黙り込むよりよい) *)
      If[AssociationQ[voice] && Lookup[voice, "Engine", ""] === "AivisSpeech",
        If[TrueQ[Quiet @ Check[
              If[Length[DownValues[SourceVault`SourceVaultVoiceAivisEnsure]] > 0,
                SourceVault`SourceVaultVoiceAivisEnsure[], False], False]],
          engineChoice = "Aivis";
          aivisEndpoint = Replace[Lookup[voice, "Model", ""], Except[_String] -> ""];
          aivisStyle = Replace[Lookup[voice, "StyleId", 0], Except[_Integer] -> 0];
          aivisName = Lookup[voice, "Name", None],
          hint = "指定された声は AivisSpeech Engine のものですが、エンジンが応答しません。" <>
            "Piper の声があればそちらへ落とします。"];
        voice = SelectFirst[
          Select[Quiet @ Check[SourceVault`SourceVaultVoices[], {}], AssociationQ],
          Lookup[#, "Engine", ""] === "PiperPlus" &, None]],
    (* else: SourceVault_voice が無い環境 *)
      scan = iFallbackVoiceScan[];
      exe = scan["Executable"];
      voice = scan["Voice"]];
    If[exe === $Failed, exe = None];
    If[!AssociationQ[voice], voice = None];

    (* 明示指定は常に勝つ *)
    exe = Replace[exeOpt, {Automatic | None :> exe, s_String :> s}];
    model = Replace[modelOpt, {
      Automatic | None :> If[voice === None, None, voice["Model"]],
      s_String :> s}];
    config = Replace[configOpt, {
      Automatic | None :> If[voice === None, None, Replace[voice["Config"], None -> ""]],
      s_String :> s}];
    (* language は声の性質。多言語の声にだけ渡す (単言語の声は拒否しうる) *)
    language = If[voice =!= None && TrueQ[Lookup[voice, "Multilingual", False]],
      Replace[Lookup[voice, "Language", ""], Except[_String] -> ""], ""];

    (* Aivis 採用時は piper の exe/model はフォールバック (無くても動く)。
       Piper が主のときだけ従来どおり欠品として扱う *)
    If[engineChoice =!= "Aivis",
      If[!(StringQ[exe] && FileExistsQ[exe]), AppendTo[missing, "TTSRuntime"]];
      If[!(StringQ[model] && FileExistsQ[model]), AppendTo[missing, "TTSVoice"]]];
    <|
      "Status" -> If[missing === {}, "OK", "Incomplete"],
      "Missing" -> missing,
      "Engine" -> engineChoice,
      "AivisEndpoint" -> aivisEndpoint,
      "AivisStyleId" -> aivisStyle,
      "Executable" -> Replace[exe, Except[_String] -> ""],
      "Model" -> Replace[model, Except[_String] -> ""],
      "Config" -> Replace[config, Except[_String] -> ""],
      "Language" -> language,
      "Voice" -> Which[
        aivisName =!= None, aivisName,
        voice === None, None,
        True, Lookup[voice, "Name", None]],
      "Layer" -> If[layer, "SourceVault_voice", "Fallback"],
      "Hint" -> hint
    |>
  ];

(* ---- 走っているワーカーの発見と停止 --------------------------------------
   state/control ファイルは実行ごとに UUID で名前が変わるので、パッケージを
   読み直した (= Private の $iProcess が None に戻った) カーネルからは、自分が
   起動したワーカーにすら手が届かなくなる。実際 2026-08-21 にワーカーが 2 つ
   並走し、同じ部屋に二重に応答して同じアカウントに二重課金していた。
   ワーカーは起動時に固定パスのロックへ pid / state / control を書くので、
   どのカーネルからでもそこを見れば発見・観測・停止できる。 *)

$iWorkerLockPath = FileNameJoin[{$TemporaryDirectory, "vrc-realtime-worker.lock"}];

iReadWorkerLock[] := Module[{value},
  If[!FileExistsQ[$iWorkerLockPath], Return[<||>]];
  value = Quiet @ Check[Import[$iWorkerLockPath, "RawJSON"], <||>];
  If[AssociationQ[value] && Lookup[value, "worker", ""] === "VRCRealtime",
    value, <||>]];

(* 生死は state ファイルの更新時刻で見る。プロセス一覧を引くと Windows で端末窓が
   ちらつくうえ遅い。ワーカーは常時 state を書き続けるので、止まっていればすぐ古びる。 *)
iWorkerLockLive[lock_Association, staleSeconds_ : 25.] := Module[{stateFile, state, age},
  stateFile = Lookup[lock, "stateFile", ""];
  If[!StringQ[stateFile] || !FileExistsQ[stateFile], Return[False]];
  state = Quiet @ Check[Import[stateFile, "RawJSON"], <||>];
  If[!AssociationQ[state], Return[False]];
  age = Quiet @ Check[AbsoluteTime[] - AbsoluteTime[
    DateObject[FromUnixTime[Lookup[state, "updatedAt", 0]], TimeZone -> $TimeZone]], $Failed];
  (* FromUnixTime は既に現地時刻なので差分だけ見ればよい。失敗したら生存扱いしない *)
  If[!NumericQ[age], Return[False]];
  age <= staleSeconds];

iWorkerLockLive[_] := False;

(* 手順は「作法どおり止める → 待つ → それでも生きていれば強制終了」。
   control ファイル経由なら相手が自分でロックを消して綺麗に落ちる。 *)
iStopWorkerFromLock[lock_Association, waitSeconds_ : 10.] :=
  Module[{controlFile, temporary, ba, strm, pid, deadline, killed = False},
    controlFile = Lookup[lock, "controlFile", ""];
    pid = Lookup[lock, "pid", None];
    If[StringQ[controlFile] && StringLength[controlFile] > 0,
      temporary = controlFile <> ".tmp";
      Quiet @ Check[
        Module[{},
          ba = ExportByteArray[<|"command" -> "stop"|>, "RawJSON"];
          strm = OpenWrite[temporary, BinaryFormat -> True];
          WithCleanup[BinaryWrite[strm, ba], Quiet @ Close[strm]];
          RenameFile[temporary, controlFile, OverwriteTarget -> True]],
        Null];
      deadline = AbsoluteTime[] + waitSeconds;
      While[AbsoluteTime[] < deadline,
        Pause[0.4];
        If[!iWorkerLockLive[lock, 3.], Break[]]]];
    If[iWorkerLockLive[lock, 3.] && IntegerQ[pid],
      Quiet @ Check[
        RunProcess[{"taskkill", "/PID", ToString[pid], "/T", "/F"}], $Failed];
      killed = True;
      Pause[1.]];
    Quiet @ Check[If[FileExistsQ[$iWorkerLockPath] &&
      Lookup[iReadWorkerLock[], "pid", None] === pid,
      DeleteFile[$iWorkerLockPath]], Null];
    <|"Status" -> If[iWorkerLockLive[lock, 3.], "StopFailed", "Stopped"],
      "PID" -> pid, "Forced" -> killed|>];

iReadState[] := Module[{path, value},
  path = $iStateFile;
  (* パッケージを読み直したカーネルは自分の state ファイルを知らない。走っている
     ワーカーのロックから引き継ぐことで、状態の観測だけは常にできるようにする。 *)
  If[!StringQ[path] || !FileExistsQ[path],
    path = Lookup[iReadWorkerLock[], "stateFile", ""]];
  If[!StringQ[path] || !FileExistsQ[path], Return[<||>]];
  value = Quiet @ Check[Import[path, "RawJSON"], <||>];
  If[AssociationQ[value], value, <||>]
];

iReadFinishedProcessDiagnostics[] := Module[{out, err},
  If[!MatchQ[$iProcess, _ProcessObject], Return[<||>]];
  If[AssociationQ[$iProcessDiagnostics] && Length[$iProcessDiagnostics] > 0,
    Return[$iProcessDiagnostics]
  ];
  out = Quiet @ Check[ReadString[$iProcess["StandardOutput"]], ""];
  err = Quiet @ Check[ReadString[$iProcess["StandardError"]], ""];
  $iProcessDiagnostics = <|
    "StandardOutput" -> If[StringQ[out], out, ""],
    "StandardError" -> If[StringQ[err], err, ""]
  |>;
  $iProcessDiagnostics
];

iSendCommand[command_Association] := Module[{json, temporary, wrote = False},
  If[!iProcessRunningQ[],
    Return[iFailure["NotRunning", "VRCRealtime worker is not running."]]
  ];
  json = ExportString[command, "RawJSON"];
  (* uv/python launchers on Windows do not always forward an inherited stdin
     pipe reliably.  New workers therefore use an atomic control file. *)
  If[StringQ[$iControlFile] && StringLength[$iControlFile] > 0,
    temporary = $iControlFile <> ".tmp";
    (* ExportByteArray で単一 UTF-8 エンコード。Export[..., "RawJSON"] 経由だと
       日本語 Windows で二重エンコードされ、Python 側が原稿を読めなくなる。 *)
    wrote = TrueQ @ Quiet @ Check[
      Module[{ba = ExportByteArray[command, "RawJSON"], strm},
        If[! ByteArrayQ[ba], Return[False, Module]];
        strm = OpenWrite[temporary, BinaryFormat -> True];
        If[Head[strm] =!= OutputStream, Return[False, Module]];
        WithCleanup[BinaryWrite[strm, ba], Quiet @ Close[strm]];
        RenameFile[temporary, $iControlFile, OverwriteTarget -> True];
        True],
      False
    ];
    If[wrote,
      Return[<|"Status" -> "Sent", "Transport" -> "ControlFile",
        "Command" -> Lookup[command, "command", None]|>]
    ]
  ];
  (* Compatibility path for a worker started by VRCRealtime 0.1.x. *)
  Quiet @ Check[
    WriteString[$iProcess, json, "\n"];
    FlushWrite[$iProcess];
    <|"Status" -> "Sent", "Transport" -> "StandardInput",
      "Command" -> Lookup[command, "command", None]|>,
    iFailure["ControlWriteFailed", "Failed to write to the worker process."]
  ]
];

Options[VRCRealtimeStart] = {
  "Model" -> "gpt-realtime-2.1",
  "Voice" -> "marin",
  "Instructions" -> "日本語で自然かつ簡潔に会話してください。",
  "VRCpilotDirectory" -> "C:\\Users\\imai_\\gitwork\\vrcpilot",
  "UVExecutable" -> Automatic,
  "InputSource" -> "VRChat",
  "InputDevice" -> Automatic,
  "OutputDevice" -> "CABLE-A Input",
  "VRChatPID" -> Automatic,
  "ChunkMilliseconds" -> 20,
  "VADThreshold" -> 0.5,
  "PrefixPaddingMilliseconds" -> 200,
  "SilenceDurationMilliseconds" -> 350,
  "ReconnectSeconds" -> 3300,
  "OutputCooldownMilliseconds" -> 1200,
  "AllowBargeIn" -> False,
  "VisualAttentionGate" -> False,
  "VisualGateFramesPerSecond" -> 8.,
  "VisualGateCenterTolerance" -> 0.18,
  (* 顔の見かけの大きさの下限。16:9 換算の画面幅に対する割合として書き、内部では
     画面「高さ」を通して角度に直す (Unity は垂直 FOV 固定・水平は aspect 従属)。
     0.07 は会話距離のオーナーに対して厳しすぎた: 実測 0.063 で 10% 足りず、
     ゲートが一度も開かないまま「声をかけても答えない」になった (2026-08-21)。
     0.05 は同じ姿勢で約 1.26 倍の距離まで許す。近すぎる値にすると室内の通行人で
     開いてしまうので、これ以上は下げないこと。実測値は状態の
     visualGateFaceFraction / visualGateFaceFractionRequired で確認できる。 *)
  "VisualGateMinFaceFraction" -> 0.05,
  "VisualGateMouthMotionThreshold" -> 0.025,
  "VisualGateAudioRMSThreshold" -> 0.008,
  "VisualGateMaxStereoImbalance" -> 0.35,
  "VisualGateHangoverMilliseconds" -> 650,
  "VisualGatePrerollMilliseconds" -> 600,
  "VisualGateRequireNameplatePulse" -> True,
  "VisualGateNameplateBlueThreshold" -> 0.002,
  "VisualGateConversationLatch" -> True,
  "VisualGateConversationIdleSeconds" -> 30.,
  "AvatarMode" -> "Standalone",
  "OwnerAccount" -> Automatic,
  "OwnerFollowEngine" -> "Formation",
  "OwnerFollowDistanceMeters" -> 1.8,
  "OwnerFollowRearAngleDegrees" -> 35.,
  "OwnerFollowOwnerBearingDegrees" -> 33.,
  "OwnerFollowMenuLocate" -> True,
  (* 空振りスイープの間に最も開けた方向へ数 m 歩く (同じ場所で旋回し続けない) *)
  "OwnerFollowSearchPatrol" -> True,
  (* 探索がこの秒数成功しなければ「待機」(静止・見かけたら接近・定期的にメニュー再確認)。
     オーナーがオフライン/参加不可とメニューで分かった時も即「待機」 *)
  "OwnerFollowIdleAfterSeconds" -> 360.,
  "OwnerFollowIdleRecheckSeconds" -> 300.,
  "OwnerFollowTrackingFPS" -> 1.5,
  "OwnerFollowTargetScreenX" -> 0.66,
  "OwnerFollowTargetNameplateHeight" -> 0.055,
  "OwnerFollowLostSeconds" -> 1.25,
  "OwnerFollowFormationSeconds" -> 4.5,
  "OwnerFollowOrbitSpeed" -> 0.32,
  "OwnerFollowSearchTurnSpeed" -> 0.42,
  "OwnerFollowAssumeSingleAvatar" -> True,
  "OwnerFollowOCRMinSide" -> 1080,
  "OwnerFollowOCRBoxThreshold" -> 0.35,
  "OwnerFollowOCRTextScore" -> 0.35,
  "OSCHost" -> "127.0.0.1",
  "OSCPort" -> 9000,
  "EnableVisionCapture" -> True,
  "VoiceCommands" -> True,
  "VoiceCommandModelDirectory" -> Automatic,
  "VoiceCommandMinConfidence" -> 0.45,
  "VisionImageMaxWidth" -> 960,
  "VisionJPEGQuality" -> 78,
  "EnableSourceVault" -> True,
  (* 既定 False: 室内の誰の質問でもクエリが外部へ出る経路になるため opt-in。
     True でも高速索引 (sourcevault_lookup) が空振りしたときだけ使われる。 *)
  "EnableVoiceWebSearch" -> False,
  "SourceVaultServiceDirectory" -> Automatic,
  "SourceVaultKinds" -> {"search", "packageapi", "llmlog", "mail", "eagle", "snapshot"},
  "SourceVaultKB" -> Automatic,
  "SourceVaultGroup" -> None,
  "SourceVaultReleaseContext" -> None,
  "SourceVaultLimit" -> 3,
  "SourceVaultVoiceWaitSeconds" -> 60.,
  (* ローカル読み上げ 1 回の上限秒数。高 PL の検索結果は数千字になることがあり、
     室内の誰もそれを飛ばせない。超えた分は文末で切って「以下は省略」と告げる *)
  "LocalSpeechMaxSeconds" -> 60.,
  (* SourceVault の Privacy.Level (0..1) がこの値以上の資料は、フォローモードで本人確認済みの
     オーナーが近く (隊形距離の 1.7 倍以内) にいるときだけ答える。経路を問わない:
     OpenAI へ渡す本文 (PL<0.5)、ローカル読み上げ (PL>=0.5)、mixed の公開半分のすべて。
     オーナー不在・別ワールド・ログアウト中は PL<0.25 の資料しか答えない。 *)
  "OwnerPresencePrivacyThreshold" -> 0.25,
  "PrivateTTSBroker" -> Automatic,
  (* Automatic: ローカル音声資産があれば使い、無ければ警告して SourceVault 連携を
     切ったまま起動する (非公開資料は取得も読み上げもされない = privacy 上は安全側)。
     True: 必須にする (無ければ起動しない)。False: 使わない。 *)
  "PrivateTTS" -> Automatic,
  (* ローカル読み上げに使う声。Automatic は SourceVault_voice が選ぶ既定の声。
     特定の声は前提にしない。上の "Voice" (OpenAI Realtime の声、既定 marin) とは
     別物なので名前を分ける — 同名にすると OptionValue が先に書いた方を返し、
     Realtime の声名でローカルの声を探して必ず失敗する。 *)
  "SpeechVoice" -> Automatic,
  "PiperExecutable" -> Automatic,
  "PiperModel" -> Automatic,
  "PiperConfig" -> Automatic,
  "SpeechLengthScale" -> Automatic,
  "SpeechNoiseScale" -> Automatic,
  "SafetyIdentifier" -> None,
  (* 既に走っているワーカーがあれば止めてから起動する。False なら
     WorkerAlreadyRunning で失敗する (二重起動を許さない)。 *)
  "ReplaceRunning" -> True,
  "Notebook" -> Automatic,
  "RequirePaidAPIApproval" -> True
};

(* 不足しているもの (`1`) を必ず出す。以前は理由を出さずヒントだけを出していて、
   「資産が無いため…」と「資産はそろっています」が並ぶ矛盾したメッセージになった。 *)
VRCRealtimeStart::replacing =
  "既に走っている VRCRealtime ワーカー (PID `1`) を停止してから起動します。";

VRCRealtimeStart::mcpstopped =
  "SourceVault MCP サービスが停止しています。起動を試みます "
  "(資料検索と発表登録簿はこのサービス越しに引くため、止まったままだと "
  "発表原稿が取得できません)。";

VRCRealtimeStart::mcpfailed =
  "SourceVault MCP サービスを起動できませんでした (`1`)。"
  "SourceVaultMCPStatus[] を確認してください。";

VRCRealtimeStart::novoice =
  "ローカル音声合成を使えないため (未解決: `1`) SourceVault 連携を切って起動します \
(非公開資料は取得も読み上げもされません)。`2`";

VRCRealtimeStart[OptionsPattern[]] := Module[
  {nb, requireApproval, apiKey, uv, vrcpilotDirectory, stateFile,
   model, voice, instructions, inputSource, inputDevice, outputDevice,
   vrchatPID, chunkMS, vadThreshold, prefixMS, silenceMS, reconnectSeconds,
   outputCooldownMS, safetyIdentifier, visualAttentionGate, visualGateFPS,
   visualGateCenterTolerance, visualGateMinFaceFraction,
   visualGateMotionThreshold, visualGateAudioRMSThreshold,
    visualGateMaxStereoImbalance, visualGateHangoverMS, visualGatePrerollMS,
    visualGateRequireNameplatePulse, visualGateNameplateBlueThreshold,
    visualGateConversationLatch, visualGateConversationIdleSeconds,
    avatarMode, ownerAccount, ownerFollowEngine, ownerFollowDistanceMeters,
    ownerFollowRearAngleDegrees, ownerFollowOwnerBearingDegrees,
    ownerFollowMenuLocate, ownerFollowSearchPatrol, ownerFollowIdleAfterSeconds,
    ownerFollowIdleRecheckSeconds, ownerFollowTrackingFPS,
    ownerFollowTargetScreenX, ownerFollowTargetNameplateHeight,
    ownerFollowLostSeconds, ownerFollowFormationSeconds,
    ownerFollowOrbitSpeed, ownerFollowSearchTurnSpeed,
    ownerFollowAssumeSingleAvatar,
    ownerFollowOCRMinSide, ownerFollowOCRBoxThreshold, ownerFollowOCRTextScore,
    oscHost, oscPort, enableVisionCapture,
    visionImageMaxWidth, visionJPEGQuality,
    voiceCommands, voiceCommandModelDirectory, voiceCommandMinConfidence,
   allowBargeIn, enableSourceVault, enableVoiceWebSearch, sourceVaultServiceDirectory,
   sourceVaultKinds, sourceVaultKB, sourceVaultGroup, sourceVaultReleaseContext,
   sourceVaultLimit, sourceVaultVoiceWaitSeconds, localSpeechMaxSeconds,
   ownerPresencePrivacyThreshold, privateTTSBroker, piperExecutable, piperModel,
   piperConfig, piperLanguage, privateTTSMode, privateTTSReady, voiceResolution,
   ttsEngine, ttsAivisEndpoint, ttsAivisStyleId,
   speechLengthScale, speechNoiseScale,
   privateAudioDir, controlFile, command, environment, process,
   startupDeadline, startupState, startupReason, startupDiagnostics,
   sourceVaultService, existingLock, replacedWorker},

  If[iProcessRunningQ[],
    Return[iFailure["AlreadyRunning", "VRCRealtime worker is already running.",
      <|"Status" -> VRCRealtimeStatus[]|>]]
  ];
  If[!FileExistsQ[$iWorkerPath],
    Return[iFailure["WorkerMissing", "VRCRealtime_worker.py was not found."]]
  ];
  If[!TrueQ[iEnsureNBAccess[]],
    Return[iFailure["NBAccessUnavailable", "NBAccess.wl could not be loaded."]]
  ];

  nb = iResolveNotebook[OptionValue["Notebook"]];
  requireApproval = TrueQ[OptionValue["RequirePaidAPIApproval"]];
  If[requireApproval,
    If[!MatchQ[nb, _NotebookObject],
      Return[iFailure["NotebookRequired",
        "Paid API approval requires a front-end notebook or an explicit Notebook option."]]
    ];
    If[!TrueQ @ Quiet @ NBAccess`NBGetNotebookPaidAPIAllowed[nb],
      Return[iFailure["PaidAPINotAllowed",
        "This notebook has not allowed paid API calls. Evaluate NBAccess`NBSetNotebookPaidAPIAllowed[EvaluationNotebook[], True] first."]]
    ]
  ];

  If[!TrueQ @ Quiet @ NBAccess`NBProviderCanAccess["openai", 0.5],
    Return[iFailure["OpenAIDisabled",
      "NBAccess provider policy does not currently allow OpenAI access."]]
  ];
  apiKey = Quiet @ NBAccess`NBGetAPIKey[
    "openai", NBAccess`PrivacySpec -> <|"AccessLevel" -> 1.0|>
  ];
  If[!StringQ[apiKey] || StringLength[apiKey] == 0,
    Return[iFailure["APIKeyMissing",
      "OPENAI_API_KEY was not found through NBAccess/SystemCredential."]]
  ];

  uv = iResolveUV[OptionValue["UVExecutable"]];
  If[!StringQ[uv] || !FileExistsQ[uv],
    Return[iFailure["UVNotFound",
      "uv executable was not found. Set \"UVExecutable\" to the full path of uv.exe."]]
  ];
  vrcpilotDirectory = OptionValue["VRCpilotDirectory"];
  If[!StringQ[vrcpilotDirectory] || !DirectoryQ[vrcpilotDirectory],
    Return[iFailure["VRCpilotDirectoryMissing",
      "The configured VRC Pilot project directory does not exist."]]
  ];

  model = OptionValue["Model"];
  voice = OptionValue["Voice"];
  instructions = OptionValue["Instructions"];
  inputSource = OptionValue["InputSource"];
  inputDevice = Replace[OptionValue["InputDevice"], Automatic -> ""];
  outputDevice = OptionValue["OutputDevice"];
  vrchatPID = Replace[OptionValue["VRChatPID"], Automatic -> 0];
  chunkMS = OptionValue["ChunkMilliseconds"];
  vadThreshold = OptionValue["VADThreshold"];
  prefixMS = OptionValue["PrefixPaddingMilliseconds"];
  silenceMS = OptionValue["SilenceDurationMilliseconds"];
  reconnectSeconds = OptionValue["ReconnectSeconds"];
  outputCooldownMS = OptionValue["OutputCooldownMilliseconds"];
  visualAttentionGate = TrueQ[OptionValue["VisualAttentionGate"]];
  visualGateFPS = OptionValue["VisualGateFramesPerSecond"];
  visualGateCenterTolerance = OptionValue["VisualGateCenterTolerance"];
  visualGateMinFaceFraction = OptionValue["VisualGateMinFaceFraction"];
  visualGateMotionThreshold = OptionValue["VisualGateMouthMotionThreshold"];
  visualGateAudioRMSThreshold = OptionValue["VisualGateAudioRMSThreshold"];
  visualGateMaxStereoImbalance = OptionValue["VisualGateMaxStereoImbalance"];
  visualGateHangoverMS = OptionValue["VisualGateHangoverMilliseconds"];
  visualGatePrerollMS = OptionValue["VisualGatePrerollMilliseconds"];
  visualGateRequireNameplatePulse =
    TrueQ[OptionValue["VisualGateRequireNameplatePulse"]];
  visualGateNameplateBlueThreshold =
    OptionValue["VisualGateNameplateBlueThreshold"];
  visualGateConversationLatch =
    TrueQ[OptionValue["VisualGateConversationLatch"]];
  visualGateConversationIdleSeconds =
    OptionValue["VisualGateConversationIdleSeconds"];
  avatarMode = OptionValue["AvatarMode"];
  ownerAccount = Replace[OptionValue["OwnerAccount"],
    Automatic -> VRCRealtime`$VRCRealtimeOwnerAccount];
  ownerFollowEngine = OptionValue["OwnerFollowEngine"];
  ownerFollowDistanceMeters = OptionValue["OwnerFollowDistanceMeters"];
  ownerFollowRearAngleDegrees = OptionValue["OwnerFollowRearAngleDegrees"];
  ownerFollowOwnerBearingDegrees = OptionValue["OwnerFollowOwnerBearingDegrees"];
  ownerFollowMenuLocate = TrueQ[OptionValue["OwnerFollowMenuLocate"]];
  ownerFollowSearchPatrol = TrueQ[OptionValue["OwnerFollowSearchPatrol"]];
  ownerFollowIdleAfterSeconds = OptionValue["OwnerFollowIdleAfterSeconds"];
  ownerFollowIdleRecheckSeconds = OptionValue["OwnerFollowIdleRecheckSeconds"];
  ownerFollowTrackingFPS = OptionValue["OwnerFollowTrackingFPS"];
  ownerFollowTargetScreenX = OptionValue["OwnerFollowTargetScreenX"];
  ownerFollowTargetNameplateHeight =
    OptionValue["OwnerFollowTargetNameplateHeight"];
  ownerFollowLostSeconds = OptionValue["OwnerFollowLostSeconds"];
  ownerFollowFormationSeconds = OptionValue["OwnerFollowFormationSeconds"];
  ownerFollowOrbitSpeed = OptionValue["OwnerFollowOrbitSpeed"];
  ownerFollowSearchTurnSpeed = OptionValue["OwnerFollowSearchTurnSpeed"];
  ownerFollowAssumeSingleAvatar =
    OptionValue["OwnerFollowAssumeSingleAvatar"];
  ownerFollowOCRMinSide = OptionValue["OwnerFollowOCRMinSide"];
  ownerFollowOCRBoxThreshold = OptionValue["OwnerFollowOCRBoxThreshold"];
  ownerFollowOCRTextScore = OptionValue["OwnerFollowOCRTextScore"];
  oscHost = OptionValue["OSCHost"];
  oscPort = OptionValue["OSCPort"];
  enableVisionCapture = TrueQ[OptionValue["EnableVisionCapture"]];
  voiceCommands = TrueQ[OptionValue["VoiceCommands"]];
  voiceCommandModelDirectory = Replace[OptionValue["VoiceCommandModelDirectory"],
    Automatic | None -> ""];
  voiceCommandMinConfidence = OptionValue["VoiceCommandMinConfidence"];
  visionImageMaxWidth = OptionValue["VisionImageMaxWidth"];
  visionJPEGQuality = OptionValue["VisionJPEGQuality"];
  allowBargeIn = OptionValue["AllowBargeIn"];
  enableSourceVault = TrueQ[OptionValue["EnableSourceVault"]];
  enableVoiceWebSearch = TrueQ[OptionValue["EnableVoiceWebSearch"]];
  (* 二重起動の遮断。パッケージを読み直すと $iProcess を失うので、走っている
     ワーカーはロック越しにしか見つけられない。 *)
  existingLock = iReadWorkerLock[];
  replacedWorker = None;
  If[AssociationQ[existingLock] && iWorkerLockLive[existingLock],
    If[!TrueQ[OptionValue["ReplaceRunning"]],
      Return[iFailure["WorkerAlreadyRunning",
        "VRCRealtime のワーカーが既に走っています。",
        <|"PID" -> Lookup[existingLock, "pid", None],
          "StateFile" -> Lookup[existingLock, "stateFile", None],
          "Hint" -> "VRCRealtimeKill[] で止めるか、"
            <> "\"ReplaceRunning\" -> True で置き換えてください。"|>]]];
    Message[VRCRealtimeStart::replacing, Lookup[existingLock, "pid", "?"]];
    replacedWorker = iStopWorkerFromLock[existingLock]];

  (* MCP が止まっていれば報告して起動を試みる。runtime ディレクトリは起動後に
     現れるので、解決はこの後で行う (順序を入れ替えると必ず空振りする)。 *)
  sourceVaultService = If[enableSourceVault, iEnsureSourceVaultService[],
    <|"Status" -> "Skipped", "Started" -> False|>];
  sourceVaultServiceDirectory = If[enableSourceVault,
    iResolveSourceVaultServiceDirectory[OptionValue["SourceVaultServiceDirectory"]],
    ""];
  sourceVaultKinds = OptionValue["SourceVaultKinds"];
  (* Graph-RAG の高速 KB。Automatic は SourceVault 側の既定 KB を使う
     ($SourceVaultKBDefaultId)、None は KB 先行ルートを使わない *)
  sourceVaultKB = Replace[OptionValue["SourceVaultKB"], {Automatic -> "", None -> ""}];
  sourceVaultGroup = Replace[OptionValue["SourceVaultGroup"], None -> ""];
  sourceVaultReleaseContext = Replace[OptionValue["SourceVaultReleaseContext"], None -> ""];
  sourceVaultLimit = OptionValue["SourceVaultLimit"];
  sourceVaultVoiceWaitSeconds = OptionValue["SourceVaultVoiceWaitSeconds"];
  localSpeechMaxSeconds = OptionValue["LocalSpeechMaxSeconds"];
  ownerPresencePrivacyThreshold =
    OptionValue["OwnerPresencePrivacyThreshold"];
  privateTTSBroker = Replace[OptionValue["PrivateTTSBroker"],
    Automatic -> $iPrivateTTSBrokerPath];
  privateTTSMode = OptionValue["PrivateTTS"];
  voiceResolution = iResolveVoice[
    OptionValue["PiperExecutable"], OptionValue["PiperModel"],
    OptionValue["PiperConfig"], OptionValue["SpeechVoice"]];
  piperExecutable = voiceResolution["Executable"];
  piperModel = voiceResolution["Model"];
  piperConfig = voiceResolution["Config"];
  piperLanguage = voiceResolution["Language"];
  ttsEngine = Lookup[voiceResolution, "Engine", "Piper"];
  ttsAivisEndpoint = Lookup[voiceResolution, "AivisEndpoint", ""];
  ttsAivisStyleId = Lookup[voiceResolution, "AivisStyleId", 0];
  speechLengthScale = Replace[OptionValue["SpeechLengthScale"],
    Automatic | None -> 0.];
  speechNoiseScale = Replace[OptionValue["SpeechNoiseScale"],
    Automatic | None -> 0.];
  privateAudioDir = FileNameJoin[{$TemporaryDirectory,
    "vrc-private-tts-" <> StringReplace[CreateUUID[], "-" -> ""]}];
  safetyIdentifier = Replace[OptionValue["SafetyIdentifier"], None -> ""];

  If[!And[
      StringQ[model], StringLength[model] > 0,
      StringQ[voice], StringLength[voice] > 0,
      StringQ[instructions], MemberQ[{"VRChat", "Microphone"}, inputSource],
      !visualAttentionGate || inputSource === "VRChat",
      StringQ[inputDevice], StringQ[outputDevice],
      IntegerQ[vrchatPID] && vrchatPID >= 0,
      IntegerQ[chunkMS] && 10 <= chunkMS <= 100,
      NumericQ[vadThreshold] && 0. <= N[vadThreshold] <= 1.,
      IntegerQ[prefixMS] && prefixMS >= 0,
      IntegerQ[silenceMS] && silenceMS >= 100,
      IntegerQ[reconnectSeconds] && reconnectSeconds >= 60,
      IntegerQ[outputCooldownMS] && outputCooldownMS >= 0,
      BooleanQ[allowBargeIn],
      BooleanQ[visualAttentionGate],
      NumericQ[visualGateFPS] && 1. <= N[visualGateFPS] <= 20.,
      NumericQ[visualGateCenterTolerance] &&
        0.05 <= N[visualGateCenterTolerance] <= 0.5,
      NumericQ[visualGateMinFaceFraction] &&
        0.03 <= N[visualGateMinFaceFraction] <= 0.4,
      NumericQ[visualGateMotionThreshold] &&
        0. <= N[visualGateMotionThreshold] <= 0.2,
      NumericQ[visualGateAudioRMSThreshold] &&
        0. <= N[visualGateAudioRMSThreshold] <= 0.5,
      NumericQ[visualGateMaxStereoImbalance] &&
        0. <= N[visualGateMaxStereoImbalance] <= 1.,
      IntegerQ[visualGateHangoverMS] && 100 <= visualGateHangoverMS <= 3000,
      IntegerQ[visualGatePrerollMS] && 0 <= visualGatePrerollMS <= 2000,
      BooleanQ[visualGateRequireNameplatePulse],
      NumericQ[visualGateNameplateBlueThreshold] &&
        0.0001 <= N[visualGateNameplateBlueThreshold] <= 0.1,
      BooleanQ[visualGateConversationLatch],
      NumericQ[visualGateConversationIdleSeconds] &&
        5. <= N[visualGateConversationIdleSeconds] <= 600.,
      MemberQ[{"Standalone", "OwnerFollow", "Presentation"}, avatarMode],
      avatarMode === "Standalone" || visualAttentionGate,
      StringQ[ownerAccount] && StringLength[StringTrim[ownerAccount]] > 0,
      MemberQ[{"Formation", "Legacy"}, ownerFollowEngine],
      NumericQ[ownerFollowDistanceMeters] &&
        0.8 <= N[ownerFollowDistanceMeters] <= 6.,
      NumericQ[ownerFollowRearAngleDegrees] &&
        0. <= N[ownerFollowRearAngleDegrees] <= 90.,
      NumericQ[ownerFollowOwnerBearingDegrees] &&
        10. <= N[ownerFollowOwnerBearingDegrees] <= 42.,
      NumericQ[ownerFollowIdleAfterSeconds] &&
        0. <= N[ownerFollowIdleAfterSeconds] <= 3600.,
      NumericQ[ownerFollowIdleRecheckSeconds] &&
        30. <= N[ownerFollowIdleRecheckSeconds] <= 3600.,
      NumericQ[ownerFollowTrackingFPS] &&
        0.25 <= N[ownerFollowTrackingFPS] <= 5.,
      NumericQ[ownerFollowTargetScreenX] &&
        0.50 <= N[ownerFollowTargetScreenX] <= 0.80,
      NumericQ[ownerFollowTargetNameplateHeight] &&
        0.005 <= N[ownerFollowTargetNameplateHeight] <= 0.15,
      NumericQ[ownerFollowLostSeconds] &&
        0.5 <= N[ownerFollowLostSeconds] <= 5.,
      NumericQ[ownerFollowFormationSeconds] &&
        1.5 <= N[ownerFollowFormationSeconds] <= 10.,
      NumericQ[ownerFollowOrbitSpeed] &&
        0.10 <= N[ownerFollowOrbitSpeed] <= 0.60,
      NumericQ[ownerFollowSearchTurnSpeed] &&
        0.08 <= N[ownerFollowSearchTurnSpeed] <= 0.50,
      MatchQ[ownerFollowAssumeSingleAvatar, True | False],
      IntegerQ[ownerFollowOCRMinSide] &&
        736 <= ownerFollowOCRMinSide <= 1440,
      NumericQ[ownerFollowOCRBoxThreshold] &&
        0.20 <= N[ownerFollowOCRBoxThreshold] <= 0.70,
      NumericQ[ownerFollowOCRTextScore] &&
        0.20 <= N[ownerFollowOCRTextScore] <= 0.70,
      StringQ[oscHost] && StringLength[oscHost] > 0,
      IntegerQ[oscPort] && 1 <= oscPort <= 65535,
      BooleanQ[enableVisionCapture],
      StringQ[voiceCommandModelDirectory],
      NumericQ[voiceCommandMinConfidence] &&
        0. <= N[voiceCommandMinConfidence] <= 1.,
      IntegerQ[visionImageMaxWidth] && 320 <= visionImageMaxWidth <= 1920,
      IntegerQ[visionJPEGQuality] && 40 <= visionJPEGQuality <= 95,
      BooleanQ[enableSourceVault], BooleanQ[enableVoiceWebSearch],
      ListQ[sourceVaultKinds] && AllTrue[sourceVaultKinds, StringQ],
      StringQ[sourceVaultKB],
      StringQ[sourceVaultGroup], StringQ[sourceVaultReleaseContext],
      IntegerQ[sourceVaultLimit] && 1 <= sourceVaultLimit <= 5,
      NumericQ[sourceVaultVoiceWaitSeconds] &&
        1. <= N[sourceVaultVoiceWaitSeconds] <= 120.,
      NumericQ[localSpeechMaxSeconds] && 5. <= N[localSpeechMaxSeconds] <= 600.,
      NumericQ[ownerPresencePrivacyThreshold] &&
        0. <= N[ownerPresencePrivacyThreshold] <= 10.,
      StringQ[safetyIdentifier]
    ],
    Return[iFailure["InvalidOptions", "One or more VRCRealtimeStart options are invalid."]]
  ];

  (* ---- ローカル音声資産の可否判定 -------------------------------------
     非公開資料 (Privacy.Level >= 0.5) はローカル合成でしか読み上げない。
     資産が無いときに OpenAI 側へ回すことは絶対にしない。したがって資産が
     無ければ SourceVault 連携そのものを切る: 非公開資料は取得も読み上げも
     されなくなるだけで、外部へ出る経路は増えない。
     "PrivateTTS" -> True なら従来どおり起動を止める。 *)
  privateTTSReady = And[
    StringQ[privateTTSBroker] && FileExistsQ[privateTTSBroker],
    voiceResolution["Status"] === "OK"];
  If[privateTTSMode === False,
    enableSourceVault = False;
    privateTTSReady = False
  ];
  If[enableSourceVault && !privateTTSReady,
    If[TrueQ[privateTTSMode],
      Return[iFailure["PrivateTTSMissing",
        "ローカル音声合成の資産が見つかりません。",
        <|"Missing" -> Lookup[voiceResolution, "Missing", {}],
          "Broker" -> privateTTSBroker,
          "Layer" -> voiceResolution["Layer"],
          "Hint" -> Replace[voiceResolution["Hint"], None ->
            "SourceVaultVoiceStatus[] で不足を確認してください。"]|>]]
    ];
    (* Automatic: 止めずに切って進む。何が起きたかは必ず知らせる *)
    Message[VRCRealtimeStart::novoice,
      Which[
        Lookup[voiceResolution, "Missing", {}] =!= {},
          StringRiffle[voiceResolution["Missing"], ", "],
        !(StringQ[privateTTSBroker] && FileExistsQ[privateTTSBroker]),
          "PrivateTTSBroker (" <> ToString[privateTTSBroker] <> ")",
        True, "unknown"],
      Replace[voiceResolution["Hint"], None ->
        "SourceVaultVoiceStatus[] で不足を確認してください。"]];
    enableSourceVault = False
  ];

  (* 音声資産の判定で SourceVault 連携が切れた場合はサービスも要らない *)
  If[enableSourceVault && !StringQ[sourceVaultServiceDirectory],
    Return[iFailure["SourceVaultServiceMissing",
      "SourceVault サービスの runtime ディレクトリが見つかりません。",
      <|"MCPService" -> sourceVaultService,
        "Hint" -> "SourceVaultMCPStatus[] を確認するか、"
          <> "\"SourceVaultServiceDirectory\" を明示指定してください。"
          <> " 資料検索が不要なら \"EnableSourceVault\" -> False で起動できます。"|>]]
  ];

  stateFile = FileNameJoin[{
    $TemporaryDirectory,
    "vrc-realtime-" <> StringReplace[CreateUUID[], "-" -> ""] <> ".json"
  }];
  controlFile = FileNameJoin[{
    $TemporaryDirectory,
    "vrc-realtime-control-" <> StringReplace[CreateUUID[], "-" -> ""] <> ".json"
  }];
  command = Join[
    {
      uv, "run", "--with", "websocket-client>=1.8,<2",
      "--with", "vosk>=0.3.45", "python",
      $iWorkerPath,
      "--state-file", stateFile,
      "--control-file", controlFile,
      "--model", model,
      "--voice", voice,
      "--input-source", ToLowerCase[inputSource],
      "--output-device", outputDevice,
      "--vrchat-pid", ToString[vrchatPID],
      "--chunk-ms", ToString[chunkMS],
      "--vad-threshold", iNumberString[vadThreshold],
      "--prefix-padding-ms", ToString[prefixMS],
      "--silence-duration-ms", ToString[silenceMS],
      "--reconnect-seconds", ToString[reconnectSeconds],
      "--output-cooldown-ms", ToString[outputCooldownMS]
    },
    If[StringLength[inputDevice] > 0,
      {"--input-device", inputDevice},
      {}
    ],
    If[StringLength[safetyIdentifier] > 0,
      {"--safety-identifier", safetyIdentifier},
      {}
    ],
    If[TrueQ[allowBargeIn], {"--allow-barge-in"}, {}],
    If[visualAttentionGate,
      {"--visual-attention-gate",
       "--visual-gate-fps", iNumberString[visualGateFPS],
       "--visual-gate-center-tolerance", iNumberString[visualGateCenterTolerance],
       "--visual-gate-min-face-fraction", iNumberString[visualGateMinFaceFraction],
       "--visual-gate-motion-threshold", iNumberString[visualGateMotionThreshold],
       "--visual-gate-audio-rms-threshold", iNumberString[visualGateAudioRMSThreshold],
       "--visual-gate-max-stereo-imbalance", iNumberString[visualGateMaxStereoImbalance],
       "--visual-gate-hangover-ms", ToString[visualGateHangoverMS],
       "--visual-gate-preroll-ms", ToString[visualGatePrerollMS],
       "--visual-gate-nameplate-blue-threshold",
         iNumberString[visualGateNameplateBlueThreshold]},
      {}
    ],
    If[visualAttentionGate && visualGateRequireNameplatePulse,
      {"--visual-gate-require-nameplate-pulse"}, {}],
    If[visualAttentionGate && visualGateConversationLatch,
      {"--visual-gate-conversation-latch",
       "--visual-gate-conversation-idle-seconds",
         iNumberString[visualGateConversationIdleSeconds]}, {}],
    {"--avatar-mode",
       Switch[avatarMode,
         "OwnerFollow", "owner-follow",
         "Presentation", "presentation",
         _, "standalone"],
     "--owner-account", ownerAccount,
     "--owner-follow-engine", ToLowerCase[ownerFollowEngine],
     "--owner-follow-distance-m", iNumberString[ownerFollowDistanceMeters],
     "--owner-follow-rear-angle", iNumberString[ownerFollowRearAngleDegrees],
     "--owner-follow-owner-bearing", iNumberString[ownerFollowOwnerBearingDegrees],
     If[ownerFollowMenuLocate, "--owner-follow-menu-locate", "--no-owner-follow-menu-locate"],
     If[ownerFollowSearchPatrol, "--owner-follow-search-patrol", "--no-owner-follow-search-patrol"],
     "--owner-follow-idle-after-seconds", iNumberString[ownerFollowIdleAfterSeconds],
     "--owner-follow-idle-recheck-seconds", iNumberString[ownerFollowIdleRecheckSeconds],
     "--owner-follow-tracking-fps", iNumberString[ownerFollowTrackingFPS],
     "--owner-follow-target-screen-x", iNumberString[ownerFollowTargetScreenX],
     "--owner-follow-target-nameplate-height",
       iNumberString[ownerFollowTargetNameplateHeight],
     "--owner-follow-lost-seconds", iNumberString[ownerFollowLostSeconds],
     "--owner-follow-formation-seconds", iNumberString[ownerFollowFormationSeconds],
     "--owner-follow-orbit-speed", iNumberString[ownerFollowOrbitSpeed],
     "--owner-follow-search-turn-speed", iNumberString[ownerFollowSearchTurnSpeed],
     If[TrueQ[ownerFollowAssumeSingleAvatar],
       "--owner-follow-assume-single-avatar",
       "--no-owner-follow-assume-single-avatar"],
     "--owner-follow-ocr-min-side", ToString[ownerFollowOCRMinSide],
     "--owner-follow-ocr-box-threshold", iNumberString[ownerFollowOCRBoxThreshold],
     "--owner-follow-ocr-text-score", iNumberString[ownerFollowOCRTextScore],
     "--osc-host", oscHost,
     "--osc-port", ToString[oscPort],
     "--vision-image-max-width", ToString[visionImageMaxWidth],
     "--vision-jpeg-quality", ToString[visionJPEGQuality],
     If[voiceCommands, "--voice-commands", "--no-voice-commands"],
     "--voice-command-min-confidence", iNumberString[voiceCommandMinConfidence]},
    If[StringLength[voiceCommandModelDirectory] > 0,
      {"--voice-command-model-dir", voiceCommandModelDirectory}, {}],
    If[enableVisionCapture, {"--enable-vision-capture"}, {}],
    If[enableSourceVault,
      Join[
        {"--enable-sourcevault",
         "--private-tts-broker", privateTTSBroker,
         "--sourcevault-service-dir", sourceVaultServiceDirectory,
         "--sourcevault-kinds", StringRiffle[sourceVaultKinds, ","],
         "--sourcevault-limit", ToString[sourceVaultLimit],
         "--sourcevault-wait-seconds", iNumberString[sourceVaultVoiceWaitSeconds],
         "--local-speech-max-seconds", iNumberString[localSpeechMaxSeconds],
         "--owner-presence-privacy-threshold",
           iNumberString[ownerPresencePrivacyThreshold],
         "--piper-executable", piperExecutable,
         "--piper-model", piperModel,
         "--piper-config", piperConfig,
         (* 声から解決した値。空なら worker 側で送らない = piper の既定に任せる *)
         "--piper-language", piperLanguage,
         "--private-audio-dir", privateAudioDir},
        If[NumericQ[speechLengthScale] && N[speechLengthScale] > 0.,
          {"--piper-length-scale", iNumberString[speechLengthScale]}, {}],
        If[NumericQ[speechNoiseScale] && N[speechNoiseScale] > 0.,
          {"--piper-noise-scale", iNumberString[speechNoiseScale]}, {}],
        If[ttsEngine === "Aivis",
          {"--tts-engine", "aivis",
           "--aivis-endpoint", ttsAivisEndpoint,
           "--aivis-style-id", ToString[ttsAivisStyleId],
           (* SpeechLengthScale (>1 = ゆっくり) を AivisSpeech の speedScale へ
              逆数で写す。未指定なら等速 (piper の既定 1.5 は写さない) *)
           "--aivis-speed-scale", iNumberString[
             If[NumericQ[speechLengthScale] && N[speechLengthScale] > 0.,
               1./N[speechLengthScale], 1.]]}, {}],
        If[StringLength[sourceVaultKB] > 0,
          {"--sourcevault-kb", sourceVaultKB}, {}],
        If[StringLength[sourceVaultGroup] > 0,
          {"--sourcevault-group", sourceVaultGroup}, {}],
        If[StringLength[sourceVaultReleaseContext] > 0,
          {"--sourcevault-release-context", sourceVaultReleaseContext}, {}],
        If[enableVoiceWebSearch, {"--enable-voice-web-search"}, {}]
      ],
      {}
    ]
  ];
  environment = Join[
    Association @ GetEnvironment[],
    <|
      "OPENAI_API_KEY" -> apiKey,
      "VRC_REALTIME_INSTRUCTIONS" -> instructions,
      "PYTHONUNBUFFERED" -> "1"
    |>
  ];

  process = Quiet @ Check[
    StartProcess[
      command,
      ProcessDirectory -> vrcpilotDirectory,
      ProcessEnvironment -> environment
    ],
    $Failed
  ];
  apiKey =.;
  environment =.;
  If[!MatchQ[process, _ProcessObject],
    Return[iFailure["StartFailed", "Failed to start the VRCRealtime worker."]]
  ];

  $iProcess = process;
  $iStateFile = stateFile;
  $iControlFile = controlFile;
  $iProcessDiagnostics = <||>;

  (* ---- 起動直後の生存確認 -------------------------------------------------
     ワーカーは引数の検証に落ちると 1 秒足らずで死ぬ。ここで見ないと
     VRCRealtimeStart は正常な Association を返し、利用者には「起動したのに
     声をかけても答えない」としか見えない (状態ファイルを開くまで理由が
     分からない)。少しだけ待って、死んでいたら理由ごと Failure にする。
     非ブロックの原則は保つ: 上限 3 秒で、生きていれば即座に抜ける。
     Module を挟まないのは Return が最内 Module から返ってしまうため。 *)
  startupDeadline = AbsoluteTime[] + 3.;
  While[AbsoluteTime[] < startupDeadline,
    Pause[0.15];
    If[ProcessStatus[process] =!= "Running", Break[]];
    startupState = iReadState[];
    If[AssociationQ[startupState] && Lookup[startupState, "status", ""] === "error",
      Break[]];
    (* 起動を通り抜けた印が出たら待たずに進む *)
    If[AssociationQ[startupState] && KeyExistsQ[startupState, "connected"], Break[]]];
  startupState = iReadState[];
  If[ProcessStatus[process] =!= "Running" ||
     (AssociationQ[startupState] && Lookup[startupState, "status", ""] === "error"),
    startupReason = If[AssociationQ[startupState],
      Lookup[startupState, "lastError", None], None];
    startupDiagnostics = iReadFinishedProcessDiagnostics[];
    Quiet @ Check[KillProcess[process], Null];
    $iProcess = None; $iStateFile = None; $iControlFile = None;
    Return[iFailure["WorkerStartupFailed",
      "ワーカーが起動直後に停止しました。",
      <|"Reason" -> Replace[startupReason, None -> "unknown"],
        "StateFile" -> stateFile,
        "Diagnostics" -> startupDiagnostics|>]]
  ];

  $iSession = <|
    "Process" -> process,
    "StateFile" -> stateFile,
    "ControlFile" -> controlFile,
    "Model" -> model,
    "Voice" -> voice,
    "InputSource" -> inputSource,
    "InputDevice" -> Replace[inputDevice, "" -> Automatic],
    "OutputDevice" -> outputDevice,
    "AllowBargeIn" -> allowBargeIn,
    "VisualAttentionGate" -> visualAttentionGate,
    "VisualGateConversationLatch" -> visualGateConversationLatch,
    "VisualGateConversationIdleSeconds" -> N[visualGateConversationIdleSeconds],
    "AvatarMode" -> avatarMode,
    "OwnerAccount" -> ownerAccount,
    "OwnerFollowEngine" -> ownerFollowEngine,
    "OwnerFollowDistanceMeters" -> N[ownerFollowDistanceMeters],
    "OwnerFollowRearAngleDegrees" -> N[ownerFollowRearAngleDegrees],
    "OwnerFollowOwnerBearingDegrees" -> N[ownerFollowOwnerBearingDegrees],
    "VisionCaptureEnabled" -> enableVisionCapture,
    "VoiceCommands" -> voiceCommands,
    "SourceVaultEnabled" -> enableSourceVault,
    "VoiceWebSearchEnabled" -> enableVoiceWebSearch,
    "SourceVaultServiceDirectory" -> Replace[sourceVaultServiceDirectory, "" -> None],
    "SourceVaultService" -> sourceVaultService,
    "ReplacedWorker" -> replacedWorker,
    "SourceVaultKinds" -> sourceVaultKinds,
    "SourceVaultKB" -> Replace[sourceVaultKB, "" -> Automatic],
    "SourceVaultVoiceWaitSeconds" -> N[sourceVaultVoiceWaitSeconds],
    "LocalSpeechMaxSeconds" -> N[localSpeechMaxSeconds],
    "SourceVaultGroup" -> Replace[sourceVaultGroup, "" -> None],
    "SourceVaultReleaseContext" -> Replace[sourceVaultReleaseContext, "" -> None],
    (* 実際に解決した声を記録する。特定の声名は埋め込まない *)
    "PrivateTTS" -> If[TrueQ[privateTTSReady] && enableSourceVault,
      <|"Engine" -> If[ttsEngine === "Aivis", "AivisSpeech", "PiperPlus"],
        "Voice" -> voiceResolution["Voice"],
        "Language" -> Replace[piperLanguage, "" -> Automatic],
        "Model" -> If[ttsEngine === "Aivis", ttsAivisEndpoint, piperModel],
        "FallbackModel" -> If[ttsEngine === "Aivis",
          Replace[piperModel, "" -> None], None],
        "ResolvedBy" -> voiceResolution["Layer"]|>,
      None],
    "Instructions" -> instructions,
    "StartedAt" -> Now
  |>;
  $iSession
];

VRCRealtimeStatus[] := Module[{state, processStatus, diagnostics, lock, adopted},
  processStatus = If[MatchQ[$iProcess, _ProcessObject],
    Quiet @ Check[ProcessStatus[$iProcess], "Unknown"],
    "NotStarted"
  ];
  (* このカーネルが起動したのでなくても、走っているワーカーは報告する。
     「NotStarted」と出しながら実際には別プロセスが喋っている、という食い違いが
     ワーカーの二重起動を見えなくしていた。 *)
  lock = iReadWorkerLock[];
  adopted = If[processStatus === "NotStarted" && iWorkerLockLive[lock],
    (processStatus = "Running"; Lookup[lock, "pid", None]), None];
  state = iReadState[];
  diagnostics = If[processStatus === "Finished",
    iReadFinishedProcessDiagnostics[],
    <||>
  ];
  Join[
    <|"ProcessStatus" -> processStatus|>,
    If[AssociationQ[state], state, <||>],
    diagnostics,
    If[adopted =!= None, <|"AdoptedWorkerPID" -> adopted|>, <||>],
    If[AssociationQ[$iSession],
      KeyTake[$iSession, {"Model", "Voice", "OutputDevice", "StartedAt"}],
      <||>
    ]
  ]
];

Options[VRCRealtimeStop] = {"WaitSeconds" -> 8.0};
VRCRealtimeStop[OptionsPattern[]] := Module[{sent, wait, deadline, status},
  wait = OptionValue["WaitSeconds"];
  If[! NumericQ[wait] || N[wait] < 0,
    Return[iFailure["InvalidWaitSeconds", "WaitSeconds must be non-negative."]]
  ];
  sent = iSendCommand[<|"command" -> "stop"|>];
  If[FailureQ[sent], Return[sent]];
  deadline = AbsoluteTime[] + N[wait];
  While[iProcessRunningQ[] && AbsoluteTime[] < deadline, Pause[0.05]];
  status = If[MatchQ[$iProcess, _ProcessObject],
    Quiet @ Check[ProcessStatus[$iProcess], "Unknown"], "NotStarted"];
  If[! iProcessRunningQ[],
    If[StringQ[$iControlFile] && FileExistsQ[$iControlFile],
      Quiet @ Check[DeleteFile[$iControlFile], Null]];
    <|"Status" -> "Stopped", "ProcessStatus" -> status|>,
    iFailure["StopTimeout",
      "VRCRealtime worker did not stop before the timeout. Use VRCRealtimeKill[] if necessary.",
      <|"WaitSeconds" -> N[wait],
        "WorkerPID" -> Lookup[iReadState[], "pid", Missing["Unknown"]]|>]
  ]
];

VRCRealtimeKill[] := Module[{result, pid, success, lock},
  If[!iProcessRunningQ[],
    (* 自分が起動したのでなくても、走っているワーカーは止められるべき。
       これが無いと、パッケージ再読込で孤児になったワーカーに手が届かない。 *)
    lock = iReadWorkerLock[];
    If[AssociationQ[lock] && iWorkerLockLive[lock],
      Return[Join[<|"Status" -> "Killed", "Adopted" -> True|>,
        iStopWorkerFromLock[lock]]]];
    Return[iFailure["NotRunning", "VRCRealtime worker is not running."]]
  ];
  pid = Quiet @ Check[$iProcess["ProcessID"], $Failed];
  result = If[$OperatingSystem === "Windows" && IntegerQ[pid],
    Quiet @ Check[
      RunProcess[{"taskkill", "/PID", ToString[pid], "/T", "/F"}],
      $Failed],
    Quiet @ Check[KillProcess[$iProcess], $Failed]
  ];
  success = If[AssociationQ[result], Lookup[result, "ExitCode", 1] === 0,
    result =!= $Failed];
  If[! TrueQ[success],
    iFailure["KillFailed", "Failed to terminate the VRCRealtime worker."],
    <|"Status" -> "Killed", "Process" -> $iProcess,
      "ProcessTreePID" -> pid|>
  ]
];

VRCRealtimeSetInstructions[text_String] := Module[{result},
  result = iSendCommand[<|"command" -> "instructions", "value" -> text|>];
  If[AssociationQ[result], $iSession["Instructions"] = text];
  result
];

VRCRealtimeMute[flag : (True | False)] :=
  iSendCommand[<|"command" -> If[flag, "mute", "unmute"]|>];

VRCRealtimeEndConversation[] :=
  iSendCommand[<|"command" -> "end-conversation"|>];

(* 実体は SourceVault_voice。ここは VRCRealtime が実際に解決する値 (声の選択や
   明示指定の反映後) を重ねて返す薄い口で、利用者が VRCRealtime 側の関数だけを
   知っていれば診断できるようにするためにある。 *)
VRCRealtimeVoiceStatus[] := Module[{layer, base, resolved},
  layer = iEnsureVoiceLayer[];
  base = If[layer,
    Quiet @ Check[SourceVault`SourceVaultVoiceStatus[], <||>],
    <|"Status" -> "Unknown", "Missing" -> {},
      "Hint" -> "SourceVault_voice.wl が見つかりません。SourceVault を更新してください。"|>];
  If[!AssociationQ[base], base = <||>];
  resolved = iResolveVoice[Automatic, Automatic, Automatic, Automatic];
  Join[
    KeyDrop[base, {"Status", "Missing"}],
    <|
      "Status" -> resolved["Status"],
      "Missing" -> resolved["Missing"],
      "ResolvedBy" -> resolved["Layer"],
      "SelectedEngine" -> Lookup[resolved, "Engine", "Piper"],
      "AivisEndpoint" -> Replace[Lookup[resolved, "AivisEndpoint", ""], "" -> None],
      "SelectedVoice" -> resolved["Voice"],
      "SelectedLanguage" -> Replace[resolved["Language"], "" -> Automatic],
      "Executable" -> Replace[resolved["Executable"], "" -> None],
      "Model" -> Replace[resolved["Model"], "" -> None],
      "PrivateTTSBroker" -> $iPrivateTTSBrokerPath,
      "PrivateTTSBrokerPresent" -> FileExistsQ[$iPrivateTTSBrokerPath],
      "Hint" -> Replace[resolved["Hint"], None :> Lookup[base, "Hint", None]]
    |>]
];

Options[VRCRealtimeInstallVoiceCommandModel] = {
  "UVExecutable" -> Automatic,
  "VRCpilotDirectory" -> "C:\\Users\\imai_\\gitwork\\vrcpilot",
  "ModelDirectory" -> Automatic
};
VRCRealtimeInstallVoiceCommandModel[opts : OptionsPattern[]] :=
  Module[{uv, dir, modelDir, command, result, out, parsed},
    uv = iResolveUV[OptionValue["UVExecutable"]];
    If[!StringQ[uv] || !FileExistsQ[uv],
      Return[iFailure["UVNotFound", "uv executable was not found."]]
    ];
    dir = OptionValue["VRCpilotDirectory"];
    If[!StringQ[dir] || !DirectoryQ[dir],
      Return[iFailure["VRCpilotDirectoryMissing", "VRCpilotDirectory does not exist."]]
    ];
    modelDir = Replace[OptionValue["ModelDirectory"], Automatic | None -> ""];
    command = Join[
      {uv, "run", "--with", "vosk>=0.3.45", "python",
       iPythonPath["VRCRealtime_voice_commands.py"], "--install"},
      If[StringQ[modelDir] && StringLength[modelDir] > 0, {"--model-dir", modelDir}, {}]
    ];
    result = RunProcess[command, ProcessDirectory -> dir];
    If[!AssociationQ[result],
      Return[iFailure["InstallFailed", "uv run did not return a result."]]
    ];
    out = StringTrim[Lookup[result, "StandardOutput", ""]];
    parsed = Quiet @ Check[
      ImportString[Last[StringSplit[out, "\n"], "{}"], "RawJSON"], $Failed];
    If[result["ExitCode"] =!= 0 || !AssociationQ[parsed],
      Return[iFailure["InstallFailed",
        "Voice command model install failed.",
        <|"ExitCode" -> result["ExitCode"],
          "StandardError" -> Lookup[result, "StandardError", ""],
          "StandardOutput" -> out|>]]
    ];
    parsed
  ];

VRCRealtimeLook[question_String : "現在のVRChat画面には何が見えますか。"] :=
  iSendCommand[<|"command" -> "look", "question" -> question|>];

(* ---- 発表登録簿 (SourceVault_slidedeck.wl) との結線 ----
   スライドと発表シナリオの中身は SlideWorkflow の管轄、その所在と公開可否は
   SourceVault の管轄、VRChat の挙動だけがここの管轄。よってこの層は原稿を
   作らず、登録簿から引いてそのまま worker へ渡すだけにする。 *)

iSlideDeckRegistryAvailableQ[] :=
  Length[DownValues[SourceVault`SourceVaultSlideDeckPresentationSpec]] > 0;

VRCRealtimeListPresentations[] := If[!iSlideDeckRegistryAvailableQ[],
  iFailure["SlideDeckRegistryUnavailable",
    "SourceVault の発表登録簿が読み込まれていません。SourceVault をロードしてください。"],
  Quiet @ Check[SourceVault`SourceVaultSlideDeckList[], {}]];

iResolvePresentationSpec[target_String] := Module[{entries, hit},
  If[!iSlideDeckRegistryAvailableQ[], Return[Missing["RegistryUnavailable"]]];
  If[AnyTrue[{"http://", "https://"}, StringStartsQ[target, #] &],
    (* URL 直指定でも、登録済みなら原稿を拾う *)
    entries = Quiet @ Check[SourceVault`SourceVaultSlideDeckList[], {}];
    If[!ListQ[entries], entries = {}];
    hit = SelectFirst[entries,
      AssociationQ[#] && Lookup[#, "url", ""] === target &, Missing["NoEntry"]];
    If[AssociationQ[hit],
      Quiet @ Check[
        SourceVault`SourceVaultSlideDeckPresentationSpec[Lookup[hit, "title", ""]],
        Missing["NoEntry"]],
      Missing["NoEntry"]],
    Quiet @ Check[SourceVault`SourceVaultSlideDeckPresentationSpec[target],
      Missing["NoEntry"]]]];

(* 原稿を worker が読む形へ整える。1 枚も本文が無ければ Null (= 従来の画像説明)。 *)
iNormalizeTalkPayload[talk_] := Module[{slides},
  If[!AssociationQ[talk], Return[Null]];
  slides = Lookup[talk, "slides", Lookup[talk, "Slides", {}]];
  If[!ListQ[slides], Return[Null]];
  slides = Select[
    Function[s, If[!AssociationQ[s], Nothing,
      <|"slide" -> With[{n = Lookup[s, "slide", Lookup[s, "Slide", 0]]},
          If[IntegerQ[n], n, 0]],
        "seconds" -> With[{v = Lookup[s, "seconds", Lookup[s, "Seconds", Null]]},
          If[NumericQ[v], N[v], Null]],
        "text" -> ToString @ Lookup[s, "text", Lookup[s, "Text", ""]]|>]] /@ slides,
    #["slide"] >= 1 && StringTrim[#["text"]] =!= "" &];
  If[slides === {}, Return[Null]];
  <|"opening" -> ToString @ Lookup[talk, "opening", Lookup[talk, "Opening", ""]],
    "closing" -> ToString @ Lookup[talk, "closing", Lookup[talk, "Closing", ""]],
    "slides" -> slides|>];

(* NavigateToConsole は既定 Automatic:
     まず今いる場所から操作パネルが使えるか確かめ、使えるならそのまま開始し、
     使えないときだけ部屋を探して移動してから開始する。
   以前の既定は False で「パネルの前に立っている」ことを前提にしていたため、
   立っていないと URL 入力の段で必ず失敗した。
   True = 常に探索してから開始、False = 探索しない (従来の厳格な挙動)。
   判定は configure_url が待つ述語と同一 (URL 入力欄のある操作パネル) にして
   あるので、「見えている」と答えたのに直後の URL 入力で失敗することはない。 *)
Options[VRCRealtimeStartPresentation] = {
  "SecondsPerSlide" -> Automatic,
  "ConsoleTimeoutSeconds" -> 30.,
  "NavigateToConsole" -> Automatic,
  "NavigationTimeoutSeconds" -> 120.,
  "DownloadTimeoutSeconds" -> 120.,
  "MaxDownloadMegabytes" -> 256.,
  "StartSlide" -> Automatic,
  "EndSlide" -> Automatic,
  "NarrationInstructions" -> Automatic,
  "Talk" -> Automatic,
  "Title" -> Automatic,
  (* スライド切り替え時に開く質疑応答の秒数。0 で質疑応答なし *)
  "QuestionSeconds" -> 5.,
  (* 全スライド終了後の質疑応答の上限秒数 (無音でこの時間が過ぎたら終了) *)
  "FinalQuestionSeconds" -> 120.
};

(* 引数は Sliden 用 mp4 の URL、または登録済みの発表タイトル
   (「計算と自然集会31」のような表記ゆれも登録簿側が吸収する)。 *)
VRCRealtimeStartPresentation[
    target_String : VRCRealtime`$VRCRealtimeDefaultSlideURL,
    OptionsPattern[]] := Module[
  {value, urlGiven, spec, url, title, seconds, consoleTimeout, navigate, mcpService,
   navigationTimeout, downloadTimeout, maxMB, start, end, narration, talk,
   questionSeconds, finalQuestionSeconds, result},
  value = StringTrim[target];
  urlGiven = AnyTrue[{"http://", "https://"}, StringStartsQ[value, #] &];
  (* 登録簿は MCP サービス越しに引く。止まっていると照合が timeout して原稿だけ
     静かに落ちるので、先に確保しておく (止まっていれば報告して起動を試みる)。 *)
  mcpService = iEnsureSourceVaultService[];
  spec = iResolvePresentationSpec[value];
  If[!urlGiven,
    If[!iSlideDeckRegistryAvailableQ[],
      Return[iFailure["SlideDeckRegistryUnavailable",
        "発表タイトルで開始するには SourceVault の発表登録簿が必要です。URL を直接渡すこともできます。"]]];
    If[!AssociationQ[spec] || !TrueQ[Lookup[spec, "found", False]],
      Return[iFailure["PresentationNotRegistered",
        "その発表は登録されていません。SlideWorkflow パレットの「登録」で登録してください。",
        <|"Query" -> value,
          "Available" -> If[AssociationQ[spec], Lookup[spec, "available", {}], {}]|>]]];
    If[!TrueQ[Lookup[spec, "released", True]],
      Return[iFailure["PresentationWithheld",
        "その発表は privacy 判定により音声発表へ出せません。",
        <|"Query" -> value|>]]];
    url = ToString @ Lookup[spec, "url", ""],
    url = value];
  If[!AnyTrue[{"http://", "https://"}, StringStartsQ[url, #] &],
    Return[iFailure["InvalidSlideURL",
      "Slide URL must be an absolute HTTP or HTTPS URL."]]];

  (* オプション明示 > 登録簿 > 既定 の順に解決する *)
  seconds = Replace[OptionValue["SecondsPerSlide"], Automatic :>
    If[AssociationQ[spec] && NumericQ[Lookup[spec, "secondsPerSlide", Null]],
      N[spec["secondsPerSlide"]], 30.]];
  start = Replace[OptionValue["StartSlide"], Automatic :>
    If[AssociationQ[spec] && IntegerQ[Lookup[spec, "startSlide", Null]],
      spec["startSlide"], 1]];
  end = Replace[OptionValue["EndSlide"], Automatic :>
    If[AssociationQ[spec] && IntegerQ[Lookup[spec, "endSlide", Null]],
      spec["endSlide"], All]];
  narration = Replace[OptionValue["NarrationInstructions"], Automatic :>
    If[AssociationQ[spec] && StringQ[Lookup[spec, "narrationInstructions", Null]],
      spec["narrationInstructions"], ""]];
  title = Replace[OptionValue["Title"], Automatic :>
    Which[
      AssociationQ[spec] && StringQ[Lookup[spec, "title", Null]] &&
        StringTrim[spec["title"]] =!= "", spec["title"],
      url === VRCRealtime`$VRCRealtimeDefaultSlideURL, "計算と自然31",
      True, Null]];
  talk = OptionValue["Talk"];
  talk = Which[
    talk === None, Null,
    talk === Automatic,
      If[AssociationQ[spec] && !TrueQ[Lookup[spec, "talkWithheld", False]],
        iNormalizeTalkPayload[Lookup[spec, "talk", Null]], Null],
    True, iNormalizeTalkPayload[talk]];

  questionSeconds = OptionValue["QuestionSeconds"];
  finalQuestionSeconds = OptionValue["FinalQuestionSeconds"];
  consoleTimeout = OptionValue["ConsoleTimeoutSeconds"];
  navigate = OptionValue["NavigateToConsole"];
  navigationTimeout = OptionValue["NavigationTimeoutSeconds"];
  downloadTimeout = OptionValue["DownloadTimeoutSeconds"];
  maxMB = OptionValue["MaxDownloadMegabytes"];
  If[!And[
      NumericQ[seconds] && 3. <= N[seconds] <= 300.,
      NumericQ[consoleTimeout] && 3. <= N[consoleTimeout] <= 180.,
      MemberQ[{Automatic, True, False}, navigate],
      NumericQ[navigationTimeout] && 10. <= N[navigationTimeout] <= 300.,
      NumericQ[downloadTimeout] && 5. <= N[downloadTimeout] <= 600.,
      NumericQ[maxMB] && 1. <= N[maxMB] <= 1024.,
      IntegerQ[start] && start >= 1,
      end === All || IntegerQ[end] && end >= start,
      StringQ[narration],
      title === Null || StringQ[title],
      talk === Null || AssociationQ[talk],
      NumericQ[questionSeconds] && 0. <= N[questionSeconds] <= 60.,
      NumericQ[finalQuestionSeconds] && 0. <= N[finalQuestionSeconds] <= 1800.],
    Return[iFailure["InvalidPresentationOptions",
      "One or more automatic presentation options are invalid."]]
  ];
  (* URL 直指定なら登録簿は要らない (原稿が付かないだけ) ので止めない。
     タイトル指定は登録簿が引けないと何も決まらないので、ここで諦める。 *)
  If[mcpService["Status"] === "StartFailed" && !urlGiven,
    Return[iFailure["SourceVaultServiceMissing",
      "SourceVault MCP サービスを起動できず、発表登録簿を引けません。",
      <|"MCPService" -> mcpService,
        "Hint" -> "SourceVaultMCPStatus[] を確認してください。"
          <> " URL を直接渡せば登録簿なしでも開始できます (原稿は付きません)。"|>]]
  ];
  result = iSendCommand[<|
    "command" -> "presentation-start",
    "url" -> url,
    "secondsPerSlide" -> N[seconds],
    "consoleTimeoutSeconds" -> N[consoleTimeout],
    "navigateToConsole" -> Replace[navigate, {Automatic -> "auto",
      True -> "always", False -> "never"}],
    "navigationTimeoutSeconds" -> N[navigationTimeout],
    "activatePresentationMode" -> True,
    "presentationTitle" -> title,
    "downloadTimeoutSeconds" -> N[downloadTimeout],
    "maxDownloadMegabytes" -> N[maxMB],
    "startSlide" -> start,
    "endSlide" -> Replace[end, All -> Null],
    "narrationInstructions" -> narration,
    "questionSeconds" -> N[questionSeconds],
    "finalQuestionSeconds" -> N[finalQuestionSeconds],
    "talk" -> talk,
    (* 原稿の無い発表を聞いたときに、未登録なのか原稿未作成なのかを後から
       区別できるようにする *)
    "catalogRoute" -> If[AssociationQ[spec] && TrueQ[Lookup[spec, "found", False]],
      "registry", "direct"]
  |>];
  If[AssociationQ[result],
    $iSession["PresentationURL"] = url;
    $iSession["PresentationTitle"] = title;
    $iSession["PresentationScripted"] = AssociationQ[talk];
    $iSession["PresentationStartedAt"] = Now
  ];
  result
];

VRCRealtimePresentationStatus[] := Module[{state},
  state = iReadState[];
  If[!AssociationQ[state], Return[<||>]];
  Join[
    <|"ProcessStatus" -> If[MatchQ[$iProcess, _ProcessObject],
       Quiet @ Check[ProcessStatus[$iProcess], "Unknown"], "NotStarted"]|>,
    KeySelect[state, StringStartsQ[#, "presentation"] &]
  ]
];

VRCRealtimeStopPresentation[] :=
  iSendCommand[<|"command" -> "presentation-stop"|>];

VRCRealtimePresentationControl[
    action : ("Pause" | "Resume" | "Stop")] :=
  iSendCommand[<|"command" -> "presentation-control",
    "value" -> ToLowerCase[action]|>];

VRCRealtimeSetMode[
    mode : ("Standalone" | "OwnerFollow" | "Presentation")] := Module[{result},
  result = iSendCommand[<|"command" -> "mode", "value" -> mode|>];
  If[AssociationQ[result], $iSession["AvatarMode"] = mode];
  result
];

VRCRealtimeSetOwnerAccount[account_String] := Module[{value, result},
  value = StringTrim[account];
  If[StringLength[value] == 0,
    Return[iFailure["InvalidOwnerAccount", "Owner account must not be empty."]]
  ];
  VRCRealtime`$VRCRealtimeOwnerAccount = value;
  result = iSendCommand[<|"command" -> "owner-account", "value" -> value|>];
  If[AssociationQ[result], $iSession["OwnerAccount"] = value];
  result
];

VRCRealtimeSetFollowDistance[target_?NumericQ] := Module[{value, legacy},
  value = N[target];
  legacy = Lookup[$iSession, "OwnerFollowEngine", "Formation"] === "Legacy";
  If[legacy,
    If[!0.005 <= value <= 0.15,
      Return[iFailure["InvalidFollowDistance",
        "Follow distance target must be between 0.005 and 0.15 (nameplate height, legacy engine)."]]
    ];
    Return[iSendCommand[<|"command" -> "follow-distance",
      "targetNameplateHeight" -> value|>]]
  ];
  (* Formation エンジン: メートル。旧来の 0.005〜0.15 の値も後方互換で受ける。 *)
  If[0.005 <= value <= 0.15,
    Return[iSendCommand[<|"command" -> "follow-distance",
      "targetNameplateHeight" -> value|>]]
  ];
  If[!0.8 <= value <= 6.,
    Return[iFailure["InvalidFollowDistance",
      "Follow distance must be between 0.8 and 6 metres."]]
  ];
  iSendCommand[<|"command" -> "follow-distance", "distanceMeters" -> value|>]
];

VRCRealtimeLocateOwner[] := iSendCommand[<|"command" -> "locate-owner"|>];

Options[VRCRealtimeSetFormation] = {"RearAngle" -> None, "OwnerBearing" -> None};
VRCRealtimeSetFormation[opts : OptionsPattern[]] := Module[{rear, bearing, payload},
  rear = OptionValue["RearAngle"];
  bearing = OptionValue["OwnerBearing"];
  If[rear =!= None && !(NumericQ[rear] && 0. <= N[rear] <= 90.),
    Return[iFailure["InvalidFormation", "RearAngle must be between 0 and 90 degrees."]]
  ];
  If[bearing =!= None && !(NumericQ[bearing] && 10. <= N[bearing] <= 42.),
    Return[iFailure["InvalidFormation", "OwnerBearing must be between 10 and 42 degrees."]]
  ];
  payload = <|"command" -> "formation"|>;
  If[rear =!= None, payload["rearAngleDeg"] = N[rear]];
  If[bearing =!= None, payload["ownerBearingDeg"] = N[bearing]];
  iSendCommand[payload]
];

VRCRealtimeMonitor[] :=
  Dynamic[Refresh[VRCRealtimeStatus[], UpdateInterval -> 1.0]];

(* ---- ワールド入場 (VRCRealtime_worldjoin.py) ----
   Home ワールドからグループインスタンスを作って入場し、Sliden の部屋まで歩く
   ところまでを、音声ブリッジ (worker) とは別の外部プロセスで行う。VRChat の
   入力を持つプロセスは同時に 1 つであるべきなので、追尾/発表中の worker が
   あれば断る。進捗は worker と同じく状態ファイル経由で読む。 *)

$iJoinProcess = None;
$iJoinStateFile = None;
$iJoinControlFile = None;
$iJoinSpecFile = None;
$iJoinSession = <||>;
$iJoinProcessDiagnostics = <||>;

ClearAll[iJoinProcessRunningQ, iReadJoinState, iWriteJSONFile,
  iReadFinishedJoinDiagnostics, iJoinResult, iSendJoinCommand];

iJoinProcessRunningQ[] :=
  MatchQ[$iJoinProcess, _ProcessObject] &&
    TrueQ @ Quiet @ Check[ProcessStatus[$iJoinProcess] === "Running", False];

iReadJoinState[] := Module[{value},
  If[!StringQ[$iJoinStateFile] || !FileExistsQ[$iJoinStateFile], Return[<||>]];
  value = Quiet @ Check[Import[$iJoinStateFile, "RawJSON"], <||>];
  If[AssociationQ[value], value, <||>]
];

(* ExportByteArray で単一 UTF-8 エンコード (iSendCommand と同じ理由:
   Export[..., "RawJSON"] は日本語 Windows で二重エンコードされる)。 *)
iWriteJSONFile[path_String, value_Association] := TrueQ @ Quiet @ Check[
  Module[{ba = ExportByteArray[value, "RawJSON"], strm},
    If[! ByteArrayQ[ba], Return[False, Module]];
    strm = OpenWrite[path, BinaryFormat -> True];
    If[Head[strm] =!= OutputStream, Return[False, Module]];
    WithCleanup[BinaryWrite[strm, ba], Quiet @ Close[strm]];
    True],
  False];

iReadFinishedJoinDiagnostics[] := Module[{out, err},
  If[!MatchQ[$iJoinProcess, _ProcessObject], Return[<||>]];
  If[AssociationQ[$iJoinProcessDiagnostics] && Length[$iJoinProcessDiagnostics] > 0,
    Return[$iJoinProcessDiagnostics]];
  out = Quiet @ Check[ReadString[$iJoinProcess["StandardOutput"]], ""];
  err = Quiet @ Check[ReadString[$iJoinProcess["StandardError"]], ""];
  $iJoinProcessDiagnostics = <|
    "StandardOutput" -> If[StringQ[out], out, ""],
    "StandardError" -> If[StringQ[err], err, ""]|>;
  $iJoinProcessDiagnostics
];

Options[VRCRealtimeJoinNGCGalleryWorld] = {
  "FavoritesList" -> "Favorite Worlds 1",
  "Group" -> "\:8a08\:7b97\:3068\:81ea\:7136",
  "GroupAccess" -> "Group+",
  "Region" -> "JP",
  "EnterSlidenRoom" -> True,
  "ApproachConsole" -> False,
  "StepTimeoutSeconds" -> 20.,
  "LoadTimeoutSeconds" -> 180.,
  (* \:5165\:5ba4\:306f\:300c\:898b\:3048\:308b\:5834\:6240\:307e\:3067\:6b69\:304f\:300d\:3067\:306f\:6e08\:307e\:306a\:3044:
     \:30ac\:30e9\:30b9\:58c1\:8d8a\:3057\:306b\:753b\:9762\:304c\:898b\:3048\:308b\:305f\:3081\:3001\:56de\:308a\:8fbc\:307f (\:6a2a\:6b69\:304d) \:306e\:5206\:3060\:3051
     \:4f59\:88d5\:304c\:8981\:308b\:3002120s \:3060\:3068 4 \:672c\:76ee\:306e\:8ff4\:56de\:3067\:5207\:308c\:308b\:3002 *)
  "NavigationTimeoutSeconds" -> 210.,
  "Wait" -> True,
  "TotalTimeoutSeconds" -> Automatic,
  "VRCpilotDirectory" -> "C:\\Users\\imai_\\gitwork\\vrcpilot",
  "UVExecutable" -> Automatic,
  "VRChatPID" -> Automatic,
  "OSCHost" -> "127.0.0.1",
  "OSCPort" -> 9000,
  "DiagnosticDirectory" -> Automatic
};

VRCRealtimeJoinNGCGalleryWorld[world_String : "NCG Gallery", OptionsPattern[]] := Module[
  {favorites, group, access, region, enterRoom, approachConsole, stepTimeout,
   loadTimeout, navigationTimeout, wait, totalTimeout, vrcpilotDirectory, uv,
   vrchatPID, oscHost, oscPort, diagnostics, tag, stateFile, controlFile,
   specFile, spec, command, environment, process, workerState},

  If[iJoinProcessRunningQ[],
    Return[iFailure["JoinAlreadyRunning",
      "VRCRealtimeJoinNGCGalleryWorld is already running. Use VRCRealtimeJoinNGCGalleryWorldStatus[] or VRCRealtimeJoinNGCGalleryWorldStop[].",
      <|"Status" -> VRCRealtimeJoinNGCGalleryWorldStatus[]|>]]];
  If[!FileExistsQ[$iWorldJoinPath],
    Return[iFailure["WorldJoinScriptMissing", "VRCRealtime_worldjoin.py was not found."]]];
  (* 追尾/発表中の worker とマウス・OSC 入力を奪い合わない *)
  If[iProcessRunningQ[],
    workerState = iReadState[];
    If[Lookup[$iSession, "AvatarMode", "Standalone"] =!= "Standalone" ||
        MemberQ[{"running", "stopping"}, Lookup[workerState, "presentationStatus", None]],
      Return[iFailure["WorkerAvatarActive",
        "The VRCRealtime worker is following the owner or presenting. Run VRCRealtimeSetMode[\"Standalone\"] (or VRCRealtimeStop[]) before VRCRealtimeJoinNGCGalleryWorld.",
        <|"AvatarMode" -> Lookup[$iSession, "AvatarMode", None],
          "PresentationStatus" -> Lookup[workerState, "presentationStatus", None]|>]]]];

  favorites = OptionValue["FavoritesList"];
  group = OptionValue["Group"];
  access = OptionValue["GroupAccess"];
  region = OptionValue["Region"];
  enterRoom = OptionValue["EnterSlidenRoom"];
  approachConsole = OptionValue["ApproachConsole"];
  stepTimeout = OptionValue["StepTimeoutSeconds"];
  loadTimeout = OptionValue["LoadTimeoutSeconds"];
  navigationTimeout = OptionValue["NavigationTimeoutSeconds"];
  wait = OptionValue["Wait"];
  vrchatPID = Replace[OptionValue["VRChatPID"], Automatic -> 0];
  oscHost = OptionValue["OSCHost"];
  oscPort = OptionValue["OSCPort"];
  If[!And[
      StringQ[world] && StringLength[StringTrim[world]] > 0,
      StringQ[favorites] && StringLength[StringTrim[favorites]] > 0,
      StringQ[group] && StringLength[StringTrim[group]] > 0,
      MemberQ[{"Group", "Group+", "Group Public"}, access],
      MemberQ[{"USW", "USE", "EU", "JP"}, region],
      BooleanQ[enterRoom], BooleanQ[approachConsole], BooleanQ[wait],
      NumericQ[stepTimeout] && 3. <= N[stepTimeout] <= 300.,
      NumericQ[loadTimeout] && 10. <= N[loadTimeout] <= 900.,
      NumericQ[navigationTimeout] && 10. <= N[navigationTimeout] <= 600.,
      IntegerQ[vrchatPID] && vrchatPID >= 0,
      StringQ[oscHost] && StringLength[oscHost] > 0,
      IntegerQ[oscPort] && 1 <= oscPort <= 65535],
    Return[iFailure["InvalidOptions", "One or more VRCRealtimeJoinNGCGalleryWorld options are invalid."]]];
  (* メニュー 9 手 + 読み込み + 入室 (+ パネル接近) の合計に余裕を足す *)
  totalTimeout = Replace[OptionValue["TotalTimeoutSeconds"], Automatic :>
    12. * N[stepTimeout] + N[loadTimeout] +
      If[enterRoom, N[navigationTimeout], 0.] +
      If[enterRoom && approachConsole, N[navigationTimeout], 0.] + 60.];
  If[!(NumericQ[totalTimeout] && N[totalTimeout] > 0.),
    Return[iFailure["InvalidOptions", "TotalTimeoutSeconds must be positive."]]];

  uv = iResolveUV[OptionValue["UVExecutable"]];
  If[!StringQ[uv] || !FileExistsQ[uv],
    Return[iFailure["UVNotFound",
      "uv executable was not found. Set \"UVExecutable\" to the full path of uv.exe."]]];
  vrcpilotDirectory = OptionValue["VRCpilotDirectory"];
  If[!StringQ[vrcpilotDirectory] || !DirectoryQ[vrcpilotDirectory],
    Return[iFailure["VRCpilotDirectoryMissing",
      "The configured VRC Pilot project directory does not exist."]]];

  tag = StringReplace[CreateUUID[], "-" -> ""];
  stateFile = FileNameJoin[{$TemporaryDirectory, "vrc-worldjoin-" <> tag <> ".json"}];
  controlFile = FileNameJoin[{$TemporaryDirectory, "vrc-worldjoin-control-" <> tag <> ".json"}];
  specFile = FileNameJoin[{$TemporaryDirectory, "vrc-worldjoin-spec-" <> tag <> ".json"}];
  diagnostics = Replace[OptionValue["DiagnosticDirectory"], Automatic :>
    FileNameJoin[{$TemporaryDirectory, "vrc-worldjoin-diag-" <> tag}]];
  If[!StringQ[diagnostics],
    Return[iFailure["InvalidOptions", "DiagnosticDirectory must be a directory path or Automatic."]]];
  (* 日本語 (グループ名) はコマンドライン引数でなく UTF-8 の JSON で渡す *)
  spec = <|
    "world" -> world,
    "favorites" -> favorites,
    "group" -> group,
    "groupAccess" -> access,
    "region" -> region,
    "enterRoom" -> enterRoom,
    "approachConsole" -> approachConsole,
    "stepTimeoutSeconds" -> N[stepTimeout],
    "loadTimeoutSeconds" -> N[loadTimeout],
    "navigationTimeoutSeconds" -> N[navigationTimeout],
    "vrchatPid" -> vrchatPID,
    "oscHost" -> oscHost,
    "oscPort" -> oscPort|>;
  If[!iWriteJSONFile[specFile, spec],
    Return[iFailure["SpecWriteFailed", "Failed to write the world join spec file."]]];

  command = {uv, "run", "python", $iWorldJoinPath,
    "--spec-file", specFile,
    "--state-file", stateFile,
    "--control-file", controlFile,
    "--diagnostic-dir", diagnostics};
  environment = Join[Association @ GetEnvironment[], <|"PYTHONUNBUFFERED" -> "1"|>];
  process = Quiet @ Check[
    StartProcess[command, ProcessDirectory -> vrcpilotDirectory,
      ProcessEnvironment -> environment],
    $Failed];
  If[!MatchQ[process, _ProcessObject],
    Return[iFailure["StartFailed", "Failed to start the VRCRealtime world join process."]]];

  $iJoinProcess = process;
  $iJoinStateFile = stateFile;
  $iJoinControlFile = controlFile;
  $iJoinSpecFile = specFile;
  $iJoinProcessDiagnostics = <||>;
  $iJoinSession = <|
    "Process" -> process,
    "StateFile" -> stateFile,
    "ControlFile" -> controlFile,
    "DiagnosticDirectory" -> diagnostics,
    "World" -> world,
    "FavoritesList" -> favorites,
    "Group" -> group,
    "GroupAccess" -> access,
    "Region" -> region,
    "EnterSlidenRoom" -> enterRoom,
    "ApproachConsole" -> approachConsole,
    "StartedAt" -> Now|>;
  If[!wait, Return[$iJoinSession]];
  iJoinResult[N[totalTimeout]]
];

(* 完了まで待つ。フェーズが変わるたびに一時セルで進捗を見せる (headless では無音)。 *)
iJoinResult[totalTimeout_?NumericQ] := Module[
  {deadline, state, status, lastPhase = None, phase, finished = False, diag},
  deadline = AbsoluteTime[] + totalTimeout;
  While[AbsoluteTime[] < deadline,
    state = iReadJoinState[];
    status = Lookup[state, "worldJoinStatus", None];
    phase = Lookup[state, "worldJoinPhase", None];
    If[StringQ[phase] && phase =!= lastPhase,
      lastPhase = phase;
      Quiet @ PrintTemporary[Row[{"VRCRealtimeJoinNGCGalleryWorld: ", phase,
        With[{steps = Lookup[state, "worldJoinSteps", {}]},
          If[ListQ[steps] && steps =!= {}, "  (" <> StringRiffle[steps, " > "] <> ")", ""]]}]]];
    If[MemberQ[{"done", "failed", "cancelled"}, status], finished = True; Break[]];
    If[!iJoinProcessRunningQ[],
      (* プロセスは終了。最後の状態書込 (tmp -> rename) と読みが競合すると一時的に
         空になるので、終端状態を数回読み直してから「落ちた」と判定する *)
      Do[
        Pause[0.25];
        state = iReadJoinState[];
        status = Lookup[state, "worldJoinStatus", None];
        finished = MemberQ[{"done", "failed", "cancelled"}, status];
        If[finished, Break[]],
        {6}];
      Break[]];
    Pause[0.4]];
  If[!finished && iJoinProcessRunningQ[],
    Quiet @ iSendJoinCommand[<|"command" -> "stop"|>];
    Return[iFailure["JoinTimeout",
      "VRCRealtimeJoinNGCGalleryWorld did not finish before TotalTimeoutSeconds; a stop was requested.",
      <|"Phase" -> Lookup[state, "worldJoinPhase", None],
        "Steps" -> Lookup[state, "worldJoinSteps", {}],
        "DiagnosticDirectory" -> Lookup[$iJoinSession, "DiagnosticDirectory", None],
        "State" -> state|>]]];
  If[!finished,
    diag = iReadFinishedJoinDiagnostics[];
    Return[iFailure["JoinProcessExited",
      "The world join process exited without reporting a result.",
      Join[<|"Phase" -> Lookup[state, "worldJoinPhase", None],
        "Steps" -> Lookup[state, "worldJoinSteps", {}],
        "DiagnosticDirectory" -> Lookup[$iJoinSession, "DiagnosticDirectory", None]|>, diag]]]];
  Switch[status,
    "done",
      <|"Status" -> "Joined",
        "World" -> Lookup[$iJoinSession, "World", None],
        "Steps" -> Lookup[state, "worldJoinSteps", {}],
        "Settings" -> Lookup[Lookup[state, "worldJoinSummary", <||>], "settings", <||>],
        "Load" -> Lookup[Lookup[state, "worldJoinSummary", <||>], "load", <||>],
        "Room" -> Lookup[Lookup[state, "worldJoinSummary", <||>], "room", None],
        "Console" -> Lookup[Lookup[state, "worldJoinSummary", <||>], "console", None],
        "Warnings" -> Lookup[state, "worldJoinWarnings", {}],
        "ElapsedSeconds" -> Lookup[state, "worldJoinElapsedSeconds", None],
        "DiagnosticDirectory" -> Lookup[$iJoinSession, "DiagnosticDirectory", None]|>,
    "cancelled",
      iFailure["JoinCancelled", "VRCRealtimeJoinNGCGalleryWorld was cancelled.",
        <|"Phase" -> Lookup[state, "worldJoinPhase", None],
          "Steps" -> Lookup[state, "worldJoinSteps", {}]|>],
    _,
      iFailure["JoinFailed",
        ToString @ Lookup[state, "worldJoinError", "VRCRealtimeJoinNGCGalleryWorld failed."],
        <|"Phase" -> Lookup[state, "worldJoinPhase", None],
          "Steps" -> Lookup[state, "worldJoinSteps", {}],
          "Warnings" -> Lookup[state, "worldJoinWarnings", {}],
          "LastTexts" -> Lookup[state, "worldJoinLastTexts", {}],
          "DiagnosticDirectory" -> Lookup[$iJoinSession, "DiagnosticDirectory", None]|>]]
];

iSendJoinCommand[command_Association] := Module[{temporary},
  If[!iJoinProcessRunningQ[],
    Return[iFailure["NotRunning", "VRCRealtimeJoinNGCGalleryWorld is not running."]]];
  If[!StringQ[$iJoinControlFile] || StringLength[$iJoinControlFile] == 0,
    Return[iFailure["ControlWriteFailed", "No control file for the world join process."]]];
  temporary = $iJoinControlFile <> ".tmp";
  If[iWriteJSONFile[temporary, command] &&
      TrueQ @ Quiet @ Check[
        RenameFile[temporary, $iJoinControlFile, OverwriteTarget -> True]; True, False],
    <|"Status" -> "Sent", "Command" -> Lookup[command, "command", None]|>,
    iFailure["ControlWriteFailed", "Failed to write the world join control file."]]
];

VRCRealtimeJoinNGCGalleryWorldStatus[] := Module[{processStatus, state},
  processStatus = If[MatchQ[$iJoinProcess, _ProcessObject],
    Quiet @ Check[ProcessStatus[$iJoinProcess], "Unknown"], "NotStarted"];
  state = iReadJoinState[];
  Join[
    <|"ProcessStatus" -> processStatus|>,
    KeyDrop[state, {"worldJoinSpec"}],
    If[processStatus === "Finished", iReadFinishedJoinDiagnostics[], <||>],
    If[AssociationQ[$iJoinSession],
      KeyTake[$iJoinSession, {"World", "DiagnosticDirectory", "StartedAt"}], <||>]]
];

Options[VRCRealtimeJoinNGCGalleryWorldStop] = {"WaitSeconds" -> 8.0};
VRCRealtimeJoinNGCGalleryWorldStop[OptionsPattern[]] := Module[{sent, wait, deadline, pid},
  wait = OptionValue["WaitSeconds"];
  If[!NumericQ[wait] || N[wait] < 0,
    Return[iFailure["InvalidWaitSeconds", "WaitSeconds must be non-negative."]]];
  sent = iSendJoinCommand[<|"command" -> "stop"|>];
  If[FailureQ[sent], Return[sent]];
  deadline = AbsoluteTime[] + N[wait];
  While[iJoinProcessRunningQ[] && AbsoluteTime[] < deadline, Pause[0.05]];
  If[iJoinProcessRunningQ[],
    (* 待っても止まらない: OCR やクリックの途中で刺さっている。強制終了する *)
    pid = Quiet @ Check[$iJoinProcess["ProcessID"], $Failed];
    If[$OperatingSystem === "Windows" && IntegerQ[pid],
      Quiet @ Check[RunProcess[{"taskkill", "/PID", ToString[pid], "/T", "/F"}], Null],
      Quiet @ Check[KillProcess[$iJoinProcess], Null]]];
  If[StringQ[$iJoinControlFile] && FileExistsQ[$iJoinControlFile],
    Quiet @ Check[DeleteFile[$iJoinControlFile], Null]];
  <|"Status" -> If[iJoinProcessRunningQ[], "StopRequested", "Stopped"],
    "Phase" -> Lookup[iReadJoinState[], "worldJoinPhase", None]|>
];

End[];
EndPackage[];
