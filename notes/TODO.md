# TODO

## Config / CLI

- [ ] tyro の MISSING デフォルト警告を抑制する(cosmetic だが出力がうるさい)

## Trajectory

- [x] ExcitedTrajectory の特異姿勢回避: q4≈0 を除外する制約を最適化に追加する (Stage 0 完了: ペナルティ法 + coeff_bound; 2026-07-02: `singularity_center`/`singularity_margin` として関節ごとに再実装, `q_min`/`q_max` とは別建て)
- [x] 関節ごとの coeff_bounds 実装 (Stage 1-3 完了: per-joint coeff_bounds, q_min/q_max 制約, compute_fourier_bounds 移植)
- [x] D-optimal 目的関数 (Stage 4 完了: objective_type で条件数/D-optimal 切替)
- [x] マルチスタート最適化 (Stage 5 完了: n_restarts, seed, early_stop_patience)
- [ ] Excited 軌道最適化の高速化(現状 ~30 秒/反復, 6DOF)
- [ ] `ExcitationTrajectory` クラス (`excitation.py`) がどこにもインポートされていないか確認し, 削除または復活させる
- [x] 条件数の列正規化(equilibration)を実装 (2026-07-07: `column_scale` 引数を追加, 既定 True. Van der Sluis/Swevers に基づく標準手法. 単位変換不変性を検証済み)
- [x] q4(pitch) の特異点回避が構造的に機能しない問題を解消する (2026-07-08: 真の特異点は pitch=±π/2 で, q_min/q_max=±π/4 が既に十分な安全マージンを確保していた. singularity_center/margin による q4=0 近傍の追加除外は冗長かつ有害だったため, 全 YAML の singularity_margin の q4 成分を 0 にして無効化. q_min/q_max による ±45° 制限は維持)
- [ ] 三角不等式の解析的バウンド(`compute_fourier_bounds`)が保守的すぎる問題への対処: dq/ddq を SLSQP の直接非線形制約にする方式を, 正しい初期値戦略(解析的バウンドを満たす解を初期値にする)で再挑戦する (2026-07-08: 過去の失敗はj5の実行不可能な設定が原因だった可能性が高いと判明. 実装方針は notes/LOGS/log_trajectory_optimization.md 参照)
- [x] `coeff_bounds[4]=0.13`(全 6 YAML で一律)を根拠のある値に見直す (2026-07-09: `excited.py` の `min(manual, analytical)` 併用ロジックを撤去し, `dq_max`/`ddq_max` かつ `use_analytical_bounds=True` の場合は解析的バウンドを唯一の情報源とするよう変更. この条件下で `coeff_bounds` も指定されていると `ValueError` を送出するガードを追加. 全 6 YAML から該当箇所の `coeff_bounds` を削除または q4=0.5(他関節と同一)に統一)
- [ ] `excited_6dof_20s_cond10/50/100.yaml` の `target_condition_number`(10/50/100)が実現可能域とかけ離れて陳腐化している問題への対処: 2026-07-11 時点で `dq_max`/`ddq_max` envelope を 2 段階でタイト化し, 到達可能な cond フロアを 2〜17 の帯から約 57 まで引き上げた (現行タイト envelope: 並進 dq/ddq=0.3/0.6, 回転(j4,j5) dq/ddq=0.5/1.0). ただし 3 YAML を同一 envelope にしたため, 差別化がむしろ逆方向に破綻している: cond10(target=10) はこのフロアより低く到達不能, cond100(target=100) は 1 反復目でフロアを下回り即発火する. cond50 のみ意図通り動作する. (詳細: notes/LOGS/log_trajectory_optimization.md 2026-07-11 エントリ, notes/ISSUES.md)
  - [ ] 差別化建て付けの再設計判断: (a) 3 YAML それぞれに別 envelope を与える(cond10 は緩め, cond50 は現行, cond100 はさらにタイト), (b) 差別化機構自体(target_condition_number による早期停止)を再設計する, のどちらを取るか判断する (2026-07-11: envelope タイトさと cond の関係を 5 点の実測データで対数線形モデル化し, cond≈25(実測 26.65)・cond<10(実測 9.64)の envelope を予測どおり再現できることを確認した. 3 YAML への具体的な envelope 割り当てはまだ未着手)
    - [x] envelope-cond ペアの三点確定 (2026-07-12: 対数線形モデルで cond10=1.3/2.1→9.64, cond50=0.35/0.6→43.96, cond100=0.22/0.37→92.04 の 3 帯が揃った. ただし独立に判明した base_freq 問題 (下記新規 TODO) により, envelope 割当ての YAML 反映は保留)
- [ ] `excited_6dof_strict.yaml` の nh スイープ(nh∈{2,3,5,10}, base_freq=0.1)を実行し, 単一調波(nh=1)の劣化を回避した非退化な本番用軌道を選定する (2026-07-09 のセッションで合意されたが未着手)
- [ ] base_freq=0.1 が同定 SNR を制限している問題への対処: FTA で cond=2.99 と cond=9.64 の TLS L2 がほぼ同水準 (0.20 vs 0.19) と判明. 加速度が f² スケールするため bf=0.1 では envelope 使用率が 11-17% にとどまり同定 SNR が不足. 対応候補は (1) bf を 0.2-0.4 Hz 帯に上げて cond と L2 の関係を再校正, (2) 文献調査で得た唯一の verbatim 数値 (Swevers 2007 CSM: f_0=0.033 Hz, sparse harmonics 20th/25th) を参考に sparse harmonics 方式に切替, (3) このまま bf=0.1 を維持し cond を独立変数とする study と割り切る (ただし L2 は 0.2 前後で頭打ち). 詳細: notes/LOGS/log_trajectory_optimization.md 2026-07-12 エントリ, notes/ISSUES.md
  - [x] duration を 20s→10s に変更 (bf=0.1 据置き) して cond 24 / cond 43 / cond 2.99 の 3 条件を再実行し Swevers 1997 (T=10s=1周期) と周期数を揃える (2026-07-12: 3 軌道生成 + sim 実施. cond は 5.8-8.1x 悪化するが TLS L2 は 0.165-0.204 帯に張り付き, 周期数一致仮説は L2 改善には効かないと判明)
  - [x] 系統的 total_mass 低推定バイアスの root cause 特定 (2026-07-12: FTA で `sensors/sensors.py:17-23` の qacc 測定ノイズ σ=3.6 m/s² が真因と特定. bf=0.1 で peak qacc ~1 m/s² と SNR 0.3:1. TLS の EIV 補償で 15% 残バイアス. 検証: `simulator.py:170` を `perturbed=False` に一時変更 → LS mass=1.116096 (誤差 3e-6), TLS L2=0.000173 で 1180x 改善. 設計変更は未実施 — 完了後にコード側は per-location revert 済み)

## Simulation / Identification

- [x] `excited_6dof.yaml` の `start_pos` を manipulator XML の `initial_state` に合わせる
- [ ] TLS 慣性パラメータ同定の精度検証: L2 誤差 0.209(spline) vs 0.140(excited)の差を定量的に評価する
- [ ] センサーノイズ (`sensors/sensors.py:17-23` の jointpos_stddev / jointvar_noise_scaler) と同定パイプラインの整合をとる設計判断: (a) `SimulatorConfig` に `perturbation: bool` フラグを追加してユーザーが実験毎に切り替え, (b) ノイズなしを default にして「実機模擬モード」だけ明示的に on, (c) MuJoCo 真値 (`d.qpos/qvel/qacc`) をメタデータ保存して事後にノイズ再合成可能にする, (d) EIV (errors-in-variables) を明示的に扱う回帰式に変更する, のどの組み合わせを取るか. FTA 結果: ノイズを切れば TLS L2=0.000173 でほぼ完全同定可能. 詳細: notes/LOGS/log_trajectory_optimization.md 2026-07-12 (続 2) エントリ, notes/ISSUES.md
  - [x] `get_unperturbed` ノイズなしデータ出力の実装 (done 2026-07-12: `SimulatorConfig.get_unperturbed` を追加し, ノイズなし複製 (`jointvars_clean` 込み) をシミュレーションから直接出力可能にした. 同一軌道 (cond=9.64) で検証: ノイズあり TLS L2=0.183 → ノイズなし TLS L2=3.2e-5 (5700x 改善). 詳細: notes/LOGS/log_trajectory_optimization.md 2026-07-12 (続 3) エントリ)
  - [x] 適切なセンサーノイズ σ を決定する (基準 (実機センサー仕様 vs 許容同定 L2) + σ スイープが必要. `sensors.py` の qacc ノイズ `jointpos_stddev * (sqrt(2)*fps)**2` という fps² 差分増幅モデルの妥当性も精査する) (2026-07-12 (続 4): fps² 差分増幅モデル自体は正しいが, 二階中心差分の係数二乗和は `sqrt(2)²=2` ではなく `[1,-2,1]` の二乗和 6 であるべきと判明し, `jointacc_noise_scaler` を `sqrt(6)*fps²` に修正した (qacc ノイズ約 22% 増). `noise_scale` ノブを追加し 6 点の基準 σ スイープを実施, TLS L2 が 0.05 を割るのは noise_scale≈0.25(qacc_sigma≈1.1 m/s²)からと判明. ただし基準 `jointpos_stddev` そのものの実機グラウンディングは未着手のまま残る (下記新規 TODO 参照). 詳細: notes/LOGS/log_trajectory_optimization.md 2026-07-12 (続 4) エントリ)
  - [x] wrench (force/torque) センサーに独立スケールのノイズを追加する (2026-07-13: `force_noise_scale`/`torque_noise_scale`/`perturb_wrench` を実装)
  - [x] `jointpos_stddev` を実機の関節エンコーダ仕様に基づいてグラウンディングする (2026-07-13/14: `force_stddev`/`torque_stddev` は Robotiq FT300-S 公称スペック (force σ=0.1N, torque σ=0.005Nm) に基づき決定済み. `jointpos_stddev` (並進 prismatic 5e-4 m, 回転 revolute 1e-3 rad. 単位が異なるため分けて評価する) は未着手のまま残る. マニピュレータ `xml_models/manipulators/sequential` が実在機の模擬か汎用仮想機かの素性確認も含む) (done 2026-08-25: 回転は UR5e のポーズ繰返し精度 ±0.03mm から逆算した 2.4e-5 rad とエンコーダ分解能 1.7e-6 rad を踏まえ, 校正残差込みの実効誤差として 1.0e-4 rad. 並進は直動軸の繰返し位置決め精度 ±5-20µm の上端として 2.0e-5 m. コミット df76d91. → notes/LOGS/2026-08-26_dataset-merge-and-noise-model.md)
  - [x] 速度・加速度のノイズ係数を実機に合わせる (2026-08-26 起票, done 同日: 現行の √2·fps / √6·fps² は「位置を平滑化せずに 2 回差分する」最悪ケースの増幅率だった. UR5e の実測記録から実効微分窓幅 32.6ms を推定し, 速度 43 (fps 非依存), 加速度 1840 (fps の 1 乗に比例, 60fps 時) に変更. 加速度は独立 2 経路で検算 (連鎖則 1841 / 直接測定 1479). ハンマーの OLS L2 は 0.0850→0.0139 (6.1 倍改善, 対応のある t=+49.7). 未コミット. → notes/LOGS/2026-08-26_dataset-merge-and-noise-model.md)
  - [ ] 実機調査で FT ノイズが運動学ノイズに対し不均衡に大きいと分かった場合に限り, 列ごとの既知 σ を重みとして `scipy.odr` に渡す scaled/generalized TLS への変更を検討する (素の `total_lstsq` は [Y|τ] 全列の等分散を仮定するため) (2026-07-13/14: kinematics=0 スイープにより, 素の TLS の異分散バイアスは wrench ノイズの絶対量ではなく Y と τ の誤差比で決まると判明した. Robotiq 公称値相当の実測ノイズ下では total_mass 誤差 <0.03% と無害で, 異分散病理が顕在化するのは kinematics ノイズがほぼゼロという非現実的条件のみだった. 優先度は低いと判断する)
  - [x] `Sensors` に seed パラメータを追加し, ノイズスイープ比較を再現可能にする (done 2026-08-26: `Sensors`/`SimulatorConfig` に `seed` を追加し CLI `--seed` で指定可能に. 未指定時も実際に引かれた値を目録ファイル最上位の `noise_seed` に記録する. 同一 seed でレンチ・回帰行列がビット単位一致することを検証済み. コミット df76d91. → notes/LOGS/2026-08-26_dataset-merge-and-noise-model.md)
  - [x] `sensors/sensors.py` のデフォルト値 (force_stddev=2N, torque_stddev=0.1Nm) を Robotiq FT300-S 公称スペック (force_stddev=0.1N, torque σ=[0.005, 0.005, 0.003]Nm) に変更する (2026-07-13/14: 値の決定は完了したがコード変更は未実施. 現行デフォルトは公称値のちょうど 20 倍過大. 2026-08-06: この未実装が下流に実害を出した — loaded_dice の torque 信号 RMS 0.023-0.033Nm に対し σ=0.1Nm で SNR≈0.3 となり摂動版が同定不能になった. 公称値なら SNR≈6 で足りる. → notes/LOGS/2026-08-06_loaded-dice-wrench-inconsistency.md) (done 2026-08-25: 作業ツリーに未コミットで存在した変更を確認しコミットした. 変更自体は 8 月上旬のセッションで実施済みだったが記録が残っていなかった)
  - [ ] 慣性パラメータ 10 個 (mass, mx/my/mz, 慣性テンソル 6 成分) のパラメータ別誤差分析を実施する (2026-07-13/14: 既存の total_mass 単体評価では wrench ノイズがどのパラメータに流れるか分離できていない. GT がゼロ近辺の成分の評価尺度 (絶対誤差か特性スケールでの正規化か) の決定も必要. 詳細: notes/ISSUES.md)
  - [x] dataset 出力ディレクトリに run 識別子を追加する (2026-07-14/15: `main.py` に `_next_run_dir` を追加し, `_build_dataset_subdir` の直後で glob→max_n+1 により `_run<N>` を自動採番するよう解消)

## Task-required base drift と励起の相互作用 (2026-07-14/15)

- [ ] Yun 2023 (arXiv:2310.12409) 精読 (最も近い null-space perturbation 手法)
- [ ] Abu-Dakka 2017 IROS "Comparison of trajectory parametrization methods" 精読
- [ ] Ayusawa 2017 ICRA "Generating persistently exciting trajectory" 精読
- [ ] In-Situ Excitation Trajectory Optimizer (Springer LNEE 2024, DOI:10.1007/978-981-95-2098-5_52) 精読 (paywall, institutional access 経由)
- [ ] cond 一覧を base 振幅 vs cond の連続関係で追加検証 (e.g., j5=4π, j5=π 等の中間点)
- [ ] Task drift excitation 論文化: 相対振幅比 $\propto T^2/(\text{turn 数})$ の数学的形式化 (proof or numerical evidence)
- [ ] `trajectories/generate.py` の tee 機構は「Loaded MuJoCo model...」の pre-optimize print をキャプチャできない (output_dir 確定前). 気になれば output_dir 解決をさらに早める

## データセット出荷 (2026-08-06)

- [ ] 出荷先 `datasets/neural-mass-fields/loaded_dice/` の `transforms.json` と train/valid/test 分割を, 同ディレクトリ同梱の `unperturbed_*.json.bak` へ差し替える (→ notes/LOGS/2026-08-06_loaded-dice-wrench-inconsistency.md)
- [ ] ~~`global_gt` をセンサ座標系へ変換して書き出すよう生成側を修正する~~ → 取り消し (2026-08-25 決定: 10 次元慣性パラメータは全て物体 aabb 系に統一. センサ系への変換は出荷前チェックの内部でのみ行う. 学習側で真値と推定値が出会う場所は全て aabb 系のため. ユーザー承認済み. → notes/DECISIONS.md 2026-08-25)
- [ ] 摂動版だけが `transforms.json` を名乗る命名規則 (`recorders/standard_recorder.py:171-173`) を見直す, または `perturb_wrench` の状態を JSON 最上位に記録して下流が検知できるようにする
  - [x] 命名規則の見直し (2026-08-25 判明: 8/3 セッションで実装済みだった. `split_file_name` + `primary_prefix` により非摂動系列があれば `unperturbed_transforms` が素の `.json` を取る. → notes/LOGS/2026-08-03_wisp-dataset-compatibility.md)
  - [ ] `perturb_wrench` の状態を JSON 最上位に記録する (未着手のまま残る)
- [ ] 出荷前の自己整合性チェックを組み込む: `‖regressor @ gt_sen − wrench‖ / ‖wrench‖` が force <1e-2 / torque <2e-2 (`global_gt` はセンサ系へ変換してから判定する. 変換なし・torque 1e-2 では正しいファイルを誤検出で弾く)
- [ ] `Sensors` の乱数種を固定し, `SimulatorConfig.config_export_path` を既定で有効にする (現状, 摂動系列も生成コマンドも再現不能)
- [ ] 重心が軸上にない物体を回帰テストに含める (hammer は重心が z 軸上・慣性テンソルほぼ対角のため座標系バグを検出できない) (2026-08-25: 単体レベルは `tests/test_iparams_transfer.py` の z 軸 180° 回転符号反転テストが部分的に担う. パイプライン全体の E2E 回帰テストは未着手)
- [ ] hammer データ (`hammer_spline_20260731_113007_run1`) が実機由来か否かを提供元に確認する (ディレクトリ構成が本リポジトリのシミュレータ出力と一致する)

## wisp 互換性対応 (2026-07-31〜08-03, 記録は 2026-08-25 に事後再構成)

→ notes/LOGS/2026-08-03_wisp-dataset-compatibility.md

- [x] 座標系の統一方針を決定する (done 2026-08-25: 10 次元慣性パラメータ 3 組は全て物体 aabb 系に統一 (8/3 実装を維持). per-frame の `regressor`/`wrench` はセンサ系のまま据え置き, これは仕様として明記. センサ系への真値変換は出荷前チェックの内部に閉じる. 分析根拠: 学習側 (pixi-wisp-container) で真値と推定値が突き合わされる経路 (score/md_mse/質量分布距離) は全て aabb 系で, `global_gt` を読む消費者は動かないスクリプト 1 本のみ. ユーザー承認済み)
- [ ] `simulators/setup.py` のカメラ距離 4→5*aabb_scale 変更の採否を判断する (理由の記録がどの文書にもない)
- [ ] wisp 側バグ 3 箇所の修正を先方に確認する: `nemd_tracker.py:562`, `md_multiview_trainer.py:678-679`, `:842,855` (慣性テンソル非対角ラベルの取り違え. 7/31 レポート §5 の正誤判定は逆だった. → wisp_handoff_20260803.md §2)
- [ ] `wandb_add_reference.py:71` のファイル名不一致 (`transform_train.json` → 実体は `transforms_train.json`) と `global_gt` からの自動採点経路 (任意改善, → wisp_handoff_20260803.md §3)
- [ ] hammer の `ground_truth.csv` の Windows 側からのサルベージ状況を確認する

- [ ] シミュレーション設定の合成順を直す: 現状「クラス既定 → YAML → CLI」の最後で, CLI 側が未指定項目も既定値で埋まった完全なオブジェクトになるため, YAML に書いた値が警告なく既定値で上書きされる. `recorder.track_cam_distance_factor` (2026-08-25 新設) を YAML で物体ごとに指定したい場合はこの修正が前提 (既存の全項目に共通の構造問題)

## Code Cleanup

- [x] `_lqr.py`(旧 LQR 実装)を削除するか残すか判断する (2026-08-25 判明: 2026-06-30 のコミット 3d85322 で削除済みだった. 現行 `controllers/lqr.py` のみ残存し `__init__.py` から参照)
- [x] `xml_models/targets/sledgehammer/object.mtl` の未コミット差分の去就を決める (2026-08-25 解消: MuJoCo は mtl を読まず実行時未使用と判明したため, ルート直下 `object.obj` ごと `_archive/sledgehammer-cad/` へ退避. 複製 PNG の `object/` は削除)
- [x] 重複・旧作業コピー 4 件の削除: ルート直下 `gt_mass_distr.csv` (sledgehammer 配下とバイト一致, 139 MB), `object_cad_gt.csv` / `object_cad_gt_2.csv` (target 外の古い作業コピー), `xml_models/targets/_sledgehammer/` (クリーンアップ前スナップショット) (2026-08-25 起票, done 同日: 削除前に `cmp` でバイト一致を再確認済み)
- [x] CAD ソースと計測 CSV の `.gitignore` 追加 (2026-08-25 解消: `xml_models/targets/_archive/` を新設して CAD ソース一式と `gt_mass_distr*.csv` を退避し, `_archive/` を .gitignore に追加. あわせて使用実績のない 17 ターゲットを `_archive/` へ git mv で退避 (残置 7: hammer, sledgehammer, loaded_dice, chair, teapot_ceramic, wooden-tray, wooden-tray-2. 全 7 つで読み込み+50 ステップ実行を検証済み). loaded_dice の変換残骸 `mjcf/` も `_archive/loaded_dice-mjcf/` へ. hammer/chair の `ground_truth.csv:Zone.Identifier` は削除)
- [x] リポジトリ直下と `configurations/trajectories/` のデバッグ残骸 13 ファイルの削除 (done 2026-08-25: `debug_*` 5 件, `spline.json/png`, `inverse_spline.json`, `debug_tyro.py`, `test_merge.py`, `tracking_*.png` 2 件, `image.png`. すべて未追跡のため履歴影響なし)

## 2026-08-26 起票 (データセット合成とノイズモデル)

- [ ] loaded_dice の慣性成分 (真値 2.55e-4 kg·m²) が推定できない原因を, ノイズ水準と励起振幅のどちらかに切り分ける (推定値の標準偏差が真値の 1〜3 倍あり seed ごとに符号が変わる. ノイズなしデータでは全 10 成分が誤差率 1% 未満で解けているため, 幾何的な励起の不足ではなく SNR の問題である可能性が高いが未検証. → notes/LOGS/2026-08-26_dataset-merge-and-noise-model.md)
- [ ] 回転方向の励起振幅の増強. 実機の加速度 SNR 中央値 15.4 に並ぶには回転の加速度ピークが約 4.0 rad/s² 必要で, 現行の純励起軌道 (0.118〜0.303 rad/s²) の 13〜34 倍にあたる. 既存 TODO の「三角不等式の解析的バウンドが保守的すぎる問題」「base_freq=0.1 が同定 SNR を制限している問題」と同じ場所を指す
- [ ] 補正 LS (Fuller のモーメント法補正, `x = (AᵀA − EᵀE)⁻¹ Aᵀb`) を `regressions/` に正式実装する (2026-08-25 の検証で L2 が LS の 12.6 分の 1, TLS の 8.3 分の 1 になった. EᵀE はデルタ法で近似する. 見積もりが ±30% ずれても TLS を上回る. → scratchpad reports/2026-08-25_231443_eiv-corrected-estimator-vs-ls-tls.md)
- [ ] 速度・加速度ノイズを位置ノイズと独立に加える現行方式の見直し. 実機では 3 者が同一の位置信号から派生するため相関し, 時間方向にも相関する. 係数の修正ではこの構造的不一致は残る
- [ ] 数十秒の静止記録を 1 本取得する. これによりノイズ推定の留保 3 件 (静止時標準偏差による第 3 の推定法, 低周波誤差とドリフトの可視化, 力覚センサのノイズと構造振動の切り分け) が同時に解消する
- [ ] スライドの信号対雑音比の表のうち位置の行 (中央値 4590) の扱いを決める. 実測から再現するには信号を関節角の絶対値 (原点を含む生の実効値) とする必要があり, 関節角の原点は校正上の約束事のため測定品質の指標として使えない. 加速度の行 (15.4) は 5% 以内で再現した
- [ ] `sensors/sensors.py` の速度・加速度係数の変更が未コミットのまま残っている. コミットするか判断する
- [ ] レンチのトルクノイズが過大に見える件の原因調査 (別エージェントが実施中). loaded_dice の慣性モーメント真値 2.55e-4 kg·m² に対しトルクノイズ 0.005 N·m が一桁以上大きく, 関節ノイズを切っても推定が成立しない
- [ ] ダイスの結果を複数の種で回して平均を取るかの判断. 配布物 `merged_wrenchonly` は種 42 の単一実現である
- [ ] `notes/dataset-generation-procedure.md` の未確認事項 1 件. 実際に打ったコマンド文字列がリポジトリに記録されておらず, E 節のコマンド列は生成物からの再構成である
