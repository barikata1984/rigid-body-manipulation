# Issues

## [2026-06-30] tyro MISSING デフォルト警告

`coefficients`/`q0` 等を `None` に変更したが, tyro が `str` 型として MISSING を検出する警告が残っている.
動作には影響しないが出力がうるさい.

## [2026-06-30] Excited 軌道最適化が遅い

6DOF で約 30 秒/反復. 実用上の問題になりうる.

## [2026-07-12] センサーノイズ (qacc σ=3.6 m/s²) が同定精度を律速

hammer 慣性同定で total_mass が LS 28-29% 低推定, TLS 14-17% 低推定という系統的バイアスを 5 条件 (envelope・T・cond いずれも異なる) で検出した. cond=2.99〜253.88 の 84 倍レンジでも TLS L2 は 0.165-0.204 帯にほぼ一定.

FTA で `sensors/sensors.py:17-23` のノイズ設定が真因と特定した. `jointpos_stddev × jointvar_noise_scaler² = 5e-4 × (√2 × 60)² = 3.6 m/s²` が qacc に加わる. `simulators/simulator.py:170` の `perturbed=True` がハードコード. bf=0.1 で実測 peak qacc は約 1 m/s² と, ノイズが信号の 3-4 倍大きい. 慣性同定は τ = M(q)q̈ を回帰するため EIV 減衰で LS が下方バイアス, TLS が部分補償するが 15% が残る. cond は Y の相対誤差増幅率のみを測り絶対 SNR を評価しないため, cond に依存しない.

決定的検証: `simulator.py:170` を一時的に `perturbed=False` に変更 → LS mass=1.116096 (誤差 3e-6, 0.0003%), LS L2=0.000173 (ノイズありの 0.204 から 1180x 改善). ノイズが唯一の主要因. 2026-07-07 の bf=0.5 setup で L2=0.018 を達成できていたのは peak qacc ~25 m/s² で SNR 十分だったため.

対応候補: (a) `SimulatorConfig` に `perturbation: bool` フラグを追加して実験ごとに切替可能にする, (b) ノイズなしを default にして「実機模擬モード」だけ明示的に on, (c) MuJoCo 真値 (`d.qpos/qvel/qacc`) をメタデータに常時保存して事後にノイズ再合成可能にする, (d) EIV を明示的に扱う回帰式 (総合最小二乗 の適切な定式化, 器差モデル込み等) に変更する. 判断は未着手 (詳細: `notes/LOGS/log_trajectory_optimization.md` 2026-07-12 (続 2) エントリ, `notes/TODO.md`).

追記 (2026-07-12): `SimulatorConfig.get_unperturbed` によりノイズなし参照出力が実装され, 同一軌道でノイズあり TLS L2=0.183 → ノイズなし TLS L2=3.2e-5 (5700x 改善) を独立に再確認した. これにより offline でのノイズモデル較正が可能になったが, 本問題自体は未解決である. 論点は「センサーノイズのモデル・大きさが同定精度の律速要因である」ことに絞られた (詳細: `notes/LOGS/log_trajectory_optimization.md` 2026-07-12 (続 3) エントリ).

追記 (2026-07-12 続 4): 上記の qacc σ=3.6 m/s² という値自体に誤りがあったと判明した. `jointacc_noise_scaler` の実装は `(sqrt(2)*fps)**2 = 2*fps**2` だったが, これは二階中心差分 `(x[k]-2x[k-1]+x[k-2])/dt**2` を構成する係数 `[1,-2,1]` の二乗和 (=6) を使うべきところを, 一階差分の係数二乗和 (=2) の二乗として誤って導出したものだった. `sqrt(6)*fps**2` に修正した結果, 既定の qacc_sigma_trans は 3.6 m/s² から約 4.4 m/s² に約 22% 増加した (fps² スケーリング自体と sqrt(6) という定数はノイズモデルの選択から数学的に確定するため, ここに調整の余地はない). 修正後の値で `noise_scale` (新設の基準 σ 倍率パラメータ) を 0.03〜1.0 で振ったところ, 既定 (noise_scale=1.0, qacc_sigma=4.4 m/s²) は TLS L2=0.259 と依然として同定に過大なノイズであり, TLS L2 が 0.05 を初めて下回るのは noise_scale≈0.25 (qacc_sigma≈1.1 m/s²) からだった. 基準となる `jointpos_stddev` の適切な値は実機のエンコーダ/FT センサー仕様に基づくグラウンディングが依然として必要であり, 本問題は未解決のまま残る. なお `Sensors` が unseeded RNG のため, 上記スイープの各点は単一実現であり run-to-run variance を含む (詳細: `notes/LOGS/log_trajectory_optimization.md` 2026-07-12 (続 4) エントリ).

追記 (2026-08-06): 未実装のまま残っていた wrench 側デフォルト (force σ=2N, torque σ=0.1Nm, 公称値の 20 倍) が下流に実害を出した. loaded_dice の torque 信号 RMS は 0.023-0.033Nm で, σ=0.1Nm 下では SNR≈0.3 となり, 摂動版データセットの最小二乗解は主慣性モーメントが負という物理的に不可能な値になる. hammer の摂動版も同じ病理を持つ (最小二乗質量比 0.632, loaded_dice は 0.613) が, 出荷されたのがノイズなし版だったため表面化していなかった. 運動学側では `dtwist_sen` の並進成分のノイズ std 4.25 が信号 RMS 3.69 を上回っており, qacc ノイズ過大という本課題の中心はそのまま残る (詳細: `notes/LOGS/2026-08-06_loaded-dice-wrench-inconsistency.md`).

追記 (2026-07-13/14): wrench (force/torque) 側は実機グラウンディングにより律速要因から外れたと判明した. Robotiq FT300-S の公称シグナルノイズ (force σ=0.1N, torque σ=0.005Nm) は現行コードのデフォルト (2N/0.1Nm) のちょうど 1/20 である. kinematics ノイズ (noise_scale=0.1) と同時に与えた場合, wrench 側の寄与は kinematics 起因の誤差フロアに完全に埋もれて検出できなかった. kinematics ノイズを厳密にゼロにした極端条件のスイープでも, Robotiq 公称値相当 (R=1) では total_mass 誤差 <0.03%, L2 ≤0.0013 と無害だった. これにより本問題の律速要因は kinematics (qacc) ノイズ側に絞られた. ただし本問題自体は未解決である (詳細: `notes/LOGS/log_trajectory_optimization.md` 2026-07-13/14 エントリ).

追記 (2026-08-26): 本問題の中心だった qacc ノイズ過大は 2 段階で解消した. 第 1 段は `jointpos_stddev` の実機グラウンディングで, 並進 5e-4→2.0e-5 m, 回転 1e-3→1.0e-4 rad とした (コミット df76d91). 第 2 段は速度・加速度の係数で, 現行の `√2·fps` / `√6·fps²` が「位置を平滑化せずに 2 回差分する」最悪ケースの増幅率であり実機と式の形が違うと判明した. UR5e の実測記録から実効微分窓幅 32.6ms を推定し, 速度 43 (fps 非依存), 加速度 1840 (fps の 1 乗に比例, 60fps 時) に変更した. 回帰行列の相対ノイズは 0.197→0.042 (4.7 分の 1), ハンマーの OLS L2 は 0.0850→0.0139 (6.1 倍改善, 種 5 個の対応のある t=+49.7) となり, mz は −47.3%→−7.4%, ixx は −69.6%→−15.6% に改善した. 残る課題は loaded_dice の慣性成分 (真値 2.55e-4 kg·m² に対し推定値の標準偏差が 1〜3 倍) と, 回転方向の励起振幅不足 (実機の加速度 SNR に並ぶには 13〜34 倍必要) である (詳細: `notes/LOGS/2026-08-26_dataset-merge-and-noise-model.md`).

追記 (2026-08-26 続): 後続の OLS 因果評価に 2 ms の状態・力覚時刻ずれが含まれると判明した。
実機準拠のノイズ過程は実装したが、同定誤差への寄与は同期修正後に再評価する。
この ISSUE の過去数値は経緯として残すが、現行 profile の最終評価には使わない。

## [2026-08-06] `global_gt` が物体座標系, `regressor` がセンサ座標系で書かれている

データセットの `global_gt` は CAD 真値を物体 (aabb) 座標系原点まわりへ移した値であり (`object_cad_gt.csv` と相対差 0), 一方 `regressor` はセンサ座標系で組まれる. 両者は `pose_sen_obj` で関係し, loaded_dice では `mx, my, iyz, izx` の符号反転に相当する. ノイズなしデータで変換を施すと force 残差 0.4105→0.0179 N, torque 残差 0.0536→0.00026 N·m, 最近傍一致 300/300 になる. hammer は重心が z 軸上・慣性テンソルほぼ対角で該当 4 成分が 1e-9 以下のため症状が出ず, 重心が軸から外れた loaded_dice で初めて露呈した. `global_gt` を予測値と突き合わせる利用側 (`wandb_add_reference.py`, 提案中の出荷前チェック) はすべて影響を受ける (詳細: `notes/LOGS/2026-08-06_loaded-dice-wrench-inconsistency.md`).

追記 (2026-08-25): 統一方針が未決定のまま指示が矛盾していることが判明した. 8/3 セッションの未コミット実装は `ls`/`tls` をセンサ系から物体 aabb 系へ寄せた (`global_gt` と揃えた) のに対し, 本 ISSUE 由来の TODO は逆に `global_gt` をセンサ系へ寄せる指示になっている. 両方実施すると同一 JSON 内で `global_gt` と `ls`/`tls` の座標系が割れる. どちらの系に統一するかの決定が先に必要である (→ `notes/LOGS/2026-08-03_wisp-dataset-compatibility.md`).

解決 (2026-08-25, ユーザー承認済み): 10 次元慣性パラメータ 3 組 (`global_gt`/`ls`/`tls`) は全て物体 aabb 系に統一し, per-frame の `regressor`/`wrench` はセンサ系のままとすることを仕様として確定した. センサ系への真値変換は出荷前チェックの内部でのみ行う. これにより 8/6 の「チェックは真値をセンサ系へ変換して行う」決定と 8/3 の「`ls`/`tls` を aabb 系で書く」実装は両立する. 矛盾していたのは決定を「書き出し時に変換する」と読み替えた TODO の 1 項目だけだった. 本 ISSUE のうち座標系の混在部分はこれで閉じ, 残るのは出荷前チェックの実装のみ (TODO 参照).

追記 (2026-08-26): 同一の座標系取り違えが, 本 ISSUE の解決後に新規作成した手順書 `notes/ols-identification-procedure.md` (コミット 32c65f5) で再発した. センサ系の OLS 解を `global_gt` と直接引き算する手順を書いており, loaded_dice で mx, my, iyz, izx の 4 成分が符号反転して現れた. `transfer_iparams` による変換を挟むと L2 は 0.01027→0.0000246 になる. 手順書と `notes/debug/2026-08-26_merged-dataset-and-ols-results.md` を訂正した (コミット 916ade4). ハンマーで表面化しなかった理由は本 ISSUE 本文と同じである (詳細: `notes/LOGS/2026-08-26_dataset-merge-and-noise-model.md`).
