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
  - [ ] wrench (force/torque) センサーに独立スケールのノイズを追加し, 回帰行列 Y とターゲット τ の両方にノイズが乗る errors-in-variables 問題として LS/TLS を比較する (`_get_wrench(perturbed=True)` は実装済みだが simulator 側が未使用)
  - [ ] `jointpos_stddev` / `force_stddev` / `torque_stddev` を実機の関節エンコーダおよび FT センサー仕様に基づいてグラウンディングする (現状は未検証の見積もり)
  - [ ] `Sensors` に seed パラメータを追加し, ノイズスイープ比較を再現可能にする

## Code Cleanup

- [ ] `_lqr.py`(旧 LQR 実装)を削除するか残すか判断する
