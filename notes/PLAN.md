# 励起軌道最適化の段階的移行計画

desktop-ur5e の制約付き最適化体系を rigid-body-manipulation に段階的に移植する.

参照元: `barikata1984/desktop-ur5e` の `src/ur5e_sim/identification/` 以下
- `constraints.py` — JointLimits, _TrajectoryCache, per-joint constraints, compute_fourier_bounds
- `objective.py` — condition_number_objective, d_optimal_objective
- `optimizer.py` — ExcitationOptimizer (multi-start, parallel, wandb)
- `workspace.py` — EE 変位, ペイロード位置, EE 速度, 干渉回避

## 現状の差分

| 機能 | desktop-ur5e | rigid-body-manipulation |
|---|---|---|
| 目的関数 | 条件数 / D-optimal 選択可 | 条件数のみ |
| ソルバー | SLSQP | L-BFGS-B + ペナルティ |
| フーリエ係数バウンド | 関節ごと × ハーモニクスごとに解析的導出 | 全関節一律 coeff_bound |
| 関節位置制約 | JointLimits + make_joint_position_constraint | ペナルティで 1 関節のみ |
| 関節速度制約 | make_joint_velocity_constraint | なし |
| 関節加速度制約 | make_joint_acceleration_constraint | なし |
| ワークスペース制約 | EE 変位上限, ペイロード, EE 速度 | なし |
| 干渉回避 | CollisionChecker | なし |
| マルチスタート | モンテカルロ N 回 + early stopping | なし (1 回) |
| 並列化 | ProcessPoolExecutor | なし |
| ロギング | wandb | print のみ |
| メイン軌道 + 励起 | なし (フーリエ単体) | あり (スプライン + 窓付きフーリエ) |

## 段階的実装

### Stage 0 (2026-06-30 完了): 特異姿勢回避の最小実装

- ペナルティ項 `weight * Σ max(0, margin - |q_j|)²` を目的関数に加算
- 全関節一律の `coeff_bound` で振幅を制限
- `singular_joint`, `singular_margin`, `coeff_bound` を ExcitedTrajectoryConfig に追加
- generate.py の OmegaConf merge バグ修正 (CLI デフォルトが YAML を上書きする問題)

結果: LQR 発散を防止. 条件数は coeff_bound の制約で悪化 (75〜1700, 設定依存).

### Stage 1 (2026-07-01 完了): 関節ごとの coeff_bounds

`coeff_bound: float` を `coeff_bounds: list[float] | float` に拡張.
float なら全関節一律 (後方互換), リストなら関節ごとに設定.

変更箇所:
- `ExcitedTrajectoryConfig.coeff_bounds`
- `ExcitedTrajectory._optimize()` のバウンド構築

YAML 例:
```yaml
coeff_bounds: [0.5, 0.5, 0.5, 0.5, 0.13, 0.5]
```

効果: 条件数が約 150 倍改善 (1693 → 11.1).

### Stage 2 (2026-07-01 完了): 関節位置の上下限制約

desktop-ur5e の `JointLimits` + `make_joint_position_constraint` に相当.
全関節に対して `q_min <= q_total <= q_max` をペナルティで課す.
`singular_joint` / `singular_margin` を置き換える汎用的な仕組み.

変更箇所:
- `ExcitedTrajectoryConfig` に `q_min`, `q_max` (リスト) を追加
- `_optimize()` のペナルティ計算を一般化

### Stage 3 (2026-07-01 完了): 解析的バウンド導出

desktop-ur5e の `compute_fourier_bounds` を移植.
速度・加速度限界から三角不等式でフーリエ係数の安全なバウンドを自動計算.

変更箇所:
- `constraints.py` を新設 (desktop-ur5e から移植)
- 窓関数の違い (64s³(1-s)³ → 256s⁴(1-s)⁴) に合わせて導出式を修正

注意: 三角不等式バウンドは保守的. スライド関節で dq_max が小さい場合, 係数上限が ~0.003 と極端に小さくなる.
手動 coeff_bounds との併用が現実的. rigid-body-manipulation はメイン軌道 + 励起の構造なので, バウンドは励起成分のみに適用する.

### Stage 4 (2026-07-01 完了): D-optimal 目的関数

desktop-ur5e の `d_optimal_objective` を移植.
`-log det(Y) = -Σ log(λ_i(Y))` を最小化 (Y = W^T W の固有値から計算).

変更箇所:
- `base_trajectory.py` に `_build_observation_matrix`, `compute_d_optimal`, `compute_objective_with_cond` を追加
- `ExcitedTrajectoryConfig.objective_type` で `"condition_number"` / `"d_optimal"` を切り替え

### Stage 5 (2026-07-01 完了): マルチスタート最適化

N 個のランダム初期値から最適化し, 条件数が最良の結果を選択.
early stopping (改善しなくなったら打ち切り) を含む.

変更箇所:
- `ExcitedTrajectoryConfig` に `n_restarts`, `seed`, `early_stop_patience` を追加
- `_optimize()` をマルチスタートループに拡張, 単一リスタートを `_run_single_optimization` に分離
- `_generate_random_x0()` で高調波ほど小さい振幅の初期値を生成

比較結果 (max_iter=5, n_restarts=3):

| 指標 | condition_number | d_optimal |
|---|---|---|
| Best Cond | 3199 | 8856 |
| L2 (TLS) | 0.173 | 0.189 |
| 最適化時間 | 221s | 41s |

D-optimal は L-BFGS-B との相性で各リスタートの収束が浅い (1-2 反復).
リスタート数・反復数を増やすか, SLSQP への移行で改善する可能性がある.

### Stage 6: 並列化 + wandb

マルチスタートを ProcessPoolExecutor で並列実行.
wandb で各リスタートの条件数推移を記録.

前提: Stage 5 の完了.

### Stage 7: ワークスペース・干渉回避制約

desktop-ur5e の `workspace.py` + `collision.py` を移植.
FK でエンドエフェクタ位置を評価し, 変位上限・干渉を制約.

実機適用時に必要. シミュレーションのみなら優先度は低い.

## API 変更 (Stage 0-3)

### ExcitedTrajectoryConfig の追加パラメータ

全てオプション. 省略時は従来と同じ動作.

| パラメータ | 型 | デフォルト | 用途 |
|---|---|---|---|
| `q_min` | `list[float] \| None` | None | 関節位置下限 (ペナルティ) |
| `q_max` | `list[float] \| None` | None | 関節位置上限 (ペナルティ) |
| `coeff_bounds` | `list[float] \| None` | None (→ 全関節 0.5) | 関節ごとのフーリエ係数バウンド |
| `dq_max` | `list[float] \| None` | None | 速度上限 (解析的バウンド導出用) |
| `ddq_max` | `list[float] \| None` | None | 加速度上限 (解析的バウンド導出用) |

### 削除されたパラメータ

| パラメータ | 置き換え先 |
|---|---|
| `singular_joint` | `q_min` / `q_max` で汎用化 |
| `singular_margin` | 同上 |
| `coeff_bound` (単数) | `coeff_bounds` (複数) |

注記 (2026-07-02): `q_min`/`q_max` は範囲内に収める包含制約であり, q4≈0 のような一点近傍を除外する用途には意味論が逆で使えないことが判明した.
そのため `singularity_center` / `singularity_margin` を関節ごとのパラメータとして再導入し, `q_min`/`q_max` とは別建てのペナルティ機構として実装した (除外領域が非凸のため SLSQP でも不等式制約ではなくペナルティ法を使用).

訂正 (2026-07-09): 上記の前提 (「q4≈0 が特異点」) 自体が誤りだったと判明した. 球面手首の角速度ヤコビアン行列式は `cos(pitch)` であり, 真の特異姿勢は pitch=±π/2 (q4=0 は最良条件).
`q_min`/`q_max` (pitch を ±45° に制限) は ±π/2 の真の特異点から十分な安全マージンを確保する設計として当初から正しく機能しており, `singularity_center`/`singularity_margin` による q4=0 近傍の追加除外は冗長かつ有害だった (main_trajectory が start/end で必ず q4=0 を通るため, marginの値によらず境界で必ず制約違反する構造的矛盾を生んでいた).
対応として `singularity_center`/`singularity_margin` の仕組み自体は削除せず, 全 YAML で q4 の `singularity_margin` を 0 にして無効化した (詳細: `notes/LOGS/log_trajectory_optimization.md` 2026-07-09 エントリ, `notes/ISSUES.md`).

追記 (2026-07-09): 同日中の別作業で, 下記「バウンドの優先順位」節が説明する `coeff_bounds`(手動)と `dq_max`/`ddq_max`(解析的)の併用(小さい方を採用)自体が, 同じ誤った前提の副作用だったと判明した. 全 6 YAML で一律だった `coeff_bounds[4]=0.13` は解析的バウンド(~0.4 rad)より 3 倍以上タイトで, 併用ロジックにより常にこの手動値が支配していた. 対応として併用ロジックを撤去し, 「バウンドの優先順位」節の内容は本セッションの修正により無効化された(詳細後述).

### YAML 設定例

```yaml
# 関節位置制約 (q4 を [-0.25π, 0.25π] に制限)
q_min: [-10.0, -10.0, -10.0, -10.0, -0.7853982, -10.0]
q_max: [10.0, 10.0, 10.0, 10.0, 0.7853982, 10.0]

# 関節ごとのフーリエ係数バウンド (q4 だけタイト)
coeff_bounds: [0.5, 0.5, 0.5, 0.5, 0.13, 0.5]

# 速度・加速度上限 (解析的バウンド導出)
dq_max: [0.2, 0.2, 0.2, 3.14159, 3.14159, 3.14159]
ddq_max: [0.4, 0.4, 0.4, 8.0, 8.0, 8.0]
```

### バウンドの優先順位

旧仕様: `coeff_bounds` (手動) と `dq_max`/`ddq_max` (解析的) の両方が指定された場合, 各関節で小さい方が採用されていた.

訂正 (2026-07-09): 上記の併用仕様自体が, `coeff_bounds[4]=0.13` という根拠のない手動値を常に支配させる副作用を生んでいたと判明した (詳細: `notes/LOGS/log_trajectory_optimization.md` 2026-07-09 エントリ, `notes/ISSUES.md`).
現行仕様: `dq_max`/`ddq_max` が設定され `use_analytical_bounds=True` (既定) の場合, 解析的バウンド (三角不等式) が `coeff_bounds` の唯一の情報源となる. この場合に `coeff_bounds` も指定すると `ValueError`. `dq_max`/`ddq_max` が未設定, または `use_analytical_bounds=False` の場合のみ, 手動 `coeff_bounds` がボックスバウンドとして機能する.

### base_freq と同定 SNR のトレードオフ (2026-07-12 追記)

envelope-cond ペア (cond10/50/100 用の 3 点) が確定した後の実地検証で, `base_freq` が cond 最適化と同定精度の間に別のトレードオフを生む設計変数であることが判明した.
Van der Sluis の等化 cond は観測行列 Y の conditioning (相対誤差増幅率) を測る指標であり, 観測信号そのものの絶対 SNR は評価しない.
慣性同定は τ = M(q)q̈ を回帰するため, 同じ振幅係数でも base_freq が低いほど加速度 q̈ が小さくなり (q̈ は f_0 の 2 乗でスケールする), 同定に使える信号が痩せる. 実際 bf=0.1 と bf=0.5 は同じ envelope でも q̈ ピークが約 25 倍異なる.
訂正 (2026-07-12 追記): 現行 YAML の bf=0.1 は当初文献調査で直接的な根拠が見つからないと判定していたが, 追加の文献チェックで誤りと判明した. Swevers 1997 (IEEE T-RA, フーリエ級数励起の原論文) が f_0=0.1 Hz, N=5, T=10s を verbatim に明記しており, 現行設定 (f_0=0.1, N=5) と完全に一致する (詳細: `notes/LOGS/log_trajectory_optimization.md` 2026-07-12 (続き) エントリ).
ただし文献的根拠の有無は同定 SNR 不足の問題を解消しない. Swevers 1997 は T=10s (1周期) だが現行は T=20s (2周期) であり, 周期数の違いが同定 SNR にどう影響するかは未検証である.
したがって cond 目標の達成 (envelope 選定) とは独立に, 適切な (f_0, N, T) の組み合わせの選定が必要である.

追記 (2026-07-12 続 2): duration=10s (Swevers 1997 と周期数を一致) で 3 条件を再実行したところ, cond は 5.8-8.1 倍悪化したが TLS L2 は 0.165-0.204 帯に張り付いたままで, 周期数一致仮説は L2 改善には効かないと判明した. 続けて FTA で全 5 sim 出力の total_mass 低推定バイアスを追跡した結果, SNR 不足の真因は base_freq ではなく `sensors/sensors.py` のセンサーノイズ側にあると判明した. qacc ノイズ標準偏差 (並進 σ=3.6 m/s², 回転 σ=7.2 rad/s²) は 2 段階微分によるノイズ増幅を想定した設計値だが, bf=0.1 の低加速度信号 (peak ~1 m/s²) に対してはノイズが信号の 3-4 倍大きく, 信号自体を圧倒する. 検証としてノイズを切った (`perturbed=False`) ところ LS mass の誤差が 3e-6 まで縮み, TLS L2 が 1180 倍改善した (詳細: `notes/LOGS/log_trajectory_optimization.md` 2026-07-12 (続 2) エントリ, `notes/ISSUES.md`). 次のステップは (a) ノイズをフラグ化して実験ごとに切替可能にする, (b) MuJoCo 真値を常に保存して事後にノイズを再合成できるようにする, (c) EIV (errors-in-variables) を明示的に扱う回帰式に変更する, の 3 択で, 設計判断が必要である. これは既存 Stage 群 (Stage 0-7) とは独立の設計軸であり Stage 番号は付けない.

### generate.py の修正

OmegaConf merge で CLI のデフォルト値が YAML の値を上書きするバグを修正.
CLI で明示的に指定されなかったフィールドは YAML の値が使われる.

## 推奨順序

Stage 1 が条件数改善への最短ルートで, 30 分程度で実装可能.
Stage 2-3 は制約の汎用化と自動化で, desktop-ur5e との構造的な差を埋める.
Stage 4-5 は最適化品質の改善で, 同定精度に直結する.
Stage 6-7 は運用品質で, 実機適用時に必要.
