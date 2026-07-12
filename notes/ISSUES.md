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
