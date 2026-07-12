# Issues

## [2026-06-30] tyro MISSING デフォルト警告

`coefficients`/`q0` 等を `None` に変更したが, tyro が `str` 型として MISSING を検出する警告が残っている.
動作には影響しないが出力がうるさい.

## [2026-06-30] Excited 軌道最適化が遅い

6DOF で約 30 秒/反復. 実用上の問題になりうる.

## [2026-07-12] base_freq=0.1 Hz が同定 SNR を律速している問題

cond=2.99 (envelope 1.5/π) と cond=9.64 (envelope 1.3/2.1) の軌道で TLS L2 がほぼ同水準 (0.204 vs 0.193) となり, cond が同定精度の予測子として機能しない状態を検出した.

FTA による根本原因: base_freq=0.1 では加速度が `q̈ ∝ (2π k f_0)²` により f_0 の 2 乗でスケールするため, envelope (ddq_max) の使用率が 11-17% にとどまり, 慣性同定 (τ = M(q)q̈ の回帰) に必要な加速度信号の SNR が不足する. Van der Sluis/Swevers の等化 cond は観測行列 Y の相対誤差増幅率のみを測り, 観測信号自体の絶対 SNR は評価しないため, cond が低くても L2 が改善しない.

対応候補: (1) base_freq を 0.2-0.4 Hz 帯に上げて cond と L2 の関係を再校正する, (2) sparse harmonics 方式 (Swevers 2007 CSM: f_0=0.033 Hz, 20th/25th harmonics のみを参考) に切り替える, (3) 現状の bf=0.1 を維持し, cond を独立変数とする study と割り切る (ただし L2 は 0.2 前後で頭打ち). 判断は未着手 (詳細: `notes/LOGS/log_trajectory_optimization.md` 2026-07-12 エントリ, `notes/TODO.md`).
