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
