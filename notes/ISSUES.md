# Issues

## [2026-06-30] tyro MISSING デフォルト警告

`coefficients`/`q0` 等を `None` に変更したが, tyro が `str` 型として MISSING を検出する警告が残っている.
動作には影響しないが出力がうるさい.

## [2026-06-30] Excited 軌道最適化が遅い

6DOF で約 30 秒/反復. 実用上の問題になりうる.

## [2026-07-09] coeff_bounds[4]=0.13 に物理的根拠がない

全 6 つの YAML 設定ファイルで `coeff_bounds` の q4(pitch) 成分が一律 `0.13` に固定されているが, この値には物理的・解析的根拠がない.

背景: 2026-07-08 に, q4 の真の運動学的特異姿勢が pitch=0 ではなく **pitch=±π/2**(手首の角速度ヤコビアン行列式 = cos(pitch))であることが判明し, pitch=0 近傍を除外しようとしていた `singularity_margin` は誤った設計と判明して無効化した(詳細は `notes/LOGS/log_trajectory_optimization.md` 2026-07-09 エントリ参照). しかし `coeff_bounds[4]=0.13` は `singularity_margin` とは別に, 同じ誤った前提(「pitch=0 が特異点」)に基づいて手動で絞られていた値であり, 未対応のまま残っている.

`excited_6dof_strict.yaml` のパラメータ(duration=5.0, num_harmonics=1, base_freq=0.3, dq_max[4]=π, ddq_max[4]=2π)で `compute_fourier_bounds`(三角不等式に基づく解析的バウンド導出)を実際に計算すると, 解析的に安全な上限は約 **0.408 rad** であり, 手動値 0.13 の 3 倍以上ゆるい. `excited.py` の `compute_fourier_bounds` 呼び出し側では手動値と解析値の小さい方(`np.minimum`)が採用されるため, 常に手動の 0.13 が支配的になり, 解析的バウンドは実質的に一度も効いていない.

YAML 内のコメント自体が「q4 tight (0.13, singularity)」と明記しており, これも誤った前提に基づく手動値だったことを示している. 全 6 ファイルで `singularity_margin` の値が 0.15/0.1 とばらついていたにもかかわらず `coeff_bounds[4]` だけは一律 0.13 で固定されていた点も, ファイルごとの解析的再計算に基づく値ではなく根拠のない決め打ち値であることを裏付ける.

対応候補: `coeff_bounds[4]` を解析的バウンド(~0.4 rad 程度)に近い値まで緩める. `notes/TODO.md` の「三角不等式の解析的バウンドが保守的すぎる問題」とも関連する.

