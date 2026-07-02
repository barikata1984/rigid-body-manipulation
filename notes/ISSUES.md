# Issues

## [2026-06-30] tyro MISSING デフォルト警告

`coefficients`/`q0` 等を `None` に変更したが, tyro が `str` 型として MISSING を検出する警告が残っている.
動作には影響しないが出力がうるさい.

## [2026-06-30] Excited 軌道最適化が遅い

6DOF で約 30 秒/反復. 実用上の問題になりうる.

## [2026-07-02] singularity_margin が coeff_bounds に対して有効に機能していない

生成された励起軌道の q4(pitch) を数値チェックしたところ, 300 フレーム中 263 フレーム(88%)が除外ゾーン `|q4| < singularity_margin (0.15)` 内に留まっていた.
原因は, `dq_max`/`ddq_max` から `compute_fourier_bounds` で解析的に導出される `coeff_bounds` が全関節で約 0.063 rad しかなく, 除外マージン 0.15 rad より小さいこと. 最適化が q4 を除外ゾーン外へ押し出す振幅の余地がほとんどない.
除外制約の実装自体は正しく動作している. 現状の `dq_max`/`ddq_max` 設定下では実効性が乏しいだけである.
今回のシミュレーションはたまたま発散しなかった(境界通過は dq4=0 の通過点を一瞬跨いだのみ)が, 振幅不足の根本問題は未解決.
対応候補(未決定): `singularity_margin` を下げる, `dq_max`/`ddq_max` を緩める, `max_iter`/`n_restarts` を増やす, またはこれらの組み合わせ.

