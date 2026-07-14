---
Title: Fourier-based optimal excitation trajectories for the dynamic identification of robots
Authors:
  - Park, Kyung-Jo
Year: 2006
Venue: Robotica
Tags:
  - "dynamic-identification"
  - "excitation-trajectory"
  - "fourier-series"
  - "d-optimality"
  - "boundary-conditions"
  - "industrial-robot"
PDF: "[[papers/Park-Robotica2006-Fourier_Optimal_Excitation/main.pdf|📃]]"
Import Date: "2026-07-15"
Read Date: 2026-07-15
Executive Summary: Swevers 型フーリエ級数励起の三つの弱点 (境界条件不整合, 速度・加速度の収束保証欠如, 収束の遅さ) を, フーリエ級数と 5 次多項式の和 $q_i(t)=\lambda_i(t)+\delta_i(t)$ で解消する励起軌道パラメタ化を提案する. 多項式係数は境界条件式から閉形式で決まる従属変数とし, フーリエ係数のみを d-最適性 (共分散行列の対数行列式) を目的関数として関節・カルテシアン制約下で最適化する. CRS A465 の 3 軸シミュレーションで最尤推定の平均相対誤差 0.66, 最大 3.26 を得, 境界条件を厳密に満たす軌道を生成できることを示した.
Citekey: Park-Robotica2006-Fourier_Optimal_Excitation
BibTeX Key: park2006fourier
DOI: 10.1017/S0263574706002712
Relevance: 5
Repository: none
Category: note
Template Version: v2.3
---

## Executive Summary
Swevers 型フーリエ級数励起の三つの弱点 (境界条件不整合, 速度・加速度の収束保証欠如, 収束の遅さ) を, フーリエ級数と 5 次多項式の和 $q_i(t)=\lambda_i(t)+\delta_i(t)$ で解消する励起軌道パラメタ化を提案する.
多項式係数は境界条件式から閉形式で決まる従属変数とし, フーリエ係数のみを d-最適性 (共分散行列の対数行列式) を目的関数として関節・カルテシアン制約下で最適化する.
CRS A465 の 3 軸シミュレーションで最尤推定の平均相対誤差 0.66, 最大 3.26 を得, 境界条件を厳密に満たす軌道を生成できることを示した.

---
## Summary

### この論文が答えた問い、あるいは解決した課題は何か？
ロボット動力学同定における励起軌道設計法として, [[papers/Swevers-TRA1997-OptimalExcitation/optimal-robot-excitation-and-identification|Swevers+ 1997]] [5] の有限フーリエ級数パラメタ化 (時間領域平均によるノイズ低減や周波数領域微分など複数の利点を持つ) には次の欠点があった (§I, §III 冒頭):
(1) 端点における境界条件を厳密には満たせない.
(2) 位置は最適解に収束しても, その時間微分 (関節速度・加速度) が最適解の微分に収束する保証がない.
(3) フーリエ級数の収束速度が最適解に依存し遅くなり得る.
本論文はこれらを, フーリエ級数の周期性・周波数解析上の利点を保ったまま解消する軌道パラメタ化を与える (§III).

### 提案手法のアプローチと、その根幹をなす要素は何か？
各関節 $i$ の位置を 5 次多項式 $\lambda_i(t)=\sum_{j=0}^{5}\lambda_{ij}t^{j}$ と, 基本周期 $t_f$ に対する $M$ 項のコサイン級数 $\delta_i(t)=\sum_{m_f=1}^{M}a_{im_f}\cos(m_f\pi t/t_f)$ の和として書き, 位置・速度・加速度の $t=0,t_f$ における 6 個の境界条件から多項式 6 係数 $\lambda_{ij}$ を閉形式でフーリエ係数の従属変数として解く (§III.1, 式 (11)–(14)). フーリエ係数 $a_{im_f}$ のみを設計変数として, 最尤推定量の共分散行列 $(F^{T}\Sigma^{-1}F)^{-1}$ の d-最適性基準 $-\log\det P$ を関節角・速度・加速度・エンドエフェクタ位置制約下で最小化する (§III.2).
- 5 次多項式部 $\lambda_i(t)$: 境界条件充足のための従属変数. フーリエ級数の定数項もここに吸収する.
- コサインのみの有限フーリエ級数 $\delta_i(t)$: 全関節で共通の基本周波数 $\omega_f=2\pi/t_f$ を用いる (合成軌道の周期性を保つため).
- d-最適性基準 $-\log\det(F^{T}\Sigma^{-1}F)^{-1}$: 尺度不変性と最尤推定パラメータの高確率密度領域体積という物理的解釈を持つスカラ化. トルクの共分散行列 $\Sigma$ で重み付けする.

### 特に参考とした既存研究と、それらと比した提案手法の新規性は何か？
主要な比較対象は [[papers/Swevers-TRA1997-OptimalExcitation/optimal-robot-excitation-and-identification|Swevers+ 1997]] [5] (サインとコサインの有限フーリエ級数のみによる周期励起) である. §III で他方式との位置づけが述べられている: Armstrong+ [8] は関節加速度の点列を自由度とし最も一般的だが自由度が過大, Gautier-Khalil [12] は疎な関節角・速度点を最適化し 5 次多項式で内挿するため軌道全体が制約や条件数指標に対して最適とは限らない. 本論文の新規性は, フーリエ級数と 5 次多項式の和という単一パラメタ化で (i) 位置・速度・加速度の境界条件を項数によらず厳密に満たし, (ii) フーリエ項の追加による収束を保証し, (iii) 導関数の収束も保証する点にある (§III.1 末尾, §V).

### どのように訓練・最適化したのか？
- **損失関数 / 最適化目的**: 最尤推定量の共分散行列 $P=(F^{T}\Sigma^{-1}F)^{-1}$ に対する d-最適性基準 $-\log\det P$ を最小化 (§III.2, 式 (10) と対応する議論). 関節角・速度・加速度上下限とエンドエフェクタ-第 1 軸間距離 $r_{ee}\ge 330\,\mathrm{mm}$ (自機衝突回避) を制約に含める. 実装上は制約充足時に負, 違反時に正となる連続関数として扱う (§III.2 末尾). Matlab Optimization Toolbox の `FMINCON` を用い, 5000 反復で停止. 初期軌道はサイクロイド運動. 検証用に `FMINSEARCH` による無制約最適化も実施 (§IV.2).
- **データセット**: N/A (シミュレーションのみ). CRS A465 産業ロボットの第 1–3 軸のみをモデル化 (§IV). 最小基底パラメータ 21 個 (剛体項 15 + 粘性摩擦 3 + クーロン摩擦 3, Table I). サンプリング 300 Hz, 基本周波数 0.2 Hz, 5 秒 = 1500 サンプル (1 周期), 3 関節 × 5 項フーリエ = 15 設計変数 (§IV.2). トルク測定ノイズは平均 0 の独立ガウス, 分散はそれぞれ $25, 16, 9\,\mathrm{N^2m^2}$ (関節 1, 2, 3). 関節角はノイズなしと仮定 (§IV.1).

### どのように検証したか？指標と結果は？
CRS A465 の 3 軸モデルで既知パラメータからトルクをシミュレートし, ガウス雑音を重畳して最尤 (実装上は重み付き最小二乗) 推定した結果を評価 (§IV). 平均相対誤差 $\varepsilon_{AV}=\frac{1}{21}\sum |( \beta^{*}_{i}-\beta_{i})/\beta^{*}_{i}|=0.66$, 最大相対誤差 $\varepsilon_{MAX}=3.26$ を得た (§IV.2, Table I 列 4, 5 に推定値と標準偏差 $\sigma_i$ を掲載). トルク再現の RMS 誤差 $\varepsilon_{RMS}$ は関節 1 で 5.4236, 関節 2 で 2.5394, 関節 3 で 2.4632 (Fig. 4). 最適化軌道は境界での位置・速度・加速度条件を厳密に満たし, サイクロイド初期軌道より高調波係数が大きくなること (Table II) が示された. カルテシアン制約下では TCP 軌道は無制約時よりコンパクトになる (Fig. 3). 著者は $\varepsilon_{AV}$ と $\varepsilon_{MAX}$ について [5], [11] の結果と比較し「reasonably accurate」と評価している.

### 検証結果に基づいた議論、明らかになった課題はあるか？
著者が明示的に述べた限界・議論は限定的で, 主要な留保は「得られた最適解は必ずしも大域最適とは限らない (near optimal solution, ..., not necessarily the global optima)」の一点である (§IV.2 中盤). 収束が保証されるのはあくまで最適化アルゴリズムの意味での最適近傍到達であり, フーリエ項数 $M$ の増加に伴う相対誤差の系統的評価や, 実機実験による検証は本論文では扱われていない (Abstract は CRS A465 の identification に言及するが本文の §IV は simulation experiment と明記). 摩擦モデルは粘性・クーロンのみで, 静摩擦や関節可撓性は含めていない (§IV.1 の記述と Table I の対応). Future work は§V に明示的に列挙されていない.

---
## 自身の研究との関連
本論文の分解形 $q_i(t)=\lambda_i(t)+\delta_i(t)$ (5 次多項式 + コサイン級数) は本プロジェクトの hammer 慣性 10 パラメータ同定で用いる 6DoF フーリエ励起 + task-required base drift の構成と形式的に同型である. しかし意味論は異なる:
- Park の $\lambda_i(t)$ は境界条件 (位置・速度・加速度) の厳密充足のための**従属変数** — フーリエ係数が決まれば 6 個の多項式係数は式 (14) で一意に定まり, 最適化変数ではない.
- 本プロジェクトの多項式部 (base drift) $q_\mathrm{base}$ は**タスク要求される目標運動** — 最適化変数でもなく境界条件から従属的に定まる量でもない, 外生的に与えられる非最適化ドリフトである.

この差の帰結として, 本プロジェクトで観測された「$q_\mathrm{base}$ の存在が条件数を桁で悪化させ, $T^{2}/(\text{turn 数})$ スケーリングで励起振幅の相対比が決まる」現象は Park の枠組みでは現れない (Park では境界値のみが問題で, 途中経路のドリフトは扱わない). 本プロジェクトの寄与は, Park および [[papers/Swevers-TRA1997-OptimalExcitation/optimal-robot-excitation-and-identification|Swevers+ 1997]] が扱わなかった「タスク要求ドリフト共存下の d-最適励起」という設定にある.

Park の d-最適性基準の使い方 ($-\log\det(F^{T}\Sigma^{-1}F)^{-1}$, トルク分散による重み付け) と, 全関節共通の基本周波数 $\omega_f=2\pi/t_f$ による周期性維持は, 本プロジェクトの目的関数構成と直接対応する参照として使える. Fourier 項数 $M$ に対する条件数改善の飽和挙動については Park は定量評価していないため, 本プロジェクトの $T^{2}/(\text{turn 数})$ スケーリングは新規知見として位置付けられる.

---
## 追加議論

---
## BibTex
<details>
<summary> Click to show/noshow the BibTex data </summary>
```bibtex
@article{park2006fourier,
  title   = {Fourier-based optimal excitation trajectories for the dynamic identification of robots},
  author  = {Park, Kyung-Jo},
  journal = {Robotica},
  volume  = {24},
  number  = {5},
  pages   = {625--633},
  year    = {2006},
  doi     = {10.1017/S0263574706002712},
  publisher = {Cambridge University Press}
}
```
</details>
