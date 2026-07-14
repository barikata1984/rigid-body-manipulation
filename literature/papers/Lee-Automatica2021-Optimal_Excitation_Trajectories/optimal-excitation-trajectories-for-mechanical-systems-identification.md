---
Title: Optimal excitation trajectories for mechanical systems identification
Authors:
  - Lee, Taeyoon
  - Lee, Bryan D.
  - Park, Frank C.
Year: 2021
Venue: Automatica
Tags:
  - "excitation-trajectory"
  - "system-identification"
  - "coordinate-invariant"
  - "riemannian-geometry"
  - "base-parameters"
  - "optimal-experiment-design"
PDF: "[[papers/Lee-Automatica2021-Optimal_Excitation_Trajectories/main.pdf|📃]]"
Import Date: "2026-07-15"
Read Date: 2026-07-15
Executive Summary: 多体機械系の慣性同定における励起軌道最適化は, 従来 A/D/E/条件数などの Fisher 情報量基準で行われてきたが, これらは D-最適性以外は座標系・基底パラメータ表現・物理単位に依存し不整合を生む. 本論文はベースパラメータ空間に affine-invariant Riemannian 計量 $H_0 = (B G_0^{-1} B^T)^{-1}$ を導入し, 正規化情報行列 $A_B^T \Sigma^{-1} A_B \cdot H_0^{-1}$ の固有値を対称関数で評価することで A/D/E/条件数のすべてを座標不変化する統一枠組みを提示する. 正規化共分散の固有分解により有効同定可能な縮約パラメータ集合も抽出でき, 29-dof Atlas V5 で 204 個中 70 個を同定できることを示す.
Citekey: Lee-Automatica2021-Optimal_Excitation_Trajectories
BibTeX Key: lee2021optimal
DOI: 10.1016/j.automatica.2021.109773
Relevance: 5
Repository: none
Category: note
Template Version: v2.3
---

## Executive Summary
多体機械系の慣性同定における励起軌道最適化は, 従来 A/D/E/条件数などの Fisher 情報量基準で行われてきたが, これらは D-最適性以外は座標系・基底パラメータ表現・物理単位に依存し不整合を生む. 本論文はベースパラメータ空間に affine-invariant Riemannian 計量 $H_0 = (B G_0^{-1} B^T)^{-1}$ を導入し, 正規化情報行列 $A_B^T \Sigma^{-1} A_B \cdot H_0^{-1}$ の固有値を対称関数で評価することで A/D/E/条件数のすべてを座標不変化する統一枠組みを提示する. 正規化共分散の固有分解により有効同定可能な縮約パラメータ集合も抽出でき, 29-dof Atlas V5 で 204 個中 70 個を同定できることを示す.

---
## Summary

### この論文が答えた問い、あるいは解決した課題は何か？
多体機械系の慣性パラメータ同定における励起軌道生成には, A-最適性・E-最適性・条件数最小化などの Fisher 情報行列 $M_B = A_B^T \Sigma^{-1} A_B$ に基づく基準が広く用いられてきた.
しかしこれらの基準は, ボディ固定座標系の位置・向き, ベースパラメータ基底 $B$ の選び方, 物理単位 (SI vs inch-pound 等) に強く依存し, 同じ機械系でも座標選択によって最適軌道と同定精度が大きく変わる (Fig. 2, Table 1 で 1-dof 例で条件数の等高線が座標によって形状が大きく変わることを示す).
[[papers/Swevers-TRA1997-OptimalExcitation/optimal-robot-excitation-and-identification|Swevers+ 1997]] を含む古典的アプローチはこの座標依存性を明示的に扱っておらず, Presse & Gautier (1993) の diagonal 正規化 $M_B \cdot [\mathrm{diag}(\Phi_B^0)]^{-2}$ もスケール不変性のみでフレーム不変性は持たない.
本論文は「座標選択に依らず物理的に意味のある励起軌道最適化基準をどう定式化するか」という問いに答える.
併せて高次元系 (例: Atlas V5 humanoid) で全ベースパラメータを励起する軌道が物理制約 (バランス等) 下で存在しない場合の, 縮約された有効同定パラメータ集合の抽出も扱う.

### 提案手法のアプローチと、その根幹をなす要素は何か？
ベースパラメータ空間 $\mathcal{N} = \{\Phi_B = B\Phi\}$ に affine-invariant Riemannian 計量を pullback で導入し, これを用いて Fisher 情報行列を正規化することで座標不変な励起基準を構成する.
正規化された情報行列の固有値の対称関数として A/D/E/条件数を再定義し, B-スプライン軌道の制御点を SQP で最適化する.
根幹要素:

- **Affine-invariant metric on full inertial space (§3.2)**: Lee & Park (2018) の pseudo-inertia 表現 $P(\phi) \in \mathcal{P}(4)$ 上の計量 $\mathrm{d}s^2 = \frac{1}{2}\mathrm{tr}((P^{-1}\mathrm{d}P)^2)$. $GL(4)$ 群作用 $G * P = GPG^T$ で不変であり, フレーム変換とスケール変換をすべて吸収する.
- **Pushforward metric $H_0$ on base parameter space (Eq. 42)**: $H_0 \triangleq (B G_0^{-1} B^T)^{-1} \in \mathbb{R}^{N_B \times N_B}$. $G_0 = G(\Phi_0)$ は nominal パラメータ (CAD 値等) で評価した block diagonal 計量.
- **Geometric normalized criterion (Prop 5)**: $J = f(\lambda(A_B^T \Sigma^{-1} A_B \cdot H_0^{-1}))$. $A = A_B B$ とすれば非零固有値は基底 $B$ の選び方に不変. これで A/D/E/条件数がすべて座標不変になる.
- **Recursive analytic gradient (§4)**: Ayusawa+ 2017 の chain rule に $H_0^{-1}$ を挟む修正 (Eq. 64, 65) で $\partial J / \partial p$ を解析的に計算し, fmincon SQP で B-スプライン制御点を最適化.
- **Reduced identification via eigendecomposition (§5)**: 正規化共分散 $C = H_0^{-1/2} (M_B)^g H_0^{-1/2}$ の固有分解から, 推定分散が閾値 $\epsilon_\lambda$ 以下の方向のみを残す線形結合 $\Phi_B^* = \Phi_B^0 + V_- w^*$ を得る. これで励起不十分な方向は事前値に固定する.

### 特に参考とした既存研究と、それらと比した提案手法の新規性は何か？
最も直接の対比対象は Presse & Gautier (1993) の diagonal 正規化 $M_B \cdot [\mathrm{diag}(\Phi_B^0)]^{-2}$ で, スケール不変性は達成するがフレーム不変性を持たず, かつ nominal 値が 0 になる座標選択 (例: 質量中心を原点に取ると 1 次モーメントが 0) で正規化が破綻する (§2.2 の 1-dof 反例と本文 §6.1 SCARA 実験の Table 1 で確認).
[[papers/Swevers-TRA1997-OptimalExcitation/optimal-robot-excitation-and-identification|Swevers+ 1997]], Armstrong 1989, Gautier & Khalil 1992 らは条件数・D-最適性を扱うが座標不変性を明示的に議論していない.
D-最適性のみは対数行列式の性質から座標変換に対して定数シフトで済み既に不変だが, A/E/条件数は不変でない (§2.2 Eq. 28-29).
本論文の新規性は, ベースパラメータ空間の Riemannian 構造から Fisher 情報行列の pullback 解釈を与え, 任意の対称関数基準を統一的に座標不変化した点にある.
また [[papers/Wensing-RAL2017-LMIPhysicalConsistency/linear-matrix-inequalities-for-physically-consistent-inertial-parameter-identification-a-statistical-perspective-on-the-mass-distribution|Wensing+ 2018]] の pseudo-inertia 物理整合性表現と Lee & Park (2018), Lee, Wensing, & Park (2019) の Riemannian 同定枠組みを励起軌道設計側に拡張した点も新しい.

### どのように訓練・最適化したのか？
- **損失関数 / 最適化目的**: 座標不変正規化基準 $J = f(\lambda(A_B^T \Sigma^{-1} A_B \cdot H_0^{-1}))$ を最小化 (Eq. 60). $f$ は A/E/条件数のいずれか. SCARA 実験では条件数基準, AMBIDEX と Atlas 実験では E-最適性基準を採用. 軌道は $q(t) = \sum_j p_j B_j(t)$ の 5 次 B-スプライン (制御点数 $n_f = 10$, 期間 $T = 10$ 秒, boundary 条件で始終端の速度・加速度・躍度を 0 固定). 制御点 $P = \{p_{ij}\}$ を MATLAB fmincon の SQP で最適化し, gradient は Ayusawa+ 2017 由来の recursive analytic gradient に $H_0^{-1}$ を組み込んで解析計算 (Eq. 64, 65). 拘束条件はジョイント角・角速度・トルク限界, humanoid では単脚 balance 拘束.
- **データセット**: N/A (合成観測・実機実験のみ, 学習データセットではない). 実験構成: (i) 2-dof SCARA 数値実験, ジョイントトルク観測にガウスノイズ $\sigma = 0.03$ Nm を付加. (ii) 4-dof AMBIDEX ケーブル駆動マニピュレータの実機実験, 摩擦モデル含む. (iii) 29-dof Atlas V5 humanoid 数値実験, 支持足の 6 軸力/モーメント観測にガウスノイズ $\sigma = 0.5$ N, $1.5$ Nm を付加, 20 個のランダムに生成した動的にフィージブルな軌道 (期間 3 秒, サンプリング 300 Hz) から 10 個を同定, 残り 10 個を検証に使用.

### どのように検証したか？指標と結果は？
検証プロトコルは 3 つの実験系で構成される.
SCARA (§6.1, Table 1): 4 種類のボディ固定座標フレームで, 提案手法 (Invariant) と Presse & Gautier diagonal 正規化と正規化なし条件数基準を比較. 提案手法は 4 フレームで一意 (Invariant 列) の RMS 誤差を返し, 特に 1 次モーメント $p^y$ と長さ $r^z$ で他手法より安定. Diagonal 正規化はフレーム {1}, {3} で prior 値が小さくなる方向に過剰正規化し性能劣化.
AMBIDEX (§6.2, Table 2): 座標不変 E-最適性で得た軌道を実機 6 回試行し, motor 1,2 で 0.0062 Nm, motor 3,4 で 0.0059 Nm の RMS トルク予測誤差を達成 (座標依存 diagonal 正規化の最良値 0.0071 と 0.0066, 非正規化の最悪値 0.0551 と 0.0667 と比較). Fig. 5 で reduced identification のサイズ $r$ を 1-30 で変化させると, 提案 metric ($H_0$) は縮約後の予測誤差が最小 0.010 Nm まで低下 (フル同定比 80% 減), Euclidean metric に置換すると劣化.
Atlas V5 humanoid (§6.3, Table 3, 4): 204 個のベースパラメータ中 70 個が閾値 $\epsilon_\lambda = 10^{-2}$ で同定可能と判定 (Fig. 8). 検証軌道での GR force 予測誤差は prior 61.768 N → full ID 0.017 N → reduced ID 0.009 N. ジョイントトルク誤差も 6 部位すべてで reduced ID が full ID より小さい (例: hip で 0.131 → 0.019 Nm).

### 検証結果に基づいた議論、明らかになった課題はあるか？
- (§7 Conclusion) 提案枠組みは線形観測モデル (慣性パラメータが力学式に線形に現れる仮定) に限定されており, 非線形観測モデルへの拡張は future work.
- (§7 Conclusion) 慣性以外の機械的パラメータ (stiffness/impedance, friction) についても同様の geometric 扱いが望ましいが本論文では扱っていない. スケール不変性は成立しそうだが, matrix-valued な spatial stiffness/impedance/damping は affine 変換下でより複雑な挙動を示す.
- (Remark 3, §3.2) 提案 metric $H_0$ の定義には nominal パラメータ $\Phi_0$ (CAD 値等) が必要で, 真値からの乖離が大きい場合の頑健性は本文中の実験では確認されているが定量的な感度解析は与えられていない (Fig. 5 では point mass 摂動による粗い prior でも reduced ID が改善することを示すのみ).
- (Remark 7, §3.2) 提示した $H_0$ は一つの選択であり, prediction error 基準等ではより自然な metric 選択が存在しうる点を著者自身が指摘 (本文中の前提として記載).

---
## 自身の研究との関連
本プロジェクトは 6-DoF 逐次型マニピュレータ + hammer の 10 慣性パラメータ同定を扱っており, task-required な非最適化 base drift $q(t) = q_\mathrm{base} + q_\mathrm{exc}$ が回帰行列の条件数を桁で悪化させ, $T^2/(\text{turn 数})$ でスケールする経験則を得ている.
本論文の $H_0$ 正規化は **パラメータ空間側の座標不変性** を扱うのに対し, 本プロジェクトの発見は **時間領域の軌道構造 (drift vs pure excitation)** の問題であり両者は直交する.
実際, D-最適性は元々座標不変なので $H_0$ 正規化しても値は定数シフトのみで, 本プロジェクトで D-最適軌道でも条件数が 24 で頭打ちする現象は $H_0$ では吸収できない.
一方, 本論文の枠組みは相補的に活用できる可能性がある:

- **正規化基準の採用**: hammer 同定でも body-fixed frame の選び方 (hammer 頭の原点 vs 柄の原点) や慣性テンソルの単位選択で条件数が変わるはずで, 提案 $H_0$ 正規化を組み込めばこの座標依存性を除去できる.
- **Reduced identification (§5)**: 本プロジェクトの base drift 起因の ill-conditioning は「そもそも励起されない方向が存在する」構造と解釈でき, 正規化共分散の固有分解で有効同定可能な線形結合のみを抽出する §5 の手続きが直接適用できる可能性がある.
- **B-スプライン + 端点固定 boundary 条件 (§4)**: 本論文は $\dot{q}(0)=\dot{q}(T)=\ddot{q}(0)=\ddot{q}(T)=0$ の endpoint-pinned 軌道を用いており, drift 成分を持たない. 本プロジェクトの task drift 問題を扱っていない点で対象問題は異なるが, drift を許容する拡張は独自研究の余地となる.

判定: 提案 metric $H_0$ は座標不変化の道具として本プロジェクトの回帰行列条件付け解析に組み込む価値があるが, base drift 起因の cond 悪化そのものは吸収しない相補的貢献.

---
## 追加議論

---
## BibTex
<details>
<summary> Click to show/noshow the BibTex data </summary>
```bibtex
@article{lee2021optimal,
  title   = {Optimal excitation trajectories for mechanical systems identification},
  author  = {Lee, Taeyoon and Lee, Bryan D. and Park, Frank C.},
  journal = {Automatica},
  volume  = {131},
  pages   = {109773},
  year    = {2021},
  doi     = {10.1016/j.automatica.2021.109773},
  publisher = {Elsevier}
}
```
</details>
