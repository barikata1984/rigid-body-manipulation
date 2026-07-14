---
Title: "Robot Model Identification and Learning: A Modern Perspective"
Authors:
  - Lee, Taeyoon
  - Kwon, Jaewoon
  - Wensing, Patrick M.
  - Park, Frank C.
Year: 2024
Venue: AR
Tags:
  - "system-identification"
  - "model-learning"
  - "rigid-body-dynamics"
  - "geometric-methods"
  - "inductive-bias"
  - "survey"
PDF: "[[papers/Lee-AR2024-Robot_Model_Identification/main.pdf|📃]]"
Import Date: "2026-07-15"
Read Date: 2026-07-15
Executive Summary: ロボットの動力学モデルを実測データから同定・学習する問題を、古典的な力学ベース同定と最近の機械学習ベース手法の両方を統一的に俯瞰する Annual Review 論文。慣性パラメータ空間の幾何学的構造（アフィン不変 Riemann 計量・pseudoinertia の LMI 表現）を活用したロバスト同定と、物理法則を帰納的バイアスとしてデータ駆動モデルに埋め込むアプローチを二本柱に据え、構造誤差・データ不完全性・実用的識別可能性という 3 つの実務的困難に対する近年の進展を整理する。
Citekey: Lee-AR2024-Robot_Model_Identification
BibTeX Key: lee2024robot
DOI: 10.1146/annurev-control-061523-102310
Relevance: 5
Repository: none
Category: note
Template Version: v2.3
---

## Executive Summary
ロボットの動力学モデルを実測データから同定・学習する問題を、古典的な力学ベース同定と最近の機械学習ベース手法の両方を統一的に俯瞰する Annual Review 論文。慣性パラメータ空間の幾何学的構造（アフィン不変 Riemann 計量・pseudoinertia の LMI 表現）を活用したロバスト同定と、物理法則を帰納的バイアスとしてデータ駆動モデルに埋め込むアプローチを二本柱に据え、構造誤差・データ不完全性・実用的識別可能性という 3 つの実務的困難に対する近年の進展を整理する。

---
## Summary

### この論文が答えた問い、あるいは解決した課題は何か？
複雑化・安全重要化するロボットタスクに耐える動力学モデルをどう構築するかという問いに対し、system identification / model learning を横断的に整理し、力学ベースの構造と機械学習の柔軟性をどう統合するかという方法論的な地図を提供することを目的とする。特に (a) 剛体力学パラメータの構造的・実用的識別可能性、(b) 有限かつ不完全なデータ下での推定誤差、(c) 純粋な黒箱化を避けつつ既存物理モデルでは捕えきれない現象（摩擦の非線形性・接触・柔軟性）にどう対処するか、という 3 つの課題に焦点を当てる。

### 提案手法のアプローチと、その根幹をなす要素は何か？
本論文はサーベイであり単一の提案手法を持たない。代わりに、非線形確率状態空間モデル $x_{t+1}=f_\theta(x_t,u_t,\omega_t)$ に対する MLE 定式化 (§2) を出発点として、ロボット同定の議論を以下の 3 軸に整理する。

- **誤差基準の 2 分法**：equation error（1 ステップ予測誤差、Eq. 4-5）と simulation error（軌道全体の積分誤差、Eq. 6）。前者はパラメータに線形で解析的、後者は非凸だが受動制御・強化学習の long-horizon 用途に整合。
- **誤差源の 3 分類**：structural error（$f\notin\mathcal{M}_f$）、random error（ノイズ）、incompleteness（情報行列 $\mathcal{I}(\theta)$ の退化）。これらは Fisher 情報行列と Cramér-Rao 下界で定量化され、Data Collection Problem（最適励起）へと接続する。
- **幾何学的枠組み**（§4）：質量慣性ベクトル $\phi\in\mathbb{R}^{10}$ の物理的整合性を pseudoinertia $P(\phi)\succ 0$ の LMI で表現し、アフィン不変 Riemann 計量 $d(\phi_1,\phi_2)^2=\tfrac12\mathrm{tr}(\mathrm{Log}((P(\phi_1)^{-1}P(\phi_2))^2))$（Eq. 14）とその Bregman 発散近似（Eq. 15-16）で座標・単位不変な距離を定義する。この計量を情報行列に適用すると、幾何 A-最適性基準 $\sigma(\mathcal{I}(\bar\theta,u_{1:N})\cdot H_0^{-1})$（Eq. 18）が得られ、軽いリンクをより励起するべきという物理的に意味のある励起設計へ通じる。
- **構造バイアス低減**（§5）：運動学・動力学同時同定（kinodynamic ID, Eq. 20; Kwon et al. の指摘に基づく）、discrepancy modeling（残差を NN や GP で吸収）、そして inductive bias（エネルギー保存 Hamiltonian/Lagrangian NN、接触の非貫入、topology のグラフ表現、SE(3)-equivariance）を用いた physics-informed 学習。

### 特に参考とした既存研究と、それらと比した提案手法の新規性は何か？
古典系として Atkeson et al. の inverse-dynamics 線形回帰、[[papers/Swevers-TRA1997-OptimalExcitation/optimal-robot-excitation-and-identification|[5]]] の最適励起、Sousa & Cortesão の SDP による物理整合性、Traversaro et al. の必要十分条件、[[papers/Wensing-RAL2017-LMIPhysicalConsistency/linear-matrix-inequalities-for-physically-consistent-inertial-parameter-identification-a-statistical-perspective-on-the-mass-distribution|[34]]] の完全 LMI 定式化を土台とする。幾何学的視点は Li & Slotine、Yoshida ら、Lee & Park の Riemann 計量、そして Lee et al. の幾何的最適性基準（本論文の主著者らの一連の研究）に依拠する。既存の総説（Reference 27 の Wu ら、Reference 14 の soft robot）に対する新規性は、(i) 物理パラメータの幾何構造を軸に整理した点、(ii) MLE 統一視点で古典同定と ML を橋渡しした点、(iii) 帰納バイアス経由での physics-informed 学習を体系的に位置づけた点にある。

### どのように訓練・最適化したのか？
N/A: 本論文はサーベイであり、独自の訓練・最適化は行わない。ただし総論として以下の目的関数群が紹介される：

- **損失関数 / 最適化目的**：MLE の $L(\theta)=-\sum_t \log p(\omega_t)$（equation error, Eq. 5）、シミュレーション誤差の $L(\theta)=\sum_i\|\hat x_i-x_s(i,\theta)\|^2$（Eq. 6）、幾何正則化付き MAP 定式 $\min_\psi L(B\psi)+\gamma\, d(\psi,\psi_0)^2$（Eq. 19）、kinodynamic 同時同定の $\min_{\theta_{\mathrm{dyn}},\theta_{\mathrm{kin}}} L_{\mathrm{dyn}}+\alpha L_{\mathrm{kin}}$（Eq. 20、重み $\alpha=\sigma_{\mathrm{dyn}}/\sigma_{\mathrm{kin}}$）。物理整合性制約として LMI $P(\phi)\succ 0$（Eq. 13）を含む凸半正定値計画が推奨される。
- **データセット**：N/A: サーベイのため独自データセットなし。

### どのように検証したか？指標と結果は？
N/A: 独自の実験検証はない。ただし引用研究の定量的知見として、Kwon et al. (Reference 9) の「粗い運動学パラメータ誤差が動力学同定の誤差を数倍〜数十倍に増幅する」という報告を §5.1 で強調し、kinodynamic 同時同定の動機として引用する。また Lee et al. (Reference 59, §4.4) の数値研究として「構造的に識別可能なパラメータのうち実用的に識別可能な割合は特に高次元系（humanoid 等）で著しく限定される」との知見を紹介する。

### 検証結果に基づいた議論、明らかになった課題はあるか？
著者は §6 Conclusion および Summary Points で以下の展望・限界を明示的に述べる：

- (§6 Conclusion) 「system identification は目的そのものではなく、targeted robotics applications に向けた道具である」と位置づけ、制御・強化学習コミュニティが進めつつある **task-aligned system identification methods** —— 制御目的・タスク目的と直接整合した同定 —— を future prospect として挙げる（References 26, 89, 90 を挙げるのみで本総説では extensively には扱わない、と明言）。
- (§2.3.2 The Data Collection Problem sidebar) 情報行列に基づく最適励起は「chicken-and-egg 問題」であり、公称パラメータなしには最適化できない点、および humanoid など安全臨界・underactuated 系では固定基座マニピュレータのように自由に励起できない点が根本的困難として挙げられる。
- (§5.3.4 Topology and graphs) グラフ構造化データ駆動モデルの任意のグラフ組合せに対する zero-shot 汎化性能は open problem として残される。
- (§5.3 Putting Physics in Data-Driven Model Learning) discrepancy modeling は元の物理モデルの多くの system characteristics を必ずしも保存しないため、広範な test 分布での慎重な検証が essential と述べる。
- (Summary Points 3) 帰納バイアスを保った data-driven augmentation は「柔軟性を高めつつ物理的整合性を保つ」方向の課題として今後の発展余地があると位置づけられる。

---
## 自身の研究との関連
本プロジェクト（6DoF 直列マニピュレータ + hammer の 10 慣性パラメータ同定）に対して、本論文は positioning の材料として重要な役割を持つ。

第一に、excitation design は本論文では §4.4 の Geometric Information Measure でのみ扱われ、Lee & Park のアフィン不変 Riemann 計量から導かれる $H_0=(BG_0^{-1}B^{\mathrm{T}})^{-1}$ 正規化情報行列（Eq. 18, Figure 3）による座標・単位不変な A-最適性が中心的に紹介される。本プロジェクトの発見である「task-required かつ非最適化 base drift」に起因する $T^2/(\text{turn 数})$ スケーリングと cond 悪化は、この幾何的枠組み内では扱われておらず、task motion と ID motion の同時実行という設定自体が本文の議論範囲外にある。

第二に、§6 Conclusion が明示する future prospect **task-aligned system identification** はまさに本プロジェクトの立ち位置と重なる。ここで著者らが引用する References 26, 89, 90（control-aligned / task-oriented identification の系譜）は要確認だが、Annual Review 級の総説が「今後の課題」として挙げている以上、本プロジェクトは「open problem に踏み込む研究」として positioning できる。

第三に、慣性 10 パラメータの物理整合性を LMI $P(\phi)\succ 0$（Eq. 13、Wensing et al. 2017）と幾何正則化 Eq. 19 で扱う枠組みは、本プロジェクトの推定器を強化する道具として直接利用可能。MuJoCo が pseudoinertia 制約の built-in validation を持つ（§4.2）という記述は実装面でも有用。

第四に、kinodynamic ID（§5.1, Eq. 20）は本プロジェクトの直接対象ではないが、Kwon et al. の「運動学誤差が動力学同定を数倍増幅する」観察は、hammer 取り付け位置の CAD 値誤差が慣性推定に与える影響として実務的に無視できない検討材料となる。

判定：本プロジェクトの中核発見（task-required 非最適化 motion の drift による cond 悪化）は本総説で言及されていない。ただし conclusion で future prospect として明記されているため、本論文は本プロジェクトの gap 主張を裏付ける権威的引用元となる。

---
## 追加議論


---
## BibTex
<details>
<summary> Click to show/noshow the BibTex data </summary>

```bibtex
@article{lee2024robot,
  author  = {Lee, Taeyoon and Kwon, Jaewoon and Wensing, Patrick M. and Park, Frank C.},
  title   = {Robot Model Identification and Learning: A Modern Perspective},
  journal = {Annual Review of Control, Robotics, and Autonomous Systems},
  volume  = {7},
  pages   = {311--334},
  year    = {2024},
  doi     = {10.1146/annurev-control-061523-102310}
}
```

</details>
