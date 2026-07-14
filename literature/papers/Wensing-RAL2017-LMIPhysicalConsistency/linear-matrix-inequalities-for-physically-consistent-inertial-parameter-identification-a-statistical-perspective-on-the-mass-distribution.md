---
Title: Linear Matrix Inequalities for Physically-Consistent Inertial Parameter Identification: A Statistical Perspective on the Mass Distribution
Authors:
  - Wensing, Patrick M.
  - Kim, Sangbae
  - Slotine, Jean-Jacques E.
Year: 2017
Venue: RAL
Tags:
  - "inertial-parameter-identification"
  - "physical-consistency"
  - "linear-matrix-inequalities"
  - "semidefinite-programming"
  - "legged-robots"
PDF: "[[papers/Wensing-RAL2017-LMIPhysicalConsistency/main.pdf|📃]]"
Import Date: "2026-07-12"
Read Date: 2026-07-12
Executive Summary: 剛体の慣性パラメータ同定において、物理的に実現可能（density realizable）な質量分布であるという制約を、非凸多様体上の最適化を要さずLinear Matrix Inequality（LMI）として定式化した。鍵となるのは、回転慣性テンソルではなく質量分布の密度加重共分散（擬慣性行列 J(π)）に基づいて物理的整合性を表現する点で、これにより凸最適化（半正定値計画）による大域最適解が保証される。楕円体境界への拡張制約も導出し、MIT Cheetah 3の脚同定実験で、制約が厳しいほど少サンプルで検証誤差が下がり過学習が減ることを確認した。
Citekey: Wensing-RAL2017-LMIPhysicalConsistency
BibTeX Key: wensing2017linear
DOI: 10.1109/LRA.2017.2729659
Relevance: 4
Repository: "none"
Category: note
Template Version: v2.3
---

## Executive Summary
剛体の慣性パラメータ同定において、物理的に実現可能（density realizable）な質量分布であるという制約を、非凸多様体上の最適化を要さずLinear Matrix Inequality（LMI）として定式化した。鍵となるのは、回転慣性テンソルではなく質量分布の密度加重共分散（擬慣性行列 J(π)）に基づいて物理的整合性を表現する点で、これにより凸最適化（半正定値計画）による大域最適解が保証される。楕円体境界への拡張制約も導出し、MIT Cheetah 3の脚同定実験で、制約が厳しいほど少サンプルで検証誤差が下がり過学習が減ることを確認した。

---
## Summary

### この論文が答えた問い、あるいは解決した課題は何か？
剛体の10個の慣性パラメータ（質量・第一質量モーメント・回転慣性テンソル）を同定する際、結果が物理的に実現可能な質量分布に対応するという「physical consistency」制約を、凸最適化の枠組みに乗る形（LMI）でどう表現するかという問題を扱った。従来、Sousa & Cortesão [13] の運動エネルギー正定値制約（physical semi-consistency）は凸だが真の物理的整合性を保証せず、Traversaro et al. [14] の完全な physical consistency 条件（三角不等式を含む）は非凸多様体（SO(3)上の回転行列パラメータ化）を要し大域最適性を保証できなかった（§I Introduction, §III）。本論文はこの両立不可能に見えたトレードオフ（凸性 vs 完全な物理的整合性）を解消する。

### 提案手法のアプローチと、その根幹をなす要素は何か？
回転慣性テンソルの三角不等式条件（Traversaro et al. [14] が示した physical consistency の必要十分条件）を、剛体の質量分布に対する「密度加重共分散」ΣC の半正定値性という等価な条件に書き換え、これを擬慣性行列 J(π)（4×4、質量・第一質量モーメント・二次モーメント行列 Σ から構成）の半正定値制約 J(π) ⪰ 0 として表現した（Theorem 3）。この J(π) はπに関して線形であるため、J(π) ⪰ 0 は真のLMIとなり、非凸な多様体パラメータ化（回転行列R、主慣性J、CoM位置cによる分解）を経由せずに済む。
- 密度加重共分散 ΣC = ∫ xc xc^T ρ(x) dx の導入（確率統計における共分散との数学的対応）とその半正定値性が三角不等式と等価であること（Proposition 1）
- 擬慣性行列 J(π) = [[Σ, h], [h^T, m]] がπに関して線形であること、およびSchur補元によりJ(π) ⪰ 0 が ΣC ⪰ 0 と等価であること（Theorem 3）
- 古典的モーメント問題（classical problem of moments, Fialkow & Nie 2010）の結果を援用し、境界楕円体S上での密度実現可能性を J(π) ⪰ 0 かつ Tr(J(π)Q) ≥ 0 という1次元線形不等式1本の追加で表現できること（Theorem 4）

### 特に参考とした既存研究と、それらと比した提案手法の新規性は何か？
主な比較対象は Sousa & Cortesão [13]（運動エネルギー正定値化によるLMI、physical semi-consistencyのみ保証）と Traversaro et al. [14]（三角不等式を含む完全な physical consistency を非凸多様体最適化で扱う）である（Table I による特徴比較）。新規性は、[14] の非凸な三角不等式条件を、質量分布の共分散という統計的解釈を経由することで凸なLMI（J(π) ⪰ 0）に変換した点にある。これにより Ayusawa et al. [23] が離散点質量近似で physical consistency と凸性を両立させていたのに対し、本論文は離散近似なしに（連続的な質量分布に対して）両立させた（Table Iの "Discrete Approx." 列における差異）。また、楕円体境界制約についても Jovic et al. [12] のCoM境界ボックス制約（一次モーメントのみ）を拡張し、二次モーメントまで含めた密度実現可能性条件（Theorem 4）を古典的モーメント問題から新たに導出した点が第二の貢献とされる。

### どのように訓練・最適化したのか？
- **損失関数 / 最適化目的**: MIT Cheetah 3脚（リンク3体+ロータ3体、計6体）の同定問題として、正則化付き非線形最小二乗損失
  min_{π,Bc,B} (1/ns) Σ_m ||Y^(m)π + Bν^(m) + Bc sign(ν^(m)) − τ^(m)||^2 + wπ||π − π̂||^2
  （式(25)）を、各体iについて Ci(πi) ⪰ 0（CoM境界制約）、J(πi) ≻ 0、Tr(J(πi)Qi) ≥ 0（楕円体密度実現可能性、Theorem 4）というLMI制約下でSDPとして解いた。摩擦項として粘性・クーロン摩擦係数の対角行列 B, Bc（式(24)）を同時推定。正則化重みは wπ = 10^-6、正則化の基準パラメータ π̂ はCADから取得。境界楕円体パラメータ Ci, Qi もCAD形状から設定（§VI）。
- **データセット**: MIT Cheetah 3の1脚（3自由度：ab/ad, hip, knee）を用いたリーグスイング実験。脚をCartesianインピーダンス制御下に置き、足先を仮想楕円体シェル上で球面角(φ, θ)により振動させて励起（Aφ=12 rad/s, Aθ=3.4 rad/s, ωφ=1.63 rad/s, ωθ=0.265 rad/s）。1 kHzサンプリング。同定には10,000サンプルを使用し、MOSEK（MATLAB）で解いた（2011 Intel Core i5 MacBook Proで1.67秒、大域最適解に到達）。検証には学習に使っていない別の10,000サンプルを使用（§VI）。

### どのように検証したか？指標と結果は？
検証は式(24)の逆算によるモータトルクのRMS誤差で行った。学習に使用していない次の10,000サンプルに対し、ab/ad, hip, kneeの各関節でRMS誤差はそれぞれ1.48, 1.69, 1.16 Nm（全体で1.46 Nm）であった。同定されたクーロン摩擦は Bc = diag(3.12, 1.25, 0.95) Nm（Fig. 4）。また、学習サンプル数 ns を変化させ、(1) 制約なし、(2) physical semi-consistency（[13]のLMI）、(3) 楕円体上のdensity realizability（本論文のTheorem 4によるLMI）の3条件で検証誤差を比較したところ（Fig. 6）、制約が厳しいほど少ないサンプル数でも検証誤差が低く抑えられ、過学習が軽減されることが定性的に示された（具体的な誤差値の表は本文になし、図示のみ）。全ての物理的整合性制約が満たされていることも確認された（§VI）。

### 検証結果に基づいた議論、明らかになった課題はあるか？
(§VII Conclusions より) 著者らは、本研究の統計的視点は数学的な類推であり、剛体自体に確率的性質があるわけではないと明示的に断っている。その上で、不確実な計測から動力学モデルを推定する（ノイズ・バイアスへの頑健性を扱う）確率的推論の枠組みに、本論文の厳密な物理的整合性制約を組み込むことを「興味深い次のステップ（an interesting next step）」として挙げている（測定ノイズの定量的な扱いは本論文のスコープ外であり、今後の課題として明示）。
(§VI 実験セクション より) ab/ad関節で17.5秒付近の推定精度が低い箇所があるが、著者はこれを「関節が静止している時間帯でありクーロン摩擦トルクの符号が信頼して予測できないため」と説明しており、モデルの欠陥というより符号不定性に起因する既知の挙動として位置づけている。
(Remark 7 より) 実験は3自由度の脚1本に限定されているが、著者は提案するLMI（4×4）は高自由度系への拡張でも計算負荷の増加が小さいと主張し、剛体数が増えるほど（Remark 6の議論から）タイトな制約の恩恵は指数的に増すと予想している。ただしこれは実験的には未検証であり、著者自身も "expected" という推測的表現を用いている。

---
## 自身の研究との関連
本論文はリアルタイム励起・オンライン慣性パラメータ同定という本プロジェクトの主題そのものではなく、その下流にある「同定問題の制約付き定式化」を扱う基礎理論として位置づけられる。[[papers/Kubus-IROS2007-On-line_Rigid_Object/on-line-rigid-object-recognition-and-pose-estimation-based-on-inertial-parameters|Kubus+ 2007]]・[[papers/Kubus-IROS2008-Recursive_Total_Least-Squares/on-line-estimation-of-inertial-parameters-using-a-recursive-total-least-squares-approach|Kubus+ 2008]]（オンライン同定・Recursive Total Least-Squares）や[[papers/Nadeau-ICRA2022-FastObjectInertial/fast-object-inertial-parameter-identification-for-collaborative-robots|Nadeau+ 2022]]・[[papers/Nadeau-ICRA2023-SumOfItsParts/the-sum-of-its-parts-visual-part-segmentation-for-inertial-parameter-identification-of-manipulated-objects|Nadeau+ 2023]]（協調ロボット向け高速同定）が主に「速く・頑健に」推定するアルゴリズム側に焦点を当てているのに対し、本論文は「推定結果が物理的に妥当か」という制約側の問題を扱っており、両者は相補的である。具体的には、オンライン同定の各更新ステップに物理的整合性のLMI制約（J(π) ⪰ 0）を組み込むことで、ノイズの多い区間での発散やありえないパラメータへの収束を防げる可能性があり、Fig. 6で示された「制約が厳しいほど少サンプルで収束する」という知見はオンライン同定の収束速度向上に直結しうる。

また、[[papers/Swevers-TRA1997-OptimalExcitation/optimal-robot-excitation-and-identification|Swevers+ 1997]]の最適励起軌道設計とは異なるレイヤーの話であり、本論文の楕円体境界制約（Theorem 4, Corollary 2）はCAD由来の形状事前知識を持つ場合に励起軌道設計と組み合わせて使うことが考えられる。[[papers/Markovsky-SigProc2007-Overview_Total_Least-Squares/overview-of-total-least-squares-methods|Markovsky+ 2007]]のTotal Least-Squares的な観測ノイズへの頑健性の議論とは対照的に、本論文は測定ノイズを明示的にモデル化しておらず（VII節で著者自身が確率的推論との統合を将来課題としている）、この点で[[papers/Hu-arXiv2025-Adaptive_Experiment_Design/adaptive-experiment-design-for-nonlinear-system-identification-with-operational-constraints|Hu+ 2025]]や[[papers/Zhang-arXiv2025-ProvablySafe/provably-safe-online-system-identification|Zhang+ 2025]]が扱うような適応的実験設計・安全性の議論とは独立した貢献である。

擬慣性行列 J(π) による4×4定式化は、6×6の空間慣性行列I(π)によるLMIより計算コストが低いとされており（§IV-C）、リアルタイム性を重視する本プロジェクトの実装（オンボードでの高頻度パラメータ更新）において、制約付きSDPを毎ステップ解く場合の計算量削減に直接活用できる可能性がある。

---
## 追加議論

---
## BibTex
<details>
<summary> Click to show/noshow the BibTex data </summary>

```bibtex
@article{wensing2017linear,
  title={Linear Matrix Inequalities for Physically-Consistent Inertial Parameter Identification: A Statistical Perspective on the Mass Distribution},
  author={Wensing, Patrick M. and Kim, Sangbae and Slotine, Jean-Jacques E.},
  journal={IEEE Robotics and Automation Letters},
  year={2017},
  doi={10.1109/LRA.2017.2729659}
}
```
</details>
