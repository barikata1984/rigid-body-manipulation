---
Title: The Sum of Its Parts: Visual Part Segmentation for Inertial Parameter Identification of Manipulated Objects
Authors:
  - Nadeau, Philippe
  - Giamou, Matthew
  - Kelly, Jonathan
Year: 2023
Venue: ICRA
Tags:
  - "inertial-parameter-identification"
  - "part-segmentation"
  - "point-cloud"
  - "collaborative-robots"
  - "force-torque-sensing"
  - "stop-and-go-motion"
PDF: "[[papers/Nadeau-ICRA2023-SumOfItsParts/main.pdf|📃]]"
Import Date: "2026-07-11"
Read Date: 2026-07-11
Executive Summary: 高速・大振幅な励起なしでは信号対雑音比が不足するという慣性パラメータ同定の課題に対し、物体形状をパーツごとの均質密度部品に分割し(HPS)、各部品の質量のみを未知数とすることで「stop-and-go」動作でも全慣性パラメータを同定可能にした。表面ベースと体積ベースのセグメンテーションを組み合わせ高速化し、20種の工具データセットとハンマーバランス実演で有効性を示したが、パーツ重心が共面になる対称物体では質量がゼロに縮退する弱点がある。
Citekey: Nadeau-ICRA2023-SumOfItsParts
BibTeX Key: nadeau2023sum
DOI: 10.1109/ICRA48891.2023.10160394
Relevance: 3
Repository: https://papers.starslab.ca/part-segmentation-for-inertial-identification/
Category: note
Template Version: v2.3
---

## Executive Summary
高速・大振幅な励起なしでは信号対雑音比が不足するという慣性パラメータ同定の課題に対し、物体形状をパーツごとの均質密度部品に分割し(HPS)、各部品の質量のみを未知数とすることで「stop-and-go」動作でも全慣性パラメータを同定可能にした。表面ベースと体積ベースのセグメンテーションを組み合わせ高速化し、20種の工具データセットとハンマーバランス実演で有効性を示したが、パーツ重心が共面になる対称物体では質量がゼロに縮退する弱点がある。

---
## Summary

### この論文が答えた問い、あるいは解決した課題は何か？
協働ロボット(cobot)は安全規格(ISO 10218, ISO/TS 15066)により低速動作しか許されないため、力覚センサ(FT)データの信号対雑音比が低く、従来の高速励起軌道に依存する慣性パラメータ同定手法が使えないという課題（§I Introduction）。本論文は、RGB-D カメラによる視覚情報を併用することで、低速・停止発進(stop-and-go)動作のみから物体の質量・重心・慣性テンソルの全パラメータを同定する手法を提案する。

### 提案手法のアプローチと、その根幹をなす要素は何か？
物体は少数の均質密度パーツから構成されるという仮定(Assumption 1: Homogeneous Density of Parts)に基づき、パーツ形状が既知であれば、物体の慣性パラメータは各パーツの質量のみの関数になることを利用する(§III)。これにより、stop-and-go 軌道で得られるランク4のレグレッサ行列でも(最大4パーツまで)質量を非負制約付き最小二乗(式7)で解けば全パラメータが復元できる。

- **Homogeneous Part Segmentation (HPS)**: 各パーツの質量 $m_{r_j}$ のみを未知数として、力覚データから重み付き最小二乗(式6, 7)で解く定式化(§III-B)
- **点群からのパーツ分割パイプライン**: 表面再構成(ball-pivoting algorithm)→テトラヘドラ化(TetGen)→Hierarchical Tetrahedra Clustering(HTC, [8])による体積ベースの凸性最大化クラスタリング(§IV)
- **初期クラスタリングによる高速化**: [9] のサーフェスベース点群クラスタリング(色・法線・位置の非類似度指標、式9)を前処理として用い、HTC が扱うクラスタ数を削減し convex hull 計算コストを低減(Algorithm 1)
- **ground truth 付きデータセット**: 20種の工具の watertight メッシュ・色付き点群・パーツラベル・慣性パラメータ（シミュレーション評価と手法比較に不可欠）

### 特に参考とした既存研究と、それらと比した提案手法の新規性は何か？
負荷同定分野では、古典的な最小二乗法(Atkeson et al. [11], 論文中で OLS と呼称)、regressor 行列のノイズを扱う recursive total least squares(Kubus et al. [12] = [[papers/Kubus-IROS2008-Recursive_Total_Least-Squares/on-line-estimation-of-inertial-parameters-using-a-recursive-total-least-squares-approach|Kubus+ 2008]])、物理的整合性を強制する制約付き最適化(Traversaro et al. [15]、Wensing et al. の LMI 手法[16] ([[papers/Wensing-RAL2017-LMIPhysicalConsistency/linear-matrix-inequalities-for-physically-consistent-inertial-parameter-identification-a-statistical-perspective-on-the-mass-distribution|Wensing+ 2017]]))、prior solution からの測地距離正則化(Lee, Wensing, Park [17]、論文中で GEO と呼称)などが挙げられている(§II-A)。パーツ分割分野では、凹凸最小規則(minima rule)に基づく手法群のサーベイ([22])、体積ベースの Hierarchical Tetrahedra Clustering(Attene et al. [8])、境界保持を重視する表面ベースのスーパーボクセル分割(Lin et al. [9])を直接の構成要素として採用している(§II-B)。

新規性は、(1) 物理的整合性を制約ではなく「パーツごとの点質量による離散化」という定式化そのものに内包させたこと、(2) [8] と [9] という異なる系統(体積ベース／表面ベース)のセグメンテーション手法を組み合わせて高速化したこと、(3) 従来の同定研究が前提としていた高速励起軌道ではなく、視覚情報を併用することで低速な stop-and-go 軌道でも全パラメータ同定を可能にした点にある。

### どのように訓練・最適化したのか？
N/A（学習ベースの手法ではなく、幾何学的最適化に基づく手法のため訓練は行われない。ただしパーツ分割は開発済み手法[8][9]の実装であり、学習成分を含まないと明記されている、§II-B「Our approach to segmentation does not apply any learned components」）。

- **最適化目的**: 式(7) の凸二次計画問題 $\min_{m} \|Am - b\|_2$ subject to $m_{r_j} \geq 0$（パーツ質量に対する非負制約付き最小二乗、MOSEK[39] で解く）。パーツセグメンテーション自体は式(8)のコスト関数（凹み量とクラスタサイズに基づく）を最小化する貪欲マージにより得られる(Algorithm 1)。
- **データセット**: コミュニティ提供の CAD ファイルから著者らが構築した、20種の一般的な工具（Allen Key, Box Wrench, Hammer, Pliers, Vise Grip 等、Table II）からなる新規データセット。各物体について watertight メッシュ、パーツラベル付き色付き点群、正解慣性パラメータ、正解パーツセグメンテーションを含む（train/test スプリットの概念はなく、シミュレーション評価用ベンチマークとして全20物体を使用、§V）。

### どのように検証したか？指標と結果は？
検証は3段階：(1) 20物体データセットでのパーツ分割・形状再構成の単体評価、(2) 4段階のノイズ条件下でのシミュレーション同定精度評価（80シナリオ）、(3) 実機(uFactory xArm 7 + RealSense D435 + Robotiq FT-300)によるハンマーバランス実演(§V)。

- 分割精度指標: undersegmentation error (USE) と global consistency error (GCE)、形状再構成は Hausdorff 距離。初期クラスタリングを加えた Algorithm 1 は、単体 HTC とほぼ同等の誤差（USE 0.1 vs 0.1, GCE 0.07 vs 0.05）でありながら計算時間を約1/3に短縮（3.48秒 vs 9.73秒、Table I）。
- 同定精度は Riemannian geodesic distance $e_\mathrm{Rie}$ と、質量・COM・慣性テンソルの割合誤差 $\bar e_m, \bar e_C, \bar e_J$ で評価。比較対象は古典的最小二乗法(OLS, [11])と幾何学的凸最適化法(GEO, [17])。Table IV によれば、ノイズなし条件では OLS が最も正確だが、ノイズが加わると OLS はほぼ常に物理的に非整合な解に収束（Low ノイズで整合解 14%、High ノイズで 2%）。HPS は全ノイズ条件で 100% 物理的整合性を維持し、COM と慣性テンソルの誤差で OLS・GEO を上回る（例えば High ノイズ条件で $\bar e_C$: HPS 1.07% vs GEO 1.58%、$\bar e_J$: HPS 15.00% vs GEO 48.49%）。一方 $\bar e_m$ は HPS がやや大きい（High ノイズで HPS 2.79% vs GEO 0.64%）。
- 実機実験では、ハンマーの点群スキャン（127枚の RGB-D 画像、約30秒）と約10秒の stop-and-go 軌道後、点群スティッチング(2.87秒)・メッシュ再構成(0.24秒)・パーツ分割(2.82秒、軌道実行と並行実行可能)を経て、パラメータ同定は約0.5秒で完了。推定 COM のみを用いて半径17.5mmの円柱ターゲット上にハンマーをバランスさせることに成功した一方、OLS と GEO は不正確な推定によりバランスに失敗した(§V-B)。

### 検証結果に基づいた議論、明らかになった課題はあるか？
(§VI Discussion より) 著者は以下の限界・議論点を明示している。

- パーツ分割が誤っていると推定誤差につながりうるが、Assumption 1 が成立する限り、過分割（同一密度パーツを複数に分けること）は式(10)の積分の加法性により同定結果に影響しない。逆に密度が異なるパーツを誤って一つとして扱うと同定は失敗する。
- OLS はノイズなし条件で最も正確だが、事前情報によるバイアスがない代わりに、ノイズが加わるとほぼ常に物理的に非整合な解へ収束する。GEO はノイズレベルによらず性能が安定しているが、これは提供された prior solution（真の質量・均質密度分布）の質が高いことに起因する可能性があると著者は述べている。
- HPS の質量誤差 $\bar e_m$ がやや大きい点について、著者は stop-and-go 動作を用いる際の近似（先行研究[4]で議論されているもの）に起因する可能性を挙げている。
- データセット内の20物体では、形状再構成品質(Hausdorff)、分割品質(USE/GCE)、同定精度($e_\mathrm{Rie}$)の間に明確な相関傾向は見られなかった。これは物体の質量と形状が信号対雑音比を大きく左右するためだと著者は説明している。
- スクリュードライバーのような対称物体では、パーツの重心が共面になる場合、最適化器が一部パーツの質量を「怠惰に」ゼロにしてしまう問題がある（式7の最小化に不要なため）。著者は、HTC の階層構造を利用してパーツ重心が共面にならないよう分割を賢く定義する改良を今後の課題として挙げている(§VI 末尾, §VII Conclusion)。
- (§VII Conclusion) 今後の方向性として、質量密度を既知材料のリストから選択する混合整数計画への定式化拡張も挙げられている。

---
## 自身の研究との関連
本論文は、realtime_excitation リポジトリが扱う「物体の慣性パラメータのリアルタイム励起同定」というテーマに対し、視覚情報（RGB-D 点群からのパーツ分割）を併用することで励起軌道の設計制約そのものを緩和するという、既存の Swevers 系(最適励起軌道設計, [[papers/Swevers-TRA1997-OptimalExcitation/optimal-robot-excitation-and-identification|Swevers+ 1997]])や Nadeau 自身の先行研究(Fast Object Inertial Parameter Identification, ICRA 2022, [4])とは異なるアプローチを提示している。

- 式(6)(7)の定式化は、Nadeau らの先行研究[4]で導入された stop-and-go 軌道・点質量離散化によるレグレッサ構成を直接引き継いでおり、両論文はセットで読む価値がある。本論文はその上に「パーツ質量のみを未知数とする」制約を追加し、レグレッサのランク不足（stop-and-go ではランク4）を、視覚的パーツ数の制限（最大4パーツ）で補うという着想である。リアルタイム励起同定において、高速軌道を安全側に緩和したい場面（人と共存する環境、脆い物体の扱いなど）でこの定式化は直接的な応用対象になる。
- Swevers の最適励起軌道設計とは対照的に、本論文は「そもそも高速な励起を避ける」方向の設計であり、両者は「励起の質を上げるか、励起への依存を減らすか」という補完関係にある。リアルタイム性を重視する自身の研究では、視覚センサからのパーツ分割コスト（実機実験で分割 2.82 秒、点群スティッチング 2.87 秒、同定自体は 0.5 秒）がボトルネックになりうる点は、リアルタイム制約下での適用可否を検討する際の具体的な参考値になる。
- 一方で、Assumption 1（パーツごとの均質密度）は関節を持つ物体や既知形状の物体を前提としており、任意物体・未知形状へのオンライン適用や、密度分布が滑らかに変化する物体には適用できない。この制約は、自身の研究がより一般的な物体・オンラインでの形状未知物体を扱う場合の差別化点・限界として意識すべきである。
- 対称物体でパーツ質量がゼロに縮退する問題（式7の最適化が構造的に持つ縮退）は、励起軌道設計側でパーツ重心の非共面性を保証するような軌道生成を行うことで回避できる可能性があり、自身の励起軌道設計研究との接点になりうる。

---
## 追加議論

---
## BibTex
<details>
<summary> Click to show/noshow the BibTex data </summary>

```bibtex
@inproceedings{nadeau2023sum,
  title     = {The Sum of Its Parts: Visual Part Segmentation for Inertial Parameter Identification of Manipulated Objects},
  author    = {Nadeau, Philippe and Giamou, Matthew and Kelly, Jonathan},
  booktitle = {2023 IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2023},
  doi       = {10.1109/ICRA48891.2023.10160394},
  eprint    = {2302.06685},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```

</details>
