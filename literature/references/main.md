# 文献データベース

プロジェクト全体の書誌情報を一元管理する。エントリは `AuthorYear_keyword` 形式の Key で参照する。

---

### Swevers1997_excitation

**Optimal Robot Excitation and Identification**
Jan Swevers, Chris Ganseman, Dilek Bilgin Tükel, Joris De Schutter, Hendrik Van Brussel — IEEE Transactions on Robotics and Automation, 1997
DOI: `10.1109/70.631234`

> フーリエ級数励起軌道パラメータ化と、条件数最小化に代わる共分散行列式最小化基準を提案した励起軌道設計の古典的原典。本サーベイの thesis 軸の出発点（q0 定数まわりの周期励起を暗黙に前提）。

### Park2006_fourier

**Fourier-Based Optimal Excitation Trajectories for the Dynamic Identification of Robots**
Kyung-Jo Park — Robotica, 2006
DOI: `10.1017/S0263574706002712`

> フーリエ級数励起と多項式軌道の分解（多項式は境界条件整合のみに使用）を提示。task drift のような非周期成分を励起の主要素として扱う枠組みは持たない。

### Kubus2007_online

**On-Line Rigid Object Recognition and Pose Estimation Based on Inertial Parameters**
Daniel Kubus, Torsten Kröger, Friedrich M. Wahl — IROS, 2007
DOI: `10.1109/IROS.2007.4399184`

> 慣性パラメータに基づくオンライン物体認識・姿勢推定。base は固定（drift なし）で、推定器側の工夫が主眼。

### Kubus2008_rtls

**On-Line Estimation of Inertial Parameters Using a Recursive Total Least-Squares Approach**
Daniel Kubus, Torsten Kröger, Friedrich M. Wahl — IROS, 2008
DOI: `10.1109/IROS.2008.4650672`

> Kubus2007 の逐次総最小二乗（RTLS）推定器版。同じく base drift を扱わない。

### Bonnet2016_dance

**Optimal Exciting Dance for Identifying Inertial Parameters of an Anthropomorphic Structure**
Vincent Bonnet, Philippe Fraisse, André Crosnier, Maxime Gautier, Alejandro González, Gentiane Venture — IEEE Transactions on Robotics, 2016
DOI: `10.1109/TRO.2016.2583062`

> 全身ヒューマノイド運動を励起そのものとして最適化する手法。task 固定を前提とせず全 DoF が設計対象である点で本プロジェクトの「task 由来 drift + 設計励起の重畳」設定と異なる。

### Lee2021_optimal

**Optimal Excitation Trajectories for Mechanical Systems Identification**
Taeyoon Lee, Bryan D. Lee, Frank C. Park — Automatica, 2021
DOI: `10.1016/j.automatica.2021.109773`

> パラメータ空間の Riemannian 座標不変性に基づく励起軌道最適化。endpoint-pinned B-spline で drift を排除する設計であり、本プロジェクトの「目的関数選択（cond vs D-opt）に依らず劣化する」という主張の直交する参照点。

### Lee2024_modern

**Robot Model Identification and Learning: A Modern Perspective**
Taeyoon Lee, Jaewoon Kwon, Patrick M. Wensing, Frank C. Park — Annual Review of Control, Robotics, and Autonomous Systems, vol. 7, 2024
DOI: `10.1146/annurev-control-061523-102310`

> 分野横断レビュー。結論部で「task-aligned system identification」を将来課題として挙げるのみで、具体的な扱いはしていない。本サーベイの gap 論証における直接的な傍証。

### Albee2022_rattle

**The RATTLE Motion Planning Algorithm for Robust Online Parametric Model Improvement With On-Orbit Validation**
Keenan Albee, Monica Ekal, Brian Coltin, Rodrigo Ventura, Richard Linares, David W. Miller — IEEE Robotics and Automation Letters, 2022
DOI: `10.1109/LRA.2022.3196957`

> 自由浮遊宇宙ロボットの動作計画コストにフィッシャー情報行列トレースを重み付き項として組み込み、タスク到達と励起を同一最適化問題内でオンラインに両立させる。task 軌道と励起を「同じコスト関数内でトレードオフする」という、本プロジェクトの重畳分析とは異なるアプローチの代表例。

### Zhang2025_provably

**Provably-Safe, Online System Identification**
Bohao Zhang, Zichang Zhou, Ram Vasudevan — arXiv:2504.21486, 2025
arXiv: `2504.21486`

> 安全軌道計画（ARMOUR）のコスト関数を回帰行列の条件数最小化に置き換え、衝突・トルク制約を保証しながらオンラインで励起軌道を反復生成する。条件数最小化を安全制約下のオンライン計画に統合する近接例。

### Park2023_nullspace

**Object-Aware Impedance Control for Human-Robot Collaborative Task with Online Object Parameter Estimation**
Jinseong Park, Yong-Sik Shin, Sanghyun Kim — arXiv:2310.12409, 2023
arXiv: `2310.12409`

> 協調搬送タスクの所望軌道のヌルスペースに固定振幅・固定周波数の摂動を注入し、並進タスクを乱さずオンライン推定を行う。本サーベイが確認した中で「task 動作と励起を重畳する」設定に最も近い公表例だが、慣性テンソルは意図的に推定対象外、条件数・スペクトル分析は行っていない。

### Ayusawa2017_condition

**Generating Persistently Exciting Trajectory Based on Condition Number Optimization**
Ko Ayusawa, Antoine Rioux, Eiichi Yoshida, Gentiane Venture, Maxime Gautier — ICRA, 2017
DOI: `10.1109/ICRA.2017.7989770`

> 条件数の解析的勾配を直接計算する効率的な励起軌道最適化手法（ヒューマノイド HRP-4）。task drift は扱わない、最適化効率化の系譜。

### AbuDakka2017_comparison

**Comparison of Trajectory Parametrization Methods with Statistical Analysis for Dynamic Parameter Identification of Serial Robot**
Fares J. Abu-Dakka, Miguel Díaz-Rodríguez — IROS, 2017
DOI: `10.1109/IROS.2017.8206479`

> フーリエ級数と Schroeder Phased Harmonic Sequence を組み合わせた軌道パラメータ化の比較研究。計算時間短縮が主眼で task drift は扱わない。

### Wang2025_insitu

**An In-Situ Excitation Trajectory Optimizer for Industrial Robots in Constrained Space with Human Collaboration**
Chengzhi Wang, Haotian Ju, Zhiyuan Yang, Tianjiao Zheng, Shize Zhao, Sikai Zhao, Dawei Liang, Hegao Cai, Jie Zhao, Yanhe Zhu — Intelligent Robotics and Applications: 18th International Conference, ICIRA 2025, Okayama, Japan, Proceedings Part II (Springer, Lecture Notes in Computer Science, vol. 16075), 2025
DOI: `10.1007/978-981-95-2098-5_52`

> 障害物・作業空間制約下でのフーリエ級数励起軌道最適化（差分進化 + 境界ペナルティ関数）。制約は空間的な衝突回避であり、task 由来の大振幅ドリフトとの重畳は扱わない。全文は paywall のため未確認（書誌情報・要約は OpenAlex/WebSearch 経由の間接確認）。

### Foster2024_locomanipulation

**Physically Consistent Online Inertial Adaptation for Humanoid Loco-Manipulation**
James Foster, Stephen McCrory, Christian DeBuys, Sylvain Bertrand, Robert J. Griffin — IROS, 2024
DOI: `10.1109/IROS58592.2024.10802012`

> ヒューマノイドのロコマニピュレーション中に物理整合拡張カルマンフィルタで慣性パラメータをオンライン推定。励起軌道設計を伴わないオブザーバベース手法。

### Mori2025_activelearning

**Safe Data Acquisition for Inertial Parameter Identification by Expanding the Motion Space**
Kenya Mori, Ko Ayusawa, Gentiane Venture — IEEE Robotics and Automation Letters, 2025
DOI: `10.1109/LRA.2025.3566621`

> 非固定ベースマニピュレータのバランス制約下で反復的（active learning）に励起動作を生成し条件数を改善。励起動作自体が制約を満たす設計であり、別途の task 軌道への重畳は扱わない。

### Cho2024_recursive

**Recursive Least Squares with Log-Determinant Divergence Regularisation for Online Inertia Identification**
Namhoon Cho, Taeyoon Lee, Hyo-Sang Shin — ICRA, 2024
DOI: `10.1109/ICRA57147.2024.10610389`

> 脚式ロボットの慣性パラメータをオンライン逐次最小二乗で推定。物理整合正則化が主眼で励起軌道設計は扱わない。

### Duan2023_trajectorygen

**Trajectory Generation for Online Payload Estimation of Robot Manipulators: A Supervised Learning Based Approach**
Xiaoming Duan, Yebin Wang, Diego Romeres, Toshiaki Koike-Akino, Philip V. Orlik — CASE, 2023
DOI: `10.1109/CASE56687.2023.10260415`

> 初期関節配置（task/環境で規定済み、設計変数ではない）から最適振幅を教師あり学習で即座に生成する、オンラインペイロード推定向け高速軌道生成。

### Tian2024_virtualconstraints

**Excitation Trajectory Optimization for Dynamic Parameter Identification Using Virtual Constraints in Hands-on Robotic System**
Huanyu Tian, Martin Huber, Christopher E. Mower, Zhe Han, Changsheng Li, Xingguang Duan, Christos Bergeles — ICRA, 2024
DOI: `10.1109/ICRA57147.2024.10610950`

> 自己衝突回避を重視した co-manipulation ロボットの励起軌道最適化。空間的制約が主眼で task drift の重畳は扱わない。

### Huang2025_hydraulic

**A Sequential Approach for Accurate Parameters Identification of Heavy-Duty Hydraulic Manipulators Ensuring Physical Feasibility**
Weidi Huang, Zhiwei Chen, Fu Zhang, Min Cheng, Ruqi Ding, Junhui Zhang, Bing Xu — IEEE Robotics and Automation Letters, 2025
DOI: `10.1109/LRA.2025.3579253`

> 低速動作しかできない重機油圧マニピュレータ向けに、パラメータ群ごとに分離した逐次励起で観測行列の条件数を改善。条件数悪化の原因は速度・トルクのハード制約であり、task 由来の非設計的ドリフトとは機構が異なる。

### HuangS2025_analytical

**An Analytical Approach for Dealing With Explicit Physical Constraints in Excitation Optimization Problems of Dynamic Identification**
Shifeng Huang, Fan Li, Xing Zhou, Molong Duan — IEEE Transactions on Robotics, 2025
DOI: `10.1109/TRO.2025.3543296`

> 初期条件・関節限界などの物理制約を、反復探索でなく決定論的なフーリエパラメータ化操作（オフセット・スケーリング・中心並進）で満たす手法。100% の実行可能解達成率と1桁の効率化を報告。

### Wensing2024_geometric

**A Geometric Characterization of Observability in Inertial Parameter Identification**
Patrick M. Wensing, Günter Niemeyer, Jean-Jacques Slotine — The International Journal of Robotics Research, 2024（arXiv 版: 1711.03896, 2017）
DOI: `10.1177/02783649241258215`

> 運動学木の慣性パラメータ識別可能性を、近似なしに幾何学的に特徴づけるアルゴリズム。励起軌道設計ではなく構造的識別可能性の解析であり、本プロジェクトの観測行列条件数の議論とは補完的だが異なる軸。

### Nadeau2022_fast

**Fast Object Inertial Parameter Identification for Collaborative Robots**
Philippe Nadeau, Matthew Giamou, Jonathan Kelly — ICRA, 2022
DOI: `10.1109/ICRA46639.2022.9916213`

> cobot の低速動作制約による低 SNR に対応する高速慣性パラメータ同定。大振幅励起そのものを回避する設計思想が本プロジェクトの「大振幅励起を前提とする」設定と対照的。

### Nadeau2023_partseg

**The Sum of Its Parts: Visual Part Segmentation for Inertial Parameter Identification of Manipulated Objects**
Philippe Nadeau, Matthew Giamou, Jonathan Kelly — ICRA, 2023
DOI: `10.1109/ICRA48891.2023.10160394`

> 物体形状のパーツ分割により stop-and-go 動作のみで全慣性パラメータを復元。大振幅励起を要求しない代替アプローチの代表例。

### Wensing2017_lmi

**Linear Matrix Inequalities for Physically-Consistent Inertial Parameter Identification: A Statistical Perspective on the Mass Distribution**
Patrick M. Wensing, Sangbae Kim, Jean-Jacques E. Slotine — IEEE Robotics and Automation Letters, 2017
DOI: `10.1109/LRA.2017.2729659`

> 質量分布の物理的整合性を凸制約（LMI）として定式化。励起軌道設計ではなくパラメータ空間の制約に関する基盤研究。

### Hu2025_adaptive

**Adaptive Experiment Design for Nonlinear System Identification with Operational Constraints**
Jingwei Hu, Dave Zachariah, Torbjörn Wigren, Petre Stoica — arXiv, 2025
arXiv: `2502.20941`

> 運用制約下でのレシーディングホライズン型適応実験計画（ロボット非依存の一般制御理論）。標準化 L-criterion によるオンライン入力設計は、task/運用制約と励起設計のトレードオフを扱う一般理論として関連。

### Leboutet2021_survey

**Inertial Parameter Identification in Robotics: A Survey**
Quentin Leboutet, Julien Roux, Alexandre Janot, J. Rogelio Guadarrama-Olvera, Gordon Cheng — Applied Sciences, 2021
DOI: `10.3390/app11094303`

> 慣性パラメータ同定分野の包括的レビュー。励起軌道設計への言及は41ページ中1段落程度に限られる。
