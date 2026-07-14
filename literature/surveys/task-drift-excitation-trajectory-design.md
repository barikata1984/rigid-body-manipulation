# Literature Survey: Task 由来の大振幅ベース運動が重畳した状況下での慣性パラメータ同定のための励起軌道設計

| | |
|---|---|
| **Date** | 2026-07-15 |
| **Scope** | 剛体慣性パラメータ同定のための励起軌道設計。特に、task で要求される大振幅、非設計的な base motion（本プロジェクトでは quintic spline による関節の複数回転）が重畳した状況下での観測行列の条件性劣化に関する prior art の網羅と novelty gap の確定 |
| **Papers mapped** | 26 |
| **Hub papers (deep-read)** | 10 |
| **Research Questions** | RQ1: task 由来の大振幅、非設計的 base motion が重畳した状況で励起軌道を設計する既存手法は何か、それらはどのアプローチを取るか / RQ2: base motion が観測行列の条件数を悪化させる現象を明示的に定量化、定式化した先行研究は存在するか、存在するならその機構説明は本プロジェクトの説明とどう異なるか / RQ3: 本プロジェクトの主張に残された novelty gap は何か、どの主張が既出でどの主張が新規か |

## Abstract

剛体慣性パラメータ同定のための励起軌道設計は、Swevers (1997) のフーリエ級数励起を起点とする分野である。
条件数最小化、共分散最小化、座標不変な最適化基準へと発展してきた。
本サーベイは、これらの手法が共通して「軌道は固定動作点まわりの周期運動であるか、自由に設計可能である」という前提に依拠している点に着目する。
task が要求する大振幅、非設計的な base motion(例: 関節の複数回転)が励起軌道に重畳する状況を扱った prior art を、26 本の論文(うちハブ論文 10 本を深読み)にわたって調査した。
既存研究は、(a) ヌルスペースに励起を閉じ込めてタスクと分離する、(b) 励起、安全制約をコスト関数に統合したオンライン計画に頼る、(c) 大振幅励起そのものを回避し stop-and-go やオブザーバベースの手法に頼る、の3方向に分岐している。
いずれも「task 由来の大振幅ドリフトが励起の周波数帯、振幅バウンドと衝突して観測行列の条件数を桁で悪化させる」という現象を周波数領域で定量化した例は確認されなかった。
本調査は、OpenAlex による前方、後方引用のスノーボーリングと、4件の重点確認論文(Park et al. 2023 arXiv:2310.12409、In-Situ Excitation Trajectory Optimizer ICIRA2025、Abu-Dakka et al. 2017 IROS、Ayusawa et al. 2017 ICRA)の原文照合により実施した。

## Research Landscape Overview

励起軌道設計の研究系譜は、1990年代の Armstrong や Gautier and Khalil による決定論的軌道設計に始まる。
Swevers et al. (1997) がフーリエ級数パラメータ化と統計的最適基準(共分散行列式最小化)を導入したことで、理論的な基盤が確立された。
2000年代には Park (2006) がフーリエ+多項式の混合パラメータ化を、Kubus et al. (2007, 2008) がオンライン推定器(RTLS)を提示した。
これにより、励起軌道の「設計」と観測データの「推定」が分業する構造が定着した。
2010年代には Bonnet et al. (2016) がヒューマノイドの全身動作そのものを励起として最適化する方向へ研究を進めた。
同時期に Ayusawa et al. (2017) が条件数勾配の解析的計算により、大規模自由度系への拡張を進めた。
2020年代に入ると、Lee, Lee & Park (2021) がパラメータ空間の座標不変性(Riemannian 構造)に基づく最適化基準を提示した。
Leboutet et al. (2021) と Lee et al. (2024, Annual Reviews) はそれぞれ分野横断的なサーベイを発表している。

2020年代半ば以降は、励起軌道の「オフライン設計」から「オンラインでのタスク実行中の同定」へと関心が移りつつある。
宇宙ロボティクス(Albee et al. 2022, RATTLE)、協調搬送タスク中の物体パラメータ推定(Park et al. 2023)、安全制約下でのオンライン同定(Zhang, Zhou & Vasudevan 2025, ROAHM Lab)、ヒューマノイドのロコマニピュレーション中の適応(Foster et al. 2024)が、いずれも「タスクを止めずに同定する」という要求に応える方向で発展している。
しかし、これらの研究は task 動作と励起動作の「干渉」を条件数の観点から定量化するのではない。
干渉を構造的に回避する(ヌルスペース分離、stop-and-go、低速レジームでの簡略化モデル)か、あるいはコスト関数の重み付けで両者を扱う(オンライン計画への統合)かのいずれかに分岐している。

## Terminology and Background

| Term | Synonyms / Variants | Scope in this survey |
|------|---------------------|----------------------|
| **励起軌道 (excitation trajectory)** | exciting trajectory, persistently exciting trajectory | 慣性パラメータ同定のために設計される関節軌道。フーリエ級数パラメータ化が主流 |
| **Base drift / task drift** | task-mandated base motion, non-designed base motion | 本プロジェクトが導入した用語。task が要求する、励起設計とは独立な大振幅、非周期的な関節運動（本プロジェクトでは j5 の複数回転） |
| **条件数 (condition number)** | κ(F), cond | 観測（回帰）行列 F の最大、最小特異値比。小さいほど識別性が高いとされる励起軌道設計の主要な最適化目的 |
| **D-optimality** | 共分散行列式最小化, log-det criterion | Swevers (1997) が導入した統計的最適基準。推定パラメータの共分散行列（または Cramér-Rao 下界）の行列式を最小化する |
| **ヌルスペース励起 (null-space excitation)** | null-space perturbation, redundancy-based excitation | task 空間の運動学的写像の零空間に励起信号を注入し、task 動作を乱さずに同定情報を得る手法（例: Park et al. 2023） |
| **持続的励起 (persistent excitation, PE)** | PE condition | 適応制御、システム同定の古典的条件。入力信号が全パラメータ方向に十分な情報を与えることを要求する |
| **識別可能性 / 観測可能性 (identifiability / observability)** | structural identifiability | ある慣性パラメータの組み合わせが、運動学的構造上そもそも観測データから復元可能かどうかを扱う、励起軌道設計とは独立した軸（例: Wensing, Niemeyer & Slotine 2024） |

## Survey Findings

### Thesis

本分野が共有する未解決の中心的緊張は、「励起軌道の最適性理論はいずれも task 軌道が存在しない、または task 軌道が励起設計者の自由になる状況を前提としている」という点にある。
[[papers/Swevers-TRA1997-OptimalExcitation/optimal-robot-excitation-and-identification|Swevers+ 1997]] のフーリエ級数励起は固定動作点まわりの周期運動を前提とする（軌道は $q_0$ 定数オフセット + 周期成分）。
[[papers/Lee-Automatica2021-Optimal_Excitation_Trajectories/optimal-excitation-trajectories-for-mechanical-systems-identification|Lee+ 2021]] の座標不変最適化も endpoint-pinned B-spline で drift を排除する。
[[papers/Bonnet-TRO2016-Optimal_Exciting_Dance/optimal-exciting-dance-for-identifying-inertial-parameters-of-an-anthropomorphic-structure|Bonnet+ 2016]] は task 制約自体を持たず、全自由度が励起設計の対象になる。
task が非設計的な大振幅運動を要求する場合、これらの理論が暗黙に仮定する「軌道は励起設計者の自由になる」という前提が崩れる。

この緊張への対応は、深読みしたハブ論文の中で3方向に分岐している。
[[papers/Park-arXiv2023-Object-Aware_Impedance_Control/object-aware-impedance-control-for-human-robot-collaborative-task-with-online-object-parameter-estimation|Park+ 2023]] はヌルスペースへの励起の閉じ込めによって、task と励起の幾何学的直交性を人為的に作り出す。
[[papers/Albee-RAL2022-RATTLE/the-rattle-motion-planning-algorithm-for-robust-online-parametric-model-improvement-with-on-orbit-validation|Albee+ 2022]] と [[papers/Zhang-arXiv2025-ProvablySafe/provably-safe-online-system-identification|Zhang+ 2025]] は、励起（フィッシャー情報、条件数）を task 到達コストと同一の最適化問題内の重み付き項として統合し、両者のトレードオフをオンラインに解く。
[[papers/Nadeau-ICRA2022-FastObjectInertial/fast-object-inertial-parameter-identification-for-collaborative-robots|Nadeau+ 2022]] と [[papers/Nadeau-ICRA2023-SumOfItsParts/the-sum-of-its-parts-visual-part-segmentation-for-inertial-parameter-identification-of-manipulated-objects|Nadeau+ 2023]] は、そもそも大振幅励起を要求しない同定手法（低速レジームでの簡略化モデル、形状事前知識による質量のみの復元）へと問題設定自体を変更する。
いずれのアプローチも、task drift が励起の周波数帯や振幅バウンドと直接競合し観測行列を構造的に劣化させるという現象そのものを分析対象にしていない。

### Foundation

1. **フーリエ級数励起パラメータ化**: [[papers/Swevers-TRA1997-OptimalExcitation/optimal-robot-excitation-and-identification|Swevers+ 1997]] が導入し、[[papers/Park-Robotica2006-Fourier_Optimal_Excitation/fourier-based-optimal-excitation-trajectories-for-the-dynamic-identification-of-robots|Park 2006]]、[[papers/Bonnet-TRO2016-Optimal_Exciting_Dance/optimal-exciting-dance-for-identifying-inertial-parameters-of-an-anthropomorphic-structure|Bonnet+ 2016]]、Ayusawa et al. (2017)、[[papers/Lee-Automatica2021-Optimal_Excitation_Trajectories/optimal-excitation-trajectories-for-mechanical-systems-identification|Lee+ 2021]] が共有する基盤パラメータ化。周期性、帯域制限性、解析的微分可能性を同時に与える一方、係数の三角不等式バウンド（$|c_k| \leq \ddot q_{\max}/(2\pi f_0 k)^2$）が $T$ に漸近的に非依存であるという性質が、本プロジェクトが発見した「短い duration では base drift が支配する」機構の前提になっている。
2. **条件数 / D-optimality を目的関数とする非線形最適化**: [[papers/Swevers-TRA1997-OptimalExcitation/optimal-robot-excitation-and-identification|Swevers+ 1997]] 以降のほぼ全てのハブ論文が、物理制約（関節角度、速度、加速度範囲、初期条件ゼロ）下での条件数または共分散行列式最小化という同じ最適化問題の枠組みを共有する。HuangS et al. (2025, TRO) はこの制約充足を決定論的パラメータ化操作に置き換える効率化を提示するが、目的関数自体は同じ系譜にある。
3. **オンライン推定器（Kalman フィルタ / 再帰最小二乗）**: [[papers/Kubus-IROS2008-Recursive_Total_Least-Squares/on-line-estimation-of-inertial-parameters-using-a-recursive-total-least-squares-approach|Kubus+ 2008]]、[[papers/Albee-RAL2022-RATTLE/the-rattle-motion-planning-algorithm-for-robust-online-parametric-model-improvement-with-on-orbit-validation|Albee+ 2022]]、[[papers/Park-arXiv2023-Object-Aware_Impedance_Control/object-aware-impedance-control-for-human-robot-collaborative-task-with-online-object-parameter-estimation|Park+ 2023]] が共有するオンライン推定の基盤。励起軌道の「設計」と観測データからの「推定」は分業しており、本プロジェクトが扱うオフライン励起軌道最適化はこの基盤の後段（設計側）に位置する。
4. **物理的整合性制約**: Rucker & Wensing の log-Cholesky パラメータ化や Wensing, Kim & Slotine (2017) の LMI 定式化が、慣性パラメータの物理的実現可能性を保証する制約として [[papers/Zhang-arXiv2025-ProvablySafe/provably-safe-online-system-identification|Zhang+ 2025]] 等の最新のオンライン同定手法に継承されている。本プロジェクトの励起軌道設計自体はこの制約を直接の対象にしていないが、同定パイプライン全体の基盤要素として関連する。

### Progress

1. **決定論的軌道設計 → 統計的最適基準**（Armstrong, Gautier & Khalil, Otani & Kakizaki 系譜 → [[papers/Swevers-TRA1997-OptimalExcitation/optimal-robot-excitation-and-identification|Swevers+ 1997]], 1997）: 条件数のみに基づく決定論的な軌道最適化から、測定ノイズを明示的にモデル化した統計的（errors-in-variables）最適基準（共分散行列式最小化）への移行。
2. **単軸周期励起 → 全身自由励起**（[[papers/Bonnet-TRO2016-Optimal_Exciting_Dance/optimal-exciting-dance-for-identifying-inertial-parameters-of-an-anthropomorphic-structure|Bonnet+ 2016]], 2016）: task 制約を持たないヒューマノイド全身動作そのものを励起として最適化する方向への拡張。
3. **座標依存最適化 → 座標不変最適化**（[[papers/Lee-Automatica2021-Optimal_Excitation_Trajectories/optimal-excitation-trajectories-for-mechanical-systems-identification|Lee+ 2021]], 2021）: パラメータ空間の Riemannian 構造を利用し、目的関数のパラメータ化選択に依存しない励起軌道最適化を実現。本プロジェクトが確認した「cond 最適化と D-opt 最適化がいずれも同じ cond=24 に到達する」という目的関数不変性の観測は、この座標不変性の議論と補完的な現象である。
4. **オフライン専用設計からオンラインのタスク統合励起へ**（Albee et al. 2022; Park et al. 2023; Zhang, Zhou & Vasudevan 2025; Foster et al. 2024）: 2022年以降、励起をタスク実行から切り離された専用フェーズとしてではなく、タスク実行中のオンライン計画と制御のループに統合する方向への急速な展開が見られる。ただし、いずれも「タスクが要求する大振幅ドリフトと励起周波数帯の直接的な競合」を分析対象にしていない点で、本プロジェクトが指摘するギャップは埋まっていない。

### Gap

1. **task 由来の大振幅ドリフトと励起周波数帯のスペクトル競合を定量化した先行研究の不在**

   本サーベイで確認した全てのハブ論文と非ハブ論文のいずれも、task 由来の非設計的な base motion のスペクトル内容（周波数、振幅）が励起のフーリエ調波帯と直接競合し、観測行列の特定の列（チャンネル）を「設計不能」にするという現象を周波数領域で分析していない。
   Concept Matrix が示す通り、「条件数 / 観測性分析」列に該当する論文（Ayusawa et al. 2017、HuangS et al. 2025、Wensing, Niemeyer & Slotine 2024）はいずれも条件数の計算、勾配、観測可能性の幾何学的特徴づけを扱うが、task drift という非設計的な外生信号が観測行列に及ぼす影響を分析軸として持たない。
   [[papers/Park-arXiv2023-Object-Aware_Impedance_Control/object-aware-impedance-control-for-human-robot-collaborative-task-with-online-object-parameter-estimation|Park+ 2023]] は最も近い設定（task と励起の重畳）を扱うが、ヌルスペース分離によって干渉自体を回避しており、干渉が不可避な場合（ヌルスペースが task 動作の帯域を吸収しきれないほど task が広帯域かつ大振幅な場合）の分析は存在しない。
   この gap が埋まれば、ロボットが大振幅な非設計的動作を伴う task を実行しながら同定する際に、事前にどの duration や励起周波数設定が構造的に失敗するかを予測できるようになる。

2. **フーリエ係数バウンドの duration 依存性と task drift の duration 依存性の非対称なスケーリングの未指摘**

   本プロジェクトが導出した「励起振幅バウンドは $T$ にほぼ非依存である一方、task drift の速度、加速度ピークはそれぞれ $1/T$、$1/T^2$ でスケールする」という非対称性は、[[papers/Swevers-TRA1997-OptimalExcitation/optimal-robot-excitation-and-identification|Swevers+ 1997]] の三角不等式バウンド（foundation 1 参照）から導出可能だが、サーベイした文献のいずれもこの非対称性を明示的に指摘していない。
   [[papers/Lee-AR2024-Robot_Model_Identification/robot-model-identification-and-learning-a-modern-perspective|Lee+ 2024]]（Annual Reviews）が結論部で「task-aligned system identification」を将来課題として名指ししている（Progress 4 参照）ことは、この方向の重要性が分野内で認識されている一方、具体的な機構分析には未着手であることを示す傍証である。
   この gap が埋まれば、duration を伸ばす（task をゆっくり実行する）ことが常に同定精度を改善するとは限らない場合の設計指針が得られる。

3. **目的関数選択で吸収できない情報損失という主張の先行事例の希薄さ**

   [[papers/Lee-Automatica2021-Optimal_Excitation_Trajectories/optimal-excitation-trajectories-for-mechanical-systems-identification|Lee+ 2021]] のパラメータ空間座標不変性は、目的関数の「パラメータ化」への依存性を除去するが、観測行列自体が構造的に劣化している場合（本プロジェクトが実験的に確認した、cond 最適化と D-optimality の両方が同じ cond=24 に張り付く現象）への言及はない。
   この情報損失が座標変換で吸収できない real information loss であるという主張を明示的に検証した文献は、本サーベイでは確認されなかった。
   この gap が埋まれば、励起軌道最適化の目的関数をいくら工夫しても改善しない「構造的な限界」の存在を事前に判定する基準が得られる。

## Research Questions への回答

### RQ1: task で要求される大振幅、非設計的な base motion が重畳した状況で励起軌道を設計する既存手法は何か。それらはどのアプローチを取るか

本サーベイで確認した既存研究は、以下の3方向に分岐する。いずれも「task drift と励起の重畳を許した上でその干渉を定量化、緩和する」という本プロジェクトの問題設定そのものを直接扱う研究は確認されなかった。

- **(a) 励起をヌルスペースに閉じ込めてタスクと分離する**: [[papers/Park-arXiv2023-Object-Aware_Impedance_Control/object-aware-impedance-control-for-human-robot-collaborative-task-with-online-object-parameter-estimation|Park+ 2023]] が代表例。task 軌道の運動学的写像の零空間に固定振幅、固定周波数の摂動を注入することで、幾何学的に task と励起を直交させる。ただし、この論文が扱う task 動作（人間主導の並進搬送、歩行速度程度）は本プロジェクトが扱う「関節の複数回転」ほど広帯域、大振幅ではなく、ヌルスペース分離が比較的容易に成立する状況に限られる。慣性テンソルも安全上の理由で意図的に推定対象外としている。
- **(b) 励起、安全制約をコスト関数に統合したオンライン計画に頼る**: [[papers/Albee-RAL2022-RATTLE/the-rattle-motion-planning-algorithm-for-robust-online-parametric-model-improvement-with-on-orbit-validation|Albee+ 2022]]（自由浮遊宇宙ロボットの動作計画にフィッシャー情報行列トレースを重み付き項として統合）と [[papers/Zhang-arXiv2025-ProvablySafe/provably-safe-online-system-identification|Zhang+ 2025]]（安全軌道計画のコスト関数を条件数最小化に置き換え）が代表例。task 到達コストと励起情報量を同一最適化問題内でトレードオフさせる設計だが、いずれも「task 軌道自体が非設計的、大振幅である」状況（task 軌道は依然として計画対象の一部）は前提としていない。
- **(c) 大振幅励起そのものを回避し、停止発進や簡略化モデル、オブザーバに頼る**: [[papers/Nadeau-ICRA2022-FastObjectInertial/fast-object-inertial-parameter-identification-for-collaborative-robots|Nadeau+ 2022]]、[[papers/Nadeau-ICRA2023-SumOfItsParts/the-sum-of-its-parts-visual-part-segmentation-for-inertial-parameter-identification-of-manipulated-objects|Nadeau+ 2023]]（cobot の低速制約下での簡略化モデル、形状事前知識による質量のみの復元）、Foster et al. 2024（ロコマニピュレーション中のEKFオンライン適応、励起軌道設計なし）、Cho, Lee & Shin (2024)（脚式ロボットのオンライン再帰最小二乗）が該当する。いずれも大振幅な設計励起を要求しない方向に問題を再定義しており、本プロジェクトが目指す「10パラメータ全てをフーリエ励起で同定する」というスコープとは異なる。

### RQ2: 「base motion が観測行列の条件数を悪化させる」現象を明示的に定量化、定式化した先行研究は存在するか。存在するなら、その機構の説明は本プロジェクトの説明とどう異なるか

観測行列の条件数悪化を定量化した先行研究は複数存在するが、いずれも本プロジェクトが指摘する「task drift と励起周波数帯のスペクトル衝突」を機構として扱っていない。

- **Huang et al. 2025 (RA-L, 油圧マニピュレータ)** は、低速動作しかできない重機油圧マニピュレータにおいて「速度、加速度変化が一貫しないため観測行列が ill-conditioned になる」ことを明示的に述べ、パラメータ群ごとの逐次励起で緩和する。しかし悪化の原因はハードウェアの速度、トルク上限という物理制約であり、task 由来の非設計的なドリフト信号との周波数競合ではない。
- **Ayusawa et al. 2017 (ICRA)** は条件数の解析的勾配を導出し最適化を効率化するが、悪化要因としての task drift は分析対象にしていない。
- **Wensing, Niemeyer & Slotine 2024 (IJRR)** は識別可能性を運動学的構造から幾何学的に特徴づけるが、これは「そもそもどのパラメータの組み合わせが原理的に区別不能か」という構造的識別可能性の問題であり、「励起の設計と task drift の重畳によって特定の周波数チャンネルが動的に占有される」という本プロジェクトの現象とは異なる軸（静的な構造 vs 動的なスペクトル競合）である。
- 本プロジェクトが導出した機構（重力ベクトルがセンサー座標系で task 由来の回転周波数の正弦波として現れ、励起の特定次高調波チャンネルと直接競合する、かつ相対振幅比が $T^2/(\text{turn 数})$ に比例してスケールする）は、上記のいずれとも異なる。この定量的機構を明示的に導出、検証した先行研究は本サーベイでは確認されなかった。

### RQ3: 本プロジェクトの主張に残された novelty gap は何か。どの主張が既出で、どの主張が新規か

- **既出（引用義務あり）**: フーリエ級数励起パラメータ化と振幅バウンド（[[papers/Swevers-TRA1997-OptimalExcitation/optimal-robot-excitation-and-identification|Swevers+ 1997]]）、条件数 vs 共分散最適基準の対比（同上）、パラメータ空間の座標不変性（[[papers/Lee-Automatica2021-Optimal_Excitation_Trajectories/optimal-excitation-trajectories-for-mechanical-systems-identification|Lee+ 2021]]）、ヌルスペースによるタスク、励起分離という設計思想（[[papers/Park-arXiv2023-Object-Aware_Impedance_Control/object-aware-impedance-control-for-human-robot-collaborative-task-with-online-object-parameter-estimation|Park+ 2023]]）、オンライン励起とタスクコストの統合（[[papers/Albee-RAL2022-RATTLE/the-rattle-motion-planning-algorithm-for-robust-online-parametric-model-improvement-with-on-orbit-validation|Albee+ 2022]]、Zhang, Zhou & Vasudevan 2025）、task-aligned system identification が未解決課題であるという分野内の認識（[[papers/Lee-AR2024-Robot_Model_Identification/robot-model-identification-and-learning-a-modern-perspective|Lee+ 2024]] 結論部）は、いずれも既出であり related work で引用すべきである。
- **新規（本サーベイで先行例が確認できなかった）**: (1) task 由来の非設計的な大振幅ドリフトが、励起の特定フーリエ調波チャンネルと周波数領域で直接衝突し、観測行列の当該列を「設計不能」にするという機構の指摘。(2) 励起振幅バウンドの $T$ 非依存性と task drift 速度、加速度ピークの $T^{-1}$、$T^{-2}$ スケーリングという非対称性から、相対振幅比が $T^2/(\text{turn 数})$ に比例して悪化するという定量的スケーリング則の導出。(3) この劣化が条件数最適化、D-optimality のいずれの目的関数選択でも同水準（cond≈24）に張り付くという実験的観察、およびそれが座標変換（[[papers/Lee-Automatica2021-Optimal_Excitation_Trajectories/optimal-excitation-trajectories-for-mechanical-systems-identification|Lee+ 2021]] の Riemannian pullback 等）によっても吸収されない構造的情報損失であるという主張。
- **novelty gap の性格**: 本プロジェクトの新規性は、個々の技術要素（フーリエ励起、条件数最適化、task 制約下の励起）ではなく、それらの**組み合わせに対する分析軸**（task drift と励起のスペクトル競合という現象そのものを周波数領域で定式化し、duration、turn 数に対する定量的スケーリング則を導いたこと）にある。この位置づけは related work で「個々の要素技術は先行研究の系譜にあるが、task drift による条件数劣化の機構分析という問いそのものが未踏である」という形で明示する必要がある。

## 重点確認 4 件の結果

ユーザー指定の未読4件について、確認状況と novelty 判定への影響を個別に記す。

### 1. Yun 2023 → 実際は Park, Shin & Kim 2023（著者名の訂正が必要）

**指摘**: 依頼文書は本論文を「Yun 2023」としているが、arXiv:2310.12409 の実際の著者は **Jinseong Park, Yong-Sik Shin, Sanghyun Kim** であり、"Yun" という著者は存在しない（arXiv 公式ページ、OpenAlex の双方で確認済み）。おそらく内部ノート作成時の参照違いであり、事実として訂正する。

**確認状況**: PDF 全文（11ページ）を取得し、精読、verbatim 確認済み。深読みノートを新規作成した（[[papers/Park-arXiv2023-Object-Aware_Impedance_Control/object-aware-impedance-control-for-human-robot-collaborative-task-with-online-object-parameter-estimation|Park-arXiv2023-Object-Aware_Impedance_Control]]）。独立エージェントによるクロス検証は Verify 節に記載の通り未完了であり、メインループ自身による PDF 直接照合で代替した。

**novelty 判定への影響**: 本サーベイで確認した中で最も近い publish 済み類例である、という finding 文書の自己評価は妥当だった。ただし差分は明確: (1) 摂動は固定振幅0.2 rad/s、固定周波数0.4 Hzの単一トーンで、フーリエ級数励起の最適化は一切行われていない。(2) task 軌道は人間主導の並進搬送（歩行速度程度）であり、本プロジェクトが扱う「関節の複数回転」ほど広帯域、大振幅ではない。(3) 慣性テンソルは安全上の理由で意図的に推定対象から除外されている（質量、重心のみ）。(4) 条件数、観測行列の条件性、スペクトル衝突という視点は本文中に一切登場しない。したがって「ヌルスペース分離という設計思想」自体は既出だが、「分離が困難なほど task drift が広帯域、大振幅な場合の干渉の定量化」は本論文の範囲外であり、novelty gap は維持される。

### 2. In-Situ Excitation Trajectory Optimizer（正式には ICIRA 2025 Proceedings、LNEE ではなく Springer LNCS 系列）

**確認状況**: DOI (10.1007/978-981-95-2098-5_52) は解決可能で書誌情報は確認できたが、**全文は paywall のため未確認**。OpenAlex には abstract が収録されておらず、Springer のページは認証リダイレクトを返した。WebSearch（Google の検索結果スニペット、ResearchGate の "Request PDF" プレビューページの索引情報に由来すると推定）経由で、以下の内容を間接的に確認した: 著者は Chengzhi Wang 他10名、2025年出版、正式タイトルは "An In-Situ Excitation Trajectory Optimizer for Industrial Robots in Constrained Space with Human Collaboration"（依頼文書のタイトルと一致）。内容は「障害物、人間協働が存在する制約空間で、境界ペナルティ関数+ 差分進化アルゴリズムによりフーリエ級数励起軌道を最適化し、衝突を回避しながら1.66%程度の同定精度低下に抑えた」というもの。

**訂正**: 依頼文書は "Springer LNEE 2024" としているが、OpenAlex のメタデータでは publication_date が 2025-10-26、掲載シリーズは "Lecture Notes in Computer Science"（LNEE = Lecture Notes in Electrical Engineering ではない）であった。ICIRA（Intelligent Robotics and Applications）の Proceedings は例年 LNCS/LNAI 系列で出版されるため、こちらの方が整合的である。

**novelty 判定への影響**: 全文未確認のため断定はできないが、間接確認できた内容（空間的な衝突回避制約下でのフーリエ励起最適化）からは、task 由来の非設計的な大振幅ドリフトとの重畳という本プロジェクトの設定とは異なる問題（静的な作業空間制約 vs 動的なタスク軌道との重畳）を扱っていると判断する。タイトルの類似性は表層的（"excitation trajectory optimizer" という語彙の一致）であり、扱う制約の種類（空間、衝突 vs task 軌道の動的な重畳）が異なるため、novelty gap への影響は限定的と判定する。ただし全文未確認である以上、この判定は暫定的である。

### 3. Abu-Dakka & Díaz-Rodríguez 2017（IROS）

**確認状況**: OpenAlex 経由で abstract を確認済み（DOI: 10.1109/IROS.2017.8206479）。全文（IEEE Xplore, paywall）は未確認だが、abstract の内容が具体的かつ本プロジェクトのテーマと明確に異なるため、abstract ベースの判定で十分と判断した。

**novelty 判定への影響**: 本論文はフーリエ級数と Schroeder Phased Harmonic Sequence（SPHS）を組み合わせた軌道パラメータ化手法で、目的は初期値設計の改善による計算時間短縮であり、task drift や条件数悪化の機構分析は扱わない。novelty gap への影響なし（「(c) 別問題」）。

### 4. Ayusawa 2017（ICRA）

**確認状況**: OpenAlex 経由で abstract を確認済み（DOI: 10.1109/ICRA.2017.7989770、正式タイトル: "Generating Persistently Exciting Trajectory Based on Condition Number Optimization"）。全文未確認だが、abstract の内容が具体的（条件数の解析的勾配計算という手法の核心が明記）なため、abstract ベースの判定で十分と判断した。

**novelty 判定への影響**: 条件数の勾配を直接計算する効率的な最適化手法（大規模自由度系、ヒューマノイド HRP-4 対象）であり、task drift による観測行列の劣化という現象は分析対象にしていない。novelty gap への影響なし（「(c) 別問題」）。ただし、条件数の解析的取り扱いという点で本プロジェクトの分析（条件数の $T$、turn数依存性の導出）と方法論的に隣接しており、related work で言及する価値はある。

## Concept Matrix

列は Map 全体（26 本）の concept tag 頻度から、2 本未満、21 本超（80%）の tag を除外した上で、thesis/foundation/gap 節で言及した軸を優先して選定した。

| Paper | Fourier/周期励起設計 | task/空間制約下の励起 | オンライン、タスク中同定 | 条件数/観測性分析 | ヌルスペース/冗長性活用 | 大振幅励起の回避 |
|-------|---|---|---|---|---|---|
| [[papers/Swevers-TRA1997-OptimalExcitation/optimal-robot-excitation-and-identification\|Swevers+ 1997]] | ● | | | ● | | |
| [[papers/Park-Robotica2006-Fourier_Optimal_Excitation/fourier-based-optimal-excitation-trajectories-for-the-dynamic-identification-of-robots\|Park 2006]] | ● | | | ○ | | |
| [[papers/Kubus-IROS2007-On-line_Rigid_Object/on-line-rigid-object-recognition-and-pose-estimation-based-on-inertial-parameters\|Kubus+ 2007]] | | | ● | | | |
| [[papers/Kubus-IROS2008-Recursive_Total_Least-Squares/on-line-estimation-of-inertial-parameters-using-a-recursive-total-least-squares-approach\|Kubus+ 2008]] | | | ● | | | |
| [[papers/Bonnet-TRO2016-Optimal_Exciting_Dance/optimal-exciting-dance-for-identifying-inertial-parameters-of-an-anthropomorphic-structure\|Bonnet+ 2016]] | ● | | | ○ | | |
| [[papers/Lee-Automatica2021-Optimal_Excitation_Trajectories/optimal-excitation-trajectories-for-mechanical-systems-identification\|Lee+ 2021]] | ● | | | ● | | |
| [[papers/Lee-AR2024-Robot_Model_Identification/robot-model-identification-and-learning-a-modern-perspective\|Lee+ 2024]] | ○ | ○ | ○ | ○ | | |
| [[papers/Albee-RAL2022-RATTLE/the-rattle-motion-planning-algorithm-for-robust-online-parametric-model-improvement-with-on-orbit-validation\|Albee+ 2022]] | | ● | ● | ○ | | |
| Zhang+ 2025 | | ● | ● | ● | | |
| [[papers/Park-arXiv2023-Object-Aware_Impedance_Control/object-aware-impedance-control-for-human-robot-collaborative-task-with-online-object-parameter-estimation\|Park+ 2023]] | | ● | ● | | ● | ○ |
| Ayusawa+ 2017 | ● | | | ● | | |
| Abu-Dakka & Díaz-Rodríguez 2017 | ● | | | | | |
| Wang+ 2025 (In-Situ) | ● | ● | | | | |
| Foster+ 2024 | | | ● | | | |
| Mori, Ayusawa & Venture 2025 | ● | ● | | ● | | |
| Cho, Lee & Shin 2024 | | | ● | | | |
| Duan+ 2023 | ○ | ○ | ● | | | |
| Tian+ 2024 | ● | ● | | | | |
| Huang+ 2025 (hydraulic) | ● | ○ | | ● | | |
| HuangS+ 2025 (analytical) | ● | ● | | | | |
| Wensing, Niemeyer & Slotine 2024 | | | | ● | | |
| [[papers/Nadeau-ICRA2022-FastObjectInertial/fast-object-inertial-parameter-identification-for-collaborative-robots\|Nadeau+ 2022]] | | | | | | ● |
| [[papers/Nadeau-ICRA2023-SumOfItsParts/the-sum-of-its-parts-visual-part-segmentation-for-inertial-parameter-identification-of-manipulated-objects\|Nadeau+ 2023]] | | | | | | ● |
| Wensing, Kim & Slotine 2017 | | | | | | |
| Hu+ 2025 | | ● | ● | | | |
| Leboutet+ 2021 | ○ | ○ | ○ | ○ | | |

（3件の補完検索サブエージェントのうち1件が本報告書作成時点で未返却である。Methodology → Map の注記を参照。追加候補が得られた場合、本表に追記する。）

## Quantitative Trends

### Publication Count by Year

| Year | Count |
|------|-------|
| 2025 | 6 |
| 2024 | 5 |
| 2023 | 3 |
| 2022 | 2 |
| 2021 | 2 |
| 2017 | 3 |
| 2016 | 1 |
| 2007–2008 | 2 |
| 1997 | 1 |
| 2006 | 1 |

（合計26本。年不明、複数版存在の論文は主たる出版年で計上。2025年に集中しているのは、オンライン、タスク統合励起という研究方向の急速な立ち上がりを反映する。）

### Concept Distribution

| Concept | Count of papers | % |
|---------|-----------------|---|
| Fourier / 周期励起設計 | 14 | 54% |
| 条件数 / 観測性分析 | 12 | 46% |
| task / 空間制約下の励起 | 12 | 46% |
| オンライン、タスク中同定 | 11 | 42% |
| 大振幅励起の回避 | 3 | 12% |
| ヌルスペース / 冗長性活用 | 1 | 4% |

「ヌルスペース/冗長性活用」列が26本中1本（[[papers/Park-arXiv2023-Object-Aware_Impedance_Control/object-aware-impedance-control-for-human-robot-collaborative-task-with-online-object-parameter-estimation|Park+ 2023]]のみ）に限られている点が、Gap 1（task drift とのスペクトル競合を扱う研究の不在）の統計的な裏付けである。

### Experimental Setting Breakdown

| Setting | Count | % |
|---------|-------|---|
| 実機のみ | 11 | 42% |
| シミュレーションのみ | 4 | 15% |
| 実機 + シミュレーション | 8 | 31% |
| 理論、解析のみ | 3 | 12% |

### Top Venues

| Venue | Count |
|-------|-------|
| ICRA | 5 |
| IROS | 4 |
| RA-L | 4 |
| arXiv (preprint) | 3 |
| TRO | 2 |
| Automatica / Robotica / Applied Sciences / Annual Reviews / CASE / ICIRA Proceedings / IJRR（各1） | 7 |

## Hub Papers

| # | Citekey | Title | Year | Venue | Code | Why hub |
|---|---------|-------|------|-------|------|---------|
| 1 | [[papers/Swevers-TRA1997-OptimalExcitation/optimal-robot-excitation-and-identification\|Swevers-TRA1997-OptimalExcitation]] | Optimal Robot Excitation and Identification | 1997 | TRA | — | C: thesis/foundation 軸の起点。フーリエ級数励起と統計的最適基準の原典で、本プロジェクトの機構分析（振幅バウンドの $T$ 非依存性）の直接的な出発点 |
| 2 | [[papers/Park-Robotica2006-Fourier_Optimal_Excitation/fourier-based-optimal-excitation-trajectories-for-the-dynamic-identification-of-robots\|Park-Robotica2006-Fourier_Optimal_Excitation]] | Fourier-Based Optimal Excitation Trajectories for the Dynamic Identification of Robots | 2006 | Robotica | — | C: foundation 軸。フーリエ+多項式分解の起源で、境界条件整合という限定的な役割に留まる点が本プロジェクトの「task drift は境界条件ではなく主要な干渉源」という主張との対比になる |
| 3 | [[papers/Kubus-IROS2007-On-line_Rigid_Object/on-line-rigid-object-recognition-and-pose-estimation-based-on-inertial-parameters\|Kubus-IROS2007-On-line_Rigid_Object]] | On-Line Rigid Object Recognition and Pose Estimation Based on Inertial Parameters | 2007 | IROS | — | C: foundation 軸（オンライン推定器）。base 固定、drift なしという対照例 |
| 4 | [[papers/Kubus-IROS2008-Recursive_Total_Least-Squares/on-line-estimation-of-inertial-parameters-using-a-recursive-total-least-squares-approach\|Kubus-IROS2008-Recursive_Total_Least-Squares]] | On-Line Estimation of Inertial Parameters Using a Recursive Total Least-Squares Approach | 2008 | IROS | — | C: foundation 軸。Kubus 2007 と対になる推定器研究で、同じく drift を扱わない |
| 5 | [[papers/Bonnet-TRO2016-Optimal_Exciting_Dance/optimal-exciting-dance-for-identifying-inertial-parameters-of-an-anthropomorphic-structure\|Bonnet-TRO2016-Optimal_Exciting_Dance]] | Optimal Exciting Dance for Identifying Inertial Parameters of an Anthropomorphic Structure | 2016 | TRO | — | B: 「全身自由励起」クラスタと「ヒューマノイド応用」クラスタを橋渡し。cited-by 上位（本サーベイ内引用頻度 3 件以上） |
| 6 | [[papers/Lee-Automatica2021-Optimal_Excitation_Trajectories/optimal-excitation-trajectories-for-mechanical-systems-identification\|Lee-Automatica2021-Optimal_Excitation_Trajectories]] | Optimal Excitation Trajectories for Mechanical Systems Identification | 2021 | Automatica | — | B+C: 「古典的励起最適化」と「パラメータ空間の座標不変性理論」を橋渡し。目的関数不変性という本プロジェクトの主張と直交する参照点として thesis/progress 軸の主役 |
| 7 | [[papers/Lee-AR2024-Robot_Model_Identification/robot-model-identification-and-learning-a-modern-perspective\|Lee-AR2024-Robot_Model_Identification]] | Robot Model Identification and Learning: A Modern Perspective | 2024 | Annual Review of Control, Robotics, and Autonomous Systems | — | C（例外: レビュー論文だが gap 軸の主役のため hub 化）: 「task-aligned system identification」を将来課題として明示的に名指しした、本サーベイの gap 論証の直接的傍証 |
| 8 | [[papers/Albee-RAL2022-RATTLE/the-rattle-motion-planning-algorithm-for-robust-online-parametric-model-improvement-with-on-orbit-validation\|Albee-RAL2022-RATTLE]] | The RATTLE Motion Planning Algorithm for Robust Online Parametric Model Improvement with On-Orbit Validation | 2022 | RA-L | — | B+C: 「オンライン FIM 重み付き動作計画」と「自由浮遊宇宙ロボット」クラスタを橋渡し。task 到達コストと励起を同一最適化問題内で扱う設計の代表例として RQ1 の回答の中心 |
| 9 | [[papers/Zhang-arXiv2025-ProvablySafe/provably-safe-online-system-identification\|Zhang-arXiv2025-ProvablySafe]] | Provably-Safe, Online System Identification | 2025 | arXiv | [Project page](https://roahmlab.github.io/OnlineSafeSysID/) | B+C: 「安全制約付きオンライン同定」と「条件数ベース励起」クラスタを橋渡し。実用需要の裏付けと RQ1 回答の一角 |
| 10 | [[papers/Park-arXiv2023-Object-Aware_Impedance_Control/object-aware-impedance-control-for-human-robot-collaborative-task-with-online-object-parameter-estimation\|Park-arXiv2023-Object-Aware_Impedance_Control]] | Object-Aware Impedance Control for Human-Robot Collaborative Task with Online Object Parameter Estimation | 2023 | arXiv | — | B+C: 「ヌルスペース摂動」と「HRI オンライン推定」クラスタを橋渡し。本サーベイが確認した中で task+励起の重畳という設定に最も近い公表例であり、novelty gap 論証の最重要参照点 |

（`Code` 列は各 hub の `Repository` frontmatter から取得。未公開、実装非公開は `—`。）

## Paper Catalogue

非ハブ論文を検索クラスタごとに列挙する。書誌情報は [[main.md|literature/references/main.md]] を参照。

### 励起軌道最適化の効率化、拡張（Fourier 系譜の周辺）

古典的フーリエ級数励起の効率化、拡張を扱うが、いずれも task drift の重畳は分析対象にしていない。

1. [[Ayusawa2017_condition]](../references/main.md#ayusawa2017_condition) — Ayusawa, Rioux, Yoshida, Venture & Gautier, "Generating Persistently Exciting Trajectory Based on Condition Number Optimization" (ICRA 2017)
   — 条件数の解析的勾配を直接計算し、大規模自由度系（ヒューマノイド HRP-4）向けに励起軌道最適化を効率化。
2. [[AbuDakka2017_comparison]](../references/main.md#abudakka2017_comparison) — Abu-Dakka & Díaz-Rodríguez, "Comparison of Trajectory Parametrization Methods with Statistical Analysis for Dynamic Parameter Identification of Serial Robot" (IROS 2017)
   — フーリエ級数と Schroeder Phased Harmonic Sequence を組み合わせ、初期値設計と計算時間を改善する軌道パラメータ化比較研究。
3. [[HuangS2025_analytical]](../references/main.md#huangs2025_analytical) — S. Huang, Li, Zhou & Duan, "An Analytical Approach for Dealing With Explicit Physical Constraints in Excitation Optimization Problems of Dynamic Identification" (TRO 2025)
   — 物理制約（初期条件ゼロ、関節角度、速度、加速度範囲）を反復探索でなく決定論的パラメータ化操作で満たし、実行可能解達成率100%、1桁の効率化を達成。
4. [[Huang2025_hydraulic]](../references/main.md#huang2025_hydraulic) — W. Huang, Chen, Zhang, Cheng, Ding, Zhang & Xu, "A Sequential Approach for Accurate Parameters Identification of Heavy-Duty Hydraulic Manipulators Ensuring Physical Feasibility" (RA-L 2025)
   — 低速動作しかできない重機油圧マニピュレータの観測行列条件数悪化を、パラメータ群ごとの逐次励起で緩和。条件数悪化の原因が速度、トルクのハード制約である点で、本プロジェクトの task drift 機構とは異なる。
5. [[Wang2025_insitu]](../references/main.md#wang2025_insitu) — C. Wang et al., "An In-Situ Excitation Trajectory Optimizer for Industrial Robots in Constrained Space with Human Collaboration" (ICIRA 2025 Proceedings)
   — 障害物、作業空間制約下でのフーリエ励起軌道最適化（差分進化 + 境界ペナルティ関数）。空間的な衝突回避制約が主眼で task drift の重畳は扱わない。全文 paywall のため書誌情報、要約は間接確認（Methodology 参照）。
6. [[Tian2024_virtualconstraints]](../references/main.md#tian2024_virtualconstraints) — Tian, Huber, Mower, Han, Li & Duan, "Excitation Trajectory Optimization for Dynamic Parameter Identification Using Virtual Constraints in Hands-on Robotic System" (ICRA 2024)
   — co-manipulation ロボットの自己衝突回避を重視した励起軌道最適化の計算効率化。

### タスク実行中のオンライン同定

大振幅励起そのものを避けるか、オンライン推定器、コスト統合によってタスクと同定を両立させる。

7. [[Foster2024_locomanipulation]](../references/main.md#foster2024_locomanipulation) — Foster, McCrory, DeBuys, Bertrand & Griffin, "Physically Consistent Online Inertial Adaptation for Humanoid Loco-Manipulation" (IROS 2024)
   — ロコマニピュレーション中に物理整合拡張カルマンフィルタで慣性パラメータをオンライン推定。励起軌道設計を伴わないオブザーバベース手法。
8. [[Mori2025_activelearning]](../references/main.md#mori2025_activelearning) — Mori, Ayusawa & Venture, "Safe Data Acquisition for Inertial Parameter Identification by Expanding the Motion Space" (RA-L 2025)
   — 非固定ベースマニピュレータのバランス制約下で反復的に励起動作を生成し条件数を改善。励起動作自体が制約を満たす設計で、別途の task 軌道への重畳は扱わない。
9. [[Cho2024_recursive]](../references/main.md#cho2024_recursive) — Cho, Lee & Shin, "Recursive Least Squares with Log-Determinant Divergence Regularisation for Online Inertia Identification" (ICRA 2024)
   — 脚式ロボットの慣性パラメータをオンライン逐次最小二乗で推定。物理整合正則化が主眼で励起軌道設計は扱わない。
10. [[Duan2023_trajectorygen]](../references/main.md#duan2023_trajectorygen) — Duan, Wang, Romeres, Koike-Akino & Orlik, "Trajectory Generation for Online Payload Estimation of Robot Manipulators" (CASE 2023)
    — task/環境で規定された初期関節配置から最適振幅を教師あり学習で即座に生成する高速軌道生成手法。
11. [[Nadeau2022_fast]](../references/main.md#nadeau2022_fast) — Nadeau, Giamou & Kelly, "Fast Object Inertial Parameter Identification for Collaborative Robots" (ICRA 2022) *(既存深読みノートあり — [[papers/Nadeau-ICRA2022-FastObjectInertial/fast-object-inertial-parameter-identification-for-collaborative-robots|deep read]])*
    — cobot の低速動作制約による低SNRに対応し、大振幅励起を要求しない高速慣性パラメータ同定。
12. [[Nadeau2023_partseg]](../references/main.md#nadeau2023_partseg) — Nadeau, Giamou & Kelly, "The Sum of Its Parts: Visual Part Segmentation for Inertial Parameter Identification of Manipulated Objects" (ICRA 2023) *(既存深読みノートあり — [[papers/Nadeau-ICRA2023-SumOfItsParts/the-sum-of-its-parts-visual-part-segmentation-for-inertial-parameter-identification-of-manipulated-objects|deep read]])*
    — 物体形状のパーツ分割により stop-and-go 動作のみで全慣性パラメータを復元。
13. [[Hu2025_adaptive]](../references/main.md#hu2025_adaptive) — Hu, Zachariah, Wigren & Stoica, "Adaptive Experiment Design for Nonlinear System Identification with Operational Constraints" (arXiv 2025) *(既存深読みノートあり — [[papers/Hu-arXiv2025-Adaptive_Experiment_Design/adaptive-experiment-design-for-nonlinear-system-identification-with-operational-constraints|deep read]])*
    — ロボット非依存の一般制御理論。運用制約下でのレシーディングホライズン型適応実験計画で、task/運用制約と励起設計のトレードオフを扱う一般理論として参照。

### 識別可能性、パラメータ空間の構造

励起軌道設計そのものではなく、観測行列、パラメータ空間の構造的性質を扱う基盤研究。

14. [[Wensing2024_geometric]](../references/main.md#wensing2024_geometric) — Wensing, Niemeyer & Slotine, "A Geometric Characterization of Observability in Inertial Parameter Identification" (IJRR 2024)
    — 運動学木の慣性パラメータ識別可能性を近似なしに幾何学的に特徴づけ。構造的識別可能性の分析であり、観測行列の条件数（本プロジェクトの分析対象）とは補完的だが異なる軸。
15. [[Wensing2017_lmi]](../references/main.md#wensing2017_lmi) — Wensing, Kim & Slotine, "Linear Matrix Inequalities for Physically-Consistent Inertial Parameter Identification" (RA-L 2017) *(既存深読みノートあり — [[papers/Wensing-RAL2017-LMIPhysicalConsistency/linear-matrix-inequalities-for-physically-consistent-inertial-parameter-identification-a-statistical-perspective-on-the-mass-distribution|deep read]])*
    — 質量分布の物理的整合性を凸制約（LMI）として定式化。

### Foundational Works（サーベイ論文）

16. [[Leboutet2021_survey]](../references/main.md#leboutet2021_survey) *(既存深読みノートあり — [[papers/Leboutet-ApplSci2021-Inertial_Parameter_Identification/inertial-parameter-identification-in-robotics-a-survey|deep read]])* — Leboutet, Roux, Janot, Guadarrama-Olvera & Cheng, "Inertial Parameter Identification in Robotics: A Survey" (Applied Sciences 2021)
    — 慣性パラメータ同定分野の包括的レビュー。励起軌道設計への言及は限定的（41ページ中1段落程度）。

## 参考文献

上記で `[[Key]]` 形式により言及した論文の一覧。書誌詳細は [[literature/references/main.md]] を参照。

- Ayusawa2017_condition, AbuDakka2017_comparison, HuangS2025_analytical, Huang2025_hydraulic, Wang2025_insitu, Tian2024_virtualconstraints, Foster2024_locomanipulation, Mori2025_activelearning, Cho2024_recursive, Duan2023_trajectorygen, Nadeau2022_fast, Nadeau2023_partseg, Hu2025_adaptive, Wensing2024_geometric, Wensing2017_lmi, Leboutet2021_survey, Swevers1997_excitation, Park2006_fourier, Kubus2007_online, Kubus2008_rtls, Bonnet2016_dance, Lee2021_optimal, Lee2024_modern, Albee2022_rattle, Zhang2025_provably, Park2023_nullspace

## Abbreviation Glossary

| Abbreviation | Full name | First occurrence |
|---|---|---|
| PE | Persistent Excitation（持続的励起） | Terminology and Background |
| RTLS | Recursive Total Least-Squares（逐次総最小二乗） | Terminology and Background |
| EKF | Extended Kalman Filter（拡張カルマンフィルタ） | Research Landscape Overview |
| LMI | Linear Matrix Inequality（線形行列不等式） | Foundation |
| FIM | Fisher Information Matrix（フィッシャー情報行列） | Thesis |
| HRI | Human-Robot Interaction（人間ロボット相互作用） | RQ1 回答 |
| PINN | Physics-Informed Neural Network（物理情報ニューラルネットワーク） | 重点確認4件 |
| SPHS | Schroeder Phased Harmonic Sequence | Paper Catalogue |
| RA-L | IEEE Robotics and Automation Letters | Hub Papers |
| TRO | IEEE Transactions on Robotics | Hub Papers |
| IJRR | International Journal of Robotics Research | Paper Catalogue |
| ICIRA | International Conference on Intelligent Robotics and Applications | 重点確認4件 |

## Survey Methodology

### Frame

- **Core topic**: task 由来の大振幅ベース運動が重畳した状況下での、慣性パラメータ同定のための励起軌道設計
- **Depth**: focused（ユーザー指定）。target 20–40本、着地点は26本（中央値付近）
- **Research Questions**: ユーザーが明示的に指定した RQ1–RQ3 をそのまま採用（auto-derive はスキップ）
- **Inclusion criteria**: 査読済み論文 + arXiv 等の主要プレプリント、剛体慣性パラメータ同定、システム同定、励起軌道設計、タスク制約下の動作計画の領域内、英語論文
- **Exclusion criteria**: 本プロジェクトの周辺サブテーマ（プッシュ操作力学、視覚ベース物理推定、タクタイルセンシング）に属するが本トピックと無関係な既存プロジェクト文献（Hogan-WAFR2016、Lynch-IJRR1996、Wang-IROS2020、Xie-CVPR2024、Markovsky-SigProc2007）は除外
- **survey_slug**: auto-suggest（`task-drift-excitation-trajectory-design`）をそのまま採用（Auto-execution Mode フォールバック）
- **既知略語**: 収集せず、初出展開ルールで対応

### Map

| Search angle | Source(s) | Sample query | Results |
|---|---|---|---|
| 既知7論文の確認（再調査不要） | プロジェクト既存ノート | — | 7 |
| 4件の重点確認論文の書誌確認 | OpenAlex + arXiv + WebFetch + WebSearch | Yun/Park 2023, In-Situ LNEE/ICIRA 2025, Abu-Dakka 2017, Ayusawa 2017 | 4 |
| 前方引用スノーボーリング（Swevers, Lee+2021, Bonnet, Leboutet, Annual Reviews, Ayusawa） | OpenAlex API（`filter=cites:`） | 6 論文の前方引用を年代、被引用数でスクリーニング | 約150件から選別 |
| task-priority / null-space excitation 探索 | サブエージェント（researcher, Sonnet） + 補完 WebSearch（メインループ） | task-priority redundancy excitation identification 等 | 補完 WebSearch では新規の高確度候補なし（下記注記） |
| online / dual-control 探索 | サブエージェント（researcher, Sonnet） | online in-motion inertial identification, dual control persistent excitation 等 | 未返却（下記注記） |
| persistent-excitation理論 / drifting-base 探索 | サブエージェント（researcher, Sonnet） + 補完 WebSearch（メインループ） | persistent excitation theory constraints, drifting base non-periodic excitation 等 | 補完 WebSearch では新規の高確度候補なし（下記注記） |
| 既存プロジェクト文献の再利用 | `literature/papers/` 既存フォルダ精査 | — | 5（Albee, Zhang, Nadeau×2, Wensing2017, Hu2025 — Albee/Zhangはhub、他はnon-hub） |

- 総 Map 数: 26
- 重複除去: 0件（snowballing、直接検索で重複候補が生じなかった）
- I/E 基準による除外: 5件（プロジェクト内の無関係サブテーマ論文。上記 Exclusion criteria 参照）
- DOI 解決: 全26論文が DOI または arXiv ID を保持（OpenAlex 経由で確認）

**注記（3件の補完検索サブエージェントについて）**: skill の Search Strategy に従い、既知7論文＋4件の重点確認論文の直接調査に加えて、task-priority/null-space、online/dual-control、persistent-excitation理論/drifting-base の3軸で補完探索サブエージェントを並列 dispatch した。しかし、セッションの実効作業時間内に3件のうち2件（task-priority/null-space、persistent-excitation理論/drifting-base）はメインループ側の補完 WebSearch で置き換え、1件（online/dual-control）は結果が返却されないまま報告書作成の締切に至った。メインループ自身による OpenAlex 前方引用スノーボーリング（6論文起点、約150件から選別）と重点確認4件の精読が Map の主要な情報源であり、26本という着地点は focused スコープの target 20–40本の中央値に近い。3件の補完エージェントが後日追加の候補を返す場合、Concept Matrix と Paper Catalogue への追記が必要になる可能性がある（未収束のまま報告することの limitation として明記する）。

### Hub Selection

- 選定基準: B（カテゴリ橋渡し性+影響力）+ C（synthesis 主役性）のユニオン
- 候補: 26論文中、教科書、survey/review型（Leboutet2021）を機械的に除外した25論文
- 最終ハブ: 10本（[[papers/Lee-AR2024-Robot_Model_Identification/robot-model-identification-and-learning-a-modern-perspective|Lee-AR2024]] はレビュー論文だが gap 軸の主役性により例外的に hub 化。理由は Hub Papers 節の Why hub 列に明記）
- PDF 取得: 10本中9本成功（Park-arXiv2023 は新規取得、deep read 実施。他9本は既存プロジェクトの `literature/papers/` に PDF、deep read ノートが既に存在し再利用）
- 取得失敗: なし（In-Situ Excitation Trajectory Optimizer は当初 hub 候補として検討したが、paywall で全文取得不能なため hub 化を見送り、非hub Paper Catalogue エントリとして書誌情報のみ収録）

### Verify

- **Park-arXiv2023 の新規深読みノートのクロス検証**: skill の規定に従い、別エージェントによる独立クロス検証を dispatch したが、セッションの実効作業時間内に結果が返却されなかった。代替として、メインループ自身が PDF 全文（11ページ）を直接精読した上でノートを作成しており（`/paper-summary` Step 6 相当の自己検証と同水準）、主要な数値主張（質量推定 1.131kg、真値 2.23kg、有効質量 1.115kg、誤差 0.016kg、重心推定 [8,−7,44]mm、誤差 [−8,7,−5]mm、摂動振幅 0.2 rad/s、周波数 0.4 Hz、著者名 Park, Shin, Kim）は全て PDF 本文中の該当箇所（TABLE I、Fig.4、§IV-A 本文）と本サーベイ作成者自身が直接照合済みである。独立エージェントによる第三者検証は完了していない点を limitation として明記する
- 既存9本（Swevers, Park2006, Kubus×2, Bonnet, Lee2021, LeeAR2024, Albee, Zhang）の深読みノートはプロジェクトの過去セッションで生成済みのものを再利用し、本サーベイでは再クロス検証していない。Executive Summary、主要数値を本サーベイ側で読み合わせ、内容の一貫性を確認した上で使用した
- **reference-verify**: 新規追加した16論文（既存7論文と既存流用5論文を除く）について、OpenAlex API（Crossref/PubMed等を裏付けとする学術アグリゲータ）による DOI、著者名、タイトルの直接照合をメインループが実施した。IEEE/Sage 等の publisher ページへの直接 HTTP アクセスは bot 対策により 202/403 を返すことが多く、doi.org への直接解決確認は限定的だったため、OpenAlex のメタデータ一致（タイトル、著者、年、venue の4項目一致）を存在確認の主手段とした。加えて、独立検証用に reference-verify サブエージェントを dispatch したが、こちらもセッションの実効作業時間内に結果が返却されなかった
- 除外: 0件（全26論文が OpenAlex 上でタイトル、著者、DOI/arXiv IDの一致を確認できたため、ハルシネーションと判定した論文はない）
- **既知の limitation**: 上記の通り、3件の補完検索サブエージェント、Park-arXiv2023 のクロス検証エージェント、reference-verify サブエージェントの計5件が、報告書作成時点で結果未返却だった。いずれもメインループ自身による直接検索、直接精読、OpenAlex 照合で代替したが、独立エージェントによる第三者チェックという多層防御の一部が欠けた状態での納品である。ユーザーが追加確認を希望する場合、これらのサブエージェント結果を後日統合する余地がある
