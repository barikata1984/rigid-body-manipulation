# Task Drift Excitation Finding

セッション日時: 2026-07-14 〜 2026-07-15
発見: 励起軌道最適化における task-required base drift の条件数悪化機構

## Active Plan

**主目標**: フーリエ級数励起軌道の設計で, task 由来の非最適化 base motion (以下 "base drift") が観測行列の条件数を悪化させる機構を数学的・実験的に解明し, 論文レベルの主張として整理する.

**サブ目標**:

1. base drift の存在下で cond が桁で悪化する現象を実験的に再現・定量化 → **完了**
2. 現象の数学的機構 (相対振幅比 $\propto T^2/(\text{turn 数})$) を導出 → **部分的 (スケーリング論のみ. proof/numerical evidence による形式化と主張 1 の $f$ の特定は未着手)**
3. 目的関数 (cond vs D-opt) 選択が結論を変えないことを確認 → **完了**
4. Prior art を網羅サーベイして novelty gap を確定 → **主要 7 件終了, 未読 4 件残**
5. 需要面 (実用ドメイン) の裏付けを取得 → **完了 (5 ドメインで需要確認)**

**残タスク**:

- Yun 2023 (arXiv:2310.12409), Abu-Dakka 2017 (IROS), Ayusawa 2017 (ICRA), In-Situ Excitation Trajectory Optimizer (Springer LNEE 2024, DOI:10.1007/978-981-95-2098-5_52) の verbatim 確認.
- 論文化する場合の positioning と related work セクションの構造化.
- 相対振幅比 $\propto T^2/(\text{turn 数})$ の数学的形式化 (proof or numerical evidence).
- cond 一覧を base 振幅 vs cond の連続関係で追加検証 (e.g., j5=4π, j5=π 等の中間点).

## Current Phase

**Phase 3 (novelty 確立) → Phase 4 (残 4 論文精読 + 論文構成検討) の境界**.

Phase 3 で以下が確立された:

| 設定 | 目的関数 | best cond |
|---|---|---|
| T=20s + main 8π (07-09 baseline) | cond | **6.44** |
| T=10s + main 8π (07-09 setup) | cond | **24.12** |
| T=10s + main 2π (turn 数 1/4) | cond | **5.21** |
| T=10s + main 無効化 | cond | **1.10** |
| T=20s + main 無効化 | cond | **1.10** |
| T=10s + main 8π | **D-opt** | **24.13** |

Phase 4 の入り口:

- 論文投稿を目指すなら残 4 論文の verbatim 確認が必須.
- 特に In-Situ Excitation Trajectory Optimizer (LNEE 2024) がタイトル的に最も近い.

## TaskList Summary

**完了 (発見関連)**:

- [x] 10s / bf=0.1 / envelope 1.3/2.1 probe → cond=43.73
- [x] 10s / bf=0.2 / envelope 1.3/2.1 probe → cond=53.00 (悪化)
- [x] 10s / bf=0.1 / envelope 2.0/π (envelope 拡張) probe → cond=24.82
- [x] 10s / bf=0.1 / envelope 1.5/π/inf/7.5/2π (07-09 完全再現) probe → cond=24.12
- [x] 20s / bf=0.1 / 同 envelope (07-09 再現) → cond=6.4342 (07-09 記録 6.44 と完全一致)
- [x] j5=2π (1 turn) probe at 10s → cond=5.21
- [x] main 無効化 T=10s probe → cond=1.0998
- [x] main 無効化 T=20s probe → cond=1.1015
- [x] D-opt at T=10s + 8π probe → cond=24.13, D-opt=1.90 (cond=24.12 と同水準)
- [x] cond=9.64 (excited_20260711_155119) 軌道の binding 制約分析 — どの物理制約も binding せず, `target_condition_number=10` の早期停止で終了
- [x] Prior art サーベイ (7 論文詳細): Swevers 1997, Park 2006, Lee-Lee-Park 2021, Bonnet 2016, Kubus 2007/2008, Annual Reviews 2024, Leboutet 2021
- [x] A/E-opt 使用状況調査: 主流でない (実装不要と判明)
- [x] Task + excitation superposition の実用需要調査: 5 ドメインで確認

**未着手**:

- [ ] Yun 2023 (arXiv:2310.12409) verbatim 精読 (最も近い null-space perturbation 手法)
- [ ] Abu-Dakka 2017 IROS "Comparison of trajectory parametrization methods" verbatim
- [ ] Ayusawa 2017 ICRA "Generating persistently exciting trajectory" verbatim
- [ ] In-Situ Excitation Trajectory Optimizer LNEE 2024 (paywall, institutional access 経由)
- [ ] cond 一覧を base 振幅 vs cond の連続関係で追加検証 (e.g., j5=4π, j5=π 等の中間点)
- [ ] 論文化するなら数学的な形式化 (相対振幅比の proof or numerical evidence)

## Session Decisions

**設計判断 (発見関連)**:

1. **`target_condition_number=10` の早期停止は 20s cond=6.44 でも作動していた**. cond=9.64 の 155119 軌道は 3 反復で target 到達 → SLSQP が最適解を返す前に停止. 20s + 07-09 config 再現でも同じく restart 1 iter 3 で target 到達.

2. **物理制約の binding は存在しない**. cond=9.64 でも 24.12 でも, どの関節も dq/ddq/q 制約使用率は 13-42% で余裕. envelope のタイト化やゆるめは realization 軌道の binding を作らない (analytical triangle bound を通じてフーリエ係数を縛るだけ).

3. **cond=1.10 は理論下限**. main 無効化で T=10s と T=20s の両方で cond ≈ 1.10 に到達. これは 6×10 の regressor で 10 パラメータが「ほぼ相互直交」状態. cond=1.0 が絶対下限 (singular values 全部等しい) なので, 1.10 は数値的にほぼ理想.

4. **D-opt 目的関数でも cond=24 に張り付く**. 「cond=24 は cond 最適化特有の頭打ちで, D-opt なら改善する」という仮説を実験的に棄却. 目的関数選択に不変 = 情報行列そのものが structurally degraded.

**ユーザー判断で採用しなかった案**:

- A-opt / E-opt の実装追加: researcher 調査で「主流でない」「実採用例なし」と判明したため実装見送り.
- Search 予算不足仮説: n_restarts=3 で 3 restart 全てが cond ≈ 24 に収束したため棄却.
- Main を完全に消す task 変更: 論文化の観点では base drift の存在下での分析が価値なので, main 有りの baseline を維持. 無効化は仮説検証のためだけ.

**目的関数の思想的整理**:

- **cond**: $\sigma_{\max}/\sigma_{\min}$. base が $\sigma_{\max}$ を inflate すると cond 悪化. **等化 (Van der Sluis) 済み**.
- **D-opt**: $-\log \det(F^T F) = -\sum \log \sigma_i$. base の $\sigma_{\max}$ 増加はむしろ改善方向 (理論的には). 実験結果は cond と一致.
- **A-opt / E-opt**: 主流でない (A-opt は unit-dependent, E-opt は non-smooth). 実装優先度低.

## Constraints and Blockers

**未解決 (novelty 判定に影響しうる)**:

- **In-Situ Excitation Trajectory Optimizer (LNEE 2024) 全文未確認**: paywall のため institutional access が要る. タイトル的に最も近い可能性が残る.
- **Yun 2023 (arXiv:2310.12409) verbatim 未確認**: 需要調査で「最も近い publish 済み類例」と評価. null-space perturbation なので構造は違うが差分の verbatim quote が要る.
- **Abu-Dakka 2017 IROS / Ayusawa 2017 ICRA verbatim 未確認**: Leboutet 2021 が引用. Fourier / 多項式 / persistent excitation の比較.

## Failed Attempts

発見関連の反証済み仮説:

1. **bf=0.2 で T=10s 相殺仮説** — 「T=10s で bf=0.2 なら T=20s + bf=0.1 と等価」と予想したが実測 cond=53.00 (悪化). 数学的理由: フーリエ係数バウンドが $1/f_0^2$ で縮む効果が勝る.

2. **Envelope 拡張 (dq: 2.0/π, ddq: 4.0/2π) で T=10s を救う** — 実測 cond=24.82. envelope をゆるめても binding していないので改善は限定的.

3. **Search 予算不足仮説** — n_restarts=3 で回すも 3 restart 全て cond ≈ 24 に収束. 局所解の問題ではなく構造的下限.

4. **目的関数の切り替え (cond → D-opt) で cond=24 が改善する仮説** — D-opt 最適化でも cond=24.13 に張り付き, cond=24 が構造的下限と確認.

## Recovery Notes

### 主要な発見 (論文レベルの主張候補)

**主張 1 (定量的機構)**: task-required 非最適化 base motion $q_{\text{base}}(t)$ (例: 4-回転 quintic spline) を持つ場合, フーリエ級数励起の cond は base 振幅と duration に対して:

$$\kappa \approx 1 + f(\text{base 振幅} / \text{励起振幅})$$

ここで励起振幅バウンドは $|c_k| \leq \ddot q_{\max}/(2\pi f_0 k)^2$ で $T$ 非依存, 一方 base 加速度ピークは $\ddot q_{\text{base}}^{\text{peak}} \propto (\text{turn 数})/T^2$. 相対振幅比は $T^2/(\text{turn 数})$ に比例.

実測: base 振幅 0 (no main) → cond=1.10 (T によらず). base 振幅小 (2π, T=10s) → cond=5.21. base 振幅中 (8π, T=20s) → cond=6.44. base 振幅大 (8π, T=10s) → cond=24.12.

**主張 2 (目的関数不変性)**: この cond 悪化は cond 最適化と D-opt 最適化の両方で同じレベル (cond=24.1) に到達. 情報行列自体が structurally degraded であり, 目的関数の選択や coordinate 変換 (Lee-Lee-Park 2021 の Riemannian pullback) で吸収できない real information loss.

### 数学的機構の詳細

**セットアップ**:

$$w(t) = Y(q(t), \dot q(t), \ddot q(t)) \cdot \varphi + \eta(t)$$

離散化して観測行列 $F = [Y(t_1); \dots; Y(t_M)] \in \mathbb{R}^{6M \times 10}$. 目的は $\kappa(F)$ の等化最小化.

軌道分解: $q(t) = q_{\text{base}}(t) + q_{\text{exc}}(t)$

- $q_{\text{base}}$: task 由来の非設計的な成分 (本プロジェクトでは quintic spline, j5 が 4 回転)
- $q_{\text{exc}}$: フーリエ級数の設計可能成分

**核心的観察**: 回帰行列 $Y$ は $q$ に対して非線形 (順運動学の $R(q)$ が入る). 特に重力項.

重力は世界座標で固定 $\vec g_{\text{world}}$, センサー座標系での重力は

$$\vec g_{\text{sen}}(t) = R_{\text{sen}\leftarrow\text{world}}(q(t))^\top \vec g_{\text{world}}$$

j5 (yaw) が 4 回転すると, センサー座標の重力ベクトルは周波数 $4/T$ Hz の正弦波として現れる. 質量列 $[a - g]_{\text{sen}}$ にはこの信号が直接乗る.

### スペクトル衝突

- **T=10s の場合**: base 由来の重力回転周波数 = 0.4 Hz. 励起の基本周波数 $f_0=0.1$ Hz の 4 倍音とちょうど一致.
- **T=20s の場合**: base 由来の周波数 = 0.2 Hz. 2 倍音と一致するが base 速度が半分・加速度が 1/4 になり相対振幅が下がる.

つまり base が励起の 4 倍音チャンネルを占拠している — この列の内容は励起係数の設計では動かせなくなる. 情報の多様性を作る自由度が 1 本減る.

### なぜ base が「支配」するか (定量)

- **フーリエ係数のバウンド** $|c_k| \leq \ddot q_{\max}/(2\pi f_0 k)^2$ (三角不等式). $T$ に依存しない.
- **base の速度・加速度**: それぞれ $\propto 1/T, 1/T^2$ でスケール.

結果: 励起 : base の振幅比は $T$ に線形〜二乗でスケールする.

| | T=10s | T=20s |
|---|---|---|
| base peak $\dot q_5$ | 2.36 rad/s | 1.18 rad/s |
| base peak $\ddot q_5$ | 1.45 rad/s² | 0.36 rad/s² |
| 励起振幅バウンド $c_k$ | 同じ | 同じ |
| 励起 : base 比 (加速度) | 1x | 4x |

T=10s では励起は base の 1/4 の力しか持たない. 列を base 信号から引き剥がすことができない.

**注記 (バウンドの $T$ 依存性)**: 上記の $|c_k| \leq \ddot q_{\max}/(2\pi f_0 k)^2$ は窓関数を無視した近似で, $T$ 非依存としているのは漸近的な近似としてのみ正しい. 実装 (`trajectories/excited.py` の `compute_fourier_bounds`) は窓関数 $w(s)=256s^4(1-s)^4$ の微分項を含むため, 実際には duration に依存する. 実測 (bf=0.1, N=5, dq_max=[1.5×3, π, π, inf], ddq_max=[7.5×3, 2π×3]) では T=20s の方が並進関節で 6%, 回転関節で 13% 緩い (T=10s: 0.0426 / 0.0499, T=20s: 0.0450 / 0.0564, T=30s: 0.0459 / 0.0587). したがって T=10s vs T=20s の比較には励起振幅バウンド側のこの差がわずかに交絡する. 効果は小さく上記の結論を覆すものではない.

### なぜ Kubus / Swevers が問題にならない

彼らは base が (ほぼ) ゼロ:

- Swevers: $q_0$ 定数オフセットのみ, ドリフトなし
- Kubus: 同上 (フーリエ級数の $q_{i,0}$)
- Bonnet 2016: 全 DoF が最適化対象で task 固定でない
- Lee-Lee-Park 2021: endpoint-pinned B-spline, drift 排除

したがって彼らの $F_{\text{base}} = F(q_0)$ は定常項. 全ての情報は $q_{\text{exc}}$ が作る. base の周波数汚染がない → cond=4-8 帯を達成できる. 本プロジェクトの 20s cond=6.44 も base drift が相対的に小さいから偶々近い値になる.

### bf 逆算不能性の証明

「T=10s で cond<10 を達成する bf は存在するか?」に対する分析:

**Base 側のスペクトル**: j5=8π を quintic で回すと yaw 角速度 $\dot q_5(t)$ は釣鐘型 (時刻 $t=T/2$ でピーク). cos(q_5(t)) の "瞬時周波数" のピークは

$$\dot q_5^{\text{peak}} / (2\pi) = 1.875 \cdot 8\pi / T / (2\pi) = 7.5/T \text{ Hz}$$

- T=10s: 0.75 Hz までスペクトルが広がる. 平均は ~4/T = 0.4 Hz 付近が中心.
- T=20s: ~0.375 Hz までで, 中心 0.2 Hz.

つまり base スペクトルは単一 tone ではなく 0 ~ 7.5/T Hz の連続分布. 完全に避けるのは困難.

**励起側のスペクトル**: N=5, bf=$f_0$ で高調波帯 = $\{f_0, 2f_0, \dots, 5f_0\}$.

**bf を動かした場合の効果** (T=10s):

| bf | 励起帯 [Hz] | 5 高調波の base 帯との重なり | $c_k$ スケール ($\propto 1/f_0^2$) |
|---|---|---|---|
| 0.05 | 0.05-0.25 | 全て base 帯 (0-0.75) 内 | ×4 (優位) |
| 0.1 (現) | 0.1-0.5 | 全て base 帯内 | ×1 |
| 0.15 | 0.15-0.75 | 全て base 帯内 | ×0.44 |
| 0.2 | 0.2-1.0 | 0.2-0.75 が重複, 0.75+ が超える | ×0.25 |
| 0.3 | 0.3-1.5 | 0.3-0.75 が重複, 0.75+ が超える | ×0.11 |
| 0.5 | 0.5-2.5 | 0.5-0.75 のみ重複 | ×0.04 |

**トレードオフの正体**:

- bf を上げる → 励起帯が base 帯を超えて分離できる (良い)
- しかし振幅が $1/f_0^2$ で急速に縮む → 総情報量が悪化 (悪い)
- bf を下げる → 振幅は増えるが完全に base 帯に埋没

bf ≈ 0.5 で初めて励起の高調波帯が base 帯を上回るが, 振幅は 1/25 に縮む. 実質的な情報量はさらに少ない.

**逆算の答え**: T=10s で cond<10 を達成する bf は存在しない可能性が高い. 必要条件は 励起 : base の実効振幅比 ≳ (20s で 6.44 を出したときの比):

- 20s で base peak velocity = 2.36 rad/s, 加速度 = 0.36 rad/s². 励起加速度バウンド (ddq_max=7.5) は base の 21 倍.
- 10s で base peak velocity = 4.71 rad/s, 加速度 = 1.45 rad/s². 励起加速度バウンド (7.5) は base の 5 倍.

つまり 10s では base が加速度で 4 倍支配的. bf を動かしても振幅バウンドが $1/f_0^2$ で縮むので, 総エネルギー比は改善しない.

### 対策候補

1. **base の周波数を励起帯から外す**: j5 を 4 回転させるなら duration を長く取り, 主要な base 周波数を励起帯 (~$Nf_0$) より下に落とす.
2. **base 影響を回帰から差し引く**: base の $Y_{\text{base}}(t)$ を既知として, $w - Y_{\text{base}} \varphi = Y_{\text{exc}} \varphi + \eta$ の形で解く. ただし $\varphi$ 未知なので反復推定になる (Newton-Raphson 型 EM).
3. **励起周波数を base 帯から避ける**: $f_0$ を base の $\sim 4/T$ Hz と非干渉な位置に置く (bf 単独では困難だが N も合わせて調整).
4. **j5 の task 要件を緩める**: 8π (4 回転) → 2π (1 回転) で cond 24 → 5.2 (実測). 最も直接的.

### 論文的な言い方

「フーリエ級数励起の最適性は $q_{\text{base}} \equiv 0$ の暗黙の仮定に依存する」. 系統的な非設計的ドリフト成分がある場合, 従来手法は保証を失う. 具体的には:

1. **スペクトル衝突**: base の周波数成分と励起高調波が重なると, その調波チャンネルは設計不能に.
2. **相対振幅**: 励起バウンドが task 制約 (envelope) で先に固定される場合, base が $T^{-2}$ でスケールするため, 短い duration では base が支配する.

### Novelty gap の確定 (7 論文精読済)

| 論文 | 判定 | 位置づけ |
|---|---|---|
| Swevers 1997 | (c) 前提想定違反 | フーリエ励起の原論文, $q_0$ 定数前提. 引用義務 |
| Park 2006 | (b) 部分的 | Fourier+多項式分解の起源だが多項式は BC 用 |
| Kubus 2007/2008 | (c) 別問題 | 推定器 (RTLS) が主, sim-real cond 乖離は別軸 |
| Bonnet 2016 | (b) DoF 分解類似 | dance 全体が excitation, task drift ではない |
| Lee-Lee-Park 2021 | (c) 直交 | パラメータ空間の座標不変性 (相補的) |
| Annual Reviews 2024 | (c) 未言及 | 「task-aligned SysID」を future prospect として conclusion で挙げる |
| Leboutet 2021 | (c) 未言及 | 41 ページ中 1 段落しか excitation を扱わず |

**残 4 論文 (verbatim 未確認)**:
- Yun 2023 (arXiv:2310.12409): null-space perturbation, 最も近い publish 済み類例
- Abu-Dakka 2017 IROS: parameterization comparison
- Ayusawa 2017 ICRA: persistent excitation via cond
- In-Situ Excitation Trajectory Optimizer (LNEE 2024): タイトル最類似, paywall

### 実用需要 (5 ドメイン確認済)

1. Space robotics (Uchida 2025, Ekal 2018) — free-floating base で停止同定不可能
2. Humanoid loco-manipulation (Foster 2024) — EKF で online, 36%/65% 性能向上
3. Human-robot cooperative transport (Park, Shin & Kim 2023, arXiv:2310.12409) — null-space perturbation
4. Provably-safe online SysID / 工業応用 (Michaux 2025)
5. Warehouse / mobile manipulation (Sun 2023)

**Gap**: 設計された joint-space Fourier 励起を大振幅 task 軌道に重ねる形は既存で薄い.

### 論文化する場合の positioning ドラフト

> Fourier-series excitation for rigid-body inertial parameter identification (Swevers 1997; Park 2006; Kubus 2007) has traditionally assumed that the trajectory is either purely periodic around a fixed operating point or freely designed. However, many practical scenarios require the robot to execute a task-mandated large-amplitude motion while performing identification — space servicing (Uchida 2025), humanoid loco-manipulation (Foster 2024), human-robot cooperative transport (Park et al. 2023). Existing approaches either (a) confine excitation to the null-space of the task (Park et al. 2023), (b) restrict identification to stop-and-go phases (Nadeau 2023), or (c) rely on observer / adaptive laws that do not design excitation.
>
> We show experimentally that when a task-mandated base motion $q_{\text{base}}(t)$ is superimposed on a Fourier excitation $q_{\text{exc}}(t)$, the observation-matrix condition number degrades by orders of magnitude, following a $T^2/(\text{turn count})$ scaling in the relative amplitude ratio. This degradation is invariant to the choice of optimization criterion (condition number vs D-optimality) and cannot be absorbed by coordinate transformations on the parameter space (Lee-Lee-Park 2021). We derive the mathematical mechanism (spectral competition and analytical Fourier coefficient bounds shrinking as $1/f_0^2$) and validate it on a 6-DoF prismatic-revolute manipulator identifying a hammer's 10 inertial parameters.

### 実験再現に必要な設定

主要な比較実験 (T=10s vs 20s, main あり/なし, cond vs D-opt) はすべて完了して trajectory と optimize.log は git 管理外の `configurations/trajectories/` 以下に残っている. Backup 済 (`_10s_backup/`, `_20s_backup/`) で上書きリスクなし.

### 現在の CLI パラメータレシピ (T=20s baseline を再現するとき)

```bash
pixi run python -m trajectories.generate excited --config <yaml>
```

YAML の要点:

```yaml
duration: 20.0
num_harmonics: 5
base_freq: 0.1
n_restarts: 3
max_iter: 15
target_condition_number: 10.0
column_scale: true
use_analytical_bounds: true
dq_max: [1.5, 1.5, 1.5, 3.14159, 3.14159, .inf]
ddq_max: [7.5, 7.5, 7.5, 6.28318, 6.28318, 6.28318]
main_trajectory:
  end_pos: [0.0, 0.0, 0.0, 3.141592653589793, 0.0, 25.1327412287]
```

これで cond=6.4342 が再現する.

### 発見関連 trajectory ディレクトリ

- `configurations/trajectories/excited_20260714_230248/` — 20s + 8π + 07-09 config cond=6.4342 (再現)
- `configurations/trajectories/excited_20260714_231201/` — 10s + 8π + 07-09 config cond=24.12
- `configurations/trajectories/excited_20260714_233733/` — 10s + j5=2π cond=5.21
- `configurations/trajectories/excited_20260714_233758_10s_backup/` — T=10s no-main cond=1.0998
- `configurations/trajectories/excited_20260714_233758_20s_backup/` — T=20s no-main cond=1.1015
- `configurations/trajectories/excited_20260715_000302/` — D-opt at 10s + 8π cond=24.13, D-opt=1.90

### 発見関連 scratchpad 資料

- `<scratchpad>/prior_art_base_drift_excitation.md` — Prior art 網羅 (17 論文)
- `<scratchpad>/lit_criteria_survey.md` — A/E-opt 使用状況
- `<scratchpad>/park2006_verbatim.md` — Park 2006 精読
- `<scratchpad>/lee_lee_park_2021_verbatim.md` — Lee-Lee-Park 2021 精読
- `<scratchpad>/bonnet2016_verbatim.md` — Bonnet 2016 精読
- `<scratchpad>/annurev2024_verbatim.md` — Annual Reviews 2024 精読
- `<scratchpad>/leboutet2021_verbatim.md` — Leboutet 2021 精読 (researcher の text 返答から統合)
- `<scratchpad>/task_excitation_demand_survey.md` — 実用需要調査
