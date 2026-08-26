# データセット生成手順

最終更新: 2026-08-26。

この文書は、剛体物体の慣性パラメータ同定と外観再構成を同時に学習させるためのデータセットを、
MuJoCo シミュレーションから生成する現行手順を記述する。すべての記述に、リポジトリ
`rigid-body-manipulation` 内のファイルパスと行番号を添える。行番号は 2026-08-26 時点の
作業ツリーのものである。

## 0. 全体像と用語

### 何のためのデータセットか

学習側 (別リポジトリ `pixi-wisp-container`) のモデルは、物体の質量分布を 3 次元格子上の場として
持ち、次の 2 つの損失で同時に学習する。

- 色の損失: 格子から描画した画像と、データセットの画像との差
  (`~/workspace/pixi-wisp-container/wisp/trainers/md_multiview_trainer.py:593-600`)
- 動力学の損失: 格子の質量分布から計算したレンチ (力 3 成分 + トルク 3 成分) と、
  データセットの計測レンチとの差 (同ファイル `:624-661`)

したがってデータセットには、**画像 + カメラ姿勢**と、**物体の運動と計測レンチ**の両方が必要である。
本手順はこの両者を、別々の軌道で実行した 2 つのシミュレーション run から作り、データセットの
段階で 1 つに合成する。

### なぜ 1 本の軌道ではなく合成なのか

2 つの損失は、軌道に対して異なる要求を持つ。色の損失は物体を全方位から均等に見る視点を要求し、
スプライン軌道がそのために設計されている。動力学の損失は慣性パラメータが計測レンチに現れる
運動を要求し、励起軌道がそのために設計されている。

従来の実装は、この 2 つを 1 本の軌道の中で両立させようとしていた。スプラインのベース軌道の上に
フーリエ励振を波形として加算する形である (`trajectories/excited.py:270` の `_build_trajectory`)。
この形では、加算された励振が視点の軌跡を乱し、物体を全方位から均等に見るという条件が保てなく
なる。外観再構成の側が損をする。

そこで、軌道を目的ごとに 1 本ずつ作り、実行も別に行い、データセットの段階で行ごとに束ねる方式に
変えた。この方式が成立する根拠は C.1 に示す学習側の構造にある。2 つの損失は同じ行に同居する
画像と動力学量を互いに参照しないので、同一行の両者が同じ時刻の同じ運動から来ている必要がない。

**未検証の補足**: 励振が視点を乱す機構について、次の説明を仮説として置いている。視点の全方位
カバレッジを作るのは物体の回転であり、併進は視線方向をほとんど変えない。併進が変えるのは
カメラと物体の距離と画面内位置であり、距離が変動すると物体の画素上の大きさが変わって実効解像度が
不均一になる。この機構は測定して確かめてはいない。確かめたのは、混ぜた軌道では全方位の視点が
保てないという事実までである。A.7 の併進成分をゼロにする判断は、この仮説にもとづく。

### 用語

- **軌道 (trajectory)**: 6 自由度マニピュレータの関節変数の時系列。各時刻について
  関節位置 `qpos`、関節速度 `qvel`、関節加速度 `qacc` の 3 つ組を持つ。
  関節 0-2 が併進 (単位 m)、関節 3-5 が回転 (単位 rad) である
  (`trajectories/base_trajectory.py:78-107` の軸ラベル、`configurations/simulations/base.yaml:14-38` の
  ゲインの並び)。
- **スプライン軌道**: 始点と終点の境界条件から多項式係数を決める軌道。本手順では画像と
  カメラ姿勢の供給に使う。
- **励起軌道 (excited trajectory)**: 慣性パラメータの推定を良条件にするよう最適化した軌道。
  本手順では動力学量の供給に使う。
- **回帰行列 (regressor)**: 剛体の運動方程式を慣性パラメータ 10 成分について線形に書いたときの
  係数行列 (6 行 10 列)。`dynamics/dynamics.py:390` の `calculate_frame_dynamics` が
  1 フレーム分を作る。
- **観測情報行列**: 全フレームの回帰行列 `A_k` を積み上げた行列 `F` について `Y = FᵀF`。
  実装は `A_k` を明示的に積み上げず `Y = Σ_k A_kᵀ A_k` として作る
  (`trajectories/base_trajectory.py:137-153`)。
- **慣性パラメータ 10 成分**: `[mass, mx, my, mz, ixx, iyy, izz, ixy, iyz, izx]`
  (`recorders/standard_recorder.py:176`)。
- **目録ファイル (transforms file)**: データセットのルートに置く JSON。カメラ内部パラメータ、
  真値、フレーム配列を持つ。学習側ローダはこれを読む。

### 構成

```
A. スプライン軌道の生成  ─┐
                          ├─→ D. シミュレーション実行 (2 回) ─→ C. データセット合成
B. 励起軌道の生成      ─┘
```

---

## A. スプライン軌道の生成 (画像・カメラ姿勢用)

実装は `trajectories/spline.py`、基底クラスは `trajectories/base_trajectory.py`、
コマンドラインの入口は `trajectories/generate.py` である。

### A.1 時間格子

基底クラスが時間格子を作る (`trajectories/base_trajectory.py:34-35`)。

```
time_steps = int(duration * fps)
time_array = linspace(0, duration, time_steps)
```

`duration = 5.0`、`fps = 60.0` なら 300 点である。`generate` は末尾が `duration` を
超える場合に 1 点落とす (`trajectories/base_trajectory.py:298-299`)。

### A.2 quintic (5 次) の係数決定

`trajectories/spline.py:67-126`。関節 j ごとに独立に、次の多項式を置く。

$$q(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$$

境界条件は 6 個で、始点と終点の位置・速度・加速度である
(`trajectories/spline.py:77-83` のコメントに同じ式が書かれている)。

$$q(0)=q_0,\quad \dot q(0)=v_0,\quad \ddot q(0)=\alpha_0$$
$$q(T)=q_1,\quad \dot q(T)=v_1,\quad \ddot q(T)=\alpha_1$$

始点側の 3 条件は係数を直接与える (`trajectories/spline.py:95-97`)。

$$a_0 = q_0,\qquad a_1 = v_0,\qquad a_2 = \alpha_0/2$$

残る 3 係数は 3 元 1 次連立方程式を解いて得る (`trajectories/spline.py:101-117`)。

$$
\begin{bmatrix}
T^3 & T^4 & T^5\\
3T^2 & 4T^3 & 5T^4\\
6T & 12T^2 & 20T^3
\end{bmatrix}
\begin{bmatrix}a_3\\a_4\\a_5\end{bmatrix}
=
\begin{bmatrix}
q_1 - (a_0 + a_1 T + a_2 T^2)\\
v_1 - (a_1 + 2 a_2 T)\\
\alpha_1 - 2 a_2
\end{bmatrix}
$$

解法は `numpy.linalg.solve` (`trajectories/spline.py:117`)。

### A.3 septic (7 次) の係数決定

`trajectories/spline.py:128-205`。多項式を 7 次に上げ、境界条件に躍度 (jerk, 加速度の
時間微分) を加えた 8 個とする。

$$q(t) = \sum_{i=0}^{7} a_i t^i$$
$$q(0)=q_0,\ \dot q(0)=v_0,\ \ddot q(0)=\alpha_0,\ \dddot q(0)=j_0$$
$$q(T)=q_1,\ \dot q(T)=v_1,\ \ddot q(T)=\alpha_1,\ \dddot q(T)=j_1$$

始点側の 4 条件が係数を直接与える (`trajectories/spline.py:164-167`)。

$$a_0=q_0,\quad a_1=v_0,\quad a_2=\alpha_0/2,\quad a_3=j_0/6$$

残る 4 係数は 4 元 1 次連立方程式を解く (`trajectories/spline.py:171-194`)。

$$
\begin{bmatrix}
T^4 & T^5 & T^6 & T^7\\
4T^3 & 5T^4 & 6T^5 & 7T^6\\
12T^2 & 20T^3 & 30T^4 & 42T^5\\
24T & 60T^2 & 120T^3 & 210T^4
\end{bmatrix}
\begin{bmatrix}a_4\\a_5\\a_6\\a_7\end{bmatrix}
=
\begin{bmatrix}
q_1 - (a_0 + a_1 T + a_2 T^2 + a_3 T^3)\\
v_1 - (a_1 + 2a_2 T + 3a_3 T^2)\\
\alpha_1 - (2a_2 + 6a_3 T)\\
j_1 - 6a_3
\end{bmatrix}
$$

境界条件のうち、設定で指定されなかったものはすべて 0 になる
(`trajectories/spline.py:51-56`)。したがって既定では始点・終点で速度・加速度・躍度が 0 の
静止 - 静止の遷移である。

### A.4 軌道値の評価

`trajectories/spline.py:207-243` が各時刻で位置・速度・加速度を多項式とその微分から求める。
quintic は `:228-232`、septic は `:233-241`。

### A.5 実際に使った設定

`configurations/trajectory_generation/spline_6dof_rot_only.yaml` の全 6 項目
(`:1-7`、コメント行 `:6` を除く)。

| 項目 | 値 | 意味 |
| --- | --- | --- |
| `config_class` | `SplineTrajectoryConfig` | この YAML が埋める設定クラス名。`BaseTrajectoryConfig.config_class` (`trajectories/base_trajectory.py:25`) |
| `type` | `quintic` | 5 次多項式を選ぶ。`trajectories/spline.py:60-65` の分岐 |
| `duration` | `5.0` | 軌道の長さ [s]。`trajectories/base_trajectory.py:17` |
| `fps` | `60.0` | 1 秒あたりのフレーム数。`trajectories/base_trajectory.py:18`。5.0 × 60.0 = 300 フレーム |
| `start_pos` | `[0, 0, 0, 0, 0, 0]` | 始点の関節位置。前 3 成分が併進 [m]、後 3 成分が回転 [rad] |
| `end_pos` | `[0, 0, 0, 3.141592653589793, 0, 25.1327412287]` | 終点の関節位置。関節 3 が π rad (半回転の傾け)、関節 5 が 8π rad (4 回転) |

`start_vel` 以下の境界条件は YAML に無いので既定の `None` となり、A.3 末尾のとおり 0 になる
(生成物に記録された設定でも `null`: `configurations/trajectories/spline_20260825_164121/trajectory.json`
の `metadata.generation_config`)。

### A.6 生成コマンドと生成物

コマンドライン入口はパッケージのスクリプト `generate-trajectory` (`pyproject.toml:23`)。

```sh
pixi run generate-trajectory spline --config configurations/trajectory_generation/spline_6dof_rot_only.yaml
```

`--config` が指定されると、`trajectories/generate.py:119-134` が
「クラスの既定値 < YAML < コマンドラインで明示指定された項目」の順に併合する。明示指定の判定は
`sys.argv` にフラグ文字列が現れるかで行う (`trajectories/generate.py:89-104`)。

`--plot-path` も `--json-path` も指定しない場合、出力先は
`configurations/trajectories/<サブコマンド名>_<YYYYmmdd_HHMMSS>/` に自動命名される
(`trajectories/generate.py:42`, `:193-196`)。実際の生成物は次のディレクトリである。

```
configurations/trajectories/spline_20260825_164121/
    trajectory.json     軌道本体 (300 フレーム、duration 5.0、fps 60.0)
    trajectory.png      位置・速度・加速度のプロット
    optimize.log        標準出力・標準エラーの複製 (trajectories/generate.py:68-86, :201)
```

`trajectory.json` の構造は `trajectories/base_trajectory.py:37-73` が決める。
`frames` は 1 フレームあたり `[qpos リスト, qvel リスト, qacc リスト]` の 3 要素配列である
(`:58`)。`metadata` には `subcommand` と、生成時の設定全体を写した `generation_config` が入る
(`trajectories/generate.py:212`)。

生成物の実測値: フレーム数 300、先頭フレームは全成分 0、末尾フレームは
`qpos = [0, 0, 0, 3.141592653589793, 0, 25.132741228700013]`、`qvel`・`qacc` は
最大 2.8e-14 で数値誤差の範囲。

### A.7 併進成分をゼロにしている点

`end_pos` の前 3 成分 (併進) が始点と同じ 0 である。すなわちこの軌道は純粋な回転で、
物体の位置は動かない。

設計意図は、この軌道の役割が外観再構成のための視点供給に限られることによる。物体を全方位から
均等に見る条件を作るのは物体の回転であり、併進は視線方向をほとんど変えない。併進が変えるのは
カメラと物体の距離と画面内の位置であって、その結果として物体が占める画素数が変動し、実効解像度が
フレームごとに不均一になる。YAML のコメント
(`configurations/trajectory_generation/spline_6dof_rot_only.yaml:6`) にも
「rotation-only for NeRF viewpoint coverage: pi tilt + 8pi turn, no translation」と記されている。
この判断の経緯は `notes/LOGS/2026-08-26_dataset-merge-and-noise-model.md:33-35` に記録がある。

---

## B. 励起軌道の生成 (動力学量用)

実装は `trajectories/excited.py`、窓関数は `trajectories/window.py`、
フーリエ級数は `trajectories/fourier.py`、目的関数は `trajectories/base_trajectory.py` にある。

### B.1 軌道の関数形

励起軌道は、ベース軌道 (`main_trajectory`) と、窓関数を掛けたフーリエ級数の和である
(`trajectories/excited.py:610-645`、特に `:641-643`)。

$$q(t) = q_{\text{main}}(t) + w(t)\, f(t)$$

ここで `f(t)` は有限フーリエ級数で、関節 j について
(`trajectories/fourier.py:25-31` の説明、実装は `:86-116`)

$$f_j(t) = q_{j,0} + \sum_{k=1}^{N} \left( a_{j,k}\sin(k\omega_b t) + b_{j,k}\cos(k\omega_b t) \right),
\qquad \omega_b = 2\pi f_b$$

`f_b` は基本周波数 `base_freq` (`trajectories/fourier.py:49`)、`N` は高調波の本数
`num_harmonics`。速度と加速度は解析的に微分して得る (`trajectories/fourier.py:112`, `:116`)。

$$\dot f_j = \sum_k k\omega_b\left(a_{j,k}\cos(k\omega_b t) - b_{j,k}\sin(k\omega_b t)\right)$$
$$\ddot f_j = -\sum_k (k\omega_b)^2\left(a_{j,k}\sin(k\omega_b t) + b_{j,k}\cos(k\omega_b t)\right)$$

励起軌道が使うフーリエ級数のオフセット `q0` はゼロベクトルに固定される
(`trajectories/excited.py:627`)。

### B.2 窓関数

`trajectories/window.py:16-79`。正規化時刻 `r = t / T` について

$$w(r) = 256\, r^4 (1-r)^4$$

(`trajectories/window.py:20`, `:51-52`)。`u = r(1-r)` と置くと `w = 256 u^4`。
時間微分は連鎖則で (`trajectories/window.py:56-66`)

$$\dot w = \frac{1024\,u^3(1-2r)}{T},\qquad
\ddot w = \frac{1024}{T^2}\left(3u^2(1-2r)^2 - 2u^3\right)$$

`r = 0` と `r = 1` で `u = 0` となるので `w`、`\dot w`、`\ddot w` がすべて 0 になる。したがって
励振成分は始点と終点で位置・速度・加速度のすべてが 0 であり、ベース軌道の境界条件を壊さない
(`trajectories/excited.py:96-99` の説明)。

窓の適用は積の微分則による (`trajectories/window.py:81-100`)。

$$Q = w q,\qquad \dot Q = \dot w q + w\dot q,\qquad \ddot Q = \ddot w q + 2\dot w \dot q + w \ddot q$$

### B.3 最適化の定式化

#### 決定変数

フーリエ係数 `a` と `b` を平坦化して連結した長さ `2 × num_joints × num_harmonics` のベクトル
(`trajectories/excited.py:546`, 復元は `:271-273` と `:595-597`)。

#### 目的関数

`objective_type` が `"condition_number"` なら観測情報行列の条件数、`"d_optimal"` なら D 最適性
(`trajectories/excited.py:62`、分岐は `trajectories/base_trajectory.py:269-273`)。

観測情報行列は各時刻の回帰行列 `A_k` から作る (`trajectories/base_trajectory.py:145-153`)。

$$Y = \sum_k A_k^{\top} A_k$$

条件数は `Y` の固有値の最大と最小の比 (`trajectories/base_trajectory.py:200-206`)。
最小固有値が 1e-9 未満のときは 1e9 を返して打ち切る (`:204-205`)。

$$\kappa = \lambda_{\max}(Y)/\lambda_{\min}(Y)$$

D 最適性は (`trajectories/base_trajectory.py:217`, 実装 `:232-234`)

$$J_D = -\log\det Y = -\sum_i \log \lambda_i(Y)$$

固有値は 1e-30 で下から抑える (`trajectories/base_trajectory.py:233`)。

#### 列等化 (equilibration)

`trajectories/base_trajectory.py:155-170`。`Y` を計算した直後、条件数や行列式を取る前に
次の対角スケーリングを掛ける。

$$D = \operatorname{diag}\left(1/\sqrt{Y_{ii}}\right),\qquad Y \leftarrow D\,Y\,D$$

実装は `d = 1/sqrt(clip(diag(Y), 1e-30, None))` を作り `Y * outer(d, d)` とする
(`trajectories/base_trajectory.py:169-170`)。`Y = FᵀF` なので `Y_ii = ‖F[:, i]‖²` であり、
この操作は積み上げ回帰行列 `F` の各列の L2 ノルムを 1 に正規化することと等価である。`F` を
明示的に作る必要はない。

入れる理由は、条件数が列ごとの単位の取り方に対して不変でないことである。慣性パラメータの 10 成分は
質量 [kg]、一次モーメント [kg·m]、慣性テンソル [kg·m²] と物理次元が異なり、kg と g、m と mm の
どちらで書くかで生の `Y` の条件数が変わる。したがって生の条件数は設計基準として一意に定まらない。
列等化した行列の条件数は単位の取り方に依らない。コード内の説明とその出典
(Van der Sluis 1969、Swevers ら 1997) は `trajectories/base_trajectory.py:160-165` に書かれている。
対角成分の下限による打ち切りは、全要素が 0 の列でのゼロ除算を避けるためである
(`trajectories/base_trajectory.py:165`, `:169`)。

有効・無効の切り替えは設定項目 `column_scale` (既定 `True`、`trajectories/excited.py:55-59`)。
無効なら `Y` をそのまま返す (`trajectories/base_trajectory.py:167-168`)。

#### フーリエ係数の解析的バウンド

`trajectories/excited.py:222-268`。関節速度の上限 `dq_max` と加速度の上限 `ddq_max` から、
係数の箱型上限を三角不等式で導く。

窓付きフーリエ軌道 `Q = w f` の速度は `\dot Q = \dot w f + w \dot f`。`|w| ≤ 1` かつ
各調波の寄与の絶対値を足し上げると、係数の絶対値をすべて共通の上限 `B` で抑えたとき
1 調波あたりの寄与は次で抑えられる。窓の時間微分の最大値を

$$\dot w_{\max} = \max_s |1024\, s^3(1-s)^3(1-2s)| / T$$

(`trajectories/excited.py:246`, `:252`。`s` は 0 から 1 までの 10,000 点の格子
`trajectories/excited.py:239`)、`k` 番目の調波の角周波数を `ω_k = 2π f_b k`
(`trajectories/excited.py:240-241`) とすると、速度側の増幅率は

$$\alpha^{\text{vel}}_k = \dot w_{\max} + \omega_k$$

(`trajectories/excited.py:253`)。sin と cos の 2 項 × `num_harmonics` 本を足すので、
各係数に許される上限は (`trajectories/excited.py:256`)

$$B_j \le \frac{\dot q_{\max,j}}{2\, N\, \alpha^{\text{vel}}_k}$$

加速度側も同様に、窓の 2 階微分の最大値を

$$\ddot w_{\max} = \max_s \left|1024\left(3s^2(1-s)^2(1-2s)^2 - 2s^3(1-s)^3\right)\right| / T^2$$

(`trajectories/excited.py:247`, `:261`) として、`\ddot Q = \ddot w f + 2\dot w \dot f + w \ddot f`
の各項から (`trajectories/excited.py:262`)

$$\alpha^{\text{acc}}_k = \ddot w_{\max} + 2\dot w_{\max}\omega_k + \omega_k^2$$
$$B_j \le \frac{\ddot q_{\max,j}}{2\, N\, \alpha^{\text{acc}}_k}$$

(`trajectories/excited.py:265`)。関節ごとに、全調波・速度側・加速度側のすべてにわたる
最小値を採る (`trajectories/excited.py:249`, `:257`, `:266`)。

このバウンドは最悪ケースを想定するので保守的である。そのため設定項目
`use_analytical_bounds` を `False` にすると、代わりに `dq_max`・`ddq_max` を SLSQP の
直接の不等式制約として課す経路が選べる (`trajectories/excited.py:47-53`, `:326-340`)。
また、解析バウンドが有効なときに手動の `coeff_bounds` を併用することは禁止されている
(`trajectories/excited.py:168-175`)。物理的根拠のある上限が 1 つだけであるべきだからである。
`dq_max`・`ddq_max` のどちらも与えず `coeff_bounds` も無い場合は全関節 0.5 が既定になる
(`trajectories/excited.py:181-182`)。

#### 関節可動域と特異姿勢の制約

- **可動域**: `q_min` / `q_max` を全関節・全時刻の合成軌道の位置に課す。最適化手法が SLSQP なら
  不等式制約として与え (`trajectories/excited.py:311-324`)、L-BFGS-B ならペナルティ項
  (重み 1e5、違反量の二乗和) として目的関数に足す (`trajectories/excited.py:350`, `:397-403`)。
  可動域制約はスカラーの `min` に集約する (`trajectories/excited.py:315`, `:322`)。
- **特異姿勢の除外**: 関節 j について `|q_j(t) - center_j| ≥ margin_j` を課す
  (`trajectories/excited.py:34-38`, `:194-205`)。これは中心 `center_j` の近傍を除外する条件で、
  `q_min`/`q_max` の包含区間とは向きが逆である。`|q - center| ≥ margin` は選言であって
  非凸なので、単一の滑らかな不等式制約にできない。したがって手法によらずペナルティで課す
  (`trajectories/excited.py:405-411`)。`margin_j = 0` の関節では無効になる
  (`trajectories/excited.py:201`)。

#### 最適化手法、マルチスタート、早期停止

- 手法は `optimizer_method` で `"L-BFGS-B"` か `"SLSQP"` (`trajectories/excited.py:63-64`、
  検証は `:111-112`)。`scipy.optimize.minimize` を呼ぶ (`trajectories/excited.py:450-464`)。
  反復上限は `max_iter` (既定 50、`trajectories/excited.py:28`, `:456`)。
- 箱型制約は全係数に `(-B_j, B_j)` を並べたもの (`trajectories/excited.py:528-532`)。
- **マルチスタート**: `n_restarts` 回だけ初期値を変えて解く (`trajectories/excited.py:544-593`)。
  1 回目の初期値は `seed` で初期化した一様乱数 `U(-0.01, 0.01)`
  (`trajectories/excited.py:211-213`, `:545-546`)、2 回目以降は調波 `k` に対して
  振幅 `0.3/(k+1)` と箱型上限の小さい方を使った一様乱数
  (`trajectories/excited.py:280-291`, `:548`)。
- **候補の比較規則**: 実行可能解を実行可能でない解より優先し、同じ実行可能性なら条件数の小さい方を
  採る (`trajectories/excited.py:490-507`)。実行可能性の許容値は 1e-3
  (`trajectories/excited.py:536`)。速度・加速度の直接制約を使わない設定では違反量が常に 0 なので、
  この規則は「NaN でない最小の条件数が勝つ」に退化する (`trajectories/excited.py:495-497` の説明)。
- **早期停止 (リスタート側)**: 改善しないリスタートが `early_stop_patience` 回続いたら打ち切る
  (`trajectories/excited.py:588-593`)。
- **早期停止 (反復側)**: `target_condition_number` が設定されている場合、各反復のコールバックで
  受理された点 `xk` の条件数を評価し直し、目標以下なら例外 `_TargetReached` を投げて
  `minimize` を抜ける (`trajectories/excited.py:78-89`, `:434-448`, `:463-469`)。目標に達したら
  リスタートのループも打ち切る (`trajectories/excited.py:576-586`)。
- 目的関数の評価は `x.tobytes()` をキーにした 1 段のキャッシュで重複計算を避ける
  (`trajectories/excited.py:366-375`)。

### B.4 実際に使った軌道

`configurations/trajectories/excited_nostop_nomain_10s/trajectory.json` の
`metadata.generation_config` の全項目。

| 項目 | 値 | 意味 |
| --- | --- | --- |
| `target_class` | `ExcitedTrajectory` | 生成するクラス (`trajectories/excited.py:75`) |
| `duration` | `10.0` | 軌道長 [s] |
| `fps` | `60.0` | 10.0 × 60.0 = 600 フレーム |
| `num_joints` | `6` | 関節数 |
| `num_harmonics` | `5` | フーリエ級数の調波数 `N` |
| `base_freq` | `0.1` | 基本周波数 `f_b` [Hz]。`duration` 10 s の逆数 |
| `coefficients` / `q0` | `null` | 初期係数を外部から与えない |
| `main_trajectory` | 5 次スプライン、`start_pos` と `end_pos` がともに全成分 0、`duration` 10.0、`fps` 60.0 | ベース軌道 (B.5 参照) |
| `manipulator` | `xml_models/manipulators/sequential` | 回帰行列を作る MuJoCo モデル |
| `object` | `xml_models/targets/hammer` | 同上、対象物体 |
| `ee_body_name` | `link6` | 手先のボディ名 |
| `max_iter` | `50` | 1 リスタートあたりの最大反復数 |
| `q_min` | `[-10, -10, -10, -10, -0.7853982, -10]` | 関節位置の下限。関節 4 のみ ±π/4 rad に制限 |
| `q_max` | `[10, 10, 10, 10, 0.7853982, 10]` | 関節位置の上限 |
| `singularity_center` | 全成分 0 | 特異姿勢の中心 |
| `singularity_margin` | 全成分 0 | 全成分 0 なので特異姿勢の除外は無効 (`trajectories/excited.py:201`) |
| `coeff_bounds` | `null` | 手動の係数上限を使わない |
| `dq_max` | `[1.5, 1.5, 1.5, 3.14159, 3.14159, Infinity]` | 関節速度上限。併進 1.5 m/s、関節 3・4 が π rad/s、関節 5 は無制限 |
| `ddq_max` | `[7.5, 7.5, 7.5, 6.28318, 6.28318, 6.28318]` | 関節加速度上限。併進 7.5 m/s²、回転 2π rad/s² |
| `use_analytical_bounds` | `true` | 解析バウンドを箱型制約に使う |
| `column_scale` | `true` | 列等化を有効にする |
| `objective_type` | `condition_number` | 目的関数は条件数 |
| `optimizer_method` | `SLSQP` | 最適化手法 |
| `n_restarts` | `8` | マルチスタート回数 |
| `seed` | `42` | 初期値の乱数種 |
| `early_stop_patience` | `8` | 改善なしを許すリスタート数 |
| `target_condition_number` | `null` | 反復側の早期停止を使わない |

`config` 項目には生成時に使った YAML の絶対パスが記録されているが、それは一時ディレクトリ
(`/tmp/claude-1000/.../scratchpad/final_nomain_10s.yaml`) で、現在は存在しない。上の表の値が
実際に使われた設定の完全な記録である。

生成ログ `configurations/trajectories/excited_nostop_nomain_10s/optimize.log` の実測値。

- 導出された解析バウンド (`:1` 行目): 関節 0-2 が 0.04258433348846626、関節 3-5 が
  0.049936154052722885。
- リスタート 8 回の条件数: 1.0174 / **1.0158** / 1.0175 / 1.0172 / 1.0274 / 1.0187 / 1.0238 / 1.0225。
  各回 1024〜1051 秒。
- 最良はリスタート 2 の条件数 1.0158、総計 8326.8 秒。

`trajectory.json` の `metadata` に記録された最終条件数は **1.015819539547215**
(記録は `trajectories/excited.py:647-651`、`generate` 側での併合は
`trajectories/base_trajectory.py:303-305`)。同じ値が
`trajectories/catalog.py` 経由で `configurations/trajectories/catalog.json` にも追記される
(`trajectories/generate.py:220-226`)。

### B.5 ベース軌道を持たない (純励起) 点

`main_trajectory` は 5 次スプラインだが、`start_pos` と `end_pos` がともに全成分 0 である。
A.2 の係数の式に `q_0 = q_1 = 0`、`v_0 = v_1 = \alpha_0 = \alpha_1 = 0` を入れると全係数が 0 になる。
すなわちベース軌道は恒等的に 0 で、合成軌道は窓付きフーリエ励振そのものである。
ディレクトリ名の `nomain` はこれを指す。

設計意図は A.7 と対になる。役割を分離した結果、この軌道の仕事は動力学量の供給だけになり、
視点の軌跡を作る必要がない。ベース軌道を残すと励振がその上に重なるだけで、条件数の改善に
寄与しない自由度が増える。ディレクトリ名の `nostop` は、B.2 の窓関数が始終端で 0 になる形を
保ったまま反復側の早期停止 (`target_condition_number`) を使わなかったことを指す
(表の `target_condition_number` が `null`)。

---

## C. データセットの合成

実装は `recorders/merge.py`。コマンド名は `merge-datasets` (`pyproject.toml:24`)。

### C.1 下流の学習側ローダの制約

ローダは `~/workspace/pixi-wisp-container/wisp/datasets/formats/nemd_standard_dataset.py`。
以下は実際にコードを読んで確認した事項である。

1. **1 行から画像と動力学量の両方を読む**。`_load_single_entry` は 1 つの `frame` 辞書を受け取り、
   同じ辞書から画像パス (`:277`)、カメラ姿勢 `transform_matrix` (`:321`)、
   `pose_sen_obj` (`:293`)、`pose_sen_obji` (`:294`)、`twist_sen` (`:295`)、
   `dtwist_sen` (`:296`)、`linacc_sen_obji` (`:297`)、レンチ (`:301`) を取り出す。
   レンチは `ft_sen` を優先し、無ければ `wrench` を読む (`:299-301`)。読み込み結果は
   フレーム方向に `torch.stack` され (`:494-502`)、`(num_frames, ...)` に整形される
   (`:642-663`)。したがって「画像だけの行」「動力学量だけの行」は表現できない。

2. **両者が損失計算で互いを参照しない**。色の損失は描画結果 `rb.rgb` と正解画像 `gt_img` だけを
   使う (`~/workspace/pixi-wisp-container/wisp/trainers/md_multiview_trainer.py:593-600`)。
   動力学の損失は、質量分布の予測値・その行の `pose_sen_obj`・`twist_sen`・`dtwist_sen`・
   `ft_sen` だけを使う (同 `:624-661`)。動力学の項はカメラ姿勢を参照せず、色の項はレンチを
   参照しない。よって同じ行に別々の run 由来の画像と動力学量を同居させても、どちらの損失も
   壊れない。

3. **カメラ内部パラメータは全フレーム共有**。焦点距離は目録ファイルの最上位の
   `camera_angle_x` / `camera_angle_y` から 1 度だけ計算され、全カメラに同じ値が渡される
   (`:505-531`, `:581-592`)。主点も最上位の `cx` / `cy` から 1 度だけ計算する (`:554-562`)。
   姿勢の平行移動成分は `aabb_scales.mean()` で割る (`:569`)。フレームごとに内部パラメータを
   変えることはできない。

なお、目録ファイルの探索は「データセット直下の `*.json` を glob し、1 個なら train、
3 個なら train/val/test、それ以外はエラー」である (`:238-262`)。この規則が、次の C.5 で述べる
`.bak` 拡張子による除外の理由になっている。

### C.2 合成の規則

`recorders/merge.py:92-164` の `merge`。

- 各 run から目録ファイルを 1 つ選ぶ。`transforms.json`、`unperturbed_transforms.json` の順に
  優先し、どちらも無ければ直下の `*.json` がちょうど 1 個であることを要求する
  (`recorders/merge.py:41-53`)。
- カメラと形状に関する項目 `camera_angle_x`、`camera_angle_y`、`cx`、`cy`、`fl_x`、`fl_y`、
  `h`、`w`、`aabb_scale` が両 run で一致することを確認する。食い違えば中断し、`--force` で
  スプライン側を採る (`recorders/merge.py:23`, `:99-103`)。
- 画像とカメラ姿勢はスプライン run の行から、動力学量は励起 run の行から採る
  (`recorders/merge.py:84-85`)。実装は「スプライン行から動力学の項目を除いたもの」に
  「励起行の動力学の項目」を上書きする形である。
- 動力学量として上書きされる項目の一覧 (`recorders/merge.py:28-38` の `DYNAMICS_KEYS`):
  `pose_sen_obj`、`pose_sen_obji`、`twist_sen`、`dtwist_sen`、`linacc_sen_obji`、
  `wrench`、`ft_sen`、`regressor`、`jointvars_clean`。このうち `regressor` はローダは読まず、
  リポジトリ内の同定スクリプト用に運ばれる (`recorders/merge.py:26-27` のコメント)。
- 画像とマスクは合成後の連番 `%04d.png` として `complete/` と `masks/` に配置する。
  同一ファイルシステムならハードリンク、失敗すればコピーする
  (`recorders/merge.py:56-60`, `:73-82`)。
- 励起 run に、スプライン run が持つ動力学の項目が欠けていれば中断する
  (`recorders/merge.py:110-112`)。
- 最上位の同定結果 `ls` と `tls` は励起 run のものを採る (`recorders/merge.py:136-138`)。
  それ以外の最上位項目はスプライン run から引き継ぐ (`recorders/merge.py:134`)。
- スプライン run に `ground_truth.csv` があればコピーする (`recorders/merge.py:160-162`)。

### C.3 フレーム数が異なる場合の等間隔抽出

合成後のフレーム数は `n = min(len(spline_frames), len(excited_frames))`
(`recorders/merge.py:107`)。両 run とも、自分の全区間から `n` 個を等間隔に抽出する
(`recorders/merge.py:115-116`)。添字の式は `recorders/merge.py:63-70`。

$$\mathrm{idx}(i) = \operatorname{round}\!\left(\frac{i\,(M-1)}{n-1}\right),\qquad i = 0,\dots,n-1$$

`M` はその run の元のフレーム数。`n = 1` のときは `[0]` を返す (`recorders/merge.py:68-69`)。

端点が保存される: `i = 0` で `idx = 0`、`i = n-1` で `idx = round((n-1)(M-1)/(n-1)) = M-1`。
すなわち抽出区間は必ず元の run の先頭フレームと末尾フレームを含み、抽出した行が軌道の全域を覆う。
`n ≤ M` である限り添字は狭義単調増加である (`recorders/merge.py:67`)。フレーム数が違う場合は
警告を標準出力に出す (`recorders/merge.py:117-123`)。

先頭から切り詰める方式ではなく等間隔抽出を採った理由は、切り詰めでは軌道の後半が丸ごと落ちる
ためである。短い方を巡回させて長い方に合わせる方式は、行が重複して損失の重みが暗黙に偏るため
却下された。経緯は `notes/LOGS/2026-08-26_dataset-merge-and-noise-model.md:26-31` にある。

### C.4 出所の記録

各行に 2 つの項目が付く (`recorders/merge.py:87-88`)。

- `image_source`: 文字列 `"spline"`
- `dynamics_source`: 文字列 `"excited"`

目録ファイルの最上位には `merge_sources` が入る (`recorders/merge.py:139-155`)。

| 項目 | 内容 |
| --- | --- |
| `image.role` | `"image+camera_pose"` |
| `image.run_dir` | 画像側 run のディレクトリ (文字列) |
| `image.frames` | 画像側 run の元のフレーム数 |
| `image.source_indices` | 採用した添字の配列 (長さ `n`) |
| `dynamics.role` | `"dynamics"` |
| `dynamics.run_dir` | 動力学側 run のディレクトリ |
| `dynamics.frames` | 動力学側 run の元のフレーム数 |
| `dynamics.source_indices` | 採用した添字の配列 |
| `merged_frames` | 合成後のフレーム数 `n` |
| `subsampling` | `"even"` |
| `dynamics_keys` | 実際に上書きされた動力学の項目名の配列 |

実データ (`datasets/hammer/merged_wrenchonly/transforms.json`) の実測値:
`image.run_dir` = `datasets/hammer/spline_rotonly_wrenchonly`、`image.frames` = 300、
`image.source_indices` = `[0, 1, 2, …, 297, 298, 299]`、
`dynamics.run_dir` = `datasets/hammer/excited_nomain10s_wrenchonly`、`dynamics.frames` = 600、
`dynamics.source_indices` = `[0, 2, 4, …, 595, 597, 599]`、`merged_frames` = 300、
`dynamics_keys` = `["pose_sen_obj", "twist_sen", "dtwist_sen", "wrench", "regressor"]`。

### C.5 補助コマンド `export-dynamics-csv`

実装は `recorders/export_dynamics_csv.py`、コマンド名は `export-dynamics-csv`
(`pyproject.toml:25`)。目録ファイルの動力学の行を平坦な表に落とす。

列の並びと 1 項目あたりのスカラー数は `recorders/export_dynamics_csv.py:21-27` で固定である。

| 列群 | スカラー数 |
| --- | --- |
| `pose_sen_obj` | 16 (4×4 行列) |
| `twist_sen` | 6 |
| `dtwist_sen` | 6 |
| `wrench` | 6 |
| `regressor` | 60 (6×10 行列) |

- 先頭列は `frame` (0 始まりの通し番号、`:49`, `:59`)。
- 目録ファイルに `merge_sources.dynamics.source_indices` があれば、2 列目に `source_index` を
  出す (`:47`, `:50-51`, `:60-61`)。合成データセットの行が元の run のどのフレームから
  来たかを辿れる。
- 以降の列名は `<キー>_<0 始まりの添字>` (`:52`)。例: `twist_sen_0` … `twist_sen_5`。
- `regressor` は既定では出力しない。`--include-regressor` で 60 列を追加する (`:44`, `:77-78`)。
- 先頭フレームに存在しないキーは列に含めない (`:44`)。
- 平坦化した要素数が想定と違えば中断する (`:64-65`)。
- 出力先は既定で `<dataset_dir>/dynamics.csv` (`:54`, `:75-76`)。

### C.6 合成データセットの見え方

合成後のディレクトリは `transforms.json` 1 つだけを直下の `*.json` として持つ
(`recorders/merge.py:157-158`)。C.1 の 3 分岐の規則により、ローダはこれを単一 split (train) の
データセットとして読む。元の run 側では、出荷しない系列に `.bak` を足して glob から外す
(`recorders/standard_recorder.py:22-32`)。

---

## D. シミュレーション実行と観測モデル

入口は `main.py`。設定は `configurations/simulations/base.yaml` を既定とし
(`simulators/simulator.py:57`)、コマンドライン引数と併合する (`main.py:60-63`)。

### D.1 実行の流れ

- モデルと物体を組み立て、真値を得る (`main.py:64`)。
- 目標軌道の JSON を読む (`main.py:66-69`)。記録の `fps` は目標軌道から引き継ぐ
  (`main.py:71-76`)。
- シミュレーションの刻み数は `n_steps = duration / m.opt.timestep`
  (`simulators/simulator.py:136`)。`xml_models/` 配下に `timestep` の指定は無い (grep で該当なし)
  ので MuJoCo の既定値が使われる。
- 各刻みでノイズ付きの関節計測を取り (`simulators/simulator.py:190`)、記録済みフレーム数が
  `d.time * fps` 以下ならフレームを 1 枚記録する (`simulators/simulator.py:192-196`)。
  制御入力を計算して 1 刻み進める (`simulators/simulator.py:200-202`)。
- 1 刻み進めた後に時刻が増加しなかった場合、または `qpos` / `qvel` / `qacc` に非有限値が
  生じた場合は中断する。発散後の短い系列を正常なデータセットとして保存しない
  (`simulators/simulator.py`)。
- 出力先は `datasets/<物体名>/<軌道ディレクトリ名>_cond<条件数>_run<N>` に自動命名される
  (`main.py:22-57`, `:85-89`)。`--recorder.dataset-dir` で明示指定もできる
  (`main.py:90-91`)。

### D.2 制御則

`simulators/simulator.py:316-336`。目標軌道 `tgt_traj` (目標の qpos / qvel / qacc) と、
ノイズ付き計測 `act_traj` から次を作る。

1. 位置の残差は MuJoCo の `mj_differentiatePos` で計算する
   (`simulators/simulator.py:318-324`)。四元数を正しく差分するためである
   (`:318` のコメント)。
2. **逆動力学フィードフォワード**: 目標軌道をそのまま逆動力学に通して必要トルクを得る
   (`simulators/simulator.py:327`)。逆動力学は
   `dynamics.setup_robot_dynamics_parameters` が返す関数である
   (`simulators/simulator.py:157`, `:164`)。
3. **LQR フィードバック**: 状態誤差は位置残差と速度誤差を連結したベクトル
   (`simulators/simulator.py:329-330`)。これにゲイン行列を掛ける
   (`simulators/simulator.py:331`)。

$$u = \tau_{\text{ff}}(q_{\text{tgt}}, \dot q_{\text{tgt}}, \ddot q_{\text{tgt}}) - K\,
\begin{bmatrix} \Delta q \\ \dot q_{\text{meas}} - \dot q_{\text{tgt}} \end{bmatrix}$$

(`simulators/simulator.py:332`)。ゲイン行列 `K` は離散時間の代数 Riccati 方程式を解いて得る
(`controllers/lqr.py:63-64`)。

$$K = \left(R + B^{\top} P B\right)^{+} B^{\top} P A$$

重み行列は設定の対角要素から作る (`controllers/lqr.py:60-62`)。実際の値は
`configurations/simulations/base.yaml:15-38` にあり、状態の重みが
併進位置 1e6 ×3、回転位置 1e3 ×3、併進速度 1e4 ×3、回転速度 10 ×3、
入力の重みが併進 0.01 ×3、回転 0.1 ×3 である。

フィードバックへ観測値を入れるかは `control_noise` で切り替える。
真の場合、位置にはセンサモデルの観測値を使い、観測誤差が制御ループを介して実際の運動へ影響する。
速度には既定でMuJoCoの即時 `qvel` を使う。これは実機制御器の内部速度推定器を同定したものではなく、
記録ログ用の34 ms差分速度を制御器内部状態と誤認しないための代理である。
`control_derived_velocity=true` の場合だけ記録用の差分速度を制御にも使うが、高ゲイン制御を発散させるため
感度診断以外には使わない。偽の場合は位置と速度の両方にMuJoCoの状態を使う。
加速度はフィードバック状態に含まれない。
この設定は記録値にノイズを加えるかどうかを変更しない。

### D.3 観測ノイズ

ノイズモデルは `noise_profile` で選ぶ。
既定値の `empirical` は実機ログに基づく観測系列を生成し、`legacy` は以前の独立正規分布を再現する。
`empirical_degraded` は劣化した力覚セッションの周辺標準偏差を使う感度試験用設定である。

`empirical` の関節位置標準偏差は次の値である。

```text
noise_scale * [2.0e-5, 2.0e-5, 2.0e-5, 1.5e-5, 1.5e-5, 1.5e-5]
               [ m,      m,      m,    rad,    rad,    rad ]
```

回転軸の値は約500 HzのUR5e静止断片で得た短期変動 `0.983e-5`〜`1.76e-5 rad` に基づく。
実機ログは全軸回転であるため、先頭3軸の並進値 `2.0e-5 m` は実測で校正されていない暫定値である。

速度と加速度へ独立な正規分布は加えない。
記録系列では同じ位置観測系列を使い、速度は幅 `T = 34 ms` の因果的な差分で求める。

$$\dot q_k^{\mathrm{obs}} =
\frac{q_k^{\mathrm{obs}}-q^{\mathrm{obs}}(t_k-T)}{T}$$

加速度は速度の後退差分へ一次ローパスフィルタを適用する。

$$\ddot q_k^{\mathrm{obs}} =
\alpha\frac{\dot q_k^{\mathrm{obs}}-\dot q_{k-1}^{\mathrm{obs}}}{\Delta t}
+(1-\alpha)\ddot q_{k-1}^{\mathrm{obs}},
\qquad
\alpha=\frac{10\Delta t}{1+10\Delta t}$$

`joint_bias_scale` は、位置の短期標準偏差に対する試行固定バイアスの倍率である。
バイアス量は未校正なので、既定値は0とする。

`empirical` の力覚ノイズは、良好なFT300-S実機ログ18断片、60 Hz換算563点から推定した。
軸順は `[Fx, Fy, Fz, Tx, Ty, Tz]` である。

| 軸 | 標準偏差 | lag-1 |
|---|---:|---:|
| Fx | 0.066834 N | 0.287277 |
| Fy | 0.083071 N | 0.156462 |
| Fz | 0.065442 N | 0.424446 |
| Tx | 0.003380 N m | 0.072729 |
| Ty | 0.003021 N m | 0.235178 |
| Tz | 0.000989 N m | 0.319951 |

6軸の同時刻共分散と軸別lag-1を使い、対角VAR(1)として生成する。

$$x_k = D x_{k-1} + \eta_k,\qquad
D=\operatorname{diag}(\rho_1,\ldots,\rho_6),\qquad
\eta_k\sim\mathcal N(0,C_0-DC_0D^\top)$$

共分散行列 `C_0` は `sensors/noise_profiles.py` に数値を省略せず保持する。
イノベーション共分散の最小固有値は `8.60e-7` であり、正定値である。
出力には力 `0.01 N`、トルク `0.001 N m` の量子化を適用する。
ログに力覚センサ固有の時刻がないため、ネイティブ100 Hzの通信過程は再現せず、同定で使う60 Hz観測系列を模擬する。
`wrench_bias_scale` は短期標準偏差に対する試行固定バイアスの倍率であり、既定値は0である。

`noise_scale`、`force_noise_scale`、`torque_noise_scale` は関節位置、力、トルクの短期標準偏差へそれぞれ掛ける。
採用したプロファイルと実効値は、各目録ファイルの最上位 `noise_model` に保存する。

### D.4 ノイズあり系列とノイズなし系列の並行記録

制御器と記録器が観測値を使うかどうかは、二つの設定で独立に切り替える。

| `control_noise` | `record_noise` | 切り分ける対象 |
|---|---|---|
| 偽 | 偽 | 励起と数値計算 |
| 真 | 偽 | 位置観測誤差が制御軌道へ与える影響 |
| 偽 | 真 | 回帰行列と力覚の観測誤差 |
| 真 | 真 | 両者を含む実機模擬条件 |

シミュレータは各時刻に一つの位置観測を生成し、制御器と記録器が同じ位置標本を共有する。
`control_noise` は制御位置が観測値とMuJoCo真値のどちらを使うかを選ぶ。
制御速度は `control_derived_velocity` で記録用差分速度とMuJoCo即時速度を切り替える。
`record_noise` は記録全体のマスタースイッチである。
`record_joint_noise` は回帰行列を観測関節状態とMuJoCo真値のどちらから作るか、
`record_wrench_noise` はレンチへ観測ノイズを加えるかを個別に選ぶ。
`perturb_wrench=false` は従来互換の力覚スイッチであり、偽なら `record_wrench_noise=true` でも
力覚は無雑音になる。
関節用と力覚用の乱数系列は分離しているため、力覚設定を変えても関節ノイズの乱数列は変わらない。

`get_unperturbed` が真の場合は、選択した記録系列に加え、MuJoCo真値から作った参照系列も保存する。
この参照系列は同一runの物理軌道を使うため、`control_noise=true` なら観測誤差の影響を受けた軌道上の無雑音観測である。
ノイズのない制御軌道を表すものではない。
参照フレームには `jointvars_clean` を保存し、画像、カメラ姿勢、物体姿勢は記録系列と共有する。

**ファイル命名の規則**: 目録ファイルは「ちょうど 1 つだけが素の `.json` 拡張子を保ち、
残りは `.bak` を足す」規則で名付けられる (`recorders/standard_recorder.py:22-32`)。
どの系列が素の名前を取るかは `primary_prefix` が決める
(`recorders/standard_recorder.py:70-72`)。`main.py:148-151` は、ノイズなし系列が書かれるときに
`primary_prefix` を `"unperturbed_transforms"` に切り替える。分割 (train / valid / test) の
目録は常に `.bak` が付く (`recorders/standard_recorder.py:30`)。

**生成後の手動リネーム**: 本文書が扱う実際の run
(`datasets/hammer/excited_nomain10s_wrenchonly` など) では、素の `transforms.json` が
ノイズあり系列 (`jointvars_clean` を持たない) で、ノイズなし系列が
`unperturbed_transforms.json.bak` になっている。これは上記の命名規則が生む配置と逆であるが、
コードの不整合ではない。合成処理 `recorders/merge.py:41-53` が素の `transforms.json` を
優先して選ぶ仕様であり、合成データセットの動力学量にはノイズあり系列を入れたいので、
シミュレーション実行の直後に次の 2 コマンドで入れ替えている。

```sh
mv unperturbed_transforms.json unperturbed_transforms.json.bak
mv transforms.json.bak transforms.json
```

この入れ替えは E 節の手順に含まれる。実行しなければ、合成データセットの動力学量は
ノイズなし系列から採られることになる。系列の判別は、素の名前ではなくフレームの内容
(`jointvars_clean` の有無) で行うのが確実である。同じ注意は
`notes/ols-identification-procedure.md:104-108` にも記録されている。

### D.5 乱数の種

- `--seed <int>` で指定する (`simulators/simulator.py:67`, `:150`)。
- 指定しない場合は OS のエントロピーから引かれる。実際に使われた値は
  `rng.bit_generator.seed_seq.entropy` として取り出され (`sensors/sensors.py:38-40`)、
  目録ファイル最上位の `noise_seed` に記録される (`simulators/simulator.py:152-154`)。
  この項目は `base_transform` に入るので、ノイズあり・なしの両系列と train / valid / test の
  全分割に同じ値が載る (`recorders/standard_recorder.py:92-103`, `:178`)。

---

## E. 実際に生成したデータセット

現在の配布物は `exports/merged_datasets_20260825.zip` である。中身は次の 5 エントリのみ
(unzip -l で確認)。

```
datasets/hammer/merged_wrenchonly/transforms.json         (1,026,607 バイト)
datasets/hammer/merged_wrenchonly/dynamics.csv            (157,954 バイト)
datasets/loaded_dice/merged_wrenchonly/transforms.json    (1,028,512 バイト)
datasets/loaded_dice/merged_wrenchonly/dynamics.csv       (157,935 バイト)
notes/ols-identification-procedure.md                     (8,673 バイト、同定手順書)
```

加えて `complete/` と `masks/` の画像が計 1211 ファイル。`ground_truth.csv` は含まれない
(元のスプライン run に無かったため。`recorders/merge.py:160-162`)。

対象物体は 2 つで、モデルは `xml_models/targets/hammer` と `xml_models/targets/loaded_dice`。

### E.1 ノイズ条件

配布物の元になった run は、**関節側のノイズを切った条件**で生成されている。すなわち
`--noise-scale 0.0` により `sensors/sensors.py:24` の `jointpos_stddev` が全成分 0 になり、
関節位置・速度・加速度にノイズが乗らない。力覚センサ側のノイズは既定倍率 1.0 のまま有効である。

この条件は生成物から実測で確認した。

| データセット | ノイズあり系列とノイズなし系列の回帰行列 | レンチの最大差 |
| --- | --- | --- |
| `datasets/hammer/spline_rotonly_wrenchonly` | 完全一致 | 0.3517 |
| `datasets/hammer/excited_nomain10s_wrenchonly` | 完全一致 | 0.3868 |
| `datasets/loaded_dice/spline_rotonly_wrenchonly` | 完全一致 | 0.3517 |
| `datasets/loaded_dice/excited_nomain10s_wrenchonly` | 完全一致 | 0.3868 |

回帰行列は関節観測 (qpos / qvel / qacc) だけの関数なので、両系列で完全に一致することは
関節側のノイズが 0 であることを意味する。一方レンチは差があるので力覚側のノイズは有効である。

比較のため、**両側にノイズを乗せた版**が `datasets/<物体>/merged_v2` として併存する。こちらの
元 run (`spline_rotonly_v2` / `excited_nomain10s_v2`) では回帰行列が両系列で一致せず、
最大差は hammer で 0.6187、loaded_dice で 0.6186 である。すなわち `merged_v2` は
`noise_scale` を既定の 1.0 のまま実行した条件に対応する。

両条件とも `noise_seed` は 42 である (目録ファイル最上位から実測)。

### E.2 ディレクトリの対応

| 役割 | 関節ノイズなし (配布物) | 関節ノイズあり |
| --- | --- | --- |
| 画像側 run | `datasets/<obj>/spline_rotonly_wrenchonly` | `datasets/<obj>/spline_rotonly_v2` |
| 動力学側 run | `datasets/<obj>/excited_nomain10s_wrenchonly` | `datasets/<obj>/excited_nomain10s_v2` |
| 合成 | `datasets/<obj>/merged_wrenchonly` | `datasets/<obj>/merged_v2` |

`<obj>` は `hammer` または `loaded_dice`。対応関係は各合成データセットの `merge_sources` から
実測で確認した (C.4 参照)。

いずれの合成データセットもフレーム数 300 で、画像側 300 フレームをそのまま使い、動力学側
600 フレームを添字 0, 2, 4, …, 599 で等間隔に半分に抽出している。各行の項目は
`file_path`、`transform_matrix`、`pose_sen_obj`、`twist_sen`、`dtwist_sen`、`wrench`、
`regressor`、`image_source`、`dynamics_source` の 9 つである (実測)。

カメラ内部パラメータは両物体で共通で、`camera_angle_x` = `camera_angle_y` = π/4、
`cx` = `cy` = 400.0、`fl_x` = `fl_y` = 965.685424949238、`h` = `w` = 800。
`aabb_scale` は hammer が 0.195004812176059、loaded_dice が 0.0345 である (実測)。

### E.3 コマンド列

以下が 2 物体分を作った手順である。軌道 2 本は物体に依らず共通で、A.6 と B.4 のとおり
すでに生成済みのものを使う。

```sh
# --- 1. hammer: 画像側 run (スプライン軌道) ---
pixi run python main.py \
    --object xml_models/targets/hammer \
    --target-trajectory configurations/trajectories/spline_20260825_164121/trajectory.json \
    --recorder.dataset-dir datasets/hammer/spline_rotonly_wrenchonly \
    --noise-scale 0.0 \
    --seed 42

# --- 2. hammer: 動力学側 run (励起軌道) ---
pixi run python main.py \
    --object xml_models/targets/hammer \
    --target-trajectory configurations/trajectories/excited_nostop_nomain_10s/trajectory.json \
    --recorder.dataset-dir datasets/hammer/excited_nomain10s_wrenchonly \
    --noise-scale 0.0 \
    --seed 42

# --- 3. hammer: 合成 ---
pixi run merge-datasets \
    --spline-dir datasets/hammer/spline_rotonly_wrenchonly \
    --excited-dir datasets/hammer/excited_nomain10s_wrenchonly \
    --out-dir datasets/hammer/merged_wrenchonly

# --- 4. hammer: 動力学量の CSV 書き出し ---
pixi run export-dynamics-csv --dataset-dir datasets/hammer/merged_wrenchonly

# --- 5-8. loaded_dice: 同じ 4 手順を --object xml_models/targets/loaded_dice で繰り返す ---
pixi run python main.py \
    --object xml_models/targets/loaded_dice \
    --target-trajectory configurations/trajectories/spline_20260825_164121/trajectory.json \
    --recorder.dataset-dir datasets/loaded_dice/spline_rotonly_wrenchonly \
    --noise-scale 0.0 --seed 42
pixi run python main.py \
    --object xml_models/targets/loaded_dice \
    --target-trajectory configurations/trajectories/excited_nostop_nomain_10s/trajectory.json \
    --recorder.dataset-dir datasets/loaded_dice/excited_nomain10s_wrenchonly \
    --noise-scale 0.0 --seed 42
pixi run merge-datasets \
    --spline-dir datasets/loaded_dice/spline_rotonly_wrenchonly \
    --excited-dir datasets/loaded_dice/excited_nomain10s_wrenchonly \
    --out-dir datasets/loaded_dice/merged_wrenchonly
pixi run export-dynamics-csv --dataset-dir datasets/loaded_dice/merged_wrenchonly

# --- 9. 配布物の作成 ---
zip -r exports/merged_datasets_20260825.zip \
    datasets/hammer/merged_wrenchonly \
    datasets/loaded_dice/merged_wrenchonly \
    notes/ols-identification-procedure.md
```

`merged_v2` 側は、上の手順で `--noise-scale 0.0` を外し (既定の 1.0 になる)、
ディレクトリ名の `_wrenchonly` を `_v2` に置き換えたものである。

各フラグの定義位置:
`--object` (`simulators/simulator.py:51`)、
`--target-trajectory` (`simulators/simulator.py:59`)、
`--recorder.dataset-dir` (`recorders/standard_recorder.py:44`)、
`--noise-scale` (`simulators/simulator.py:64`)、
`--seed` (`simulators/simulator.py:67`)、
`--spline-dir` / `--excited-dir` / `--out-dir` / `--force` (`recorders/merge.py:167-176`)、
`--dataset-dir` / `--out-csv` / `--include-regressor` (`recorders/export_dynamics_csv.py:71-78`)。

**未確認**: 実際に打たれたコマンドの文字列そのものは、リポジトリ内のどの成果物にも記録されて
いない (シミュレーション run は設定を書き出さない。`config_export_path`
(`simulators/simulator.py:58`) は使われていない)。上のコマンド列は、コマンドライン定義と
生成物から再構成したものである。ただし次の各点は生成物から実測で裏付けた:
対象物体 2 つ、使用した軌道ファイル 2 本 (`merge_sources` の run ディレクトリ名と
フレーム数 300 / 600 の一致)、出力ディレクトリ名、関節ノイズが 0 であること (E.1 の表)、
乱数の種が 42 であること、`dynamics.csv` が回帰行列を含まない列構成であること
(実測で 36 列 = `frame` + `source_index` + `pose_sen_obj` 16 + `twist_sen` 6 +
`dtwist_sen` 6 + `wrench` 6、データ行 300。`--include-regressor` を付けた場合の 96 列ではない)。

### E.4 補足: この配布物に付随する同定手順

zip に同梱されている `notes/ols-identification-procedure.md` は、この合成データセットから
通常の最小二乗で慣性パラメータ 10 成分を推定する手順を記した別文書である。特に、回帰行列と
レンチが力覚センサ座標系で記録されているのに対して真値 `global_gt` が物体 AABB 座標系で
定義されているため、比較の前に座標変換が必要である点が記されている。

---

## F. 未確認事項の一覧

1. E.3 に記したコマンド文字列そのもの。成果物からの再構成であり、フラグの綴りや
   `pixi run` の環境指定 (`-e dev` の有無など) は裏付けられていない。

解決済みの項目を参考までに残す。

- 目録ファイル名の向き: コードの不整合ではなく、シミュレーション実行の直後に手動で
  入れ替えているためである (D.4 と E.3 を参照)。
- 積分刻み幅: `xml_models/` に `timestep` の指定が無いため MuJoCo の既定値が使われる。
  実測して 0.002 s (500 Hz) であることを確認した。記録の毎秒フレーム数 60 に対し、
  1 フレームあたり約 8.3 ステップが進む。
