# rigid_body_xl

Test project to get familiar with mujoco

## Setup

```sh
pixi install
```

[direnv](https://direnv.net/) を使っている場合, ディレクトリに入ると自動で環境が有効化される. 初回のみ `direnv allow` が必要.

Just memo:

```text
parent's body id of:
      worldbody (body_id==0): 0,
          link1 (body_id==1): 0,
          link2 (body_id==2): 1,
          link3 (body_id==3): 2,
          link4 (body_id==4): 3,
          link5 (body_id==5): 4,
          link6 (body_id==6): 5,
 target/ = worldbody in
     object.xml (body_id==7): 6,
  target/object (body_id==8): 7,
```

## Trajectory Generation

You can generate trajectories (Spline, Fourier, Excited) using the `generate-trajectory` command.

### Basic Usage

```sh
pixi run generate-trajectory spline --start-pos 1.0 1.0 1.0 0.0 0.0 0.0 --end-pos 0.2 1.4 0.6 3.14 0.0 25.13
```

### Using Configuration File

Specify the trajectory type as a subcommand, then use `--config` to load a YAML configuration file:

```sh
pixi run generate-trajectory spline --config configurations/trajectory_generation/spline_6dof.yaml
```

CLI arguments override values from the config file:

```sh
pixi run generate-trajectory spline --config configurations/trajectory_generation/spline_6dof.yaml --duration 10.0
```

Supported subcommands: `spline`, `fourier`, `excited`.

## Joint observation noise

Joint observations add independent zero-mean Gaussian
noise directly to MuJoCo position, velocity, and acceleration. It does not
finite-difference or smooth joint observations. Standard deviations per axis:

| Component | Position | Velocity | Acceleration |
| --- | --- | --- | --- |
| Translation | 0.00001 m | 0.0005 m/s | 0.075 m/s² |
| Rotation | 0.00015 rad | 0.0075 rad/s | 1.125 rad/s² |

The velocity and acceleration scales are respectively 50 and 7500 times the
position standard deviation. FT noise retains the empirical correlated,
quantized model. `--no-record-noise` uses raw MuJoCo observations.
Control independently selects the same noisy state or the raw MuJoCo state
with `--control-noise` or `--no-control-noise`. Position differencing, acceleration
smoothing, and the legacy noise profiles have been removed.

For a paired 2×3 observation plot without image/video recording, run:

```sh
pixi run python -m experiments.compare_joint_noise \
  --object xml_models/targets/sledgehammer \
  --target-trajectory /path/to/trajectory.json \
  --no-control-noise --seed 42 \
  --recorder.dataset-dir /tmp/joint_noise_comparison
```

This compares noisy and clean observations of the same motion and writes PNG/SVG
plots, joint-state arrays, configuration, and an OLS comparison CSV.
