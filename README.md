# rigid_body_xl

Test project to get familiar with mujoco

## Requirements installation

```sh
pip install -r requirements.txt
```

In addition to the above, you may need to run the following to avoid [this error which would occur when creating cv2 videowriter](https://stackoverflow.com/questions/72540359/glibcxx-3-4-30-not-found-for-librosa-in-conda-virtual-environment-after-tryin)

```sh
conda install -c conda-forge gcc=12.1.0
```

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
uv run generate-trajectory spline --start-pos 1.0 1.0 1.0 0.0 0.0 0.0 --end-pos 0.2 1.4 0.6 3.14 0.0 25.13
```

### Using Configuration File

Specify the trajectory type as a subcommand, then use `--config` to load a YAML configuration file:

```sh
uv run generate-trajectory spline --config configurations/trajectory_generation/spline_6dof.yaml
```

CLI arguments override values from the config file:

```sh
uv run generate-trajectory spline --config configurations/trajectory_generation/spline_6dof.yaml --duration 10.0
```

Supported subcommands: `spline`, `fourier`, `excited`.
