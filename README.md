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

Generate trajectories using the `generate-trajectory` command (powered by Hydra).

### Basic Usage

```sh
# Default spline trajectory
uv run generate-trajectory

# Select trajectory type
uv run generate-trajectory trajectory=fourier

# Override parameters
uv run generate-trajectory trajectory=spline duration=10.0 fps=120
```

### Using Configuration Files

Configuration files are located in `configurations/trajectory_generation/`:

```sh
# Use spline_6dof config
uv run generate-trajectory trajectory=spline_6dof

# Use fourier_6dof config
uv run generate-trajectory trajectory=fourier_6dof
```

### Available Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `trajectory` | Trajectory type/config | `spline`, `fourier`, `spline_6dof` |
| `duration` | Duration in seconds | `10.0` |
| `fps` | Frames per second | `120` |
| `show_plot` | Show plot window | `true` |
| `plot_path` | Save plot to file | `output.png` |
| `json_path` | Save trajectory JSON | `output.json` |
| `type` | Spline type | `quintic`, `septic` |
| `num_harmonics` | Fourier harmonics | `5` |

### Math Expressions in YAML

You can use `${pi:}` and `${eval:...}` in configuration files:

```yaml
end_pos:
  - ${pi:}           # π (3.14159...)
  - ${eval:8*pi}     # 8π
  - ${eval:pi/2}     # π/2
```

### Examples

```sh
# Spline with septic interpolation
uv run generate-trajectory trajectory=spline type=septic duration=5.0 plot_path=spline.png

# Fourier with custom harmonics
uv run generate-trajectory trajectory=fourier num_harmonics=7 duration=8.0 json_path=fourier.json

# Show plot interactively
uv run generate-trajectory trajectory=spline_6dof show_plot=true
```

