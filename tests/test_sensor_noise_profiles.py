import numpy as np
import pytest
from mujoco._structs import MjData, MjModel

from sensors.noise_profiles import NOISE_PROFILE
from sensors.sensors import Sensors
from simulators.simulator import SimulatorConfig

_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="link">
      <joint name="j0" type="slide" axis="1 0 0"/>
      <joint name="j1" type="slide" axis="0 1 0"/>
      <joint name="j2" type="slide" axis="0 0 1"/>
      <joint name="j3" type="hinge" axis="1 0 0"/>
      <joint name="j4" type="hinge" axis="0 1 0"/>
      <joint name="j5" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <site name="s"/>
    </body>
  </worldbody>
  <sensor>
    <force name="force" site="s"/>
    <torque name="torque" site="s"/>
  </sensor>
</mujoco>
"""


@pytest.fixture
def model_data():
    model = MjModel.from_xml_string(_XML)
    return model, MjData(model)


def test_joint_sample_is_shared_at_one_simulation_time(model_data):
    model, data = model_data
    sensors = Sensors(model, data, fps=60.0, seed=8)
    first = sensors.sample_jointvars()
    second = sensors.sample_jointvars()
    assert np.array_equal(first, second)


def test_joint_rng_is_independent_of_wrench_consumption(model_data):
    model, data_a = model_data
    data_b = MjData(model)
    sensors_a = Sensors(model, data_a, fps=60.0, seed=12)
    sensors_b = Sensors(model, data_b, fps=60.0, seed=12)

    for sample_index in range(20):
        time = sample_index * model.opt.timestep
        data_a.time = time
        data_b.time = time
        assert np.array_equal(sensors_a.sample_jointvars(), sensors_b.sample_jointvars())
        sensors_a.get("wrench", perturbed=True)


def test_empirical_wrench_matches_checked_in_statistics(model_data):
    model, data = model_data
    sensors = Sensors(model, data, fps=60.0, seed=123)
    samples = []
    for sample_index in range(20_000):
        data.time = sample_index / 60.0
        samples.append(sensors.get("wrench", perturbed=True))
    samples = np.asarray(samples)

    quantization = np.array([0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
    assert np.allclose(samples / quantization, np.round(samples / quantization), atol=1e-10)

    profile = NOISE_PROFILE
    expected_std = np.asarray(profile.wrench_stddev)
    assert np.allclose(samples.std(axis=0), expected_std, rtol=0.12)

    lag1 = np.array([np.corrcoef(samples[:-1, axis], samples[1:, axis])[0, 1] for axis in range(6)])
    assert np.allclose(lag1, profile.wrench_lag1, atol=0.06)

    correlation = np.corrcoef(samples, rowvar=False)
    expected_correlation = np.asarray(profile.wrench_correlation)
    assert correlation[1, 3] == pytest.approx(expected_correlation[1, 3], abs=0.06)
    assert correlation[0, 4] == pytest.approx(expected_correlation[0, 4], abs=0.06)


def test_translation_and_rotation_scales_are_independent(model_data):
    model, data = model_data
    sensors = Sensors(
        model,
        data,
        fps=60.0,
        seed=5,
        translation_noise_scale=0.0,
        rotation_noise_scale=2.0,
    )
    assert np.array_equal(sensors.jointpos_stddev[:3], np.zeros(3))
    assert np.allclose(sensors.jointpos_stddev[3:], 3.0e-4)


def test_four_cell_noise_switches_are_explicit():
    config = SimulatorConfig()
    assert config.control_noise is True
    assert config.record_noise is True
    assert config.record_joint_noise is True
    assert config.record_wrench_noise is True
    assert config.translation_noise_scale == 1.0
    assert config.rotation_noise_scale == 1.0


def test_independent_50_150_noise_tracks_raw_states_with_independent_residuals(model_data):
    model, data = model_data
    sensors = Sensors(model, data, fps=60, seed=42)
    sigma = np.array([1e-5] * 3 + [1.5e-4] * 3)[None, :] * np.array([1, 50, 7500])[:, None]
    residuals = []
    for i in range(10000):
        data.time = i * 0.002
        data.qpos[:] = np.sin(i * 0.1)
        data.qvel[:] = np.cos(i * 0.3)
        data.qacc[:] = i % 7 - 3
        raw = np.stack((data.qpos, data.qvel, data.qacc))
        observed = sensors.sample_jointvars()
        residuals.append((observed - raw) / sigma)
        if i == 0:
            np.testing.assert_array_equal(observed, sensors.sample_jointvars())
            np.testing.assert_array_equal(sensors.get("jointvars", perturbed=False), raw)
    residuals = np.array(residuals).reshape(-1, 18)
    np.testing.assert_allclose(residuals.std(axis=0), 1, atol=0.04)
    assert np.max(np.abs(residuals.mean(axis=0))) < 0.04
    assert np.max(np.abs(np.corrcoef(residuals, rowvar=False) - np.eye(18))) < 0.04
    assert abs(np.corrcoef(residuals[:-1, 0], residuals[1:, 0])[0, 1]) < 0.04
    np.testing.assert_allclose(sensors.metadata()["jointvel_stddev"], sigma[1])
    np.testing.assert_allclose(sensors.metadata()["jointacc_stddev"], sigma[2])


def test_zero_joint_noise_returns_raw_states_without_history(model_data):
    model, data = model_data
    sensors = Sensors(model, data, fps=60, noise_scale=0, seed=42)
    for i in range(4):
        data.time = i * 0.002
        data.qpos[:] = i
        data.qvel[:] = 3 - i
        data.qacc[:] = 7 * i
        np.testing.assert_array_equal(sensors.sample_jointvars(), np.stack((data.qpos, data.qvel, data.qacc)))
