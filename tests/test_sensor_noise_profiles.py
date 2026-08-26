import json

import numpy as np
import pytest
from mujoco._structs import MjData, MjModel

from sensors.noise_profiles import get_noise_profile
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


def test_empirical_joint_velocity_is_derived_from_one_position_stream(model_data):
    model, data = model_data
    velocity = np.array([0.02, -0.01, 0.03, 0.4, -0.2, 0.1])
    data.qvel[:] = velocity
    sensors = Sensors(model, data, fps=60.0, noise_scale=0.0, seed=4, noise_profile="empirical")

    for sample_index in range(30):
        data.time = sample_index * model.opt.timestep
        data.qpos[:] = velocity * data.time
        observation = sensors.sample_jointvars()
        assert np.allclose(observation[0], data.qpos)
        assert np.allclose(observation[1], velocity)
        assert np.allclose(observation[2], 0.0, atol=1e-12)


def test_empirical_stationary_joint_noise_has_hardware_scale(model_data):
    model, data = model_data
    sensors = Sensors(model, data, fps=60.0, seed=7, noise_profile="empirical")
    samples = []
    for sample_index in range(20_000):
        data.time = sample_index * model.opt.timestep
        samples.append(sensors.sample_jointvars())
    samples = np.asarray(samples)[2_000:, :, 3:]

    assert np.allclose(samples[:, 0].std(axis=0), 1.5e-5, rtol=0.08)
    assert np.allclose(samples[:, 1].std(axis=0), 6.24e-4, rtol=0.10)
    assert np.allclose(samples[:, 2].std(axis=0), 6.2e-3, rtol=0.15)

    metadata = sensors.metadata()
    assert metadata["profile"] == "empirical"
    assert metadata["joint_model"] == "derived"
    json.dumps(metadata)


def test_joint_sample_is_shared_at_one_simulation_time(model_data):
    model, data = model_data
    sensors = Sensors(model, data, fps=60.0, seed=8, noise_profile="empirical")
    first = sensors.sample_jointvars()
    second = sensors.sample_jointvars()
    assert np.array_equal(first, second)


def test_control_observation_uses_noisy_position_but_instantaneous_velocity(model_data):
    model, data = model_data
    data.qvel[:] = np.arange(6) + 0.25
    data.qacc[:] = np.arange(6) - 0.5
    sensors = Sensors(model, data, fps=60.0, seed=8, noise_profile="empirical")

    recorded = sensors.sample_jointvars()
    control = sensors.sample_control_jointvars()

    assert np.array_equal(control[0], recorded[0])
    assert np.array_equal(control[1], data.qvel)
    assert np.array_equal(control[2], data.qacc)
    assert not np.array_equal(control[1], recorded[1])
    assert np.array_equal(sensors.sample_control_jointvars(derived_velocity=True), recorded)


def test_joint_rng_is_independent_of_wrench_consumption(model_data):
    model, data_a = model_data
    data_b = MjData(model)
    sensors_a = Sensors(model, data_a, fps=60.0, seed=12, noise_profile="empirical")
    sensors_b = Sensors(model, data_b, fps=60.0, seed=12, noise_profile="empirical")

    for sample_index in range(20):
        time = sample_index * model.opt.timestep
        data_a.time = time
        data_b.time = time
        assert np.array_equal(sensors_a.sample_jointvars(), sensors_b.sample_jointvars())
        sensors_a.get("wrench", perturbed=True)


def test_empirical_wrench_matches_checked_in_statistics(model_data):
    model, data = model_data
    sensors = Sensors(model, data, fps=60.0, seed=123, noise_profile="empirical")
    samples = []
    for sample_index in range(20_000):
        data.time = sample_index / 60.0
        samples.append(sensors.get("wrench", perturbed=True))
    samples = np.asarray(samples)

    quantization = np.array([0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
    assert np.allclose(samples / quantization, np.round(samples / quantization), atol=1e-10)

    profile = get_noise_profile("empirical")
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
        noise_profile="empirical",
        translation_noise_scale=0.0,
        rotation_noise_scale=2.0,
    )
    assert np.array_equal(sensors.jointpos_stddev[:3], np.zeros(3))
    assert np.allclose(sensors.jointpos_stddev[3:], 3.0e-5)


def test_four_cell_noise_switches_are_explicit():
    config = SimulatorConfig()
    assert config.noise_profile == "empirical"
    assert config.control_noise is True
    assert config.control_derived_velocity is False
    assert config.record_noise is True
    assert config.record_joint_noise is True
    assert config.record_wrench_noise is True
    assert config.translation_noise_scale == 1.0
    assert config.rotation_noise_scale == 1.0


def test_legacy_profile_and_unknown_profile():
    legacy = get_noise_profile("legacy")
    assert legacy.joint_model == "independent_gaussian"
    assert legacy.wrench_model == "independent_gaussian"
    with pytest.raises(ValueError, match="Unknown noise profile"):
        get_noise_profile("not-a-profile")
