import numpy as np
import pytest
from mujoco._structs import MjData, MjModel

from sensors.sensors import Sensors

_XML = """
<mujoco>
  <worldbody>
    <body name="link">
      <joint name="j" type="slide" axis="1 0 0"/>
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
    m = MjModel.from_xml_string(_XML)
    return m, MjData(m)


def _wrench_stream(model_data, seed, n=5):
    m, d = model_data
    sensors = Sensors(m, d, fps=30.0, seed=seed)
    return sensors.seed, np.array([sensors.get("wrench", perturbed=True) for _ in range(n)])


def test_same_seed_reproduces_noise(model_data):
    _, a = _wrench_stream(model_data, 42)
    _, b = _wrench_stream(model_data, 42)
    assert np.array_equal(a, b)


def test_different_seed_changes_noise(model_data):
    _, a = _wrench_stream(model_data, 42)
    _, c = _wrench_stream(model_data, 99)
    assert not np.array_equal(a, c)


def test_seed_is_recorded(model_data):
    assert _wrench_stream(model_data, 42)[0] == 42
    # No seed given: the entropy actually drawn is exposed so the run can be replayed
    assert isinstance(_wrench_stream(model_data, None)[0], int)
