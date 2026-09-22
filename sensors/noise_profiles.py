from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseProfile:
    """Parameters for a joint-state and wrench observation model."""

    name: str
    joint_model: str
    jointpos_stddev: tuple[float, ...]
    wrench_model: str
    wrench_stddev: tuple[float, ...]
    wrench_lag1: tuple[float, ...]
    wrench_correlation: tuple[tuple[float, ...], ...]
    wrench_sample_rate_hz: float
    wrench_quantization: tuple[float, ...]
    jointvel_noise_scaler: float = 50.0
    jointacc_noise_scaler: float = 7500.0


_FT300_GOOD_CORRELATION = (
    (1.0, -0.1243749925814589, 0.2468252620755953, 0.0780073129869112, -0.2960004638408706, 0.0055611010236395),
    (-0.1243749925814589, 1.0, -0.2476863180824558, 0.5433630519250063, -0.0981031263453956, 0.0903804273651691),
    (0.2468252620755953, -0.2476863180824558, 1.0, 0.2579717751774338, 0.2588044783669669, -0.0951006359634579),
    (0.0780073129869112, 0.5433630519250063, 0.2579717751774338, 1.0, 0.1427854190040238, 0.0292830624578525),
    (-0.2960004638408706, -0.0981031263453956, 0.2588044783669669, 0.1427854190040238, 1.0, 0.0263623493863203),
    (0.0055611010236395, 0.0903804273651691, -0.0951006359634579, 0.0292830624578525, 0.0263623493863203, 1.0),
)

_FT300_GOOD_LAG1 = (
    0.2872770496752695,
    0.1564620400551030,
    0.4244461300071886,
    0.07272933628257847,
    0.2351782887624900,
    0.3199505458004590,
)


NOISE_PROFILE = NoiseProfile(
    name="independent_50_150",
    joint_model="independent_gaussian",
    jointpos_stddev=(1.0e-5, 1.0e-5, 1.0e-5, 1.5e-4, 1.5e-4, 1.5e-4),
    wrench_model="var1_quantized",
    # FT 300-S manual §5.3: standard deviations measured over one second.
    wrench_stddev=(0.1, 0.1, 0.1, 0.005, 0.005, 0.003),
    wrench_lag1=_FT300_GOOD_LAG1,
    wrench_correlation=_FT300_GOOD_CORRELATION,
    wrench_sample_rate_hz=60.0,
    wrench_quantization=(0.01, 0.01, 0.01, 0.001, 0.001, 0.001),
)


def covariance_from_profile(profile: NoiseProfile, force_scale: float, torque_scale: float) -> np.ndarray:
    stddev = np.asarray(profile.wrench_stddev, dtype=float)
    stddev *= np.array([force_scale] * 3 + [torque_scale] * 3)
    correlation = np.asarray(profile.wrench_correlation, dtype=float)
    return correlation * np.outer(stddev, stddev)
