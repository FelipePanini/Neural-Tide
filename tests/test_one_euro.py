"""One Euro filter behaviour."""

from __future__ import annotations

import numpy as np

from core.filters import OneEuroFilter

RATE_HZ = 30.0


def _run(filt: OneEuroFilter, samples: np.ndarray) -> np.ndarray:
    return np.array([filt(s, i / RATE_HZ) for i, s in enumerate(samples)])


def test_first_sample_passes_through() -> None:
    filt = OneEuroFilter(1.0, 10.0)
    x = np.array([[0.3, 0.7]])
    np.testing.assert_allclose(filt(x, 0.0), x)


def test_constant_signal_is_unchanged() -> None:
    samples = np.tile([[0.25, 0.5, -0.1]], (60, 1, 1))
    out = _run(OneEuroFilter(1.0, 10.0), samples)
    np.testing.assert_allclose(out, samples, atol=1e-12)


def test_reduces_jitter_of_a_still_hand() -> None:
    rng = np.random.default_rng(0)
    samples = 0.5 + rng.normal(0.0, 0.003, (300, 21, 2))
    out = _run(OneEuroFilter(1.0, 10.0), samples)
    assert out[30:].std() < 0.45 * samples[30:].std()  # a 1 Hz low-pass alone reaches ~0.31


def test_fast_motion_lags_less_with_higher_beta() -> None:
    t = np.arange(90) / RATE_HZ
    ramp = np.stack((t * 1.5, np.zeros_like(t)), axis=1)[:, None, :]  # 1.5 screens per second
    lag_low = np.abs(_run(OneEuroFilter(1.0, 0.0), ramp) - ramp)[-1].max()
    lag_high = np.abs(_run(OneEuroFilter(1.0, 10.0), ramp) - ramp)[-1].max()
    assert lag_high < 0.25 * lag_low


def test_converges_after_a_step() -> None:
    samples = np.concatenate((np.zeros((10, 1, 2)), np.ones((60, 1, 2))))
    out = _run(OneEuroFilter(1.0, 10.0), samples)
    np.testing.assert_allclose(out[-1], 1.0, atol=1e-3)


def test_repeated_timestamp_keeps_previous_estimate() -> None:
    filt = OneEuroFilter(1.0, 10.0)
    filt(np.zeros((1, 2)), 0.0)
    first = filt(np.ones((1, 2)), 0.1)
    np.testing.assert_allclose(filt(np.full((1, 2), 5.0), 0.1), first)


def test_cutoff_uses_vector_speed_so_axes_get_equal_smoothing() -> None:
    filt = OneEuroFilter(1.0, 10.0)
    filt(np.zeros((1, 2)), 0.0)
    out = filt(np.array([[0.1, 0.1]]), 1 / RATE_HZ)
    assert out[0, 0] == out[0, 1]
