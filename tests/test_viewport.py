"""Camera-image ↔ world ↔ pixel mapping."""

from __future__ import annotations

import numpy as np
import pytest

from core.viewport import Viewport


@pytest.mark.parametrize("image_aspect", [4 / 3, 16 / 9, 21 / 9])
def test_image_centre_maps_to_world_origin(image_aspect: float) -> None:
    vp = Viewport(1280, 720, image_aspect)
    np.testing.assert_allclose(vp.image_to_world(np.array([0.5, 0.5])), [0.0, 0.0], atol=1e-12)


def test_same_aspect_maps_corners_to_corners() -> None:
    vp = Viewport(1280, 720, 16 / 9)
    corners = vp.image_to_world(np.array([[0.0, 0.0], [1.0, 1.0]]))
    np.testing.assert_allclose(corners, [[-vp.aspect, 1.0], [vp.aspect, -1.0]])


def test_cover_fit_crops_the_taller_image() -> None:
    vp = Viewport(1280, 720, 4 / 3)  # 4:3 camera in a 16:9 window: top/bottom cropped
    u0, v0, du, dv = vp.uv_rect
    assert (u0, du) == (0.0, 1.0)
    assert dv == pytest.approx(0.75)
    assert vp.image_to_world(np.array([0.5, 0.0]))[1] > 1.0  # top edge of the image is off-screen


@pytest.mark.parametrize("image_aspect", [4 / 3, 16 / 9])
def test_screen_to_image_round_trips_through_world_to_pixels(image_aspect: float) -> None:
    vp = Viewport(1280, 720, image_aspect)
    for px, py in [(0, 0), (640, 360), (1000, 123)]:
        uv = np.array(vp.screen_to_image(px, py))
        np.testing.assert_allclose(vp.world_to_pixels(vp.image_to_world(uv)), [px, py], atol=1e-9)
