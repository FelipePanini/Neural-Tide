"""F1 debug overlay: smoothed vs raw landmarks, hand labels and a stats panel."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from config import RGBA, RGBA8, DebugConfig
from core.easing import damp
from core.hand import HAND_CONNECTIONS, INDEX_TIP, RIGHT, THUMB_TIP, WRIST, SourceStatus, finger_bends
from core.hand_state import HandState
from render.renderer import Renderer

_BONES = np.array(HAND_CONNECTIONS)
_MIN_PINCH_LINE = 0.02


@dataclass(frozen=True)
class DebugInfo:
    """Everything the stats panel shows besides the hands themselves."""

    render_fps: float
    source: SourceStatus
    mode: str
    gesture: str
    background: str
    details: tuple[str, ...] = ()  # lines from the active effect
    hints: tuple[str, ...] = ()


def _faded(color: RGBA, alpha: float) -> tuple[float, float, float, float]:
    return (color[0], color[1], color[2], color[3] * alpha)


def _side(hand: HandState) -> str:
    return "direita" if hand.handedness == RIGHT else "esquerda"


class DebugOverlay:
    """Draws on top of everything; fades in and out when toggled."""

    def __init__(self, renderer: Renderer, cfg: DebugConfig) -> None:
        self._renderer = renderer
        self._cfg = cfg
        self._visible = cfg.start_visible
        self._opacity = 1.0 if cfg.start_visible else 0.0
        self._hud_lines: list[tuple[str, RGBA8]] = []
        self._hud_age = float("inf")

    @property
    def visible(self) -> bool:
        return self._visible

    def toggle(self) -> None:
        self._visible = not self._visible

    def draw(self, hands: Sequence[HandState], info: DebugInfo, dt: float) -> None:
        self._opacity = damp(self._opacity, 1.0 if self._visible else 0.0, self._cfg.fade_tau_s, dt)
        if self._opacity < 0.005:
            return
        self._renderer.use_alpha_blending()
        self._hud_age += dt
        if self._hud_age >= 1.0 / self._cfg.hud_refresh_hz:
            self._hud_lines = self._build_hud(hands, info)
            self._hud_age = 0.0
        self._draw_panel()  # first, so hands passing behind it stay bright
        self._draw_hands(hands)
        self._draw_labels(hands)

    def _draw_hands(self, hands: Sequence[HandState]) -> None:
        cfg = self._cfg
        prims = self._renderer.primitives
        viewport = self._renderer.viewport
        for hand in hands:
            alpha = hand.presence * self._opacity
            pts = hand.points
            prims.add_lines(pts[_BONES[:, 0]], pts[_BONES[:, 1]], cfg.bone_px, _faded(cfg.bone_color, alpha), viewport)
            if hand.pinch > _MIN_PINCH_LINE:
                prims.add_lines(
                    pts[THUMB_TIP],
                    pts[INDEX_TIP],
                    cfg.pinch_line_px,
                    _faded(cfg.pinch_color, alpha * hand.pinch),
                    viewport,
                )
            prims.add_points(hand.raw_points, cfg.raw_landmark_px, _faded(cfg.raw_color, alpha))
            prims.add_points(pts, cfg.landmark_px, _faded(cfg.filtered_color, alpha))
            tip_colors = [
                _faded(cfg.tip_extended_color if up else cfg.tip_curled_color, alpha) for up in hand.fingers
            ]
            prims.add_points(hand.tips, cfg.tip_px, np.array(tip_colors))
        prims.flush(viewport)

    def _draw_labels(self, hands: Sequence[HandState]) -> None:
        cfg = self._cfg
        text = self._renderer.text
        viewport = self._renderer.viewport
        for hand in hands:
            label = f"{_side(hand)} · {hand.finger_count}"
            tex = text.texture([(label, cfg.text_color)], cfg.label_font_size, background=cfg.panel_color, padding=4)
            x, y = viewport.world_to_pixels(hand.points[WRIST])
            text.draw(
                tex,
                (x, y + cfg.label_offset_px),
                self._renderer.size,
                anchor=(0.5, 0.0),
                opacity=hand.presence * self._opacity,
            )

    def _build_hud(self, hands: Sequence[HandState], info: DebugInfo) -> list[tuple[str, RGBA8]]:
        src = info.source
        lines = ["NEURAL TIDE · debug [F1]", f"render    {info.render_fps:5.1f} fps"]
        if src.label == "webcam":
            res = f"{src.resolution[0]}x{src.resolution[1]}" if src.resolution else "--"
            lines.append(f"câmera    {src.camera_fps:5.1f} fps  {res}")
            lines.append(f"tracking  {src.tracking_fps:5.1f} fps  {src.inference_ms:4.1f} ms")
        else:
            lines.append(f"entrada   {src.label}  {src.tracking_fps:5.1f} Hz")
        visible = [h for h in hands if h.visible]
        lines.append(f"mãos      {len(visible)}")
        for hand in visible:
            pattern = "".join("|" if up else "_" for up in hand.fingers)
            bends = " ".join(f"{b:3.0f}" for b in finger_bends(hand.world))
            lines.append(f"  {_side(hand):8s} {pattern}  pinça {hand.pinch * 100:3.0f}%  dobras {bends}")
        lines.append(f"modo      {info.mode}")
        lines += [f"          {detail}" for detail in info.details]
        lines += [f"gesto     {info.gesture}", f"fundo     {info.background}  [C]"]
        lines += list(info.hints)

        color = self._cfg.text_color
        result = [(line, color) for line in lines]
        if src.error:
            result.append((f"! {src.error}", self._cfg.warning_color))
        return result

    def _draw_panel(self) -> None:
        cfg = self._cfg
        text = self._renderer.text
        tex = text.texture(self._hud_lines, cfg.font_size, background=cfg.panel_color, padding=cfg.panel_padding_px)
        text.draw(tex, (cfg.margin_px, cfg.margin_px), self._renderer.size, opacity=self._opacity)
