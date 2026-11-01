"""Central configuration: every tunable number of Neural Tide lives here.

All sections are frozen dataclasses so a value can only change by building a
new config (``dataclasses.replace``), which keeps threads from seeing
half-updated settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
SHADERS_DIR = ROOT_DIR / "render" / "shaders"
RECORDINGS_DIR = ROOT_DIR / "recordings"

RGB = tuple[float, float, float]
RGBA = tuple[float, float, float, float]
RGBA8 = tuple[int, int, int, int]


@dataclass(frozen=True)
class ModelFile:
    """A MediaPipe model: where it lives on disk and where to download it."""

    path: Path
    url: str


@dataclass(frozen=True)
class ModelsConfig:
    hand_landmarker: ModelFile = ModelFile(
        MODELS_DIR / "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task",
    )
    # Used by mode 7 (particle body); optional until then.
    selfie_segmenter: ModelFile = ModelFile(
        MODELS_DIR / "selfie_segmenter.tflite",
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter/float16/latest/selfie_segmenter.tflite",
    )
    download_chunk_bytes: int = 1 << 16
    download_timeout_s: float = 30.0


@dataclass(frozen=True)
class WindowConfig:
    title: str = "Neural Tide"
    width: int = 1280
    height: int = 720
    fullscreen: bool = False
    resizable: bool = True
    vsync: bool = True
    max_fps: int = 0  # 0 = only vsync limits the frame rate
    max_dt: float = 1.0 / 20.0  # clamp long frames so the simulation never explodes
    gl_version: tuple[int, int] = (3, 3)


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True  # selfie view: moving your right hand moves the right side of the screen
    backend: str = "auto"  # auto | dshow | msmf | avfoundation | v4l2 | any
    fourcc: str = "MJPG"  # lets most notebook webcams deliver 720p at 30 fps
    buffer_size: int = 1  # keep only the newest frame in the driver queue
    retry_delay_s: float = 1.0
    retry_max_delay_s: float = 5.0  # retries back off up to this delay
    read_failures_before_reopen: int = 30
    read_failure_sleep_s: float = 0.01


@dataclass(frozen=True)
class OneEuroConfig:
    """One Euro filter parameters (Casiez et al. 2012).

    ``min_cutoff`` (Hz) sets how smooth a still hand is (lower = smoother, more
    lag); ``beta`` sets how fast the cutoff rises with speed (higher = less lag
    when moving fast). Units of ``beta`` follow the filtered coordinates.
    """

    min_cutoff: float
    beta: float
    d_cutoff: float = 1.0


@dataclass(frozen=True)
class TrackerConfig:
    num_hands: int = 2
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # MediaPipe Tasks 1.2 labels handedness as seen in a NON-mirrored photo
    # (checked on the official sample image), so labels are swapped whenever
    # the camera feed is mirrored. Flip this if the overlay names the wrong hand.
    model_expects_mirrored: bool = False
    # Normalised image coordinates (0..1 across the frame).
    image_filter: OneEuroConfig = field(default_factory=lambda: OneEuroConfig(1.0, 10.0))
    # World landmarks are in metres around the hand centre.
    world_filter: OneEuroConfig = field(default_factory=lambda: OneEuroConfig(1.0, 20.0))
    # A hand missing for less than this keeps its slot (reported as stale).
    lost_timeout_s: float = 0.25
    # Max palm-centre jump (normalised image units) still matched to the same slot.
    match_max_distance: float = 0.25
    # Extra matching cost when the handedness label disagrees with the slot.
    handedness_penalty: float = 0.1
    # Weight of the newest handedness vote in the per-slot moving average.
    handedness_ema: float = 0.25
    inference_ema: float = 0.1
    idle_wait_s: float = 0.1  # how long the tracker waits for a new camera frame


@dataclass(frozen=True)
class HandGeometryConfig:
    """Thresholds that turn landmarks into finger states and pinch strength."""

    finger_max_bend_deg: float = 100.0  # summed joint bend below this = finger extended
    thumb_max_bend_deg: float = 70.0
    thumb_min_spread: float = 1.1  # thumb tip→pinky MCP distance / palm width
    pinch_open_ratio: float = 0.6  # thumb–index distance / palm size: pinch starts
    pinch_closed_ratio: float = 0.25  # ... pinch is complete


@dataclass(frozen=True)
class HandStateConfig:
    """Per-frame easing applied on the render thread (on top of One Euro)."""

    position_tau_s: float = 0.03
    velocity_tau_s: float = 0.08
    pinch_tau_s: float = 0.06
    presence_in_tau_s: float = 0.12
    presence_out_tau_s: float = 0.35
    presence_epsilon: float = 0.01


@dataclass(frozen=True)
class MockConfig:
    """Mouse-driven fake hand (``python main.py --mock``)."""

    hand_height: float = 0.34  # wrist-to-fingertip span as a fraction of the image height
    curl_tau_s: float = 0.08
    pinch_tau_s: float = 0.06
    tilt_deg: float = 10.0  # the hand leans outwards as it moves to the sides
    jitter: float = 0.002  # landmark noise (image units) so the One Euro filter has work to do
    default_fingers: int = 5
    # Scrolling this much (one mouse-wheel notch = 1.0) changes the finger count by one.
    wheel_step: float = 1.0
    # Pause after each change so a trackpad swipe does not skip straight from 0 to 5.
    wheel_cooldown_s: float = 0.15
    seed: int = 7


@dataclass(frozen=True)
class BackgroundConfig:
    start_with_camera: bool = True
    fade_tau_s: float = 0.25
    camera_dim: float = 0.3
    camera_desaturate: float = 0.6
    camera_tint: RGB = (0.75, 0.92, 1.1)
    vignette: float = 0.9


@dataclass(frozen=True)
class DebugConfig:
    start_visible: bool = True
    fade_tau_s: float = 0.12
    hud_refresh_hz: float = 10.0
    rate_window_s: float = 1.0
    font_names: str = "consolas,menlo,dejavusansmono,couriernew,monospace"
    font_size: int = 16
    label_font_size: int = 14
    text_cache_size: int = 96
    margin_px: int = 14
    panel_padding_px: int = 10
    line_spacing_px: int = 3
    label_offset_px: float = 26.0
    panel_color: RGBA8 = (4, 10, 20, 175)
    text_color: RGBA8 = (175, 235, 255, 255)
    warning_color: RGBA8 = (255, 190, 90, 255)
    landmark_px: float = 7.0
    raw_landmark_px: float = 5.0
    tip_px: float = 12.0
    bone_px: float = 2.0
    pinch_line_px: float = 3.0
    filtered_color: RGBA = (0.35, 0.95, 1.0, 0.95)
    raw_color: RGBA = (1.0, 0.3, 0.6, 0.6)
    bone_color: RGBA = (0.3, 0.75, 1.0, 0.6)
    tip_extended_color: RGBA = (0.6, 1.0, 0.8, 1.0)
    tip_curled_color: RGBA = (0.3, 0.45, 0.6, 0.8)
    pinch_color: RGBA = (1.0, 0.85, 0.3, 1.0)


@dataclass(frozen=True)
class RenderConfig:
    max_points: int = 4096
    max_line_segments: int = 4096
    line_feather_px: float = 1.5  # extra quad width used for anti-aliasing


@dataclass(frozen=True)
class Config:
    models: ModelsConfig = field(default_factory=ModelsConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    geometry: HandGeometryConfig = field(default_factory=HandGeometryConfig)
    hand_state: HandStateConfig = field(default_factory=HandStateConfig)
    mock: MockConfig = field(default_factory=MockConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    render: RenderConfig = field(default_factory=RenderConfig)


CONFIG = Config()
