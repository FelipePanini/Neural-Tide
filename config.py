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
class GestureConfig:
    """Gesture state machine: debounce, holds, swipes and cooldowns."""

    debounce_frames: int = 4  # a pose must be seen this many frames in a row to count
    cooldown_s: float = 0.8  # after any discrete action, nothing else fires for this long
    presence_min: float = 0.5  # hands still fading in or out are ignored
    # Holds only progress while the palm is (almost) still.
    hold_max_speed: float = 0.45  # world units per second
    hold_decay_rate: float = 3.0  # held seconds lost per second of movement
    count_hold_s: float = 0.5  # 1–5 fingers → mode 1–5
    freeze_hold_s: float = 1.0  # open palm → freeze / unfreeze
    horns_hold_s: float = 0.35  # index + pinky → next palette
    fist_hold_s: float = 0.2  # fist → explode
    # Pinch with hysteresis; the index must reach forward (a fist curls it back).
    pinch_on: float = 0.75
    pinch_off: float = 0.5
    pinch_min_index_reach: float = 0.95  # index tip↔wrist distance / palm size
    # Swipe: open hand travelling sideways quickly.
    swipe_window_s: float = 0.3
    swipe_min_distance: float = 0.45  # world units within the window
    swipe_min_speed: float = 1.2  # world units per second at the end of the swipe
    swipe_axis_ratio: float = 2.0  # horizontal travel must beat vertical by this factor
    swipe_open_share: float = 0.6  # share of the window the hand must be open
    swipe_left_is_next: bool = True  # like turning a page
    # Distance between two hands → intensity.
    two_hand_near: float = 0.4
    two_hand_far: float = 2.4
    intensity_near: float = 0.45
    intensity_far: float = 1.8


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
    start_visible: bool = False
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
class UIConfig:
    """On-screen feedback: progress rings, toasts and badges."""

    ring_radius_px: float = 24.0
    ring_width_px: float = 3.5
    ring_quad_scale: float = 2.0  # quad half-size / ring radius (room for the glow and flash)
    ring_intensity: float = 1.1
    ring_fade_tau_s: float = 0.12
    ring_progress_tau_s: float = 0.04
    flash_duration_s: float = 0.5
    label_font_size: int = 15
    label_offset_px: float = 36.0
    toast_font_size: int = 26
    toast_duration_s: float = 1.6
    toast_fade_tau_s: float = 0.18
    toast_top_px: float = 44.0
    badge_font_size: int = 16
    link_px: float = 1.5
    link_alpha: float = 0.4
    link_fade_tau_s: float = 0.25
    text_color: RGBA8 = (225, 242, 255, 255)
    panel_color: RGBA8 = (4, 10, 20, 150)
    padding_px: int = 8


@dataclass(frozen=True)
class RenderConfig:
    max_points: int = 4096
    max_line_segments: int = 4096
    max_glows: int = 256
    max_rings: int = 8
    line_feather_px: float = 1.5  # extra quad width used for anti-aliasing


@dataclass(frozen=True)
class PostFXConfig:
    """Trails (decay instead of clearing), bloom and tone mapping."""

    trail_tau_s: float = 0.16  # the trail loses ~63% of its light every this many seconds
    trail_floor_per_s: float = 0.12  # subtracted over time so faint trails really reach black
    trail_diffusion_per_s: float = 7.0  # light bleeding into neighbouring pixels (silkier trails)
    bloom_levels: int = 5
    bloom_threshold: float = 0.55
    bloom_knee: float = 0.45
    bloom_intensity: float = 0.9
    exposure: float = 1.4
    whiteout: float = 0.3  # how much the brightest spots bleach towards white
    dither: float = 1.0 / 255.0  # breaks up banding in dark gradients
    # Mode changes send a liquid ripple across the effect layer.
    ripple_duration_s: float = 1.4
    ripple_max_radius: float = 2.6  # in screen heights
    ripple_width: float = 0.09
    ripple_strength: float = 0.035  # peak displacement, in screen heights
    ripple_rim: float = 0.35  # brightness of the ripple's leading edge


@dataclass(frozen=True)
class Palette:
    name: str
    primary: RGB
    secondary: RGB
    highlight: RGB  # fast or energetic elements
    glow: RGB


PALETTES: tuple[Palette, ...] = (
    Palette("neon", primary=(0.05, 0.85, 1.0), secondary=(0.55, 0.25, 1.0), highlight=(1.0, 0.6, 0.95), glow=(0.3, 0.65, 1.0)),
    Palette("matrix", primary=(0.1, 1.0, 0.35), secondary=(0.0, 0.5, 0.2), highlight=(0.8, 1.0, 0.75), glow=(0.2, 0.9, 0.4)),
    Palette("âmbar", primary=(1.0, 0.6, 0.1), secondary=(1.0, 0.25, 0.05), highlight=(1.0, 0.92, 0.65), glow=(1.0, 0.5, 0.15)),
    Palette("gelo", primary=(0.7, 0.9, 1.0), secondary=(0.3, 0.55, 1.0), highlight=(1.0, 1.0, 1.0), glow=(0.55, 0.8, 1.0)),
)


@dataclass(frozen=True)
class PaletteConfig:
    palettes: tuple[Palette, ...] = PALETTES
    transition_tau_s: float = 0.6


@dataclass(frozen=True)
class ModesConfig:
    initial: int = 0
    crossfade_tau_s: float = 0.6
    hidden_opacity: float = 0.002  # below this a fading mode stops updating and drawing
    freeze_tau_s: float = 0.35  # time slows down into (and speeds up out of) a freeze
    intensity_tau_s: float = 0.35  # following the distance between two hands
    intensity_release_tau_s: float = 1.5  # drifting back to neutral when a hand leaves
    neutral_intensity: float = 1.0


@dataclass(frozen=True)
class FlowFieldConfig:
    """Curl-noise current shared by particle effects."""

    grid_rows: int = 48  # the field is evaluated on a grid and sampled bilinearly
    margin: float = 0.3  # the grid extends this far past the screen edges (world units)
    scale: float = 1.1  # noise features per world unit
    octaves: int = 2
    lacunarity: float = 2.1
    gain: float = 0.45
    evolution: float = 0.07  # noise time per second: how fast the currents reshape
    speed: float = 0.2  # RMS flow speed, world units per second
    swell_amplitude: float = 0.3  # slow breathing of the whole tide
    swell_period_s: float = 14.0
    drift_speed: float = 0.05  # uniform tidal drift whose direction wanders slowly
    drift_turn_period_s: float = 40.0
    seed: int = 3


@dataclass(frozen=True)
class DataTideConfig:
    """Mode 1: thousands of particles riding the tide, stirred by the hands."""

    particle_count: int = 24000
    seed: int = 11
    # Particles relax towards the local flow with this time constant
    # (a damped spring on velocity), varied per particle for an organic feel.
    follow_tau_s: float = 0.35
    follow_variation: float = 0.4
    max_speed: float = 3.0
    life_min_s: float = 4.0
    life_max_s: float = 9.0
    fade_in_s: float = 0.8
    fade_out_s: float = 1.0
    margin: float = 0.12  # leaving the screen by this much recycles a particle
    # Each hand is a smooth Lamb–Oseen vortex plus a current along its motion.
    vortex_circulation: float = 1.6  # world units² per second
    vortex_core: float = 0.12
    vortex_reach: float = 0.75
    right_hand_spin: float = 1.0  # +1 = counter-clockwise; the left hand spins the other way
    current_gain: float = 0.9  # share of the hand's velocity passed to nearby particles
    current_reach: float = 0.35
    # Pinch turns the vortex into a drain.
    pinch_spin_boost: float = 0.6
    pinch_core_scale: float = 0.6
    pinch_suction: float = 0.7  # inward speed at full pinch, world units per second
    pinch_drain: float = 4.0  # extra ageing inside the core while pinching
    # With no hands, a faint phantom vortex wanders around so the tide never sits still.
    idle_circulation: float = 0.6
    idle_path: tuple[float, float] = (0.6, 0.45)  # Lissajous size (share of half width / height)
    idle_period_s: float = 23.0
    idle_fade_tau_s: float = 1.2
    # Look
    particle_px: float = 2.0
    streak_s: float = 0.05  # streak length = speed × this (motion blur)
    max_streak_px: float = 36.0
    intensity: float = 0.09
    secondary_share: float = 0.3  # share of particles in the palette's secondary colour
    fast_speed: float = 1.2  # speed at which particles take the highlight colour
    glow_radius: float = 0.12
    glow_intensity: float = 0.08
    glow_pinch_boost: float = 1.2
    # Fist: outward burst of velocity plus a flash.
    explode_speed: float = 2.6
    explode_radius: float = 0.7
    flash_duration_s: float = 0.7
    flash_radius: float = 0.55
    flash_intensity: float = 0.5


@dataclass(frozen=True)
class NeuralNetConfig:
    """Mode 2: floating nodes linked when close; a pinch fires a pulse through the net."""

    node_count: int = 170
    seed: int = 21
    trail_tau_s: float = 0.1  # shorter trails than the tide: links are long-lived shapes
    trail_diffusion_per_s: float = 4.0
    drift_speed: float = 0.07  # nodes ride a slow curl-noise current
    follow_tau_s: float = 0.8
    max_speed: float = 1.5
    inset: float = 0.08  # soft walls keep nodes this far inside the screen
    wall_stiffness: float = 3.0
    spacing: float = 0.14  # closer nodes push each other apart
    spacing_strength: float = 0.3
    link_distance: float = 0.27
    # Hands: the palm parts the net gently, its motion drags it, fingertips plug in.
    hand_push: float = 0.3
    hand_push_reach: float = 0.28
    hand_current_gain: float = 0.5
    hand_current_reach: float = 0.35
    finger_link_distance: float = 0.33
    touch_radius: float = 0.06  # a fingertip this close to a node fires a small pulse
    touch_energy: float = 0.45
    # Pulses hop along links, losing energy on every hop.
    pulse_speed: float = 1.1  # world units per second along a link
    pulse_energy: float = 1.0
    hop_decay: float = 0.9
    min_energy: float = 0.12
    refractory_s: float = 0.6
    activity_decay_s: float = 0.45
    idle_pulse_rate: float = 0.35  # spontaneous pulses per second keep the net alive
    idle_pulse_energy: float = 0.55
    explode_speed: float = 1.6
    explode_radius: float = 0.7
    # Look
    node_radius: float = 0.04
    node_intensity: float = 0.22
    core_radius: float = 0.012
    core_intensity: float = 0.6
    active_intensity: float = 1.4
    line_px: float = 1.3
    line_intensity: float = 0.5
    active_line_boost: float = 2.5
    spark_radius: float = 0.03
    spark_intensity: float = 1.3
    finger_line_intensity: float = 0.55
    fingertip_radius: float = 0.03
    fingertip_intensity: float = 0.35
    max_links: int = 3000


@dataclass(frozen=True)
class HoloGridConfig:
    """Mode 3: a perspective wireframe surface where fingertips make real waves."""

    cols: int = 128
    rows: int = 96
    trail_tau_s: float = 0.05  # almost no trail and no diffusion: the wireframe stays crisp
    trail_diffusion_per_s: float = 0.0
    half_width: float = 8.0  # the plane spans x = ±half_width
    near: float = 1.0  # and z = -near … -far in front of the camera
    far: float = 15.0
    eye_height: float = 1.6
    look_distance: float = 6.0
    fov_y_deg: float = 55.0
    # 2D wave equation on the grid
    wave_speed: float = 2.4  # world units per second
    damping: float = 0.5  # per second
    edge_cells: int = 10  # sponge layer at the borders so waves leave instead of bouncing
    edge_damping: float = 6.0
    max_substep_s: float = 0.02  # keeps the explicit solver stable (CFL)
    finger_force: float = 5.0  # downward push of each raised fingertip
    finger_speed_gain: float = 5.0  # extra push per world unit / s of fingertip speed
    finger_radius: float = 0.22
    pinch_impulse: float = 3.0
    pinch_radius: float = 0.35
    explode_impulse: float = 7.0
    explode_radius: float = 0.6
    rain_rate: float = 1.2  # idle drops per second
    rain_impulse: float = 0.9
    rain_radius: float = 0.25
    swell_amplitude: float = 0.05  # gentle ocean swell added on top of the simulation
    swell_wavelength: float = 3.5
    swell_speed: float = 0.8
    # Look
    height_scale: float = 1.0
    line_px: float = 1.2
    line_intensity: float = 0.22
    crest_intensity: float = 0.9
    fill_intensity: float = 0.03
    fog_density: float = 0.12
    edge_fade: float = 0.15  # fraction of the width over which the side edges fade out
    contact_radius: float = 0.05
    contact_intensity: float = 0.35
    horizon_glow_intensity: float = 0.03
    horizon_glow_radius: float = 0.4
    horizon_glow_count: int = 17


@dataclass(frozen=True)
class Config:
    models: ModelsConfig = field(default_factory=ModelsConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    geometry: HandGeometryConfig = field(default_factory=HandGeometryConfig)
    hand_state: HandStateConfig = field(default_factory=HandStateConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    mock: MockConfig = field(default_factory=MockConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    postfx: PostFXConfig = field(default_factory=PostFXConfig)
    palette: PaletteConfig = field(default_factory=PaletteConfig)
    modes: ModesConfig = field(default_factory=ModesConfig)
    flow: FlowFieldConfig = field(default_factory=FlowFieldConfig)
    data_tide: DataTideConfig = field(default_factory=DataTideConfig)
    neural: NeuralNetConfig = field(default_factory=NeuralNetConfig)
    holo: HoloGridConfig = field(default_factory=HoloGridConfig)


CONFIG = Config()
