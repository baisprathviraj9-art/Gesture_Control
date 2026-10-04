"""
Mouse cursor, clicking, and scrolling controller.

Features:
- Time-aware and speed-adaptive cursor smoothing: steady at rest, responsive at speed.
- Usable active camera margin so cursor reaches all screen edges comfortably.
- Hysteresis & grace period: brief tracking dropout or noisy landmark does not reset smoother or snap cursor.
- One-finger cursor movement: index finger extended, middle/ring/pinky folded.
- Two-finger scrolling: index and middle fingers extended, ring/pinky folded.
  - On entry, the average vertical position of fingertips 8 and 12 is latched as neutral.
  - Holding fingers above neutral continuously scrolls up; below scrolls down.
  - Speed scales quadratically with distance from neutral (configurable min/max rates).
  - Returning to neutral stops scrolling immediately.
  - Cursor is frozen during scrolling.
- Thumb clicking: deliberate lowered-to-raised transition performs one left click.
  - Entering cursor mode with an already raised thumb neither clicks nor prevents movement.
  - Cursor freezes at last stable position during deliberate click gesture recognition.
  - Thumb must return to lowered position to rearm another click.
- Two-finger scrolling takes explicit priority over thumb clicking.
- Preserves PyAutoGUI fail-safe and handles either hand and hand rotations.
"""

import math
import time
import pyautogui


def _euclidean_dist(p1, p2) -> float:
    """Calculate 2D Euclidean distance between two landmarks."""
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    return math.hypot(dx, dy)


def is_scroll_pose(hand_landmarks) -> bool:
    """
    Return True if hand forms the two-finger scroll pose:
    - Index finger extended (landmark 8)
    - Middle finger extended (landmark 12)
    - Ring (16) and pinky (20) folded into palm
    - Thumb (4) unrestricted

    Evaluated using orientation-invariant Euclidean distances.
    """
    if hand_landmarks is None or len(hand_landmarks) < 21:
        return False

    wrist = hand_landmarks[0]

    idx_tip_dist = _euclidean_dist(hand_landmarks[8], wrist)
    idx_pip_dist = _euclidean_dist(hand_landmarks[6], wrist)
    idx_mcp_dist = _euclidean_dist(hand_landmarks[5], wrist)
    idx_extended = (idx_tip_dist > idx_pip_dist) and (idx_pip_dist > idx_mcp_dist * 0.9)

    mid_tip_dist = _euclidean_dist(hand_landmarks[12], wrist)
    mid_pip_dist = _euclidean_dist(hand_landmarks[10], wrist)
    mid_mcp_dist = _euclidean_dist(hand_landmarks[9], wrist)
    mid_extended = (mid_tip_dist > mid_pip_dist) and (mid_pip_dist > mid_mcp_dist * 0.9)

    if not (idx_extended and mid_extended):
        return False

    ring_tip_dist = _euclidean_dist(hand_landmarks[16], wrist)
    ring_pip_dist = _euclidean_dist(hand_landmarks[14], wrist)
    ring_tip_to_mcp = _euclidean_dist(hand_landmarks[16], hand_landmarks[13])
    ring_pip_to_mcp = _euclidean_dist(hand_landmarks[14], hand_landmarks[13])
    ring_folded = (ring_tip_dist < ring_pip_dist) or (ring_tip_to_mcp < ring_pip_to_mcp)

    pky_tip_dist = _euclidean_dist(hand_landmarks[20], wrist)
    pky_pip_dist = _euclidean_dist(hand_landmarks[18], wrist)
    pky_tip_to_mcp = _euclidean_dist(hand_landmarks[20], hand_landmarks[17])
    pky_pip_to_mcp = _euclidean_dist(hand_landmarks[18], hand_landmarks[17])
    pky_folded = (pky_tip_dist < pky_pip_dist) or (pky_tip_to_mcp < pky_pip_to_mcp)

    return ring_folded and pky_folded


def is_three_finger_screenshot_pose(hand_landmarks) -> bool:
    """
    Return True if the hand forms the 3-finger screenshot pose:
    - Index (8), middle (12), and ring (16) fingers extended.
    - Pinky (20) folded into palm.
    - Thumb (4) position unrestricted.

    Evaluated using orientation-invariant Euclidean distances suitable for
    either hand and normal in-plane rotations.
    """
    if hand_landmarks is None or len(hand_landmarks) < 21:
        return False

    wrist = hand_landmarks[0]

    def _finger_extended(tip_idx, pip_idx, mcp_idx):
        tip_dist = _euclidean_dist(hand_landmarks[tip_idx], wrist)
        pip_dist = _euclidean_dist(hand_landmarks[pip_idx], wrist)
        mcp_dist = _euclidean_dist(hand_landmarks[mcp_idx], wrist)
        return (tip_dist > pip_dist) and (pip_dist > mcp_dist * 0.9)

    if not (
        _finger_extended(8, 6, 5)
        and _finger_extended(12, 10, 9)
        and _finger_extended(16, 14, 13)
    ):
        return False

    pky_tip_dist = _euclidean_dist(hand_landmarks[20], wrist)
    pky_pip_dist = _euclidean_dist(hand_landmarks[18], wrist)
    pky_tip_to_mcp = _euclidean_dist(hand_landmarks[20], hand_landmarks[17])
    pky_pip_to_mcp = _euclidean_dist(hand_landmarks[18], hand_landmarks[17])
    pky_folded = (pky_tip_dist < pky_pip_dist) or (pky_tip_to_mcp < pky_pip_to_mcp)

    return pky_folded


def is_scroll_candidate(hand_landmarks) -> bool:
    """Return True if hand is in or entering a two-finger scroll pose."""
    if hand_landmarks is None or len(hand_landmarks) < 21:
        return False

    if is_three_finger_screenshot_pose(hand_landmarks):
        return False

    if is_scroll_pose(hand_landmarks):
        return True

    wrist = hand_landmarks[0]
    idx_extended = _euclidean_dist(hand_landmarks[8], wrist) > _euclidean_dist(hand_landmarks[6], wrist)
    mid_extended = _euclidean_dist(hand_landmarks[12], wrist) > _euclidean_dist(hand_landmarks[10], wrist)

    if not (idx_extended and mid_extended):
        return False

    ring_folded = (_euclidean_dist(hand_landmarks[16], wrist) < _euclidean_dist(hand_landmarks[14], wrist)) or (
        _euclidean_dist(hand_landmarks[16], hand_landmarks[13]) < _euclidean_dist(hand_landmarks[14], hand_landmarks[13])
    )
    pky_folded = (_euclidean_dist(hand_landmarks[20], wrist) < _euclidean_dist(hand_landmarks[18], wrist)) or (
        _euclidean_dist(hand_landmarks[20], hand_landmarks[17]) < _euclidean_dist(hand_landmarks[18], hand_landmarks[17])
    )

    return ring_folded or pky_folded


def is_pointing_pose(hand_landmarks) -> bool:
    """
    Return True if the hand forms a one-finger pointing pose:
    - Index finger extended (landmark 8)
    - Middle (12), ring (16), pinky (20) folded into the palm
    - Thumb (4) unrestricted
    """
    if hand_landmarks is None or len(hand_landmarks) < 21:
        return False

    wrist = hand_landmarks[0]

    idx_tip_dist = _euclidean_dist(hand_landmarks[8], wrist)
    idx_pip_dist = _euclidean_dist(hand_landmarks[6], wrist)
    idx_mcp_dist = _euclidean_dist(hand_landmarks[5], wrist)
    idx_extended = (idx_tip_dist > idx_pip_dist) and (idx_pip_dist > idx_mcp_dist * 0.9)

    if not idx_extended:
        return False

    mid_tip_dist = _euclidean_dist(hand_landmarks[12], wrist)
    mid_pip_dist = _euclidean_dist(hand_landmarks[10], wrist)
    mid_tip_to_mcp = _euclidean_dist(hand_landmarks[12], hand_landmarks[9])
    mid_pip_to_mcp = _euclidean_dist(hand_landmarks[10], hand_landmarks[9])
    mid_folded = (mid_tip_dist < mid_pip_dist) or (mid_tip_to_mcp < mid_pip_to_mcp)

    ring_tip_dist = _euclidean_dist(hand_landmarks[16], wrist)
    ring_pip_dist = _euclidean_dist(hand_landmarks[14], wrist)
    ring_tip_to_mcp = _euclidean_dist(hand_landmarks[16], hand_landmarks[13])
    ring_pip_to_mcp = _euclidean_dist(hand_landmarks[14], hand_landmarks[13])
    ring_folded = (ring_tip_dist < ring_pip_dist) or (ring_tip_to_mcp < ring_pip_to_mcp)

    pky_tip_dist = _euclidean_dist(hand_landmarks[20], wrist)
    pky_pip_dist = _euclidean_dist(hand_landmarks[18], wrist)
    pky_tip_to_mcp = _euclidean_dist(hand_landmarks[20], hand_landmarks[17])
    pky_pip_to_mcp = _euclidean_dist(hand_landmarks[18], hand_landmarks[17])
    pky_folded = (pky_tip_dist < pky_pip_dist) or (pky_tip_to_mcp < pky_pip_to_mcp)

    return mid_folded and ring_folded and pky_folded


def is_pointing_candidate(hand_landmarks) -> bool:
    """Return True if the hand is in or entering a pointing pose."""
    if hand_landmarks is None or len(hand_landmarks) < 21:
        return False

    if is_pointing_pose(hand_landmarks):
        return True

    wrist = hand_landmarks[0]
    idx_tip_dist = _euclidean_dist(hand_landmarks[8], wrist)
    idx_pip_dist = _euclidean_dist(hand_landmarks[6], wrist)
    idx_extended = idx_tip_dist > idx_pip_dist

    if not idx_extended:
        return False

    mid_tip_dist = _euclidean_dist(hand_landmarks[12], wrist)
    mid_pip_dist = _euclidean_dist(hand_landmarks[10], wrist)
    mid_tip_to_mcp = _euclidean_dist(hand_landmarks[12], hand_landmarks[9])
    mid_pip_to_mcp = _euclidean_dist(hand_landmarks[10], hand_landmarks[9])
    mid_folded = (mid_tip_dist < mid_pip_dist) or (mid_tip_to_mcp < mid_pip_to_mcp)

    ring_folded = (_euclidean_dist(hand_landmarks[16], wrist) < _euclidean_dist(hand_landmarks[14], wrist)) or (
        _euclidean_dist(hand_landmarks[16], hand_landmarks[13]) < _euclidean_dist(hand_landmarks[14], hand_landmarks[13])
    )
    pky_folded = (_euclidean_dist(hand_landmarks[20], wrist) < _euclidean_dist(hand_landmarks[18], wrist)) or (
        _euclidean_dist(hand_landmarks[20], hand_landmarks[17]) < _euclidean_dist(hand_landmarks[18], hand_landmarks[17])
    )

    folded_count = sum([1 for f in (mid_folded, ring_folded, pky_folded) if f])
    return folded_count >= 2


def is_mouse_mode_candidate(hand_landmarks) -> bool:
    """Return True if hand is candidate for pointing/clicking, scrolling, or right-clicking."""
    if is_three_finger_screenshot_pose(hand_landmarks):
        return False
    return (
        is_scroll_candidate(hand_landmarks)
        or is_pointing_candidate(hand_landmarks)
        or is_right_click_pose(hand_landmarks)
    )


def is_open_palm_pose(hand_landmarks) -> bool:
    """
    Return True if the hand forms an open palm with all four fingers extended:
    - Index (8), middle (12), ring (16), and pinky (20) fingers extended.
    - Thumb (4) unrestricted.

    Uses scale-relative Euclidean distances suitable for either hand and
    normal in-plane rotations.
    """
    if hand_landmarks is None or len(hand_landmarks) < 21:
        return False

    wrist = hand_landmarks[0]

    def _finger_extended(tip_idx, pip_idx, mcp_idx):
        tip_dist = _euclidean_dist(hand_landmarks[tip_idx], wrist)
        pip_dist = _euclidean_dist(hand_landmarks[pip_idx], wrist)
        mcp_dist = _euclidean_dist(hand_landmarks[mcp_idx], wrist)
        return (tip_dist > pip_dist) and (pip_dist > mcp_dist * 0.9)

    return (
        _finger_extended(8, 6, 5)
        and _finger_extended(12, 10, 9)
        and _finger_extended(16, 14, 13)
        and _finger_extended(20, 18, 17)
    )


SCROLL_SPEED_MULTIPLIER: float = 10.0
BASE_MIN_SCROLL_STEPS_PER_SECOND: float = 4.0
BASE_MAX_SCROLL_STEPS_PER_SECOND: float = 22.0
BASE_MAX_SCROLL_STEPS_PER_FRAME: int = 3

DEFAULT_MIN_SCROLL_STEPS_PER_SECOND: float = BASE_MIN_SCROLL_STEPS_PER_SECOND * SCROLL_SPEED_MULTIPLIER  
DEFAULT_MAX_SCROLL_STEPS_PER_SECOND: float = BASE_MAX_SCROLL_STEPS_PER_SECOND * SCROLL_SPEED_MULTIPLIER  
DEFAULT_MAX_SCROLL_STEPS_PER_FRAME: int = int(round(BASE_MAX_SCROLL_STEPS_PER_FRAME * SCROLL_SPEED_MULTIPLIER))

OLD_PINCH_CONTACT_THRESHOLD = 0.25
OLD_PINCH_APPROACH_THRESHOLD = 0.32
OLD_PINCH_RELEASE_THRESHOLD = 0.36

DEFAULT_PINCH_CONTACT_THRESHOLD = 0.18
DEFAULT_PINCH_APPROACH_THRESHOLD = 0.28
DEFAULT_PINCH_RELEASE_THRESHOLD = 0.35
DEFAULT_PINCH_CONFIRMATION_TIME = 0.20
DEFAULT_PINCH_SEPARATION_TIME = 0.15
DEFAULT_PINCH_FILTER_ALPHA = 0.60
DEFAULT_PINCH_CONTACT_JITTER_TIME = 0.06
DEFAULT_PINCH_CONTACT_JITTER_MARGIN = 0.03


def get_right_click_pose_status(hand_landmarks) -> tuple[bool, str]:
    """
    Evaluates whether hand forms the thumb-to-middle right-click pose and returns
    (is_valid, reason).
    """
    if hand_landmarks is None or len(hand_landmarks) < 21:
        return False, "Missing hand landmarks"

    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]
    hand_scale = _euclidean_dist(wrist, middle_mcp)
    if hand_scale < 1e-4:
        return False, "Hand scale too small"

    idx_tip_dist = _euclidean_dist(hand_landmarks[8], wrist)
    idx_pip_dist = _euclidean_dist(hand_landmarks[6], wrist)
    if idx_tip_dist <= idx_pip_dist:
        return False, "Index finger not extended"

    ring_tip_dist = _euclidean_dist(hand_landmarks[16], wrist)
    ring_pip_dist = _euclidean_dist(hand_landmarks[14], wrist)
    ring_tip_to_mcp = _euclidean_dist(hand_landmarks[16], hand_landmarks[13])
    ring_pip_to_mcp = _euclidean_dist(hand_landmarks[14], hand_landmarks[13])
    ring_folded = (ring_tip_dist < ring_pip_dist) or (ring_tip_to_mcp < ring_pip_to_mcp)
    if not ring_folded:
        return False, "Ring finger not folded"

    pky_tip_dist = _euclidean_dist(hand_landmarks[20], wrist)
    pky_pip_dist = _euclidean_dist(hand_landmarks[18], wrist)
    pky_tip_to_mcp = _euclidean_dist(hand_landmarks[20], hand_landmarks[17])
    pky_pip_to_mcp = _euclidean_dist(hand_landmarks[18], hand_landmarks[17])
    pky_folded = (pky_tip_dist < pky_pip_dist) or (pky_tip_to_mcp < pky_pip_to_mcp)
    if not pky_folded:
        return False, "Pinky finger not folded"

    return True, "Valid right-click pose"


def is_right_click_pose(hand_landmarks) -> bool:
    """
    Return True if the hand forms a pose for thumb-to-middle right-click pinch:
    - Index finger (8) extended.
    - Ring (16) and pinky (20) fingers folded.
    - Middle finger (12) free to bend toward thumb.
    - Thumb (4) unrestricted.
    """
    valid, _ = get_right_click_pose_status(hand_landmarks)
    return valid


def get_thumb_middle_pinch_ratio(hand_landmarks) -> float:
    """
    Calculate Euclidean distance between thumb tip (4) and middle fingertip (12)
    normalized by palm scale (wrist 0 to middle MCP 9).
    Returns 999.0 if landmarks are invalid or palm scale is near zero.
    """
    if hand_landmarks is None or len(hand_landmarks) < 21:
        return 999.0

    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]
    hand_scale = _euclidean_dist(wrist, middle_mcp)
    if hand_scale < 1e-4:
        return 999.0

    thumb_tip = hand_landmarks[4]
    middle_tip = hand_landmarks[12]
    return _euclidean_dist(thumb_tip, middle_tip) / hand_scale


THUMB_RAISE_RATIO = 0.38
THUMB_LOWER_RATIO = 0.26


def get_thumb_spread_ratio(hand_landmarks) -> float:
    """
    Calculate thumb tip (4) to index MCP (5) distance normalized by palm scale
    (wrist 0 to middle MCP 9). Works for either hand and across hand rotations.
    """
    if hand_landmarks is None or len(hand_landmarks) < 21:
        return 0.0

    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]
    thumb_tip = hand_landmarks[4]
    index_mcp = hand_landmarks[5]

    hand_scale = _euclidean_dist(wrist, middle_mcp)
    if hand_scale < 1e-4:
        return 0.0

    return _euclidean_dist(thumb_tip, index_mcp) / hand_scale


def is_thumb_raised(hand_landmarks) -> bool:
    """Return True if thumb is raised outward from palm."""
    return get_thumb_spread_ratio(hand_landmarks) >= THUMB_RAISE_RATIO


def is_thumb_lowered(hand_landmarks) -> bool:
    """Return True if thumb is resting / lowered close to palm."""
    return get_thumb_spread_ratio(hand_landmarks) <= THUMB_LOWER_RATIO


SWIPE_THUMB_MIN_SPREAD_RATIO = 0.30


def is_swipe_palm_pose(hand_landmarks) -> bool:
    """
    Return True if the hand forms a true five-finger open palm required for ordinary swipes:
    - All five fingers (index, middle, ring, pinky, and thumb) open and extended.
    - Specifically rejects if pinky is folded (e.g. 3-finger screenshot pose).
    - Specifically rejects if thumb is resting against or close to the palm.
    - Specifically rejects pointing, scrolling, or pinching.
    - Uses palm-scale-normalized Euclidean distances invariant to hand (left/right)
      and normal in-plane rotations.
    """
    if hand_landmarks is None or len(hand_landmarks) < 21:
        return False

    if is_three_finger_screenshot_pose(hand_landmarks):
        return False

    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]
    hand_scale = _euclidean_dist(wrist, middle_mcp)
    if hand_scale < 1e-4:
        return False

    def _finger_extended(tip_idx, pip_idx, mcp_idx):
        tip_dist = _euclidean_dist(hand_landmarks[tip_idx], wrist)
        pip_dist = _euclidean_dist(hand_landmarks[pip_idx], wrist)
        mcp_dist = _euclidean_dist(hand_landmarks[mcp_idx], wrist)
        tip_to_mcp = _euclidean_dist(hand_landmarks[tip_idx], hand_landmarks[mcp_idx])
        pip_to_mcp = _euclidean_dist(hand_landmarks[pip_idx], hand_landmarks[mcp_idx])
        return (tip_dist > pip_dist) and (pip_dist > mcp_dist * 0.9) and (tip_to_mcp > pip_to_mcp)

    if not (
        _finger_extended(8, 6, 5)
        and _finger_extended(12, 10, 9)
        and _finger_extended(16, 14, 13)
        and _finger_extended(20, 18, 17)
    ):
        return False

    pky_tip_dist = _euclidean_dist(hand_landmarks[20], wrist)
    pky_pip_dist = _euclidean_dist(hand_landmarks[18], wrist)
    pky_tip_to_mcp = _euclidean_dist(hand_landmarks[20], hand_landmarks[17])
    pky_pip_to_mcp = _euclidean_dist(hand_landmarks[18], hand_landmarks[17])
    if (pky_tip_dist <= pky_pip_dist) or (pky_tip_to_mcp <= pky_pip_to_mcp):
        return False

    if is_thumb_lowered(hand_landmarks):
        return False

    thumb_spread = get_thumb_spread_ratio(hand_landmarks)
    if thumb_spread < SWIPE_THUMB_MIN_SPREAD_RATIO:
        return False

    thumb_tip_dist = _euclidean_dist(hand_landmarks[4], wrist)
    thumb_mcp_dist = _euclidean_dist(hand_landmarks[2], wrist)
    if thumb_tip_dist <= thumb_mcp_dist * 0.95:
        return False

    return True


DEFAULT_ACTIVE_BOUNDS = (0.22, 0.12, 0.78, 0.60)

def map_coordinates_to_screen(
    norm_x: float,
    norm_y: float,
    screen_w: int,
    screen_h: int,
    margin_x: float = 0.12,
    margin_y: float = 0.12,
    margin: float | None = None,
    active_bounds: tuple[float, float, float, float] | None = None,
) -> tuple[int, int]:
    """
    Map normalised coordinates to screen pixel coordinates clamped to boundaries.
    Uses configurable active margins so the entire screen can be reached comfortably
    without moving the hand to extreme edges of the camera view.
    """
    if active_bounds is not None:
        left, top, right, bottom = active_bounds
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise ValueError("active_bounds must be within the camera frame")
        norm_x = (norm_x - left) / (right - left)
        norm_y = (norm_y - top) / (bottom - top)
    elif margin is not None:
        margin_x = margin
        margin_y = margin

    if active_bounds is None:
        if margin_x > 0.0 and (1.0 - 2.0 * margin_x) > 1e-4:
            norm_x = (norm_x - margin_x) / (1.0 - 2.0 * margin_x)
        if margin_y > 0.0 and (1.0 - 2.0 * margin_y) > 1e-4:
            norm_y = (norm_y - margin_y) / (1.0 - 2.0 * margin_y)

    screen_x = int(round(norm_x * (screen_w - 1)))
    screen_y = int(round(norm_y * (screen_h - 1)))

    clamped_x = max(0, min(screen_w - 1, screen_x))
    clamped_y = max(0, min(screen_h - 1, screen_y))
    return clamped_x, clamped_y


class CursorSmoother:
    """
    Time-aware and speed-adaptive exponential moving average smoother.
    Provides steady tremor elimination at rest while snapping instantly on fast movement.
    """

    def __init__(
        self,
        alpha: float | None = None,
        min_alpha: float = 0.20,
        max_alpha: float = 0.85,
        speed_threshold: float = 1200.0,
        deadzone: float = 2.0,
    ):
        if alpha is not None:
            self.min_alpha = max(0.01, min(1.0, alpha))
            self.max_alpha = max(self.min_alpha, max_alpha)
        else:
            self.min_alpha = min_alpha
            self.max_alpha = max_alpha

        self.speed_threshold = max(1.0, speed_threshold)
        self.deadzone = max(0.0, deadzone)

        self.smoothed_x: float | None = None
        self.smoothed_y: float | None = None
        self.last_sent_x: int | None = None
        self.last_sent_y: int | None = None
        self.last_time: float | None = None

    def reset(self):
        self.smoothed_x = None
        self.smoothed_y = None
        self.last_sent_x = None
        self.last_sent_y = None
        self.last_time = None

    def smooth(
        self,
        target_x: int,
        target_y: int,
        timestamp: float | None = None,
    ) -> tuple[int, int, bool]:
        now = timestamp if timestamp is not None else time.time()

        if self.smoothed_x is None or self.smoothed_y is None or self.last_time is None:
            self.smoothed_x = float(target_x)
            self.smoothed_y = float(target_y)
            out_x = int(round(self.smoothed_x))
            out_y = int(round(self.smoothed_y))
            self.last_sent_x = out_x
            self.last_sent_y = out_y
            self.last_time = now
            return out_x, out_y, True

        dt = max(0.010, now - self.last_time)
        self.last_time = now

        dist_to_target = math.hypot(target_x - self.smoothed_x, target_y - self.smoothed_y)
        speed = dist_to_target / dt

        speed_ratio = min(1.0, speed / self.speed_threshold)
        base_alpha = self.min_alpha + speed_ratio * (self.max_alpha - self.min_alpha)

        time_factor = dt / 0.0333
        effective_alpha = 1.0 - math.pow(max(0.0, 1.0 - base_alpha), max(0.1, time_factor))
        effective_alpha = max(0.05, min(1.0, effective_alpha))

        self.smoothed_x += effective_alpha * (target_x - self.smoothed_x)
        self.smoothed_y += effective_alpha * (target_y - self.smoothed_y)

        out_x = int(round(self.smoothed_x))
        out_y = int(round(self.smoothed_y))

        dist_moved = math.hypot(out_x - self.last_sent_x, out_y - self.last_sent_y)
        if dist_moved < self.deadzone:
            return self.last_sent_x, self.last_sent_y, False

        self.last_sent_x = out_x
        self.last_sent_y = out_y
        return out_x, out_y, True


class ScrollController:
    """Scroll continuously according to two-finger height from a neutral pose."""

    def __init__(
        self,
        neutral_deadzone: float = 0.025,
        full_speed_distance: float = 0.20,
        min_steps_per_second: float = DEFAULT_MIN_SCROLL_STEPS_PER_SECOND,
        max_steps_per_second: float = DEFAULT_MAX_SCROLL_STEPS_PER_SECOND,
        scroll_clicks_per_step: int = 1,
        max_steps_per_frame: int = DEFAULT_MAX_SCROLL_STEPS_PER_FRAME,
        speed_multiplier: float = 1.0,
        pyautogui_module=pyautogui,
    ):
        effective_min = min_steps_per_second * speed_multiplier
        effective_max = max_steps_per_second * speed_multiplier
        if not 0 <= neutral_deadzone < full_speed_distance:
            raise ValueError("full_speed_distance must exceed neutral_deadzone")
        if not 0 < effective_min <= effective_max:
            raise ValueError("scroll rates must be positive and ordered")

        self.neutral_deadzone = neutral_deadzone
        self.full_speed_distance = full_speed_distance
        self.min_steps_per_second = effective_min
        self.max_steps_per_second = effective_max
        self.scroll_clicks_per_step = scroll_clicks_per_step
        self.max_steps_per_frame = int(round(max_steps_per_frame * speed_multiplier))
        self.speed_multiplier = speed_multiplier
        self.pag = pyautogui_module

        self.neutral_y: float | None = None
        self.last_time: float | None = None
        self.step_fraction = 0.0
        self.direction = 0
        self.pending_direction = 0
        self.pending_frames = 0
        self.current_rate = 0.0

    def reset(self):
        self.neutral_y = None
        self.last_time = None
        self.step_fraction = 0.0
        self.direction = 0
        self.pending_direction = 0
        self.pending_frames = 0
        self.current_rate = 0.0

    def reanchor(self, hand_landmarks, timestamp: float | None = None):
        """Resume timing after tracking loss without changing neutral."""
        self.last_time = timestamp if timestamp is not None else time.monotonic()
        self.pending_direction = 0
        self.pending_frames = 0

    def process(self, hand_landmarks, timestamp: float | None = None) -> tuple[int, bool]:
        """
        Scroll based on vertical offset from the latched neutral position.
        Moving upward (decreasing Y) -> scroll UP (positive clicks).
        Moving downward (increasing Y) -> scroll DOWN (negative clicks).
        Speed scales quadratically with distance from neutral.

        Returns: (scroll_amount, did_scroll)
        """
        if hand_landmarks is None or len(hand_landmarks) < 21:
            self.reset()
            return 0, False

        now = timestamp if timestamp is not None else time.monotonic()
        avg_y = (hand_landmarks[8].y + hand_landmarks[12].y) / 2.0

        if self.neutral_y is None:
            self.neutral_y = avg_y
            self.last_time = now
            return 0, False

        dt = min(max(now - self.last_time, 0.0), 0.10)
        self.last_time = now
        offset = avg_y - self.neutral_y

        if abs(offset) <= self.neutral_deadzone:
            self.direction = 0
            self.pending_direction = 0
            self.pending_frames = 0
            self.step_fraction = 0.0
            self.current_rate = 0.0
            return 0, False

        direction = 1 if offset < 0 else -1
        distance = min(
            1.0,
            (abs(offset) - self.neutral_deadzone)
            / (self.full_speed_distance - self.neutral_deadzone),
        )
        self.current_rate = self.min_steps_per_second + (
            self.max_steps_per_second - self.min_steps_per_second
        ) * distance * distance

        if direction != self.direction:
            if direction == self.pending_direction:
                self.pending_frames += 1
            else:
                self.pending_direction = direction
                self.pending_frames = 1

            if self.pending_frames < 2:
                return 0, False

            self.direction = direction
            self.pending_direction = 0
            self.pending_frames = 0
            self.step_fraction = 0.0
            steps = 1
        else:
            self.pending_direction = 0
            self.pending_frames = 0
            self.step_fraction += self.current_rate * dt
            steps = min(int(self.step_fraction), self.max_steps_per_frame)
            self.step_fraction -= steps

        if steps == 0:
            return 0, False

        total_scroll = direction * steps * self.scroll_clicks_per_step
        try:
            try:
                self.pag.scroll(total_scroll, _pause=False)
            except TypeError:
                self.pag.scroll(total_scroll)
        except self.pag.FailSafeException:
            raise
        except Exception:
            return 0, False

        return total_scroll, True


class RightClickController:
    """
    State machine recognizing deliberate thumb-to-middle-finger pinch for right-click.

    States:
      - 'disarmed': initial state, or when separation beyond release threshold has not been confirmed for 150ms.
      - 'armed': fingertips separated beyond release threshold for >= 150ms; ready for deliberate contact.
      - 'approaching': valid pose with filtered pinch ratio <= approach threshold. Cursor is frozen.
      - 'confirming': collecting 200ms of valid contact; brief distance jitter pauses confirmation.
      - 'clicked': right click dispatched once; waiting for separation (150ms) before rearming.
    """

    def __init__(
        self,
        contact_threshold: float = DEFAULT_PINCH_CONTACT_THRESHOLD,
        approach_threshold: float = DEFAULT_PINCH_APPROACH_THRESHOLD,
        release_threshold: float = DEFAULT_PINCH_RELEASE_THRESHOLD,
        confirmation_duration: float = DEFAULT_PINCH_CONFIRMATION_TIME,
        separation_duration: float = DEFAULT_PINCH_SEPARATION_TIME,
        filter_alpha: float = DEFAULT_PINCH_FILTER_ALPHA,
        pyautogui_module=pyautogui,
        contact_jitter_duration: float = DEFAULT_PINCH_CONTACT_JITTER_TIME,
        contact_jitter_margin: float = DEFAULT_PINCH_CONTACT_JITTER_MARGIN,
    ):
        if not (0 < contact_threshold < approach_threshold < release_threshold):
            raise ValueError(
                "Thresholds must satisfy: 0 < contact_threshold < approach_threshold < release_threshold"
            )
        if confirmation_duration <= 0 or separation_duration <= 0:
            raise ValueError("Durations must be positive")
        if contact_jitter_duration < 0 or contact_jitter_margin < 0:
            raise ValueError("Contact jitter limits must be nonnegative")

        self.contact_threshold = contact_threshold
        self.approach_threshold = approach_threshold
        self.release_threshold = release_threshold
        self.confirmation_duration = confirmation_duration
        self.separation_duration = separation_duration
        self.filter_alpha = filter_alpha
        self.pag = pyautogui_module
        self.contact_jitter_duration = contact_jitter_duration
        self.contact_jitter_threshold = min(
            approach_threshold, contact_threshold + contact_jitter_margin
        )

        self.state: str = "disarmed"
        self.armed: bool = False
        self.contact_start_time: float | None = None
        self.separation_start_time: float | None = None
        self.separation_elapsed: float = 0.0
        self.pinch_ratio: float = 999.0
        self.raw_pinch_ratio: float = 999.0
        self.contact_elapsed: float = 0.0
        self.contact_last_time: float | None = None
        self.contact_last_valid: bool = False
        self.contact_jitter_start_time: float | None = None
        self.is_pose_valid: bool = False
        self.pose_rejection_reason: str = ""

    @property
    def is_approaching(self) -> bool:
        return self.state in ("approaching", "confirming", "clicked", "clicked_held")

    @property
    def confirmation_progress(self) -> float:
        if self.state == "confirming" and self.contact_start_time is not None:
            return min(1.0, max(0.0, self.contact_elapsed / self.confirmation_duration))
        if self.state in ("clicked", "clicked_held"):
            return 1.0
        return 0.0

    @property
    def separation_progress(self) -> float:
        if not self.armed and self.separation_start_time is not None:
            return min(1.0, max(0.0, self.separation_elapsed / self.separation_duration))
        if self.armed:
            return 1.0
        return 0.0

    def _clear_contact(self):
        self.contact_start_time = None
        self.contact_elapsed = 0.0
        self.contact_last_time = None
        self.contact_last_valid = False
        self.contact_jitter_start_time = None

    def reset(self, require_release: bool = True):
        self._clear_contact()
        self.separation_start_time = None
        self.separation_elapsed = 0.0
        self.pinch_ratio = 999.0
        self.raw_pinch_ratio = 999.0
        self.is_pose_valid = False
        self.pose_rejection_reason = ""
        if require_release:
            self.armed = False
            self.state = "disarmed"
        else:
            self.armed = True
            self.state = "armed"

    def on_tracking_lost(self, prolonged: bool = False):
        self._clear_contact()
        self.separation_start_time = None
        self.separation_elapsed = 0.0
        self.pinch_ratio = 999.0
        self.raw_pinch_ratio = 999.0
        if prolonged:
            self.armed = False
            self.state = "disarmed"
        else:
            if self.state == "confirming":
                self.state = "approaching" if self.armed else "disarmed"

    def process(self, hand_landmarks, timestamp: float | None = None) -> tuple[bool, bool]:
        now = timestamp if timestamp is not None else time.monotonic()

        if hand_landmarks is None or len(hand_landmarks) < 21:
            self.is_pose_valid = False
            self.pose_rejection_reason = "Missing hand landmarks"
            self.on_tracking_lost(prolonged=False)
            return False, False

        valid_pose, rejection_reason = get_right_click_pose_status(hand_landmarks)
        self.is_pose_valid = valid_pose
        self.pose_rejection_reason = rejection_reason

        raw_ratio = get_thumb_middle_pinch_ratio(hand_landmarks)
        self.raw_pinch_ratio = raw_ratio

        if raw_ratio == 999.0 or not valid_pose:
            self.pinch_ratio = 999.0
            self._clear_contact()
            self.separation_start_time = None
            self.separation_elapsed = 0.0
            if self.state in ("approaching", "confirming"):
                self.state = "armed" if self.armed else "disarmed"
            return False, False

        if self.pinch_ratio == 999.0:
            self.pinch_ratio = raw_ratio
        else:
            self.pinch_ratio = self.filter_alpha * raw_ratio + (1.0 - self.filter_alpha) * self.pinch_ratio

        ratio = self.pinch_ratio

        if ratio >= self.release_threshold and raw_ratio >= self.release_threshold:
            self._clear_contact()
            if not self.armed:
                if self.separation_start_time is None:
                    self.separation_start_time = now
                    self.separation_elapsed = 0.0
                else:
                    self.separation_elapsed = now - self.separation_start_time
                    if self.separation_elapsed >= self.separation_duration:
                        self.armed = True
                        self.state = "armed"
                        self.separation_start_time = None
                        self.separation_elapsed = self.separation_duration
            else:
                self.state = "armed"
                self.separation_start_time = None
                self.separation_elapsed = self.separation_duration
            return False, False

        self.separation_start_time = None
        self.separation_elapsed = 0.0

        if self.state in ("clicked", "clicked_held") or not self.armed:
            freeze = ratio <= self.approach_threshold or raw_ratio <= self.approach_threshold
            return False, freeze

        is_contact = (ratio <= self.contact_threshold) and (raw_ratio <= self.contact_threshold)

        if self.state == "confirming":
            if self.contact_jitter_start_time is not None and (
                now - self.contact_jitter_start_time > self.contact_jitter_duration + 1e-9
            ):
                self._clear_contact()
                self.state = "armed"
            elif not is_contact and max(ratio, raw_ratio) <= self.contact_jitter_threshold:
                if self.contact_jitter_start_time is None:
                    self.contact_jitter_start_time = now
                self.contact_last_time = now
                self.contact_last_valid = False
                return False, True

        if is_contact:
            if self.state != "confirming":
                self._clear_contact()
                self.state = "confirming"
                self.contact_start_time = now
                self.contact_last_time = now
                self.contact_last_valid = True
                return False, True
            else:
                if self.contact_last_valid and self.contact_last_time is not None:
                    self.contact_elapsed += max(0.0, now - self.contact_last_time)
                self.contact_last_time = now
                self.contact_last_valid = True
                self.contact_jitter_start_time = None
                if self.contact_elapsed + 1e-9 >= self.confirmation_duration:
                    success = self._dispatch_right_click()
                    if success:
                        self.state = "clicked"
                        self.armed = False
                        self._clear_contact()
                        self.separation_start_time = None
                        self.separation_elapsed = 0.0
                        return True, True
                    else:
                        self._clear_contact()
                        self.state = "approaching"
                        return False, True
                else:
                    return False, True

        elif ratio <= self.approach_threshold or raw_ratio <= self.approach_threshold:
            self.state = "approaching"
            self._clear_contact()
            return False, True

        else:
            self.state = "armed"
            self._clear_contact()
            return False, False

    def _dispatch_right_click(self) -> bool:
        """
        Dispatches right click via PyAutoGUI.
        Returns True if successful, False if dispatch failed.
        Re-raises FailSafeException immediately.
        """
        failsafe_cls = getattr(self.pag, "FailSafeException", None)
        try:
            try:
                self.pag.click(button="right", _pause=False)
            except TypeError:
                self.pag.click(button="right")
            return True
        except Exception as e:
            if failsafe_cls is not None and isinstance(e, failsafe_cls):
                raise
            import logging
            logging.getLogger(__name__).warning("Right click dispatch failed: %s", e)
            return False


class MouseController:
    """
    Coordinates cursor movement, thumb clicking, and two-finger scrolling.
    Explicit priority:
      1. Two-finger scrolling takes priority over thumb clicking.
      2. Clicking is available only in one-finger cursor mode.
      3. Cursor freezes during deliberate click recognition and during scrolling.
      4. Hysteresis grace periods prevent noisy landmarks or 1-frame dropouts
         from resetting smoother or snapping coordinates.
    """

    def __init__(
        self,
        screen_size: tuple[int, int] | None = None,
        smoother: CursorSmoother | None = None,
        scroll_controller: ScrollController | None = None,
        right_click_controller: RightClickController | None = None,
        scroll_speed_multiplier: float = 1.0,
        min_scroll_steps_per_second: float = DEFAULT_MIN_SCROLL_STEPS_PER_SECOND,
        max_scroll_steps_per_second: float = DEFAULT_MAX_SCROLL_STEPS_PER_SECOND,
        pinch_contact_threshold: float = DEFAULT_PINCH_CONTACT_THRESHOLD,
        pinch_approach_threshold: float = DEFAULT_PINCH_APPROACH_THRESHOLD,
        pinch_release_threshold: float = DEFAULT_PINCH_RELEASE_THRESHOLD,
        pinch_confirmation_time: float = DEFAULT_PINCH_CONFIRMATION_TIME,
        pinch_separation_time: float = DEFAULT_PINCH_SEPARATION_TIME,
        pinch_filter_alpha: float = DEFAULT_PINCH_FILTER_ALPHA,
        pyautogui_module=pyautogui,
        margin_x: float | None = None,
        margin_y: float | None = None,
        margin: float | None = None,
        active_bounds: tuple[float, float, float, float] = DEFAULT_ACTIVE_BOUNDS,
        max_uncertain_frames: int = 3,
        max_hand_lost_frames: int = 1,
        click_freeze_duration: float = 0.12,
        left_click_delay: float = 0.0,
    ):
        self.screen_size = screen_size
        self.smoother = smoother or CursorSmoother()
        self.pag = pyautogui_module
        self.scroll_ctrl = scroll_controller or ScrollController(
            min_steps_per_second=min_scroll_steps_per_second,
            max_steps_per_second=max_scroll_steps_per_second,
            speed_multiplier=scroll_speed_multiplier,
            pyautogui_module=pyautogui_module,
        )
        self.right_click_ctrl = right_click_controller or RightClickController(
            contact_threshold=pinch_contact_threshold,
            approach_threshold=pinch_approach_threshold,
            release_threshold=pinch_release_threshold,
            confirmation_duration=pinch_confirmation_time,
            separation_duration=pinch_separation_time,
            filter_alpha=pinch_filter_alpha,
            pyautogui_module=pyautogui_module,
        )
        self.left_click_delay = left_click_delay
        self.pending_left_click: bool = False
        self.pending_left_click_time: float | None = None

        if margin is not None or margin_x is not None or margin_y is not None:
            mx = margin if margin is not None else (margin_x if margin_x is not None else 0.12)
            my = margin if margin is not None else (margin_y if margin_y is not None else 0.12)
            self.active_bounds = (mx, my, 1.0 - mx, 1.0 - my)
        else:
            self.active_bounds = active_bounds
        left, top, right, bottom = self.active_bounds
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise ValueError("active_bounds must be within the camera frame")
        self.margin_x = left
        self.margin_y = top

        self.max_uncertain_frames = max_uncertain_frames
        self.max_hand_lost_frames = max_hand_lost_frames
        self.click_freeze_duration = click_freeze_duration

        self.mode: str = "IDLE"
        self.is_active: bool = False

        self.uncertain_frames: int = 0
        self.hand_lost_frames: int = 0
        self.scroll_reanchor_required: bool = False

        self.click_armed: bool = False
        self.click_freeze_until: float = 0.0
        self.last_stable_x: int = 0
        self.last_stable_y: int = 0

        self.raw_tip_x: float = 0.0
        self.raw_tip_y: float = 0.0
        self.target_screen_x: int = 0
        self.target_screen_y: int = 0
        self.thumb_ratio: float = 0.0
        self.thumb_state: str = "neutral"
        self.last_action: str = ""
        self.last_action_time: float = 0.0

    def get_screen_size(self) -> tuple[int, int]:
        if self.screen_size is not None:
            return self.screen_size
        try:
            return self.pag.size()
        except Exception:
            return 1920, 1080

    def process_landmarks(
        self,
        hand_landmarks,
        timestamp: float | None = None,
    ) -> tuple[str, bool, int, bool]:
        """
        Processes hand landmarks for scrolling, cursor movement, and thumb clicking.

        Returns:
            (mode, clicked, scroll_amount, scrolled)
        """
        now = timestamp if timestamp is not None else time.monotonic()

        if hand_landmarks is None or len(hand_landmarks) < 21:
            if self.mode in ("CURSOR", "SCROLL") and self.hand_lost_frames < self.max_hand_lost_frames:
                self.hand_lost_frames += 1
                if self.mode == "SCROLL":
                    self.scroll_reanchor_required = True
                    self.scroll_ctrl.pending_direction = 0
                    self.scroll_ctrl.pending_frames = 0
                self.right_click_ctrl.on_tracking_lost(prolonged=False)
                return self.mode, False, 0, False

            self.on_pose_lost()
            return "IDLE", False, 0, False

        self.hand_lost_frames = 0

        if is_three_finger_screenshot_pose(hand_landmarks):
            self.on_pose_lost(hand_landmarks)
            return "IDLE", False, 0, False

        tip = hand_landmarks[8]
        self.raw_tip_x = tip.x
        self.raw_tip_y = tip.y
        self.thumb_ratio = get_thumb_spread_ratio(hand_landmarks)
        thumb_raised = is_thumb_raised(hand_landmarks)
        thumb_lowered = is_thumb_lowered(hand_landmarks)
        self.thumb_state = "raised" if thumb_raised else ("lowered" if thumb_lowered else "neutral")

        rc_clicked, rc_freeze = self.right_click_ctrl.process(hand_landmarks, timestamp=now)
        if rc_clicked:
            self.last_action = "Right Click"
            self.last_action_time = now
            self.click_armed = False
            self.pending_left_click = False
            self.pending_left_click_time = None

        if rc_freeze:
            if self.mode != "CURSOR":
                self.mode = "CURSOR"
                self.smoother.reset()
                self.scroll_ctrl.reset()
                self.scroll_reanchor_required = False
            self.pending_left_click = False
            self.pending_left_click_time = None
            self.uncertain_frames = 0
            self.is_active = True
            return "CURSOR", rc_clicked, 0, False

        if is_scroll_pose(hand_landmarks):
            self.uncertain_frames = 0
            if self.mode != "SCROLL":
                self.smoother.reset()
                self.scroll_ctrl.reset()
                self.click_armed = False
                self.mode = "SCROLL"
                self.is_active = True

            if self.scroll_reanchor_required:
                self.scroll_ctrl.reanchor(hand_landmarks, timestamp=now)
                self.scroll_reanchor_required = False
                return "SCROLL", False, 0, False

            scroll_val, scrolled = self.scroll_ctrl.process(hand_landmarks, timestamp=now)
            if scrolled:
                self.last_action = "Scroll Up" if scroll_val > 0 else "Scroll Down"
                self.last_action_time = now

            return "SCROLL", False, scroll_val, scrolled

        if is_pointing_pose(hand_landmarks):
            self.uncertain_frames = 0
            entering_cursor_mode = (self.mode != "CURSOR")
            if entering_cursor_mode:
                self.mode = "CURSOR"
                self.is_active = True
                self.smoother.reset()
                self.scroll_ctrl.reset()
                self.scroll_reanchor_required = False
                self.click_armed = False

            if thumb_lowered:
                self.click_armed = True

            clicked = False
            pinch_ratio = self.right_click_ctrl.pinch_ratio
            not_forming_pinch = pinch_ratio >= self.right_click_ctrl.release_threshold

            if thumb_raised and self.click_armed and not_forming_pinch:
                if self.left_click_delay > 0.0:
                    if not self.pending_left_click:
                        self.pending_left_click = True
                        self.pending_left_click_time = now
                    elif (now - (self.pending_left_click_time or now)) >= self.left_click_delay:
                        try:
                            try:
                                self.pag.click(_pause=False)
                            except TypeError:
                                self.pag.click()
                            clicked = True
                            self.last_action = "Left Click"
                            self.last_action_time = now
                            self.click_freeze_until = now + self.click_freeze_duration
                        except self.pag.FailSafeException:
                            raise
                        except Exception:
                            pass
                        self.click_armed = False
                        self.pending_left_click = False
                        self.pending_left_click_time = None
                else:
                    try:
                        try:
                            self.pag.click(_pause=False)
                        except TypeError:
                            self.pag.click()
                        clicked = True
                        self.last_action = "Left Click"
                        self.last_action_time = now
                        self.click_freeze_until = now + self.click_freeze_duration
                    except self.pag.FailSafeException:
                        raise
                    except Exception:
                        pass
                    self.click_armed = False
            elif not thumb_raised:
                self.pending_left_click = False
                self.pending_left_click_time = None

            if clicked or (now < self.click_freeze_until):
                return "CURSOR", clicked, 0, False

            screen_w, screen_h = self.get_screen_size()
            raw_x, raw_y = map_coordinates_to_screen(
                tip.x, tip.y, screen_w, screen_h,
                active_bounds=self.active_bounds
            )
            raw_x = min(max(raw_x, 1), screen_w - 2)
            raw_y = min(max(raw_y, 1), screen_h - 2)
            out_x, out_y, should_move = self.smoother.smooth(raw_x, raw_y, timestamp=now)
            self.target_screen_x = out_x
            self.target_screen_y = out_y

            if should_move:
                self.last_stable_x = out_x
                self.last_stable_y = out_y
                try:
                    try:
                        self.pag.moveTo(out_x, out_y, _pause=False)
                    except TypeError:
                        self.pag.moveTo(out_x, out_y)
                except self.pag.FailSafeException:
                    raise
                except Exception:
                    pass

            return "CURSOR", False, 0, False

        if self.mode == "SCROLL" and self.uncertain_frames < self.max_uncertain_frames:
            self.uncertain_frames += 1
            self.scroll_reanchor_required = True
            self.scroll_ctrl.pending_direction = 0
            self.scroll_ctrl.pending_frames = 0
            return "SCROLL", False, 0, False

        if self.mode == "CURSOR" and is_pointing_candidate(hand_landmarks):
            if self.uncertain_frames < self.max_uncertain_frames:
                self.uncertain_frames += 1
                return "CURSOR", False, 0, False

        if is_right_click_pose(hand_landmarks):
            if self.mode != "CURSOR":
                self.mode = "CURSOR"
                self.smoother.reset()
                self.scroll_ctrl.reset()
                self.scroll_reanchor_required = False
            self.click_armed = False
            self.pending_left_click = False
            self.pending_left_click_time = None
            self.uncertain_frames = 0
            self.is_active = True
            return "CURSOR", False, 0, False

        self.on_pose_lost(hand_landmarks)
        return "IDLE", False, 0, False

    def on_pose_lost(self, hand_landmarks=None):
        """Called when pointing or scrolling ends, hand is lost, or tracking drops."""
        self.mode = "IDLE"
        self.is_active = False
        self.click_armed = False
        self.pending_left_click = False
        self.pending_left_click_time = None
        self.uncertain_frames = 0
        self.hand_lost_frames = 0
        self.click_freeze_until = 0.0
        self.smoother.reset()
        self.scroll_ctrl.reset()
        self.scroll_reanchor_required = False
        if hand_landmarks is None or not is_right_click_pose(hand_landmarks):
            self.right_click_ctrl.on_tracking_lost(prolonged=True)
