"""
Automated unit and integration tests for mouse cursor, clicking, and scrolling.

Covers:
- Pointing pose detection (index extended, middle/ring/pinky folded, thumb unrestricted)
- Scroll pose detection (index and middle extended, ring/pinky folded, thumb unrestricted)
- Rejection of non-target poses (open palm, fist, three fingers)
- Rotation and scale invariance
- Coordinate mapping and primary screen clamping
- Exponential smoothing progression and pixel deadzone suppression
- Cursor movement dispatch via mocked PyAutoGUI
- Deliberate lowered-to-raised thumb click detection (1 click per raise)
- Rearming after thumb returns to lowered position
- No click when entering cursor mode with an already raised thumb
- Cursor position frozen at last stable position while thumb click is recognized
- Two-finger vertical scrolling (upward = scroll up, downward = scroll down)
- Scroll deadzone and accumulated distance threshold (jitter rejection)
- Cursor frozen during scrolling
- Explicit gesture priority: scrolling takes priority over thumb clicking
- Hand loss and mode transition cleanup (no click, jump, or scroll burst)
- PyAutoGUI fail-safe exception handling
- Swipe suppression throughout cursor, click, and scroll gestures

All tests mock PyAutoGUI to ensure the real mouse cursor and buttons are NEVER touched.

Run with:
    .venv\\Scripts\\python.exe -m pytest test_mouse_controller.py -v
"""

import math
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mouse_controller import (
    CursorSmoother,
    DEFAULT_MAX_SCROLL_STEPS_PER_FRAME,
    DEFAULT_MAX_SCROLL_STEPS_PER_SECOND,
    DEFAULT_MIN_SCROLL_STEPS_PER_SECOND,
    DEFAULT_PINCH_APPROACH_THRESHOLD,
    DEFAULT_PINCH_CONFIRMATION_TIME,
    DEFAULT_PINCH_CONTACT_THRESHOLD,
    DEFAULT_PINCH_FILTER_ALPHA,
    DEFAULT_PINCH_RELEASE_THRESHOLD,
    DEFAULT_PINCH_SEPARATION_TIME,
    MouseController,
    RightClickController,
    SCROLL_SPEED_MULTIPLIER,
    ScrollController,
    get_right_click_pose_status,
    get_thumb_middle_pinch_ratio,
    get_thumb_spread_ratio,
    is_mouse_mode_candidate,
    is_open_palm_pose,
    is_pointing_candidate,
    is_pointing_pose,
    is_right_click_pose,
    is_scroll_candidate,
    is_scroll_pose,
    is_thumb_lowered,
    is_thumb_raised,
    map_coordinates_to_screen,
)
from main import (
    ACTION_DISPLAY_DURATION,
    GestureState,
    PalmSwipeDetector,
    detect_swipe,
    draw_hud,
    get_palm_center,
    perform_gesture_action,
)


class FakeLandmark:
    """Minimal normalized landmark."""
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def make_hand(
    index_extended: bool = True,
    middle_folded: bool = True,
    ring_folded: bool = True,
    pinky_folded: bool = True,
    thumb_raised: bool = False,
    thumb_x: float | None = None,
    thumb_y: float | None = None,
    wrist_x: float = 0.5,
    wrist_y: float = 0.8,
) -> list:
    """
    Construct a synthetic hand with 21 landmarks.
    Wrist is positioned at (wrist_x, wrist_y).
    Index MCP is at (wrist_x - 0.05, wrist_y - 0.20).
    Middle MCP is at (wrist_x, wrist_y - 0.20).
    """
    lms = [FakeLandmark(wrist_x, wrist_y) for _ in range(21)]
    lms[0] = FakeLandmark(wrist_x, wrist_y)

    mcp_y = wrist_y - 0.20
    lms[5] = FakeLandmark(wrist_x - 0.05, mcp_y)
    lms[9] = FakeLandmark(wrist_x, mcp_y)
    lms[13] = FakeLandmark(wrist_x + 0.05, mcp_y)
    lms[17] = FakeLandmark(wrist_x + 0.10, mcp_y)

    def set_finger(tip_idx, pip_idx, base_x, is_extended):
        if is_extended:
            lms[pip_idx] = FakeLandmark(base_x, wrist_y - 0.35)
            lms[tip_idx - 1] = FakeLandmark(base_x, wrist_y - 0.45)
            lms[tip_idx] = FakeLandmark(base_x, wrist_y - 0.60)
        else:
            lms[pip_idx] = FakeLandmark(base_x, wrist_y - 0.35)
            lms[tip_idx - 1] = FakeLandmark(base_x, wrist_y - 0.28)
            lms[tip_idx] = FakeLandmark(base_x, wrist_y - 0.22)

    set_finger(8, 6, wrist_x - 0.05, index_extended)

    set_finger(12, 10, wrist_x, not middle_folded)

    set_finger(16, 14, wrist_x + 0.05, not ring_folded)

    set_finger(20, 18, wrist_x + 0.10, not pinky_folded)

    if thumb_x is not None and thumb_y is not None:
        tx, ty = thumb_x, thumb_y
    elif thumb_raised:
        tx, ty = wrist_x - 0.18, mcp_y
    else:
        tx, ty = wrist_x - 0.07, mcp_y + 0.01

    lms[1] = FakeLandmark(wrist_x - 0.04, wrist_y - 0.08)
    lms[2] = FakeLandmark(wrist_x - 0.06, wrist_y - 0.14)
    lms[3] = FakeLandmark(tx + 0.02, ty + 0.04)
    lms[4] = FakeLandmark(tx, ty)

    return lms


def rotate_landmarks(landmarks: list, angle_deg: float) -> list:
    """Rotate landmarks around the wrist (landmark 0)."""
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    w = landmarks[0]
    rotated = []
    for lm in landmarks:
        dx = lm.x - w.x
        dy = lm.y - w.y
        rx = w.x + dx * cos_a - dy * sin_a
        ry = w.y + dx * sin_a + dy * cos_a
        rotated.append(FakeLandmark(rx, ry))
    return rotated


def translate_hand(hand: list, dx: float, dy: float) -> list:
    """Translate all landmarks by (dx, dy) preserving relative geometry."""
    return [FakeLandmark(lm.x + dx, lm.y + dy) for lm in hand]


def make_open_palm(wrist_x: float = 0.5, wrist_y: float = 0.8, thumb_raised: bool = True) -> list:
    """Construct synthetic landmarks for a fully extended open palm."""
    return make_hand(
        index_extended=True,
        middle_folded=False,
        ring_folded=False,
        pinky_folded=False,
        thumb_raised=thumb_raised,
        wrist_x=wrist_x,
        wrist_y=wrist_y,
    )


def mirror_landmarks(landmarks: list) -> list:
    """Mirror hand horizontally around wrist_x."""
    wx = landmarks[0].x
    return [FakeLandmark(wx - (lm.x - wx), lm.y) for lm in landmarks]


def make_pinch_hand(
    pinch_ratio: float = 0.15,
    wrist_x: float = 0.5,
    wrist_y: float = 0.8,
) -> list:
    """
    Construct synthetic hand in right-click pose:
    - Index extended (tip at y=0.20, PIP at y=0.45, wrist at y=0.80)
    - Ring and pinky folded
    - Wrist at (wrist_x, wrist_y), middle MCP at (wrist_x, wrist_y - 0.20). Palm scale = 0.20.
    - Thumb tip (4) and middle tip (12) separated by (pinch_ratio * palm_scale).
    """
    hand = make_hand(
        index_extended=True,
        middle_folded=True,
        ring_folded=True,
        pinky_folded=True,
        wrist_x=wrist_x,
        wrist_y=wrist_y,
    )
    scale = 0.20
    target_dist = pinch_ratio * scale

    mx, my = wrist_x, wrist_y - 0.25
    hand[12] = FakeLandmark(mx, my)

    tx, ty = mx - target_dist, my
    hand[4] = FakeLandmark(tx, ty)

    return hand


class TestPointingPoseDetection:
    """Verifies pointing pose classification and candidate detection."""

    def test_pointing_pose_detected_when_index_extended_others_folded(self):
        hand = make_hand(index_extended=True, middle_folded=True, ring_folded=True, pinky_folded=True)
        assert is_pointing_pose(hand) is True
        assert is_pointing_candidate(hand) is True
        assert is_mouse_mode_candidate(hand) is True

    def test_open_palm_not_detected_as_pointing(self):
        hand = make_hand(index_extended=True, middle_folded=False, ring_folded=False, pinky_folded=False)
        assert is_pointing_pose(hand) is False
        assert is_pointing_candidate(hand) is False

    def test_fist_not_detected_as_pointing(self):
        hand = make_hand(index_extended=False, middle_folded=True, ring_folded=True, pinky_folded=True)
        assert is_pointing_pose(hand) is False
        assert is_pointing_candidate(hand) is False

    def test_peace_sign_not_detected_as_pointing(self):
        hand = make_hand(index_extended=True, middle_folded=False, ring_folded=True, pinky_folded=True)
        assert is_pointing_pose(hand) is False
        assert is_scroll_pose(hand) is True

    def test_three_fingers_not_detected_as_pointing(self):
        hand = make_hand(index_extended=True, middle_folded=False, ring_folded=False, pinky_folded=True)
        assert is_pointing_pose(hand) is False
        assert is_pointing_candidate(hand) is False

    def test_thumb_position_unrestricted_for_pointing(self):
        hand_lowered = make_hand(thumb_raised=False)
        hand_raised = make_hand(thumb_raised=True)
        assert is_pointing_pose(hand_lowered) is True
        assert is_pointing_pose(hand_raised) is True

    def test_rotation_invariance_pointing(self):
        base_hand = make_hand(index_extended=True, middle_folded=True, ring_folded=True, pinky_folded=True)
        for angle in (30, 45, 90, 135, 180, 270):
            rot_hand = rotate_landmarks(base_hand, angle)
            assert is_pointing_pose(rot_hand) is True, f"Failed pointing at angle {angle}"

    def test_empty_or_none_landmarks_handled_safely(self):
        assert is_pointing_pose(None) is False
        assert is_pointing_pose([]) is False
        assert is_pointing_candidate(None) is False
        assert is_pointing_candidate([]) is False


class TestScrollPoseDetection:
    """Verifies two-finger scroll pose classification and candidate detection."""

    def test_scroll_pose_detected_when_index_and_middle_extended(self):
        hand = make_hand(index_extended=True, middle_folded=False, ring_folded=True, pinky_folded=True)
        assert is_scroll_pose(hand) is True
        assert is_scroll_candidate(hand) is True
        assert is_mouse_mode_candidate(hand) is True

    def test_scroll_pose_rejected_when_middle_folded(self):
        hand = make_hand(index_extended=True, middle_folded=True, ring_folded=True, pinky_folded=True)
        assert is_scroll_pose(hand) is False

    def test_scroll_pose_rejected_when_ring_extended(self):
        hand = make_hand(index_extended=True, middle_folded=False, ring_folded=False, pinky_folded=True)
        assert is_scroll_pose(hand) is False

    def test_scroll_pose_rejected_when_pinky_extended(self):
        hand = make_hand(index_extended=True, middle_folded=False, ring_folded=True, pinky_folded=False)
        assert is_scroll_pose(hand) is False

    def test_scroll_pose_thumb_unrestricted(self):
        hand_lowered = make_hand(index_extended=True, middle_folded=False, thumb_raised=False)
        hand_raised = make_hand(index_extended=True, middle_folded=False, thumb_raised=True)
        assert is_scroll_pose(hand_lowered) is True
        assert is_scroll_pose(hand_raised) is True

    def test_rotation_invariance_scroll(self):
        base_hand = make_hand(index_extended=True, middle_folded=False, ring_folded=True, pinky_folded=True)
        for angle in (30, 45, 90, 135, 180, 270):
            rot_hand = rotate_landmarks(base_hand, angle)
            assert is_scroll_pose(rot_hand) is True, f"Failed scroll at angle {angle}"


class TestThumbRatioAndState:
    """Verifies thumb spread ratio, raised/lowered states, and invariance."""

    def test_thumb_lowered_detection(self):
        hand = make_hand(thumb_raised=False)
        ratio = get_thumb_spread_ratio(hand)
        assert ratio <= 0.26
        assert is_thumb_lowered(hand) is True
        assert is_thumb_raised(hand) is False

    def test_thumb_raised_detection(self):
        hand = make_hand(thumb_raised=True)
        ratio = get_thumb_spread_ratio(hand)
        assert ratio >= 0.38
        assert is_thumb_raised(hand) is True
        assert is_thumb_lowered(hand) is False

    def test_thumb_hysteresis_deadband(self):
        hand = make_hand(thumb_x=0.45 - 0.20 * 0.32, thumb_y=0.60)
        ratio = get_thumb_spread_ratio(hand)
        assert 0.26 < ratio < 0.38
        assert is_thumb_raised(hand) is False
        assert is_thumb_lowered(hand) is False

    def test_rotation_invariance_thumb_spread(self):
        hand_low = make_hand(thumb_raised=False)
        hand_high = make_hand(thumb_raised=True)
        for angle in (30, 60, 90, 150, 180, 270):
            rot_low = rotate_landmarks(hand_low, angle)
            rot_high = rotate_landmarks(hand_high, angle)
            assert is_thumb_lowered(rot_low) is True, f"Failed lowered at angle {angle}"
            assert is_thumb_raised(rot_high) is True, f"Failed raised at angle {angle}"

    def test_either_hand_left_and_right(self):
        right_hand_raised = make_hand(thumb_x=0.32, thumb_y=0.60)
        assert is_thumb_raised(right_hand_raised) is True

        left_hand_raised = make_hand(thumb_x=0.58, thumb_y=0.60)
        assert is_thumb_raised(left_hand_raised) is True


class TestCoordinateMapping:
    """Verifies normalized-to-screen coordinate mapping, corners, and clamping."""

    def test_screen_corners_mapped_accurately(self):
        sw, sh = 1920, 1080
        assert map_coordinates_to_screen(0.0, 0.0, sw, sh) == (0, 0)
        assert map_coordinates_to_screen(1.0, 0.0, sw, sh) == (sw - 1, 0)
        assert map_coordinates_to_screen(0.0, 1.0, sw, sh) == (0, sh - 1)
        assert map_coordinates_to_screen(1.0, 1.0, sw, sh) == (sw - 1, sh - 1)

    def test_screen_center(self):
        sw, sh = 1920, 1080
        cx, cy = map_coordinates_to_screen(0.5, 0.5, sw, sh)
        assert abs(cx - sw // 2) <= 1
        assert abs(cy - sh // 2) <= 1

    def test_out_of_bounds_clamped_to_screen(self):
        sw, sh = 1920, 1080
        assert map_coordinates_to_screen(-0.25, -0.10, sw, sh) == (0, 0)
        assert map_coordinates_to_screen(1.30, 1.45, sw, sh) == (sw - 1, sh - 1)

    def test_mirrored_movement_tracks_horizontally(self):
        sw, sh = 1920, 1080
        x1, _ = map_coordinates_to_screen(0.20, 0.5, sw, sh)
        x2, _ = map_coordinates_to_screen(0.50, 0.5, sw, sh)
        x3, _ = map_coordinates_to_screen(0.80, 0.5, sw, sh)
        assert x1 < x2 < x3

    def test_screen_edge_mapping_with_active_margins(self):
        sw, sh = 1920, 1080
        x_left, y_top = map_coordinates_to_screen(0.12, 0.12, sw, sh, margin_x=0.12, margin_y=0.12)
        assert x_left == 0
        assert y_top == 0

        x_right, y_bot = map_coordinates_to_screen(0.88, 0.88, sw, sh, margin_x=0.12, margin_y=0.12)
        assert x_right == sw - 1
        assert y_bot == sh - 1

        cx, cy = map_coordinates_to_screen(0.50, 0.50, sw, sh, margin_x=0.12, margin_y=0.12)
        assert abs(cx - sw // 2) <= 1
        assert abs(cy - sh // 2) <= 1

        assert map_coordinates_to_screen(0.02, 0.05, sw, sh, margin_x=0.12, margin_y=0.12) == (0, 0)
        assert map_coordinates_to_screen(0.98, 0.95, sw, sh, margin_x=0.12, margin_y=0.12) == (sw - 1, sh - 1)


class TestCursorSmoother:
    """Verifies EMA smoothing, snap-on-start, and deadzone suppression."""

    def test_first_point_snaps_directly_without_jump(self):
        smoother = CursorSmoother(alpha=0.35, deadzone=2.0)
        out_x, out_y, moved = smoother.smooth(800, 600)
        assert out_x == 800
        assert out_y == 600
        assert moved is True

    def test_smoothing_progression_converges_steadily(self):
        smoother = CursorSmoother(alpha=0.4, deadzone=0.0)
        smoother.smooth(100, 100)

        x1, y1, _ = smoother.smooth(500, 500)
        assert 100 < x1 < 500
        assert 100 < y1 < 500

        x2, y2, _ = smoother.smooth(500, 500)
        assert x1 < x2 < 500
        assert y1 < y2 < 500

    def test_deadzone_suppresses_micro_movements(self):
        smoother = CursorSmoother(alpha=1.0, deadzone=5.0)
        smoother.smooth(500, 500)

        out_x, out_y, moved = smoother.smooth(501, 500)
        assert moved is False
        assert (out_x, out_y) == (500, 500)

        out_x2, out_y2, moved2 = smoother.smooth(510, 500)
        assert moved2 is True
        assert (out_x2, out_y2) == (510, 500)

    def test_reset_clears_state_for_fresh_snap(self):
        smoother = CursorSmoother(alpha=0.35)
        smoother.smooth(100, 100)
        smoother.reset()

        out_x, out_y, moved = smoother.smooth(900, 900)
        assert (out_x, out_y) == (900, 900)
        assert moved is True

    def test_fast_cursor_movement_adapts_and_responds_promptly(self):
        smoother = CursorSmoother(min_alpha=0.20, max_alpha=0.85, speed_threshold=1200.0)
        smoother.smooth(200, 200, timestamp=1.0)

        out_x, out_y, moved = smoother.smooth(1200, 200, timestamp=1.033)
        assert moved is True
        assert out_x > 900


class TestScrollController:
    """Verifies neutral-anchor position-based scrolling."""

    def test_first_frame_establishes_neutral_without_scrolling(self):
        mock_pag = MagicMock()
        sc = ScrollController(pyautogui_module=mock_pag)

        hand = make_hand(index_extended=True, middle_folded=False)
        hand[8] = FakeLandmark(0.45, 0.40)
        hand[12] = FakeLandmark(0.50, 0.40)
        scroll_val, scrolled = sc.process(hand, timestamp=1.0)

        assert scrolled is False
        assert scroll_val == 0
        assert sc.neutral_y == pytest.approx(0.40)
        mock_pag.scroll.assert_not_called()

    def test_small_offset_inside_deadzone_does_not_scroll(self):
        mock_pag = MagicMock()
        sc = ScrollController(neutral_deadzone=0.025, pyautogui_module=mock_pag)

        hand = make_hand(index_extended=True, middle_folded=False)
        hand[8] = FakeLandmark(0.45, 0.40)
        hand[12] = FakeLandmark(0.50, 0.40)
        sc.process(hand, timestamp=1.0)

        for i in range(10):
            hand[8] = FakeLandmark(0.45, 0.41)
            hand[12] = FakeLandmark(0.50, 0.41)
            scroll_val, scrolled = sc.process(hand, timestamp=1.0 + (i + 1) * 0.033)
            assert scrolled is False

        mock_pag.scroll.assert_not_called()

    def test_holding_above_neutral_continuously_scrolls_up(self):
        mock_pag = MagicMock()
        sc = ScrollController(
            neutral_deadzone=0.025,
            full_speed_distance=0.20,
            min_steps_per_second=4.0,
            max_steps_per_second=22.0,
            pyautogui_module=mock_pag,
        )

        hand = make_hand(index_extended=True, middle_folded=False)
        hand[8] = FakeLandmark(0.45, 0.50)
        hand[12] = FakeLandmark(0.50, 0.50)
        sc.process(hand, timestamp=1.0)

        hand[8] = FakeLandmark(0.45, 0.38)
        hand[12] = FakeLandmark(0.50, 0.38)

        up_scrolls = 0
        t = 1.033
        for _ in range(30):
            scroll_val, scrolled = sc.process(hand, timestamp=t)
            if scrolled:
                assert scroll_val > 0
                up_scrolls += 1
            t += 0.033

        assert up_scrolls > 0
        assert sc.direction == 1

    def test_holding_below_neutral_continuously_scrolls_down(self):
        mock_pag = MagicMock()
        sc = ScrollController(
            neutral_deadzone=0.025,
            pyautogui_module=mock_pag,
        )

        hand = make_hand(index_extended=True, middle_folded=False)
        hand[8] = FakeLandmark(0.45, 0.40)
        hand[12] = FakeLandmark(0.50, 0.40)
        sc.process(hand, timestamp=1.0)

        hand[8] = FakeLandmark(0.45, 0.52)
        hand[12] = FakeLandmark(0.50, 0.52)

        down_scrolls = 0
        t = 1.033
        for _ in range(30):
            scroll_val, scrolled = sc.process(hand, timestamp=t)
            if scrolled:
                assert scroll_val < 0
                down_scrolls += 1
            t += 0.033

        assert down_scrolls > 0
        assert sc.direction == -1

    def test_farther_distance_produces_more_scroll_steps(self):
        """Fingers held farther from neutral should scroll faster."""
        mock_pag_near = MagicMock()
        mock_pag_far = MagicMock()

        def count_scroll_steps(pag, offset_y, frames=30):
            sc = ScrollController(
                neutral_deadzone=0.025,
                full_speed_distance=0.20,
                pyautogui_module=pag,
            )
            hand = make_hand(index_extended=True, middle_folded=False)
            hand[8] = FakeLandmark(0.45, 0.50)
            hand[12] = FakeLandmark(0.50, 0.50)
            sc.process(hand, timestamp=1.0)

            hand[8] = FakeLandmark(0.45, 0.50 - offset_y)
            hand[12] = FakeLandmark(0.50, 0.50 - offset_y)
            total = 0
            t = 1.033
            for _ in range(frames):
                sv, scrolled = sc.process(hand, timestamp=t)
                if scrolled:
                    total += abs(sv)
                t += 0.033
            return total

        near_steps = count_scroll_steps(mock_pag_near, 0.06)
        far_steps = count_scroll_steps(mock_pag_far, 0.18)

        assert far_steps > near_steps

    def test_returning_to_neutral_stops_then_reversing_works(self):
        mock_pag = MagicMock()
        sc = ScrollController(neutral_deadzone=0.025, pyautogui_module=mock_pag)

        hand = make_hand(index_extended=True, middle_folded=False)
        hand[8] = FakeLandmark(0.45, 0.50)
        hand[12] = FakeLandmark(0.50, 0.50)
        sc.process(hand, timestamp=1.0)

        hand[8] = FakeLandmark(0.45, 0.38)
        hand[12] = FakeLandmark(0.50, 0.38)
        t = 1.033
        for _ in range(10):
            sc.process(hand, timestamp=t)
            t += 0.033

        hand[8] = FakeLandmark(0.45, 0.50)
        hand[12] = FakeLandmark(0.50, 0.50)
        scroll_val, scrolled = sc.process(hand, timestamp=t)
        assert scrolled is False
        assert sc.direction == 0
        assert sc.current_rate == 0.0
        t += 0.033

        hand[8] = FakeLandmark(0.45, 0.62)
        hand[12] = FakeLandmark(0.50, 0.62)
        down_scrolls = 0
        for _ in range(20):
            scroll_val, scrolled = sc.process(hand, timestamp=t)
            if scrolled:
                assert scroll_val < 0
                down_scrolls += 1
            t += 0.033

        assert down_scrolls > 0

    def test_tracking_dropout_does_not_burst_or_move_neutral(self):
        """A reanchor after dropout should not cause a scroll burst or shift neutral."""
        mock_pag = MagicMock()
        sc = ScrollController(neutral_deadzone=0.025, pyautogui_module=mock_pag)

        hand = make_hand(index_extended=True, middle_folded=False)
        hand[8] = FakeLandmark(0.45, 0.50)
        hand[12] = FakeLandmark(0.50, 0.50)
        sc.process(hand, timestamp=1.0)
        original_neutral = sc.neutral_y

        hand[8] = FakeLandmark(0.45, 0.38)
        hand[12] = FakeLandmark(0.50, 0.38)
        t = 1.033
        for _ in range(5):
            sc.process(hand, timestamp=t)
            t += 0.033

        sc.reanchor(hand, timestamp=t)
        t += 0.033

        assert sc.neutral_y == pytest.approx(original_neutral)

        scroll_val, scrolled = sc.process(hand, timestamp=t)
        if scrolled:
            assert abs(scroll_val) <= 3


class TestMouseControllerCursor:
    """Verifies cursor movement dispatch with mocked PyAutoGUI."""

    def test_cursor_moves_when_pointing_pose_is_active_and_thumb_lowered(self):
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        hand = make_hand(index_extended=True, middle_folded=True, thumb_raised=False)
        hand[8] = FakeLandmark(0.45, 0.20)

        mode, clicked, scroll_val, scrolled = ctrl.process_landmarks(hand)
        assert mode == "CURSOR"
        assert ctrl.is_active is True
        assert clicked is False
        assert scrolled is False
        mock_pag.moveTo.assert_called_once()

    def test_cursor_does_not_move_when_pose_is_not_pointing(self):
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        hand = make_hand(index_extended=True, middle_folded=False, ring_folded=False, pinky_folded=False)
        mode, clicked, scroll_val, scrolled = ctrl.process_landmarks(hand)

        assert mode == "IDLE"
        assert ctrl.is_active is False
        mock_pag.moveTo.assert_not_called()

    def test_pose_transition_stops_cursor_and_resets_smoother(self):
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        pointing_hand = make_hand(thumb_raised=False)
        ctrl.process_landmarks(pointing_hand)
        assert ctrl.is_active is True
        assert mock_pag.moveTo.call_count == 1

        open_palm = make_hand(index_extended=True, middle_folded=False, ring_folded=False, pinky_folded=False)
        ctrl.process_landmarks(open_palm)
        assert ctrl.is_active is False
        assert mock_pag.moveTo.call_count == 1

    def test_thumb_already_raised_on_mode_entry_moves_cursor_without_clicking(self):
        """Entering cursor mode with an already raised thumb moves cursor without clicking."""
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        hand_raised = make_hand(index_extended=True, middle_folded=True, thumb_raised=True)
        hand_raised[8] = FakeLandmark(0.40, 0.30)
        mode, clicked, _, _ = ctrl.process_landmarks(hand_raised)

        assert mode == "CURSOR"
        assert clicked is False
        mock_pag.click.assert_not_called()
        mock_pag.moveTo.assert_called_once()


class TestMouseControllerThumbClick:
    """Verifies one click per thumb raise, rearming on lower, entry rejection, and cursor freeze."""

    def test_one_click_per_thumb_raise(self):
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        hand_lowered = make_hand(thumb_raised=False)
        mode, clicked, _, _ = ctrl.process_landmarks(hand_lowered)
        assert mode == "CURSOR"
        assert clicked is False
        assert mock_pag.click.call_count == 0

        hand_raised = make_hand(thumb_raised=True)
        mode, clicked, _, _ = ctrl.process_landmarks(hand_raised)
        assert clicked is True
        assert mock_pag.click.call_count == 1

        for _ in range(5):
            mode, clicked, _, _ = ctrl.process_landmarks(hand_raised)
            assert clicked is False

        assert mock_pag.click.call_count == 1

    def test_rearming_after_thumb_lowering(self):
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        ctrl.process_landmarks(make_hand(thumb_raised=False))
        ctrl.process_landmarks(make_hand(thumb_raised=True))
        assert mock_pag.click.call_count == 1

        _, clicked_lowered, _, _ = ctrl.process_landmarks(make_hand(thumb_raised=False))
        assert clicked_lowered is False
        assert mock_pag.click.call_count == 1

        _, clicked_second, _, _ = ctrl.process_landmarks(make_hand(thumb_raised=True))
        assert clicked_second is True
        assert mock_pag.click.call_count == 2

    def test_no_click_on_mode_entry_with_raised_thumb(self):
        """Entering cursor mode with an already raised thumb must NOT click."""
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        ctrl.on_pose_lost()

        hand_raised = make_hand(thumb_raised=True)
        mode, clicked, _, _ = ctrl.process_landmarks(hand_raised)
        assert mode == "CURSOR"
        assert clicked is False
        mock_pag.click.assert_not_called()

        for _ in range(4):
            _, c, _, _ = ctrl.process_landmarks(hand_raised)
            assert c is False

        mock_pag.click.assert_not_called()

        ctrl.process_landmarks(make_hand(thumb_raised=False))

        _, click_after_rearm, _, _ = ctrl.process_landmarks(make_hand(thumb_raised=True))
        assert click_after_rearm is True
        mock_pag.click.assert_called_once()

    def test_cursor_frozen_at_stable_position_during_thumb_raise(self):
        """Cursor must stay at last stable position while thumb click is recognized."""
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        hand = make_hand(thumb_raised=False)
        hand[8] = FakeLandmark(0.40, 0.40)
        ctrl.process_landmarks(hand)
        assert mock_pag.moveTo.call_count == 1
        initial_move_args = mock_pag.moveTo.call_args[0]

        hand[8] = FakeLandmark(0.43, 0.38)
        hand_raised = make_hand(thumb_raised=True)
        hand_raised[8] = FakeLandmark(0.43, 0.38)

        mode, clicked, _, _ = ctrl.process_landmarks(hand_raised)
        assert clicked is True
        mock_pag.click.assert_called_once()

        assert mock_pag.moveTo.call_count == 1
        assert mock_pag.moveTo.call_args[0] == initial_move_args

        hand_raised[8] = FakeLandmark(0.60, 0.60)
        ctrl.process_landmarks(hand_raised)
        assert mock_pag.moveTo.call_count == 1


class TestMouseControllerScrolling:
    """Verifies two-finger scrolling execution, cursor freeze, and thumb inactivity."""

    def test_two_finger_scroll_up_and_down(self):
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        hand = make_hand(index_extended=True, middle_folded=False)
        mode, _, scroll_val, scrolled = ctrl.process_landmarks(hand, timestamp=1.0)
        assert mode == "SCROLL"
        assert scrolled is False

        hand_up = [FakeLandmark(lm.x, lm.y) for lm in hand]
        hand_up[8] = FakeLandmark(hand[8].x, 0.08)
        hand_up[12] = FakeLandmark(hand[12].x, 0.08)

        up_scrolls = 0
        t = 1.033
        for _ in range(20):
            mode, _, scroll_val, scrolled = ctrl.process_landmarks(hand_up, timestamp=t)
            assert mode == "SCROLL"
            if scrolled:
                assert scroll_val > 0
                up_scrolls += 1
            t += 0.033
        assert up_scrolls > 0

        hand_down = [FakeLandmark(lm.x, lm.y) for lm in hand]
        hand_down[8] = FakeLandmark(hand[8].x, 0.40)
        hand_down[12] = FakeLandmark(hand[12].x, 0.40)

        down_scrolls = 0
        for _ in range(20):
            mode, _, scroll_val, scrolled = ctrl.process_landmarks(hand_down, timestamp=t)
            assert mode == "SCROLL"
            if scrolled:
                assert scroll_val < 0
                down_scrolls += 1
            t += 0.033
        assert down_scrolls > 0

    def test_cursor_movement_frozen_during_scrolling(self):
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        hand = make_hand(index_extended=True, middle_folded=False)
        t = 1.0
        for i in range(10):
            hand[8] = FakeLandmark(0.20 + i * 0.04, 0.50 - i * 0.03)
            hand[12] = FakeLandmark(0.25 + i * 0.04, 0.50 - i * 0.03)
            mode, _, _, _ = ctrl.process_landmarks(hand, timestamp=t)
            assert mode == "SCROLL"
            t += 0.033

        mock_pag.moveTo.assert_not_called()

    def test_cursor_and_thumb_click_inactive_during_scrolling(self):
        """Scrolling with raised thumb must NOT click or move cursor."""
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        scroll_hand = make_hand(index_extended=True, middle_folded=False, thumb_raised=True)
        scroll_hand[8] = FakeLandmark(0.45, 0.50)
        scroll_hand[12] = FakeLandmark(0.50, 0.50)
        ctrl.process_landmarks(scroll_hand, timestamp=1.0)

        scroll_hand[8] = FakeLandmark(0.45, 0.35)
        scroll_hand[12] = FakeLandmark(0.50, 0.35)
        t = 1.033
        for _ in range(10):
            mode, clicked, _, _ = ctrl.process_landmarks(scroll_hand, timestamp=t)
            assert mode == "SCROLL"
            assert clicked is False
            t += 0.033

        mock_pag.click.assert_not_called()
        mock_pag.moveTo.assert_not_called()


class TestPriorityAndTransitions:
    """Verifies scrolling priority, mode transitions, and hand loss cleanup."""

    def test_two_finger_scrolling_takes_priority_over_thumb_clicking(self):
        """Scroll pose with raised thumb must scroll, NOT click."""
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        scroll_hand_raised_thumb = make_hand(
            index_extended=True,
            middle_folded=False,
            thumb_raised=True,
        )

        mode, clicked, _, _ = ctrl.process_landmarks(scroll_hand_raised_thumb)
        assert mode == "SCROLL"
        assert clicked is False
        mock_pag.click.assert_not_called()

    def test_transition_from_cursor_to_scroll_and_back(self):
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        pointing = make_hand(index_extended=True, middle_folded=True, thumb_raised=False)
        mode1, _, _, _ = ctrl.process_landmarks(pointing)
        assert mode1 == "CURSOR"

        scroll = make_hand(index_extended=True, middle_folded=False)
        mode2, _, _, _ = ctrl.process_landmarks(scroll)
        assert mode2 == "SCROLL"
        assert ctrl.smoother.smoothed_x is None

        mode3, _, _, _ = ctrl.process_landmarks(pointing)
        assert mode3 == "CURSOR"
        assert ctrl.scroll_ctrl.neutral_y is None

    def test_noisy_pointing_pose_tolerates_intermittent_frames(self):
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        pointing = make_hand(index_extended=True, middle_folded=True, thumb_raised=False)
        mode, _, _, _ = ctrl.process_landmarks(pointing)
        assert mode == "CURSOR"
        assert mock_pag.moveTo.call_count == 1

        noisy = make_hand(index_extended=True, middle_folded=True, thumb_raised=False)
        noisy[20] = FakeLandmark(0.60, 0.44)
        assert is_pointing_pose(noisy) is False
        assert is_pointing_candidate(noisy) is True
        assert is_scroll_pose(noisy) is False

        mode_noisy, _, _, _ = ctrl.process_landmarks(noisy)
        assert mode_noisy == "CURSOR"
        assert ctrl.uncertain_frames == 1
        assert mock_pag.moveTo.call_count == 1
        assert ctrl.smoother.smoothed_x is not None

        mode_clean, _, _, _ = ctrl.process_landmarks(pointing)
        assert mode_clean == "CURSOR"
        assert ctrl.uncertain_frames == 0

    def test_one_frame_tracking_loss_holds_position_safely(self):
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        pointing = make_hand(thumb_raised=False)
        ctrl.process_landmarks(pointing)
        assert ctrl.is_active is True
        assert mock_pag.moveTo.call_count == 1

        mode_drop, _, _, _ = ctrl.process_landmarks(None)
        assert mode_drop == "CURSOR"
        assert mock_pag.moveTo.call_count == 1
        assert ctrl.smoother.smoothed_x is not None

    def test_recovery_without_cursor_jump_after_dropout(self):
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        pointing1 = make_hand(thumb_raised=False)
        pointing1[8] = FakeLandmark(0.40, 0.40)
        ctrl.process_landmarks(pointing1)
        pos1 = mock_pag.moveTo.call_args[0]

        ctrl.process_landmarks(None)

        pointing2 = make_hand(thumb_raised=False)
        pointing2[8] = FakeLandmark(0.41, 0.40)
        ctrl.process_landmarks(pointing2)
        pos2 = mock_pag.moveTo.call_args[0]

        assert abs(pos2[0] - pos1[0]) < 100

    def test_hand_loss_resets_state_and_prevents_bursts(self):
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        pointing = make_hand(thumb_raised=False)
        ctrl.process_landmarks(pointing)
        assert ctrl.is_active is True

        ctrl.process_landmarks(None)
        mode, clicked, scroll_val, scrolled = ctrl.process_landmarks(None)
        assert mode == "IDLE"
        assert ctrl.is_active is False
        assert ctrl.click_armed is False
        assert ctrl.smoother.smoothed_x is None

        raised_hand = make_hand(thumb_raised=True)
        mode_ret, clicked_ret, _, _ = ctrl.process_landmarks(raised_hand)
        assert mode_ret == "CURSOR"
        assert clicked_ret is False
        mock_pag.click.assert_not_called()

    def test_failsafe_exception_propagates_from_cursor_move(self):
        """FailSafeException from moveTo must propagate to the application boundary."""
        mock_pag = MagicMock()
        class FakeFailSafe(Exception):
            pass
        mock_pag.FailSafeException = FakeFailSafe
        mock_pag.moveTo.side_effect = FakeFailSafe("Corner triggered")

        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)
        pointing_hand = make_hand(thumb_raised=False)

        with pytest.raises(FakeFailSafe):
            ctrl.process_landmarks(pointing_hand)


class TestSwipeSuppression:
    """Verifies that mouse mode movements (pointing/scrolling) suppress swipe detection."""

    @patch("main.perform_gesture_action")
    def test_pointing_pose_suppresses_swipe_actions(self, mock_action):
        detector = PalmSwipeDetector()
        base_pointing_hand = make_hand(index_extended=True, middle_folded=True, thumb_raised=False)

        for i in range(15):
            dx = i * 0.03
            moving_hand = translate_hand(base_pointing_hand, dx, 0.0)
            state = detector.update(moving_hand)
            assert state == GestureState.CURSOR_CONTROL

        mock_action.assert_not_called()
        assert detector.last_detected_gesture == ""

    @patch("main.perform_gesture_action")
    def test_scroll_pose_suppresses_swipe_actions(self, mock_action):
        detector = PalmSwipeDetector()
        base_scroll_hand = make_hand(index_extended=True, middle_folded=False)

        for i in range(15):
            dy = -i * 0.03
            moving_hand = translate_hand(base_scroll_hand, 0.0, dy)
            state = detector.update(moving_hand)
            assert state == GestureState.SCROLL_CONTROL

        mock_action.assert_not_called()
        assert detector.last_detected_gesture == ""

    def test_candidate_pose_clears_stale_history_on_entry(self):
        detector = PalmSwipeDetector()

        neutral_hand = make_open_palm()
        for _ in range(4):
            detector.update(neutral_hand)
        assert len(detector.palm_history) == 4

        pointing_hand = make_hand(index_extended=True, middle_folded=True)
        detector.update(pointing_hand)

        assert len(detector.palm_history) == 0
        assert detector.pointing_suppression is True

    @patch("main.perform_gesture_action")
    def test_exiting_mouse_mode_requires_fresh_movement_before_swipe(self, mock_action):
        detector = PalmSwipeDetector()

        pointing_hand = make_hand(index_extended=True, middle_folded=True)
        for _ in range(5):
            detector.update(pointing_hand)

        open_palm = make_hand(index_extended=True, middle_folded=False, ring_folded=False, pinky_folded=False, thumb_raised=True)
        state = detector.update(open_palm)

        assert state == GestureState.TRACKING
        mock_action.assert_not_called()
        assert len(detector.palm_history) == 1

    def test_brief_tracking_dropout_does_not_trigger_swipe(self):
        detector = PalmSwipeDetector()
        pointing_hand = make_hand(index_extended=True, middle_folded=True)

        detector.update(pointing_hand)
        detector.on_hand_lost()
        assert len(detector.palm_history) == 0

        state = detector.update(pointing_hand)
        assert state == GestureState.CURSOR_CONTROL
        assert len(detector.palm_history) == 0


class TestSwipeSuppressionWithMouseMode:
    """Verifies swipe suppression via the active_mouse_mode parameter passed
    from MouseController through the main camera loop."""

    @patch("main.perform_gesture_action")
    def test_no_swipe_during_scroll_grace_frames(self, mock_action):
        """Even when the current pose resembles an open palm, no swipe fires
        while the mouse controller reports SCROLL mode (e.g. grace period)."""
        detector = PalmSwipeDetector()

        scroll_hand = make_hand(index_extended=True, middle_folded=False)
        for _ in range(5):
            detector.update(scroll_hand, active_mouse_mode="SCROLL")

        open_palm = make_hand(
            index_extended=True, middle_folded=False,
            ring_folded=False, pinky_folded=False,
        )
        for i in range(10):
            moved = translate_hand(open_palm, 0.0, -i * 0.03)
            state = detector.update(moved, active_mouse_mode="SCROLL")
            assert "Swipe" not in state
            assert detector.last_detected_gesture == ""

        mock_action.assert_not_called()

    @patch("main.perform_gesture_action")
    def test_leaving_mouse_mode_requires_fresh_swipe_movement(self, mock_action):
        """After exiting CURSOR/SCROLL mode, the swipe detector must require
        fresh movement history before executing any swipe command."""
        detector = PalmSwipeDetector()

        pointing = make_hand(index_extended=True, middle_folded=True)
        for i in range(10):
            moved = translate_hand(pointing, i * 0.02, 0.0)
            detector.update(moved, active_mouse_mode="CURSOR")

        neutral = make_open_palm()
        state = detector.update(neutral, active_mouse_mode="IDLE")
        assert "Swipe" not in state
        mock_action.assert_not_called()
        assert len(detector.palm_history) == 1


class TestFailSafePropagation:
    """Verifies that FailSafeException propagates from every action site
    to the application boundary without being swallowed."""

    def _make_failsafe_pag(self):
        """Create a mock pyautogui with a FakeFailSafe exception class."""
        mock_pag = MagicMock()
        class FakeFailSafe(Exception):
            pass
        mock_pag.FailSafeException = FakeFailSafe
        return mock_pag, FakeFailSafe

    def test_failsafe_propagates_from_cursor_move(self):
        mock_pag, FakeFailSafe = self._make_failsafe_pag()
        mock_pag.moveTo.side_effect = FakeFailSafe("Corner triggered")

        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)
        with pytest.raises(FakeFailSafe):
            ctrl.process_landmarks(make_hand(thumb_raised=False))

    def test_failsafe_propagates_from_thumb_click(self):
        mock_pag, FakeFailSafe = self._make_failsafe_pag()
        mock_pag.click.side_effect = FakeFailSafe("Corner triggered")

        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)
        ctrl.process_landmarks(make_hand(thumb_raised=False))
        with pytest.raises(FakeFailSafe):
            ctrl.process_landmarks(make_hand(thumb_raised=True))

    def test_failsafe_propagates_from_scroll(self):
        mock_pag, FakeFailSafe = self._make_failsafe_pag()
        mock_pag.scroll.side_effect = FakeFailSafe("Corner triggered")

        sc = ScrollController(pyautogui_module=mock_pag)
        hand = make_hand(index_extended=True, middle_folded=False)
        hand[8] = FakeLandmark(0.45, 0.50)
        hand[12] = FakeLandmark(0.50, 0.50)
        sc.process(hand, timestamp=1.0)

        hand[8] = FakeLandmark(0.45, 0.30)
        hand[12] = FakeLandmark(0.50, 0.30)
        t = 1.033
        with pytest.raises(FakeFailSafe):
            for _ in range(10):
                sc.process(hand, timestamp=t)
                t += 0.033

    def test_failsafe_from_keyboard_action_propagates(self):
        """FailSafeException from perform_gesture_action must propagate."""
        class FakeFailSafe(Exception):
            pass

        with patch("main.pyautogui") as mock_pag:
            mock_pag.FailSafeException = FakeFailSafe
            mock_pag.press.side_effect = FakeFailSafe("Corner triggered")

            with pytest.raises(FakeFailSafe):
                perform_gesture_action(GestureState.SWIPE_RIGHT, target_mode_powerpoint=False)


class TestMainLoopIntegration:
    """Tests the coordination between MouseController and PalmSwipeDetector
    as performed by the main camera loop, using mocked desktop input."""

    def _simulate_frame(self, mouse_ctrl, detector, hand_landmarks, timestamp):
        """Simulate one iteration of the main camera loop coordination."""
        if hand_landmarks is not None:
            mode, clicked, scroll_val, scrolled = mouse_ctrl.process_landmarks(
                hand_landmarks, timestamp=timestamp
            )
            state_label = detector.update(hand_landmarks, active_mouse_mode=mode)
        else:
            mode, clicked, scroll_val, scrolled = mouse_ctrl.process_landmarks(
                None, timestamp=timestamp
            )
            detector.on_hand_lost()
            if mode in ("CURSOR", "SCROLL"):
                state_label = (
                    GestureState.CURSOR_CONTROL if mode == "CURSOR"
                    else GestureState.SCROLL_CONTROL
                )
            else:
                state_label = detector.state
        return mode, clicked, scroll_val, scrolled, state_label

    def test_one_missing_frame_preserves_cursor_smoother(self):
        """A single frame without hand detection must preserve the cursor smoother."""
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)
        detector = PalmSwipeDetector()

        pointing = make_hand(thumb_raised=False)
        pointing[8] = FakeLandmark(0.40, 0.30)
        self._simulate_frame(ctrl, detector, pointing, 1.0)
        assert ctrl.smoother.smoothed_x is not None
        saved_smooth = ctrl.smoother.smoothed_x

        mode, clicked, scroll_val, scrolled, _ = self._simulate_frame(
            ctrl, detector, None, 1.033
        )
        assert mode == "CURSOR"
        assert clicked is False
        assert scrolled is False
        assert ctrl.smoother.smoothed_x == saved_smooth

    def test_one_missing_frame_preserves_scroll_neutral(self):
        """Scroll neutral_y must survive a single tracking dropout."""
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)
        detector = PalmSwipeDetector()

        scroll_hand = make_hand(index_extended=True, middle_folded=False)
        self._simulate_frame(ctrl, detector, scroll_hand, 1.0)
        assert ctrl.scroll_ctrl.neutral_y is not None
        saved_neutral = ctrl.scroll_ctrl.neutral_y

        mode, _, _, _, _ = self._simulate_frame(ctrl, detector, None, 1.033)
        assert mode == "SCROLL"
        assert ctrl.scroll_ctrl.neutral_y == saved_neutral

    def test_prolonged_hand_loss_resets_state(self):
        """Multiple consecutive frames without a hand must fully reset the controller."""
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)
        detector = PalmSwipeDetector()

        pointing = make_hand(thumb_raised=False)
        self._simulate_frame(ctrl, detector, pointing, 1.0)
        assert ctrl.is_active is True

        mode1, _, _, _, _ = self._simulate_frame(ctrl, detector, None, 1.033)
        assert mode1 == "CURSOR"

        mode2, _, _, _, _ = self._simulate_frame(ctrl, detector, None, 1.066)
        assert mode2 == "IDLE"
        assert ctrl.is_active is False
        assert ctrl.smoother.smoothed_x is None

    def test_recovery_produces_no_unintended_input(self):
        """Recovering from a brief dropout must not trigger a click, swipe, or scroll burst."""
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)
        detector = PalmSwipeDetector()

        pointing = make_hand(thumb_raised=False)
        pointing[8] = FakeLandmark(0.40, 0.30)
        self._simulate_frame(ctrl, detector, pointing, 1.0)

        self._simulate_frame(ctrl, detector, None, 1.033)

        pointing2 = make_hand(thumb_raised=False)
        pointing2[8] = FakeLandmark(0.41, 0.30)
        mode, clicked, scroll_val, scrolled, _ = self._simulate_frame(
            ctrl, detector, pointing2, 1.066
        )
        assert mode == "CURSOR"
        assert clicked is False
        assert scrolled is False
        mock_pag.click.assert_not_called()

    @patch("main.perform_gesture_action")
    def test_failsafe_reaches_application_boundary(self, mock_action):
        """FailSafe during cursor movement must propagate through the main-loop
        coordination rather than being caught silently."""
        mock_pag = MagicMock()
        class FakeFailSafe(Exception):
            pass
        mock_pag.FailSafeException = FakeFailSafe

        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)
        detector = PalmSwipeDetector()

        pointing = make_hand(thumb_raised=False)
        self._simulate_frame(ctrl, detector, pointing, 1.0)

        mock_pag.moveTo.side_effect = FakeFailSafe("Corner triggered")
        pointing2 = make_hand(thumb_raised=False)
        pointing2[8] = FakeLandmark(0.50, 0.40)

        with pytest.raises(FakeFailSafe):
            self._simulate_frame(ctrl, detector, pointing2, 1.033)

        mock_action.assert_not_called()


class TestOpenPalmSwipeRequirements:
    """Verifies that swipe detection strictly requires an open palm,
    rejects moving fists and invalid poses, and accepts rotated/mirrored palms."""

    @patch("main.perform_gesture_action")
    def test_moving_fist_does_not_trigger_swipe(self, mock_action):
        """A moving fist must not trigger swipe commands or collect history."""
        detector = PalmSwipeDetector()
        fist = make_hand(index_extended=False, middle_folded=True, ring_folded=True, pinky_folded=True)

        for i in range(8):
            moved = translate_hand(fist, i * 0.03, 0.0)
            state = detector.update(moved)
            assert state == GestureState.TRACKING
            assert len(detector.palm_history) == 0

        mock_action.assert_not_called()

    @patch("main.perform_gesture_action")
    def test_invalid_finger_poses_do_not_trigger_swipes_or_leave_history(self, mock_action):
        """Invalid finger poses must not trigger swipes or leave reusable movement history."""
        detector = PalmSwipeDetector()

        fist = make_hand(index_extended=False, middle_folded=True, ring_folded=True, pinky_folded=True)
        detector.update(fist)
        detector.update(translate_hand(fist, 0.05, 0.0))
        assert len(detector.palm_history) == 0

        three_fingers = make_hand(index_extended=True, middle_folded=False, ring_folded=False, pinky_folded=True)
        detector.update(translate_hand(three_fingers, 0.10, 0.0))
        assert len(detector.palm_history) == 0

        open_palm = make_open_palm()
        t = 1000.0
        with patch("main.time.time", side_effect=lambda: t):
            state = detector.update(translate_hand(open_palm, 0.15, 0.0))
            assert state == GestureState.TRACKING
            # In confirmation window immediately upon returning from 3-finger pose
            assert len(detector.palm_history) == 0

            # Advance past 120 ms confirmation window -> fresh history begins
            t += 0.15
            state = detector.update(translate_hand(open_palm, 0.16, 0.0))
            assert state == GestureState.TRACKING
            assert len(detector.palm_history) == 1
            mock_action.assert_not_called()

    @patch("main.perform_gesture_action")
    def test_open_palm_swipe_in_all_four_directions(self, mock_action):
        """An open-palm swipe must still work in all four cardinal directions."""
        for dx, dy, expected_gesture in [
            (0.03, 0.0, GestureState.SWIPE_RIGHT),
            (-0.03, 0.0, GestureState.SWIPE_LEFT),
            (0.0, -0.03, GestureState.SWIPE_UP),
            (0.0, 0.03, GestureState.SWIPE_DOWN),
        ]:
            mock_action.reset_mock()
            detector = PalmSwipeDetector()
            palm = make_open_palm(wrist_x=0.5, wrist_y=0.7)
            triggered = False
            for i in range(6):
                moved = translate_hand(palm, i * dx, i * dy)
                state = detector.update(moved)
                if state == expected_gesture:
                    triggered = True
                    break
            assert triggered is True, f"Failed to trigger {expected_gesture}"
            mock_action.assert_called_once_with(expected_gesture)

    def test_representative_mirrored_and_rotated_open_palms_accepted(self):
        """Representative mirrored and rotated open palms must be accepted by is_open_palm_pose."""
        base_palm = make_open_palm()
        assert is_open_palm_pose(base_palm) is True

        for angle in (-45, -30, 30, 45, 90, 180):
            rotated = rotate_landmarks(base_palm, angle)
            assert is_open_palm_pose(rotated) is True, f"Failed at {angle} deg rotation"

        mirrored = mirror_landmarks(base_palm)
        assert is_open_palm_pose(mirrored) is True

        fist = make_hand(index_extended=False, middle_folded=True, ring_folded=True, pinky_folded=True)
        assert is_open_palm_pose(fist) is False

        pointing = make_hand(index_extended=True, middle_folded=True, ring_folded=True, pinky_folded=True)
        assert is_open_palm_pose(pointing) is False

        scroll = make_hand(index_extended=True, middle_folded=False, ring_folded=True, pinky_folded=True)
        assert is_open_palm_pose(scroll) is False

        three_fingers = make_hand(index_extended=True, middle_folded=False, ring_folded=False, pinky_folded=True)
        assert is_open_palm_pose(three_fingers) is False

        assert is_open_palm_pose(None) is False
        assert is_open_palm_pose([]) is False
        assert is_open_palm_pose([FakeLandmark(0.5, 0.5)] * 10) is False


class TestScrollReversalConfirmation:
    """Verifies that reversing scroll direction requires consecutive frames,
    resets pending state on direction match or neutral, and clears on dropout."""

    def test_down_up_down_sequence_does_not_reverse_direction(self):
        """After upward scrolling, the sequence down/up/down must not reverse direction."""
        mock_pag = MagicMock()
        ctrl = ScrollController(pyautogui_module=mock_pag)

        hand = make_hand(index_extended=True, middle_folded=False)
        hand[8] = FakeLandmark(0.45, 0.50)
        hand[12] = FakeLandmark(0.55, 0.50)
        ctrl.process(hand, timestamp=1.0)
        assert ctrl.neutral_y == 0.50

        hand_up = [FakeLandmark(lm.x, lm.y) for lm in hand]
        hand_up[8] = FakeLandmark(0.45, 0.40)
        hand_up[12] = FakeLandmark(0.55, 0.40)
        ctrl.process(hand_up, timestamp=1.05)
        ctrl.process(hand_up, timestamp=1.10)
        assert ctrl.direction == 1

        hand_down = [FakeLandmark(lm.x, lm.y) for lm in hand]
        hand_down[8] = FakeLandmark(0.45, 0.60)
        hand_down[12] = FakeLandmark(0.55, 0.60)
        amt, scrolled = ctrl.process(hand_down, timestamp=1.15)
        assert scrolled is False
        assert ctrl.direction == 1
        assert ctrl.pending_direction == -1
        assert ctrl.pending_frames == 1

        ctrl.process(hand_up, timestamp=1.20)
        assert ctrl.direction == 1
        assert ctrl.pending_direction == 0
        assert ctrl.pending_frames == 0

        amt, scrolled = ctrl.process(hand_down, timestamp=1.25)
        assert scrolled is False
        assert ctrl.direction == 1
        assert ctrl.pending_direction == -1
        assert ctrl.pending_frames == 1

    def test_two_consecutive_downward_frames_reverse_direction(self):
        """Two consecutive downward frames must successfully reverse scroll direction."""
        mock_pag = MagicMock()
        ctrl = ScrollController(pyautogui_module=mock_pag)

        hand = make_hand(index_extended=True, middle_folded=False)
        hand[8] = FakeLandmark(0.45, 0.50)
        hand[12] = FakeLandmark(0.55, 0.50)
        ctrl.process(hand, timestamp=1.0)

        hand_up = [FakeLandmark(lm.x, lm.y) for lm in hand]
        hand_up[8] = FakeLandmark(0.45, 0.40)
        hand_up[12] = FakeLandmark(0.55, 0.40)
        ctrl.process(hand_up, timestamp=1.05)
        ctrl.process(hand_up, timestamp=1.10)
        assert ctrl.direction == 1

        hand_down = [FakeLandmark(lm.x, lm.y) for lm in hand]
        hand_down[8] = FakeLandmark(0.45, 0.60)
        hand_down[12] = FakeLandmark(0.55, 0.60)
        ctrl.process(hand_down, timestamp=1.15)
        assert ctrl.direction == 1
        assert ctrl.pending_frames == 1

        amt, scrolled = ctrl.process(hand_down, timestamp=1.20)
        assert scrolled is True
        assert ctrl.direction == -1
        assert amt < 0
        mock_pag.scroll.assert_called_with(-1, _pause=False)

    def test_neutral_frames_and_tracking_gaps_clear_pending_confirmation(self):
        """Neutral frames and tracking gaps must clear pending reversal confirmation."""
        mock_pag = MagicMock()
        ctrl = ScrollController(pyautogui_module=mock_pag)

        hand = make_hand(index_extended=True, middle_folded=False)
        hand[8] = FakeLandmark(0.45, 0.50)
        hand[12] = FakeLandmark(0.55, 0.50)
        ctrl.process(hand, timestamp=1.0)

        hand_up = [FakeLandmark(lm.x, lm.y) for lm in hand]
        hand_up[8] = FakeLandmark(0.45, 0.40)
        hand_up[12] = FakeLandmark(0.55, 0.40)
        ctrl.process(hand_up, timestamp=1.05)
        ctrl.process(hand_up, timestamp=1.10)
        assert ctrl.direction == 1

        hand_down = [FakeLandmark(lm.x, lm.y) for lm in hand]
        hand_down[8] = FakeLandmark(0.45, 0.60)
        hand_down[12] = FakeLandmark(0.55, 0.60)

        ctrl.process(hand_down, timestamp=1.15)
        assert ctrl.pending_frames == 1

        ctrl.process(hand, timestamp=1.20)
        assert ctrl.pending_direction == 0
        assert ctrl.pending_frames == 0

        ctrl.process(hand_down, timestamp=1.25)
        assert ctrl.pending_frames == 1

        ctrl.reanchor(hand, timestamp=1.30)
        assert ctrl.pending_direction == 0
        assert ctrl.pending_frames == 0
        assert ctrl.neutral_y == 0.50


class TestHUDActionBadgeTiming:
    """Verifies that draw_hud action badge appears, expires after display duration,
    and accurately computes age with monotonic timestamps."""

    @patch("main.cv2.putText")
    @patch("main.cv2.rectangle")
    def test_hud_action_badge_expiry_monotonic(self, mock_rect, mock_put_text):
        """Action badge must show within duration, expire afterward, and reject negative age."""
        import numpy as np

        frame = np.zeros((700, 1000, 3), dtype=np.uint8)
        detector = PalmSwipeDetector()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=MagicMock())

        ctrl.last_action = "Left Click"
        ctrl.last_action_time = 100.0

        draw_hud(frame, detector, "Tracking palm...", ctrl, fps=30.0, now_mono=100.2)
        click_calls = [
            call for call in mock_put_text.call_args_list
            if "Action: Left Click" in str(call)
        ]
        assert len(click_calls) > 0

        mock_put_text.reset_mock()
        draw_hud(frame, detector, "Tracking palm...", ctrl, fps=30.0, now_mono=100.65)
        click_calls = [
            call for call in mock_put_text.call_args_list
            if "Action: Left Click" in str(call)
        ]
        assert len(click_calls) == 0

        mock_put_text.reset_mock()
        draw_hud(frame, detector, "Tracking palm...", ctrl, fps=30.0, now_mono=99.0)
        click_calls = [
            call for call in mock_put_text.call_args_list
            if "Action: Left Click" in str(call)
        ]
        assert len(click_calls) == 0


class TestRightClickPoseDetection:
    """Verifies geometric detection and scale/rotation invariance for thumb-to-middle pinch."""

    def test_valid_pinch_pose_detected_when_index_extended_ring_pinky_folded(self):
        hand = make_pinch_hand(pinch_ratio=0.15)
        assert is_right_click_pose(hand) is True
        ratio = get_thumb_middle_pinch_ratio(hand)
        assert abs(ratio - 0.15) < 1e-4

    def test_pinch_pose_rejected_when_index_folded(self):
        hand = make_pinch_hand(pinch_ratio=0.15)
        hand[8] = FakeLandmark(0.45, 0.60)
        assert is_right_click_pose(hand) is False

    def test_pinch_pose_rejected_when_ring_extended(self):
        hand = make_pinch_hand(pinch_ratio=0.15)
        hand[16] = FakeLandmark(0.55, 0.20)
        assert is_right_click_pose(hand) is False

    def test_pinch_pose_rejected_when_pinky_extended(self):
        hand = make_pinch_hand(pinch_ratio=0.15)
        hand[20] = FakeLandmark(0.60, 0.20)
        assert is_right_click_pose(hand) is False

    def test_pinch_ratio_normalization_and_scale_invariance(self):
        hand1 = make_pinch_hand(pinch_ratio=0.25, wrist_x=0.5, wrist_y=0.8)
        hand2 = make_hand(
            index_extended=True, middle_folded=True, ring_folded=True, pinky_folded=True,
            wrist_x=0.5, wrist_y=0.9
        )
        scale2 = 0.30
        hand2[5] = FakeLandmark(0.45, 0.60)
        hand2[9] = FakeLandmark(0.50, 0.60)
        hand2[13] = FakeLandmark(0.55, 0.60)
        hand2[17] = FakeLandmark(0.60, 0.60)
        hand2[12] = FakeLandmark(0.50, 0.65)
        hand2[4] = FakeLandmark(0.50 - 0.25 * scale2, 0.65)

        r1 = get_thumb_middle_pinch_ratio(hand1)
        r2 = get_thumb_middle_pinch_ratio(hand2)
        assert abs(r1 - 0.25) < 1e-4
        assert abs(r2 - 0.25) < 1e-4
        assert is_right_click_pose(hand1) is True
        assert is_right_click_pose(hand2) is True

    def test_pinch_pose_rotation_and_mirror_invariance(self):
        hand = make_pinch_hand(pinch_ratio=0.15)
        assert is_right_click_pose(hand) is True

        for angle in (-45, -30, 30, 45, 90, 180):
            rot = rotate_landmarks(hand, angle)
            assert is_right_click_pose(rot) is True, f"Failed at {angle} deg"
            r = get_thumb_middle_pinch_ratio(rot)
            assert abs(r - 0.15) < 1e-4, f"Ratio mismatch at {angle} deg"

        mirrored = mirror_landmarks(hand)
        assert is_right_click_pose(mirrored) is True
        r_mir = get_thumb_middle_pinch_ratio(mirrored)
        assert abs(r_mir - 0.15) < 1e-4

    def test_absent_or_empty_landmarks_handled_safely(self):
        assert is_right_click_pose(None) is False
        assert is_right_click_pose([]) is False
        assert is_right_click_pose([FakeLandmark(0.5, 0.5)] * 10) is False
        assert get_thumb_middle_pinch_ratio(None) == 999.0
        assert get_thumb_middle_pinch_ratio([]) == 999.0


class TestRightClickController:
    """Verifies contact confirmation, non-repeating hold, release rearming,
    closed-pinch entry rejection, and tracking loss handling."""

    def test_deliberate_contact_triggers_one_right_click_after_confirmation(self):
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        # Arming requires 150ms of continuous separation beyond release threshold (0.35)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        assert ctrl.armed is False
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)
        assert ctrl.armed is True
        assert ctrl.state == "armed"

        # Moving to contact
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.18)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.19)
        clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.20)
        assert clicked is False
        assert freeze is True
        assert ctrl.state == "confirming"
        assert ctrl.contact_start_time == 1.19
        mock_pag.click.assert_not_called()

        # Under 200ms of contact confirmation does not click
        clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.30)
        assert clicked is False
        assert freeze is True
        assert ctrl.state == "confirming"
        mock_pag.click.assert_not_called()

        # Reaching 200ms continuous contact confirms and fires exactly one right-click
        clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.41)
        assert clicked is True
        assert freeze is True
        assert ctrl.state == "clicked"
        assert ctrl.armed is False
        mock_pag.click.assert_called_once_with(button="right", _pause=False)

    def test_holding_contact_does_not_repeat(self):
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        # Arm with separation
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)
        assert ctrl.armed is True

        # Contact and confirm
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.18)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.19)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.20)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.41)
        assert mock_pag.click.call_count == 1

        # Holding contact for sustained time does not repeat
        for t in (1.50, 1.60, 1.70, 2.0):
            clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=t)
            assert clicked is False
            assert freeze is True
            assert ctrl.state == "clicked"

        assert mock_pag.click.call_count == 1

    def test_separation_followed_by_contact_permits_another_right_click(self):
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        # Arm and perform first click
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.18)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.19)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.20)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.41)
        assert mock_pag.click.call_count == 1

        # Separate beyond release threshold (0.35) - requires 150ms continuous separation to rearm
        clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.48)
        clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.50)
        assert clicked is False
        assert ctrl.state == "clicked"
        assert ctrl.armed is False

        # After 150ms of confirmed separation, becomes armed
        clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.66)
        assert clicked is False
        assert freeze is False
        assert ctrl.armed is True
        assert ctrl.state == "armed"

        # Deliberate re-contact and confirmation yields second click
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.68)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.69)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.70)
        clicked2, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.91)
        assert clicked2 is True
        assert mock_pag.click.call_count == 2

    def test_entering_with_already_closed_pinch_does_not_click(self):
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        # Entering tracking with fingertips already close remains disarmed
        for t in (1.0, 1.05, 1.15, 1.30, 1.50):
            clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=t)
            assert clicked is False
            assert ctrl.state == "disarmed"
            assert ctrl.armed is False

        mock_pag.click.assert_not_called()

        # Separating for >= 150ms arms the controller
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.58)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.60)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.76)
        assert ctrl.armed is True
        assert ctrl.state == "armed"

        # Now contact confirms and clicks
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.78)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.79)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.80)
        clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=2.01)
        assert clicked is True
        mock_pag.click.assert_called_once()

    def test_brief_contact_or_invalid_pose_cancels_confirmation(self):
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        # Arm with separation
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)

        # Start contact
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.18)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.19)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.20)
        assert ctrl.state == "confirming"

        # Contact breaks to approach (0.26)
        ctrl.process(make_pinch_hand(pinch_ratio=0.26), timestamp=1.25)
        clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.26), timestamp=1.26)
        assert clicked is False
        assert ctrl.state == "approaching"
        assert ctrl.contact_start_time is None
        mock_pag.click.assert_not_called()

        # Re-contact starts a fresh 200ms confirmation window
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.28)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.29)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.30)
        assert ctrl.state == "confirming"

        # At 1.40 only 100ms has elapsed, so no click
        clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.40)
        assert clicked is False
        mock_pag.click.assert_not_called()

        # Reaching 200ms confirms click
        clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.51)
        assert clicked is True
        mock_pag.click.assert_called_once()

    def test_tracking_dropout_resets_confirmation_and_preserves_clicked_latch(self):
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        # Arm with separation
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)

        # Contact begins
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.18)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.19)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.20)
        assert ctrl.state == "confirming"

        # Brief tracking loss resets confirmation
        ctrl.on_tracking_lost(prolonged=False)
        assert ctrl.contact_start_time is None

        # Recovery contact must confirm for full 200ms
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.23)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.24)
        clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.25)
        assert clicked is False
        mock_pag.click.assert_not_called()

        clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.46)
        assert clicked is True
        mock_pag.click.assert_called_once()

        # Tracking lost while in clicked state preserves clicked latch
        ctrl.on_tracking_lost(prolonged=False)
        clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.55)
        assert clicked is False
        assert mock_pag.click.call_count == 1

    def test_prolonged_tracking_loss_requires_release_before_arming(self):
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        ctrl.on_tracking_lost(prolonged=True)
        assert ctrl.armed is False
        assert ctrl.state == "disarmed"

        clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.15), timestamp=1.0)
        assert clicked is False
        clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.15), timestamp=1.20)
        assert clicked is False
        mock_pag.click.assert_not_called()


class TestRightClickContactTolerance:
    """Light contact is accepted without counting separation or jitter as contact."""

    def test_light_contact_clicks_once_at_different_camera_frame_rates(self):
        for fps in (15, 30, 60):
            for ratio in (0.15, 0.17):
                mock_pag = MagicMock()
                ctrl = RightClickController(pyautogui_module=mock_pag)
                ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=0.0)
                ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=0.16)
                assert ctrl.armed is True

                click_count = 0
                for frame in range(1, fps + 1):
                    clicked, freeze = ctrl.process(
                        make_pinch_hand(pinch_ratio=ratio),
                        timestamp=0.16 + frame / fps,
                    )
                    click_count += int(clicked)
                    assert freeze is True

                assert click_count == 1, (fps, ratio)
                mock_pag.click.assert_called_once_with(button="right", _pause=False)

    def test_distances_just_outside_contact_never_click(self):
        for ratio in (0.181, 0.19, 0.21, 0.24):
            mock_pag = MagicMock()
            ctrl = RightClickController(pyautogui_module=mock_pag)
            ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
            ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)

            for frame in range(30):
                clicked, freeze = ctrl.process(
                    make_pinch_hand(pinch_ratio=ratio),
                    timestamp=1.20 + frame / 30,
                )
                assert clicked is False
                assert freeze is True
                assert ctrl.state == "approaching"
                assert ctrl.confirmation_progress == 0.0

            mock_pag.click.assert_not_called()

    def test_brief_borderline_jitter_pauses_without_reset_or_premature_click(self):
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)
        ctrl.reset(require_release=False)
        ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=1.12)
        assert ctrl.contact_elapsed == pytest.approx(0.12)

        clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.195), timestamp=1.14)
        assert clicked is False
        assert freeze is True
        assert ctrl.state == "confirming"
        assert ctrl.contact_start_time == 1.0
        assert ctrl.confirmation_progress == pytest.approx(0.60)

        ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=1.18)
        assert ctrl.contact_elapsed == pytest.approx(0.12)
        assert ctrl.contact_jitter_start_time is None
        ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=1.22)
        clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=1.259)
        assert clicked is False
        assert ctrl.contact_elapsed == pytest.approx(0.199)
        mock_pag.click.assert_not_called()

        clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=1.26)
        assert clicked is True
        assert freeze is True
        mock_pag.click.assert_called_once_with(button="right", _pause=False)

    def test_persistent_borderline_gap_resets_and_requires_fresh_contact(self):
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)
        ctrl.reset(require_release=False)
        ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=1.12)
        ctrl.process(make_pinch_hand(pinch_ratio=0.19), timestamp=1.14)
        assert ctrl.state == "confirming"

        for timestamp in (1.21, 1.30, 1.50):
            clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.19), timestamp=timestamp)
            assert clicked is False
            assert ctrl.state == "approaching"
            assert ctrl.contact_elapsed == 0.0
            assert ctrl.contact_start_time is None
        mock_pag.click.assert_not_called()

        ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=1.52)
        clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=1.69)
        assert clicked is False
        clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=1.73)
        assert clicked is True
        mock_pag.click.assert_called_once()

    def test_gap_beyond_jitter_band_cancels_immediately(self):
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)
        ctrl.reset(require_release=False)
        ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=1.12)

        clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.22), timestamp=1.14)
        assert clicked is False
        assert freeze is True
        assert ctrl.state == "approaching"
        assert ctrl.contact_start_time is None
        assert ctrl.contact_elapsed == 0.0
        assert ctrl.contact_jitter_start_time is None
        mock_pag.click.assert_not_called()

    def test_tracking_loss_clears_paused_contact_timer(self):
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)
        ctrl.reset(require_release=False)
        ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=1.12)
        ctrl.process(make_pinch_hand(pinch_ratio=0.195), timestamp=1.14)

        clicked, freeze = ctrl.process(None, timestamp=1.15)
        assert clicked is False
        assert freeze is False
        assert ctrl.contact_elapsed == 0.0
        assert ctrl.contact_last_time is None
        assert ctrl.contact_last_valid is False
        assert ctrl.contact_jitter_start_time is None

        ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=1.18)
        clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=1.30)
        assert clicked is False
        mock_pag.click.assert_not_called()
        clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=1.39)
        assert clicked is True
        mock_pag.click.assert_called_once()

    def test_contact_timer_accepts_zero_timestamp_and_exact_duration(self):
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)
        ctrl.reset(require_release=False)
        ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=0.0)
        clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=0.199)
        assert clicked is False
        clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.17), timestamp=0.20)
        assert clicked is True
        mock_pag.click.assert_called_once()

    def test_light_contact_freezes_cursor_and_does_not_scroll_or_left_click(self):
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)
        separated = make_pinch_hand(pinch_ratio=0.50)
        ctrl.process_landmarks(separated, timestamp=1.0)
        ctrl.process_landmarks(separated, timestamp=1.16)
        assert ctrl.right_click_ctrl.armed is True
        mock_pag.moveTo.reset_mock()

        clicks = 0
        for frame in range(30):
            mode, clicked, scroll_value, scrolled = ctrl.process_landmarks(
                make_pinch_hand(pinch_ratio=0.17), timestamp=1.20 + frame / 30,
            )
            assert mode == "CURSOR"
            assert scroll_value == 0
            assert scrolled is False
            clicks += int(clicked)

        assert clicks == 1
        mock_pag.click.assert_called_once_with(button="right", _pause=False)
        mock_pag.moveTo.assert_not_called()
        mock_pag.scroll.assert_not_called()


class TestRightClickIntegration:
    """Verifies priority coordination, cursor freeze, left-click / scroll suppression,
    swipe suppression, and fail-safe propagation."""

    def test_cursor_movement_stops_during_approach_and_confirmation(self):
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        # Arm with separation
        hand_sep = make_pinch_hand(pinch_ratio=0.50)
        ctrl.process_landmarks(hand_sep, timestamp=1.0)
        ctrl.process_landmarks(hand_sep, timestamp=1.16)
        assert mock_pag.moveTo.call_count >= 1
        last_move_count = mock_pag.moveTo.call_count

        # Moving in approach range freezes cursor
        hand_app = make_pinch_hand(pinch_ratio=0.26)
        hand_app[8] = FakeLandmark(0.40, 0.40)
        ctrl.process_landmarks(hand_app, timestamp=1.17)
        ctrl.process_landmarks(hand_app, timestamp=1.18)
        ctrl.process_landmarks(hand_app, timestamp=1.19)
        last_move_count = mock_pag.moveTo.call_count
        mode, clicked, _, _ = ctrl.process_landmarks(hand_app, timestamp=1.20)
        assert mode == "CURSOR"
        assert clicked is False
        assert mock_pag.moveTo.call_count == last_move_count

        # Contact also freezes cursor
        hand_contact = make_pinch_hand(pinch_ratio=0.15)
        hand_contact[8] = FakeLandmark(0.45, 0.45)
        ctrl.process_landmarks(hand_contact, timestamp=1.25)
        assert mock_pag.moveTo.call_count == last_move_count

    def test_forming_pinch_triggers_neither_left_click_nor_scroll(self):
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        pointing_lowered = make_hand(index_extended=True, middle_folded=True, thumb_raised=False)
        ctrl.process_landmarks(pointing_lowered, timestamp=1.0)
        ctrl.process_landmarks(pointing_lowered, timestamp=1.16)
        assert ctrl.click_armed is True

        pinch_app = make_pinch_hand(pinch_ratio=0.26)
        ctrl.process_landmarks(pinch_app, timestamp=1.18)
        mode, clicked, scroll_val, scrolled = ctrl.process_landmarks(pinch_app, timestamp=1.20)
        assert mode == "CURSOR"
        assert clicked is False
        assert scrolled is False
        assert scroll_val == 0
        mock_pag.click.assert_not_called()
        mock_pag.scroll.assert_not_called()

        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.11), timestamp=1.23)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.11), timestamp=1.24)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.11), timestamp=1.25)
        mode, clicked, scroll_val, scrolled = ctrl.process_landmarks(
            make_pinch_hand(pinch_ratio=0.11), timestamp=1.46
        )
        assert clicked is True
        assert scrolled is False
        mock_pag.click.assert_called_once_with(button="right", _pause=False)
        mock_pag.scroll.assert_not_called()

    def test_swipes_suppressed_during_right_click_recognition(self):
        detector = PalmSwipeDetector()
        mock_action = MagicMock()

        pinch = make_pinch_hand(pinch_ratio=0.26)
        for i in range(8):
            moved = translate_hand(pinch, i * 0.03, 0.0)
            state = detector.update(moved, active_mouse_mode="CURSOR")
            assert state == GestureState.CURSOR_CONTROL
            assert len(detector.palm_history) == 0

        mock_action.assert_not_called()

    def test_left_click_scroll_and_cursor_recover_after_right_click(self):
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        # Arm and execute right click
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.11), timestamp=1.18)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.11), timestamp=1.19)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.11), timestamp=1.20)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.11), timestamp=1.41)
        assert mock_pag.click.call_count == 1
        mock_pag.click.assert_called_with(button="right", _pause=False)
        assert ctrl.click_armed is False

        # Release and rearm (150ms continuous separation)
        hand_sep = make_pinch_hand(pinch_ratio=0.50)
        ctrl.process_landmarks(hand_sep, timestamp=1.43)
        ctrl.process_landmarks(hand_sep, timestamp=1.45)
        ctrl.process_landmarks(hand_sep, timestamp=1.62)
        assert ctrl.right_click_ctrl.state == "armed"

        # Thumb lowered arms left click
        ctrl.process_landmarks(make_hand(thumb_raised=False), timestamp=1.65)
        assert ctrl.click_armed is True

        # Thumb raised fires left click
        mock_pag.click.reset_mock()
        ctrl.process_landmarks(make_hand(thumb_raised=True), timestamp=1.70)
        mock_pag.click.assert_called_once_with(_pause=False)

        # Scrolling still activates smoothly
        scroll_hand = make_hand(index_extended=True, middle_folded=False)
        mode, _, _, _ = ctrl.process_landmarks(scroll_hand, timestamp=1.75)
        assert mode == "SCROLL"

    def test_failsafe_exception_propagates_from_right_click(self):
        mock_pag = MagicMock()
        class FakeFailSafe(Exception):
            pass
        mock_pag.FailSafeException = FakeFailSafe
        mock_pag.click.side_effect = FakeFailSafe("Corner fail-safe triggered")

        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.11), timestamp=1.18)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.11), timestamp=1.19)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.11), timestamp=1.20)

        with pytest.raises(FakeFailSafe):
            ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.11), timestamp=1.41)

    @patch("main.cv2.putText")
    @patch("main.cv2.rectangle")
    def test_hud_displays_right_click_action_and_approach_state(self, mock_rect, mock_put_text):
        import numpy as np

        frame = np.zeros((700, 1000, 3), dtype=np.uint8)
        detector = PalmSwipeDetector()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=MagicMock())

        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.25), timestamp=1.20)
        draw_hud(frame, detector, GestureState.CURSOR_CONTROL, ctrl, fps=30.0, now_mono=1.20)
        approach_calls = [
            call for call in mock_put_text.call_args_list
            if "Right Click: Approach" in str(call) or "APPROACH" in str(call) or "Pinch:" in str(call)
        ]
        assert len(approach_calls) > 0

        ctrl.last_action = "Right Click"
        ctrl.last_action_time = 10.0
        mock_put_text.reset_mock()
        draw_hud(frame, detector, GestureState.CURSOR_CONTROL, ctrl, fps=30.0, now_mono=10.2)
        click_calls = [
            call for call in mock_put_text.call_args_list
            if "Action: Right Click" in str(call)
        ]
        assert len(click_calls) > 0


class TestIncreasedScrollSpeedAndCap:
    """Verifies that scrolling runs at 10x the original baseline rate, frame caps allow full throughput,
    and neutral stops / recovery remain rock solid."""

    def test_scroll_rate_emits_approximately_10x_previous_total_steps(self):
        """With identical finger positions and elapsed time, scrolling emits ~10x total wheel steps."""
        mock_pag_base = MagicMock()
        mock_pag_fast = MagicMock()

        sc_base = ScrollController(
            min_steps_per_second=4.0,
            max_steps_per_second=22.0,
            max_steps_per_frame=3,
            pyautogui_module=mock_pag_base,
        )
        sc_fast = ScrollController(pyautogui_module=mock_pag_fast)

        hand = make_hand(index_extended=True, middle_folded=False)
        hand[8] = FakeLandmark(0.45, 0.50)
        hand[12] = FakeLandmark(0.50, 0.50)
        sc_base.process(hand, timestamp=1.0)
        sc_fast.process(hand, timestamp=1.0)

        hand[8] = FakeLandmark(0.45, 0.25)
        hand[12] = FakeLandmark(0.50, 0.25)

        total_base = 0
        total_fast = 0
        t = 1.033
        for _ in range(30):
            sv_b, s_b = sc_base.process(hand, timestamp=t)
            sv_f, s_f = sc_fast.process(hand, timestamp=t)
            if s_b:
                total_base += abs(sv_b)
            if s_f:
                total_fast += abs(sv_f)
            t += 0.0333

        ratio = total_fast / max(1, total_base)
        assert 9.0 <= ratio <= 11.0

    def test_per_frame_cap_does_not_restrict_maximum_rate_at_normal_fps(self):
        """Verify that at 30 FPS, the per-frame cap allows full 220 steps/s throughput without clipping."""
        mock_pag = MagicMock()
        sc = ScrollController(pyautogui_module=mock_pag)

        hand = make_hand(index_extended=True, middle_folded=False)
        hand[8] = FakeLandmark(0.45, 0.50)
        hand[12] = FakeLandmark(0.50, 0.50)
        sc.process(hand, timestamp=1.0)

        hand[8] = FakeLandmark(0.45, 0.25)
        hand[12] = FakeLandmark(0.50, 0.25)

        total_steps = 0
        frames_above_3 = 0
        t = 1.0333
        for _ in range(30):
            sv, scrolled = sc.process(hand, timestamp=t)
            if scrolled:
                steps = abs(sv)
                total_steps += steps
                if steps > 3:
                    frames_above_3 += 1
            t += 0.0333

        assert frames_above_3 > 20
        assert total_steps >= 190

    def test_neutral_stops_immediately_at_10x_speed(self):
        """Returning fingers to neutral stops scrolling immediately."""
        mock_pag = MagicMock()
        sc = ScrollController(pyautogui_module=mock_pag)

        hand = make_hand(index_extended=True, middle_folded=False)
        hand[8] = FakeLandmark(0.45, 0.50)
        hand[12] = FakeLandmark(0.50, 0.50)
        sc.process(hand, timestamp=1.0)

        hand[8] = FakeLandmark(0.45, 0.25)
        hand[12] = FakeLandmark(0.50, 0.25)
        sc.process(hand, timestamp=1.033)
        sc.process(hand, timestamp=1.066)

        hand[8] = FakeLandmark(0.45, 0.50)
        hand[12] = FakeLandmark(0.50, 0.50)
        scroll_val, scrolled = sc.process(hand, timestamp=1.100)
        assert scrolled is False
        assert scroll_val == 0
        assert sc.direction == 0
        assert sc.current_rate == 0.0

    def test_tracking_recovery_at_10x_speed_clamps_catchup_burst(self):
        """Tracking gaps do not produce runaway bursts at 10x scroll rate."""
        mock_pag = MagicMock()
        sc = ScrollController(pyautogui_module=mock_pag)

        hand = make_hand(index_extended=True, middle_folded=False)
        hand[8] = FakeLandmark(0.45, 0.50)
        hand[12] = FakeLandmark(0.50, 0.50)
        sc.process(hand, timestamp=1.0)

        hand[8] = FakeLandmark(0.45, 0.25)
        hand[12] = FakeLandmark(0.50, 0.25)
        sc.process(hand, timestamp=1.033)

        sc.reanchor(hand, timestamp=1.533)
        scroll_val, scrolled = sc.process(hand, timestamp=1.566)
        if scrolled:
            assert abs(scroll_val) <= DEFAULT_MAX_SCROLL_STEPS_PER_FRAME

    def test_multiplier_exposed_and_applied_only_to_rate(self):
        """Verify configurable constant SCROLL_SPEED_MULTIPLIER applies to rate, not clicks per step."""
        assert SCROLL_SPEED_MULTIPLIER == 10.0
        assert DEFAULT_MIN_SCROLL_STEPS_PER_SECOND == 40.0
        assert DEFAULT_MAX_SCROLL_STEPS_PER_SECOND == 220.0
        assert DEFAULT_MAX_SCROLL_STEPS_PER_FRAME == 30

        sc = ScrollController()
        assert sc.scroll_clicks_per_step == 1
        assert sc.min_steps_per_second == 40.0
        assert sc.max_steps_per_second == 220.0
        assert sc.max_steps_per_frame == 30


class TestGentlePinchSensitivityAndInteractions:
    """Verifies deliberate gentle contact sensitivity, rejection of incidental proximity,
    ordering of thresholds, arming/confirmation hysteresis, and HUD diagnostics."""

    def test_gentle_contact_at_ratio_0_11_triggers_right_click(self):
        """Pinch ratio of 0.11 is deliberate contact and clicks after confirmation."""
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        # Arm with separation
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)
        assert ctrl.armed is True

        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.17)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.18)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.20)
        assert ctrl.state == "confirming"

        clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.41)
        assert clicked is True
        assert freeze is True
        mock_pag.click.assert_called_once_with(button="right", _pause=False)

    def test_clearly_separated_fingertips_at_ratio_0_19_never_right_click(self):
        """Visibly separated fingertips (ratio=0.19, previously misclassified at 0.21) never click."""
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)
        assert ctrl.armed is True

        # Holding separated fingertips at ratio 0.19 stays in approach range, never confirming
        for t in (1.20, 1.30, 1.40, 1.50, 1.60, 1.70):
            clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.19), timestamp=t)
            assert clicked is False
            assert freeze is True
            assert ctrl.state == "approaching"
            assert ctrl.contact_start_time is None

        mock_pag.click.assert_not_called()

    def test_accidental_proximity_at_ratio_0_24_is_rejected_and_does_not_click(self):
        """Incidental proximity (ratio=0.24, previously false-triggered at 0.25) is rejected."""
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        # Arm with separation
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)
        assert ctrl.armed is True

        # At ratio 0.24, hand is in approach range (0.18 < 0.24 <= 0.28), not contact
        for t in (1.20, 1.30, 1.40, 1.50, 1.70):
            clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.24), timestamp=t)
            assert clicked is False
            assert ctrl.state != "confirming"

        mock_pag.click.assert_not_called()

    def test_clearly_separated_fingertips_at_ratio_0_38_do_not_click(self):
        """Normal pointing / resting lowered thumb (ratio=0.38) never clicks."""
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)
        for t in (1.20, 1.30, 1.40, 1.50):
            clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.38), timestamp=t)
            assert clicked is False
            assert freeze is False
            assert ctrl.state == "armed"

        mock_pag.click.assert_not_called()

    def test_approach_range_ratio_0_26_freezes_cursor_without_clicking(self):
        """Pinch ratio in approach range (0.26) freezes cursor but never fires click."""
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)
        ctrl.process(make_pinch_hand(pinch_ratio=0.26), timestamp=1.17)
        ctrl.process(make_pinch_hand(pinch_ratio=0.26), timestamp=1.18)
        ctrl.process(make_pinch_hand(pinch_ratio=0.26), timestamp=1.19)
        for t in (1.20, 1.30, 1.40, 1.50):
            clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.26), timestamp=t)
            assert clicked is False
            assert freeze is True
            assert ctrl.state == "approaching"

        mock_pag.click.assert_not_called()

    def test_threshold_jitter_around_contact_cancels_confirmation(self):
        """Sustained movement into approach (0.26) resets contact confirmation."""
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)

        # Contact starts confirmation
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.17)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.18)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.20)
        assert ctrl.state == "confirming"

        # Movement into approach range breaks contact confirmation
        ctrl.process(make_pinch_hand(pinch_ratio=0.26), timestamp=1.27)
        ctrl.process(make_pinch_hand(pinch_ratio=0.26), timestamp=1.28)
        assert ctrl.state == "approaching"
        assert ctrl.contact_start_time is None

        # Re-entering contact restarts the confirmation clock
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.30)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.31)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.32)
        assert ctrl.state == "confirming"

        # Not yet 200ms of continuous contact
        clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.42)
        assert clicked is False
        mock_pag.click.assert_not_called()

        # Completes 200ms confirmation
        clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.53)
        assert clicked is True
        mock_pag.click.assert_called_once()

    def test_sustained_gentle_contact_does_not_repeat_and_rearms_after_150ms_separation(self):
        """Holding gentle contact (0.11) does not repeat clicks; requires 150ms separation beyond 0.35 to rearm."""
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.17)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.18)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.20)
        clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.41)
        assert clicked is True
        assert mock_pag.click.call_count == 1

        for t in range(10):
            clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.45 + t * 0.033)
            assert clicked is False
            assert freeze is True
            assert ctrl.state == "clicked"

        assert mock_pag.click.call_count == 1

        # Separating for >= 150ms rearms
        ctrl.process(make_pinch_hand(pinch_ratio=0.40), timestamp=1.78)
        ctrl.process(make_pinch_hand(pinch_ratio=0.40), timestamp=1.80)
        assert ctrl.armed is False
        ctrl.process(make_pinch_hand(pinch_ratio=0.40), timestamp=1.96)
        assert ctrl.state == "armed"
        assert ctrl.armed is True

        # Second deliberate click
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.97)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.98)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=2.00)
        clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=2.21)
        assert clicked is True
        assert mock_pag.click.call_count == 2

    def test_right_click_never_triggers_left_click_scrolling_or_swipes(self):
        """Right-click sequence triggers neither left-click, scrolling, nor swipe actions."""
        mock_pag = MagicMock()
        detector = PalmSwipeDetector()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        pointing = make_hand(thumb_raised=False)
        ctrl.process_landmarks(pointing, timestamp=0.90)
        ctrl.process_landmarks(pointing, timestamp=1.06)
        assert ctrl.right_click_ctrl.armed is True

        pinch_app = make_pinch_hand(pinch_ratio=0.26)
        ctrl.process_landmarks(pinch_app, timestamp=1.09)
        mode, clicked, scroll_val, scrolled = ctrl.process_landmarks(pinch_app, timestamp=1.10)
        assert mode == "CURSOR"
        assert clicked is False
        assert scrolled is False
        assert scroll_val == 0
        state = detector.update(pinch_app, active_mouse_mode=mode)
        assert state == GestureState.CURSOR_CONTROL
        assert len(detector.palm_history) == 0

        pinch_cont = make_pinch_hand(pinch_ratio=0.11)
        ctrl.process_landmarks(pinch_cont, timestamp=1.14)
        ctrl.process_landmarks(pinch_cont, timestamp=1.15)
        mode, clicked, scroll_val, scrolled = ctrl.process_landmarks(pinch_cont, timestamp=1.36)
        assert clicked is True
        assert scrolled is False

        assert mock_pag.click.call_count == 1
        mock_pag.click.assert_called_with(button="right", _pause=False)
        mock_pag.scroll.assert_not_called()

        state = detector.update(pinch_cont, active_mouse_mode=mode)
        assert state == GestureState.CURSOR_CONTROL
        assert len(detector.palm_history) == 0

    @patch("main.cv2.putText")
    @patch("main.cv2.rectangle")
    def test_hud_displays_normalized_pinch_distance_thresholds_state_and_progress(self, mock_rect, mock_put_text):
        """Diagnostic HUD renders pinch distance, thresholds, recognition state, and confirmation progress."""
        import numpy as np
        frame = np.zeros((700, 1000, 3), dtype=np.uint8)
        detector = PalmSwipeDetector()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=MagicMock())

        hand_norm = make_hand(thumb_raised=False)
        ctrl.process_landmarks(hand_norm, timestamp=1.0)
        ctrl.process_landmarks(hand_norm, timestamp=1.16)
        draw_hud(frame, detector, GestureState.CURSOR_CONTROL, ctrl, fps=30.0, now_mono=1.16)

        rendered_lines = [str(call) for call in mock_put_text.call_args_list]
        contact_label = f"Contact:<={DEFAULT_PINCH_CONTACT_THRESHOLD:.2f}"
        release_label = f"Rel:>={DEFAULT_PINCH_RELEASE_THRESHOLD:.2f}"
        has_pinch_dist = any(
            "Pinch:" in line and contact_label in line and release_label in line
            for line in rendered_lines
        )
        assert has_pinch_dist, f"Expected Pinch distance and thresholds in HUD, got: {rendered_lines}"
        has_state = any("RC State: ARMED" in line for line in rendered_lines)
        assert has_state, f"Expected RC State in HUD, got: {rendered_lines}"

        mock_put_text.reset_mock()
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.11), timestamp=1.17)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.11), timestamp=1.18)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.11), timestamp=1.20)
        draw_hud(frame, detector, GestureState.CURSOR_CONTROL, ctrl, fps=30.0, now_mono=1.20)
        rendered_lines2 = [str(call) for call in mock_put_text.call_args_list]
        has_confirming = any("RC State: CONFIRMING" in line for line in rendered_lines2)
        assert has_confirming, f"Expected CONFIRMING state in HUD, got: {rendered_lines2}"

    def test_touching_fingertips_in_arbitrary_pose_does_not_right_click(self):
        """Touching fingertips when index is folded or ring is extended never triggers right click."""
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        # Arm with separation in valid pose
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)
        assert ctrl.armed is True

        # Ring finger extended (invalid right click pose) even with close fingertips
        invalid_hand = make_pinch_hand(pinch_ratio=0.11)
        invalid_hand[16] = FakeLandmark(0.55, 0.20)  # extend ring finger
        for t in (1.20, 1.30, 1.40, 1.50):
            clicked, freeze = ctrl.process(invalid_hand, timestamp=t)
            assert clicked is False
            assert freeze is False
            assert ctrl.contact_start_time is None

        mock_pag.click.assert_not_called()

    def test_separation_less_than_150ms_does_not_arm(self):
        """Separation held for less than 150ms (e.g. 100ms) does not arm the gesture."""
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        # Disarmed initially
        assert ctrl.armed is False
        assert ctrl.state == "disarmed"

        # Separate for only 100ms
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.10)
        assert ctrl.armed is False
        assert ctrl.state == "disarmed"

        # Immediate contact does not trigger confirmation because controller is not armed
        clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.15)
        assert clicked is False
        assert ctrl.state == "disarmed"
        mock_pag.click.assert_not_called()

    def test_contact_less_than_200ms_does_not_click(self):
        """Contact held for 190ms (< 200ms) before releasing does not dispatch a click."""
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        # Arm
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)
        assert ctrl.armed is True

        # Filtered contact starts at 1.19; 190ms until 1.38 is below confirmation.
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.18)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.19)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.20)
        clicked, _ = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.38)
        assert clicked is False
        assert ctrl.state == "confirming"

        # Release at 1.40 breaks contact
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.40)
        assert ctrl.contact_start_time is None
        mock_pag.click.assert_not_called()


class TestRightClickFixVerification:
    """Comprehensive regression and verification test suite for right-click recognition:
    - Intermediate middle-finger bending arming through MouseController.
    - Gentle deliberate contact with realistic distance noise.
    - Brief contact rejection.
    - Accidental proximity rejection and cursor freeze.
    - Already-closed entry protection.
    - Sustained contact (no repeat).
    - Release and confirmed separation before rearming.
    - Tracking loss and missing landmark safety (no stale filter state).
    - rc_freeze honored in MouseController without unintended input.
    - Dispatch failure reporting and fail-safe propagation.
    - Diagnostics (rejection reason, progress, raw/filtered distance).
    """

    def test_intermediate_middle_finger_bending_arms_through_mouse_controller(self):
        """Hand in valid right-click pose with intermediate middle finger bending preserves arming."""
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        # Create hand with intermediate middle finger:
        # PIP=(0.50, 0.45), TIP=(0.50, 0.45)
        # is_right_click_pose is True, but is_pointing_pose is False and is_scroll_pose is False
        hand = make_pinch_hand(pinch_ratio=0.50)
        hand[10] = FakeLandmark(0.50, 0.45)
        hand[12] = FakeLandmark(0.50, 0.45)

        assert is_right_click_pose(hand) is True
        assert is_pointing_pose(hand) is False
        assert is_scroll_pose(hand) is False

        # Frame 1: starts separation confirmation
        mode1, clicked1, _, _ = ctrl.process_landmarks(hand, timestamp=1.0)
        assert mode1 == "CURSOR"
        assert clicked1 is False
        assert ctrl.is_active is True
        assert ctrl.right_click_ctrl.armed is False
        assert ctrl.right_click_ctrl.separation_start_time == 1.0

        # Frame 2: after 160ms (>= 150ms), arms successfully through MouseController!
        mode2, clicked2, _, _ = ctrl.process_landmarks(hand, timestamp=1.16)
        assert mode2 == "CURSOR"
        assert clicked2 is False
        assert ctrl.right_click_ctrl.armed is True
        assert ctrl.right_click_ctrl.state == "armed"
        # Cursor should not have moved because hand is in intermediate pose, not pointing pose
        assert mock_pag.moveTo.call_count == 0

    def test_gentle_deliberate_contact_with_realistic_distance_noise_clicks(self):
        """Realistic landmark noise (+/- 0.015) during contact does not break confirmation with 0.60 filter."""
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        # Arm with separation
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)
        assert ctrl.armed is True

        # Transition into gentle contact (base ~ 0.11)
        ctrl.process(make_pinch_hand(pinch_ratio=0.26), timestamp=1.17)
        ctrl.process(make_pinch_hand(pinch_ratio=0.15), timestamp=1.18)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.19)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.20)
        assert ctrl.state == "confirming"

        # Realistic noisy distances around 0.11 (fluctuating between 0.10 and 0.13, well <= 0.14)
        noisy_ratios = [0.105, 0.125, 0.110, 0.130, 0.115]
        t = 1.23
        for ratio in noisy_ratios:
            clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=ratio), timestamp=t)
            assert clicked is False
            assert freeze is True
            assert ctrl.state == "confirming"
            t += 0.03

        # At t=1.42 (> 200ms of continuous contact confirmation from 1.20), click fires
        clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.42)
        assert clicked is True
        assert freeze is True
        mock_pag.click.assert_called_once_with(button="right", _pause=False)

    def test_stale_smoothed_contact_cannot_complete_when_raw_fingertips_separate(self):
        """When raw fingertips separate, old smoothed value cannot complete confirmation or fire click."""
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)
        assert ctrl.armed is True

        # Contact begins at 1.20
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.18)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.19)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.20)
        assert ctrl.state == "confirming"

        # At 1.35 (150ms in), fingers suddenly separate to approach range (raw=0.22)
        # Smoothed ratio: 0.60 * 0.22 + 0.40 * 0.11 = 0.132 + 0.044 = 0.176 (> 0.14)
        # Even if filter lagged below 0.14, raw_ratio is 0.22 > 0.14
        clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.22), timestamp=1.35)
        assert clicked is False
        assert ctrl.state == "approaching"
        assert ctrl.contact_start_time is None

        # At 1.42 (past original 200ms mark), no click is ever dispatched
        clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.22), timestamp=1.42)
        assert clicked is False
        mock_pag.click.assert_not_called()

    def test_brief_contact_rejected_without_click(self):
        """Brief contact (< 200ms) before returning to approach range does not fire click."""
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)

        # Brief contact for ~40ms
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.18)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.19)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.20)
        assert ctrl.state == "confirming"

        # Release to approach at 1.25
        ctrl.process(make_pinch_hand(pinch_ratio=0.27), timestamp=1.25)
        ctrl.process(make_pinch_hand(pinch_ratio=0.27), timestamp=1.27)
        assert ctrl.state == "approaching"
        assert ctrl.contact_start_time is None
        mock_pag.click.assert_not_called()

    def test_accidental_proximity_in_approach_range_freezes_without_click(self):
        """Holding fingertips in approach range (~0.25) freezes cursor but never dispatches click."""
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)
        # Settle into approach range
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.25), timestamp=1.17)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.25), timestamp=1.18)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.25), timestamp=1.19)
        initial_moves = mock_pag.moveTo.call_count

        # Sustained approach range (ratio=0.25, contact is <=0.14)
        for i in range(15):
            t = 1.20 + i * 0.033
            hand = make_pinch_hand(pinch_ratio=0.25)
            hand[8] = FakeLandmark(0.35 + i * 0.01, 0.40)
            mode, clicked, scroll_val, scrolled = ctrl.process_landmarks(hand, timestamp=t)
            assert mode == "CURSOR"
            assert clicked is False
            assert scrolled is False
            assert mock_pag.moveTo.call_count == initial_moves

        mock_pag.click.assert_not_called()

    def test_already_closed_entry_produces_no_click_until_confirmed_separation(self):
        """Entering tracking with already-closed pinch produces no click; requires 150ms separation to arm."""
        mock_pag = MagicMock()
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        # Start with closed pinch
        closed_hand = make_pinch_hand(pinch_ratio=0.11)
        for i in range(10):
            t = 1.0 + i * 0.033
            mode, clicked, _, _ = ctrl.process_landmarks(closed_hand, timestamp=t)
            assert mode == "CURSOR"
            assert clicked is False
            assert ctrl.right_click_ctrl.armed is False

        mock_pag.click.assert_not_called()

        # Separate beyond release threshold (0.35)
        sep_hand = make_pinch_hand(pinch_ratio=0.50)
        ctrl.process_landmarks(sep_hand, timestamp=1.38)
        ctrl.process_landmarks(sep_hand, timestamp=1.40)
        assert ctrl.right_click_ctrl.armed is False
        ctrl.process_landmarks(sep_hand, timestamp=1.56)
        assert ctrl.right_click_ctrl.armed is True

        # Now deliberate contact confirms and clicks
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.11), timestamp=1.58)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.11), timestamp=1.59)
        ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.11), timestamp=1.60)
        mode, clicked, _, _ = ctrl.process_landmarks(make_pinch_hand(pinch_ratio=0.11), timestamp=1.81)
        assert clicked is True
        mock_pag.click.assert_called_once_with(button="right", _pause=False)

    def test_tracking_loss_clears_filter_and_prevents_stale_clicks(self):
        """Tracking loss resets filter ratio so stale contact is never carried over."""
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        # Arm with separation
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)
        assert ctrl.armed is True

        # Contact begins
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.18)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.19)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.20)
        assert ctrl.state == "confirming"
        assert ctrl.pinch_ratio <= 0.14

        # Tracking lost: landmarks become None
        ctrl.process(None, timestamp=1.25)
        assert ctrl.contact_start_time is None
        assert ctrl.pinch_ratio == 999.0
        assert ctrl.raw_pinch_ratio == 999.0

        # When hand reappears separated, filter initializes cleanly without blending with old 0.11
        ctrl.process(make_pinch_hand(pinch_ratio=0.45), timestamp=1.30)
        assert abs(ctrl.pinch_ratio - 0.45) < 1e-4
        assert abs(ctrl.raw_pinch_ratio - 0.45) < 1e-4
        assert ctrl.state == "armed"
        mock_pag.click.assert_not_called()

    def test_missing_landmarks_and_zero_scale_handled_safely(self):
        """Missing landmarks or near-zero palm scale never generate clicks and set pose rejection reason."""
        mock_pag = MagicMock()
        ctrl = RightClickController(pyautogui_module=mock_pag)

        # Empty / short list
        clicked, freeze = ctrl.process([], timestamp=1.0)
        assert clicked is False
        assert freeze is False
        assert ctrl.is_pose_valid is False
        assert ctrl.pose_rejection_reason == "Missing hand landmarks"

        # None landmarks
        clicked, freeze = ctrl.process(None, timestamp=1.1)
        assert clicked is False
        assert freeze is False
        assert ctrl.is_pose_valid is False
        assert ctrl.pose_rejection_reason == "Missing hand landmarks"

        # Degenerate hand scale
        hand = [FakeLandmark(0.5, 0.5)] * 21
        clicked, freeze = ctrl.process(hand, timestamp=1.2)
        assert clicked is False
        assert freeze is False
        assert ctrl.is_pose_valid is False
        assert ctrl.pose_rejection_reason == "Hand scale too small"

    def test_rc_freeze_prevents_cursor_output_and_suppresses_competing_input(self):
        """When rc_freeze is True, cursor does not move, left click cannot occur, and scroll cannot occur."""
        mock_pag = MagicMock()
        mock_pag.size.return_value = (1920, 1080)
        ctrl = MouseController(screen_size=(1920, 1080), pyautogui_module=mock_pag)

        # Hand in approach range (0.26)
        hand = make_pinch_hand(pinch_ratio=0.26)

        ctrl.process_landmarks(hand, timestamp=1.0)
        mode, clicked, scroll_val, scrolled = ctrl.process_landmarks(hand, timestamp=1.03)

        assert mode == "CURSOR"
        assert clicked is False
        assert scroll_val == 0
        assert scrolled is False
        # Neither moveTo nor click nor scroll were dispatched
        assert mock_pag.moveTo.call_count == 0
        assert mock_pag.click.call_count == 0
        assert mock_pag.scroll.call_count == 0

    def test_dispatch_failure_reporting(self):
        """Ordinary dispatch exception logs error, returns False, and does not claim click or set clicked state."""
        mock_pag = MagicMock()
        class FakeFailSafe(Exception): pass
        mock_pag.FailSafeException = FakeFailSafe
        mock_pag.click.side_effect = RuntimeError("Desktop input subsystem failed")

        ctrl = RightClickController(pyautogui_module=mock_pag)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.0)
        ctrl.process(make_pinch_hand(pinch_ratio=0.50), timestamp=1.16)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.18)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.19)
        ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.20)

        with patch("logging.Logger.warning") as mock_log:
            clicked, freeze = ctrl.process(make_pinch_hand(pinch_ratio=0.11), timestamp=1.41)
            assert clicked is False
            assert freeze is True
            assert ctrl.state != "clicked"
            mock_log.assert_called_once()


    def test_pose_rejection_reasons(self):
        """get_right_click_pose_status returns specific informative reasons for rejection."""
        # Valid pose
        hand = make_pinch_hand(pinch_ratio=0.50)
        valid, reason = get_right_click_pose_status(hand)
        assert valid is True
        assert reason == "Valid right-click pose"

        # Missing landmarks
        valid, reason = get_right_click_pose_status(None)
        assert valid is False
        assert "Missing" in reason

        # Index folded
        hand_idx_folded = make_pinch_hand(pinch_ratio=0.50)
        hand_idx_folded[8] = FakeLandmark(0.50, 0.75) # folded below PIP
        valid, reason = get_right_click_pose_status(hand_idx_folded)
        assert valid is False
        assert "Index" in reason

        # Ring extended
        hand_ring_ext = make_pinch_hand(pinch_ratio=0.50)
        hand_ring_ext[16] = FakeLandmark(0.55, 0.20) # extended
        valid, reason = get_right_click_pose_status(hand_ring_ext)
        assert valid is False
        assert "Ring" in reason

        # Pinky extended
        hand_pky_ext = make_pinch_hand(pinch_ratio=0.50)
        hand_pky_ext[20] = FakeLandmark(0.60, 0.20) # extended
        valid, reason = get_right_click_pose_status(hand_pky_ext)
        assert valid is False
        assert "Pinky" in reason



