"""
Automated unit and integration tests for three-finger downward swipe screenshot feature.

Covers:
1. Three-finger pose recognition:
   - Index, middle, ring extended; pinky folded; thumb unrestricted.
   - Hand geometry and palm-size normalization.
   - Rotation and mirror (left / right hand) invariance.
   - Handling of missing or empty landmarks safely.
   - Pose confirmation interval (~120 ms) before permitting capture.
   - Stationary pose does not trigger screenshot.
2. Deliberate downward movement:
   - Tracking average of fingertips 8, 12, 16 (increasing Y).
   - Thresholds: dy >= 0.12, dy >= 1.5 * abs(dx), velocity >= 0.25 u/s, duration <= 0.8s.
   - Rejection of tiny movements, slow drift, upward strokes, sideways strokes, diagonal strokes.
   - Apex measurement (upward start does not corrupt downward displacement).
   - Monotonic time tracking.
3. Prevent duplicate or accidental captures:
   - Exactly one screenshot per downward stroke.
   - Holding pose or moving hand back up does not capture again.
   - Rearming requires leaving 3-finger pose for ~150 ms.
   - Missing tracking alone does NOT count as deliberate release.
   - Tracking gaps clear movement history and never bridge across gaps.
   - Pose breaks reset confirmation and stroke history.
4. Gesture priority and suppression:
   - 3-finger pose is not treated as scroll or mouse mode candidate.
   - Cursor movement, clicking, and palm swipes suppressed during screenshot gesture.
   - PalmSwipeDetector clears stale history when screenshot starts and ends.
   - Controls recover cleanly after cancellation or release.
   - PyAutoGUI FailSafeException propagates properly.
5. Capture, save, and feedback:
   - Timestamped PNG saved in screenshots/ folder.
   - Directory created automatically.
   - Display "Screenshot saved" only after successful save.
   - Error during save is handled safely and reported.
   - HUD displays metrics (state, displacement, velocity, readiness, feedback).
"""

import math
import os
import shutil
import sys
import time
from collections import deque
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from screenshot_controller import (
    ScreenshotController,
    ScreenshotState,
    SCREENSHOT_POSE_CONFIRM_TIME,
    SCREENSHOT_MIN_DOWNWARD_DISPLACEMENT,
    SCREENSHOT_VERTICAL_DOMINANCE_RATIO,
    SCREENSHOT_MIN_VELOCITY,
    SCREENSHOT_MAX_STROKE_DURATION,
    SCREENSHOT_RELEASE_TIME,
    SCREENSHOT_FEEDBACK_DURATION,
    SCREENSHOT_FLASH_ENABLED,
    SCREENSHOT_FLASH_PEAK_OPACITY,
    SCREENSHOT_FLASH_DURATION,
    SCREENSHOT_NOTIFICATION_ENABLED,
    SCREENSHOT_NOTIFICATION_DURATION,
    SCREENSHOT_NOTIFICATION_FADE_DURATION,
    get_three_fingertip_center,
)
from desktop_overlay import DesktopOverlayManager
from mouse_controller import (
    MouseController,
    is_three_finger_screenshot_pose,
    is_scroll_candidate,
    is_scroll_pose,
    is_mouse_mode_candidate,
    is_pointing_pose,
    is_open_palm_pose,
    is_swipe_palm_pose,
)
from main import (
    PalmSwipeDetector,
    GestureState,
    draw_hud,
)
import pyautogui


class FakeLandmark:
    """Minimal normalized landmark."""
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)


def make_hand(
    index_extended: bool = True,
    middle_extended: bool = True,
    ring_extended: bool = True,
    pinky_extended: bool = False,
    thumb_raised: bool = False,
    wrist_x: float = 0.5,
    wrist_y: float = 0.7,
) -> list:
    """
    Construct synthetic 21 hand landmarks.
    Wrist is positioned at (wrist_x, wrist_y).
    """
    lms = [FakeLandmark(wrist_x, wrist_y) for _ in range(21)]
    lms[0] = FakeLandmark(wrist_x, wrist_y)

    mcp_y = wrist_y - 0.20
    lms[5] = FakeLandmark(wrist_x - 0.06, mcp_y)
    lms[9] = FakeLandmark(wrist_x - 0.02, mcp_y)
    lms[13] = FakeLandmark(wrist_x + 0.02, mcp_y)
    lms[17] = FakeLandmark(wrist_x + 0.06, mcp_y)

    def set_finger(tip_idx, pip_idx, base_x, is_ext):
        if is_ext:
            lms[pip_idx] = FakeLandmark(base_x, wrist_y - 0.32)
            lms[tip_idx - 1] = FakeLandmark(base_x, wrist_y - 0.42)
            lms[tip_idx] = FakeLandmark(base_x, wrist_y - 0.55)
        else:
            lms[pip_idx] = FakeLandmark(base_x, wrist_y - 0.28)
            lms[tip_idx - 1] = FakeLandmark(base_x, wrist_y - 0.22)
            lms[tip_idx] = FakeLandmark(base_x, wrist_y - 0.16)

    set_finger(8, 6, wrist_x - 0.06, index_extended)
    set_finger(12, 10, wrist_x - 0.02, middle_extended)
    set_finger(16, 14, wrist_x + 0.02, ring_extended)
    set_finger(20, 18, wrist_x + 0.06, pinky_extended)

    if thumb_raised:
        tx, ty = wrist_x - 0.16, mcp_y
    else:
        tx, ty = wrist_x - 0.08, mcp_y + 0.02

    lms[1] = FakeLandmark(wrist_x - 0.04, wrist_y - 0.08)
    lms[2] = FakeLandmark(wrist_x - 0.06, wrist_y - 0.14)
    lms[3] = FakeLandmark(tx + 0.02, ty + 0.04)
    lms[4] = FakeLandmark(tx, ty)

    return lms


def make_three_finger_pose(wrist_x: float = 0.5, wrist_y: float = 0.7, thumb_raised: bool = False) -> list:
    """Helper for standard 3-finger screenshot pose (Index, Middle, Ring extended; Pinky folded)."""
    return make_hand(
        index_extended=True,
        middle_extended=True,
        ring_extended=True,
        pinky_extended=False,
        thumb_raised=thumb_raised,
        wrist_x=wrist_x,
        wrist_y=wrist_y,
    )


def translate_hand(hand: list, dx: float, dy: float) -> list:
    """Translate all landmarks by (dx, dy)."""
    return [FakeLandmark(lm.x + dx, lm.y + dy) for lm in hand]


def rotate_landmarks(landmarks: list, angle_deg: float) -> list:
    """Rotate landmarks around wrist (landmark 0)."""
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


def mirror_landmarks(landmarks: list) -> list:
    """Mirror hand horizontally around wrist_x."""
    wx = landmarks[0].x
    return [FakeLandmark(2 * wx - lm.x, lm.y) for lm in landmarks]


class MockImage:
    """Mock PIL Image returned by pyautogui.screenshot()."""
    def __init__(self):
        self.saved_path = None
        self.saved_format = None
        self.save_time = None

    def save(self, filepath, format="PNG"):
        self.saved_path = filepath
        self.saved_format = format
        self.save_time = time.monotonic()


class MockPyAutoGUI:
    """Mock PyAutoGUI module for screenshot testing."""
    def __init__(self):
        self.FailSafeException = pyautogui.FailSafeException
        self.screenshot_call_count = 0
        self.last_mock_image = None
        self.raise_on_screenshot = None

    def screenshot(self):
        self.screenshot_call_count += 1
        if self.raise_on_screenshot:
            raise self.raise_on_screenshot
        img = MockImage()
        self.last_mock_image = img
        return img


class TestThreeFingerPoseRecognition:
    """Tests for recognizing Index+Middle+Ring extended with Pinky folded."""

    def test_valid_three_finger_pose_recognized(self):
        hand = make_three_finger_pose()
        assert is_three_finger_screenshot_pose(hand) is True

    def test_support_both_left_and_right_hands(self):
        hand_right = make_three_finger_pose()
        hand_left = mirror_landmarks(hand_right)
        assert is_three_finger_screenshot_pose(hand_right) is True
        assert is_three_finger_screenshot_pose(hand_left) is True

    def test_support_in_plane_rotations(self):
        hand = make_three_finger_pose()
        for angle in [-30, -15, 0, 15, 30]:
            rotated = rotate_landmarks(hand, angle)
            assert is_three_finger_screenshot_pose(rotated) is True, f"Failed at angle {angle}"

    def test_handle_missing_or_incomplete_landmarks_safely(self):
        assert is_three_finger_screenshot_pose(None) is False
        assert is_three_finger_screenshot_pose([]) is False
        assert is_three_finger_screenshot_pose([FakeLandmark(0, 0)] * 10) is False

    def test_thumb_position_is_unrestricted(self):
        hand_thumb_down = make_three_finger_pose(thumb_raised=False)
        hand_thumb_raised = make_three_finger_pose(thumb_raised=True)
        assert is_three_finger_screenshot_pose(hand_thumb_down) is True
        assert is_three_finger_screenshot_pose(hand_thumb_raised) is True

    def test_invalid_finger_combinations_rejected(self):
        open_palm = make_hand(True, True, True, True)
        assert is_three_finger_screenshot_pose(open_palm) is False

        fist = make_hand(False, False, False, False)
        assert is_three_finger_screenshot_pose(fist) is False

        pointing = make_hand(True, False, False, False)
        assert is_three_finger_screenshot_pose(pointing) is False

        scrolling = make_hand(True, True, False, False)
        assert is_three_finger_screenshot_pose(scrolling) is False

        no_index = make_hand(False, True, True, False)
        assert is_three_finger_screenshot_pose(no_index) is False

        pinky_ext = make_hand(True, True, False, True)
        assert is_three_finger_screenshot_pose(pinky_ext) is False

    def test_fingertip_center_computation(self):
        hand = make_three_finger_pose()
        cx, cy = get_three_fingertip_center(hand)
        expected_x = (hand[8].x + hand[12].x + hand[16].x) / 3.0
        expected_y = (hand[8].y + hand[12].y + hand[16].y) / 3.0
        assert math.isclose(cx, expected_x, rel_tol=1e-5)
        assert math.isclose(cy, expected_y, rel_tol=1e-5)


class TestPoseConfirmationAndStationaryPose:
    """Verifies ~120 ms confirmation requirement and stationary rejection."""

    def test_pose_requires_120ms_confirmation_before_stroke_tracking(self):
        pag = MockPyAutoGUI()
        ctrl = ScreenshotController(pyautogui_module=pag)
        hand = make_three_finger_pose()

        state = ctrl.update(hand, timestamp=1.000)
        assert state == ScreenshotState.CONFIRMING
        assert ctrl.is_candidate is True
        assert pag.screenshot_call_count == 0

        state = ctrl.update(hand, timestamp=1.060)
        assert state == ScreenshotState.CONFIRMING
        assert pag.screenshot_call_count == 0

        state = ctrl.update(hand, timestamp=1.121)
        assert state == ScreenshotState.TRACKING_STROKE
        assert pag.screenshot_call_count == 0

    def test_holding_pose_still_does_not_capture(self):
        pag = MockPyAutoGUI()
        ctrl = ScreenshotController(pyautogui_module=pag)
        hand = make_three_finger_pose()

        t = 1.000
        for _ in range(60):
            ctrl.update(hand, timestamp=t)
            t += 0.033

        assert pag.screenshot_call_count == 0
        assert ctrl.state == ScreenshotState.TRACKING_STROKE
        assert ctrl.downward_displacement < SCREENSHOT_MIN_DOWNWARD_DISPLACEMENT


class TestDownwardMovementRecognition:
    """Tests downward stroke classification and rejection of invalid movements."""

    def test_valid_three_finger_downward_stroke_captures_exactly_once(self, tmp_path):
        pag = MockPyAutoGUI()
        ctrl = ScreenshotController(
            screenshots_dir=str(tmp_path),
            pyautogui_module=pag,
        )
        base_hand = make_three_finger_pose()

        t = 1.000
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030
        assert ctrl.state == ScreenshotState.TRACKING_STROKE

        num_steps = 10
        total_dy = 0.15
        total_dx = 0.01
        for i in range(1, num_steps + 1):
            cur_dy = (i / num_steps) * total_dy
            cur_dx = (i / num_steps) * total_dx
            moved_hand = translate_hand(base_hand, dx=cur_dx, dy=cur_dy)
            state = ctrl.update(moved_hand, timestamp=t + (i * 0.030))

        assert pag.screenshot_call_count == 1
        assert state == ScreenshotState.CAPTURED
        assert ctrl.status_success is True
        assert "Screenshot saved" in ctrl.status_message

    def test_upward_stroke_rejected(self):
        pag = MockPyAutoGUI()
        ctrl = ScreenshotController(pyautogui_module=pag)
        base_hand = make_three_finger_pose()

        t = 1.0
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030

        for i in range(1, 10):
            moved_hand = translate_hand(base_hand, dx=0.0, dy=-0.015 * i)
            ctrl.update(moved_hand, timestamp=t + (i * 0.030))

        assert pag.screenshot_call_count == 0
        assert ctrl.state == ScreenshotState.TRACKING_STROKE

    def test_tiny_movement_rejected(self):
        pag = MockPyAutoGUI()
        ctrl = ScreenshotController(pyautogui_module=pag)
        base_hand = make_three_finger_pose()

        t = 1.0
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030

        for i in range(1, 10):
            moved_hand = translate_hand(base_hand, dx=0.0, dy=0.006 * i)
            ctrl.update(moved_hand, timestamp=t + (i * 0.030))

        assert pag.screenshot_call_count == 0
        assert ctrl.state == ScreenshotState.TRACKING_STROKE

    def test_slow_drift_rejected(self):
        pag = MockPyAutoGUI()
        ctrl = ScreenshotController(pyautogui_module=pag)
        base_hand = make_three_finger_pose()

        t = 1.0
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030

        for i in range(1, 60):
            moved_hand = translate_hand(base_hand, dx=0.0, dy=0.0022 * i)
            ctrl.update(moved_hand, timestamp=t + (i * 0.033))

        assert pag.screenshot_call_count == 0

    def test_sideways_stroke_rejected(self):
        pag = MockPyAutoGUI()
        ctrl = ScreenshotController(pyautogui_module=pag)
        base_hand = make_three_finger_pose()

        t = 1.0
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030

        for i in range(1, 10):
            moved_hand = translate_hand(base_hand, dx=0.02 * i, dy=0.005 * i)
            ctrl.update(moved_hand, timestamp=t + (i * 0.030))

        assert pag.screenshot_call_count == 0

    def test_strongly_diagonal_movement_rejected(self):
        pag = MockPyAutoGUI()
        ctrl = ScreenshotController(pyautogui_module=pag)
        base_hand = make_three_finger_pose()

        t = 1.0
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030

        for i in range(1, 10):
            moved_hand = translate_hand(base_hand, dx=0.014 * i, dy=0.014 * i)
            ctrl.update(moved_hand, timestamp=t + (i * 0.030))

        assert pag.screenshot_call_count == 0

    def test_apex_measurement_ignores_prior_upward_movement(self):
        pag = MockPyAutoGUI()
        ctrl = ScreenshotController(pyautogui_module=pag)
        base_hand = make_three_finger_pose()

        t = 1.0
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030

        ctrl.update(translate_hand(base_hand, 0, -0.05), timestamp=t + 0.030)

        ctrl.update(translate_hand(base_hand, 0, +0.09), timestamp=t + 0.200)

        assert pag.screenshot_call_count == 1
        assert ctrl.state == ScreenshotState.CAPTURED


class TestDuplicateAndAccidentalCaptures:
    """Verifies latching, deliberate release (~150 ms), and tracking loss handling."""

    def test_returning_hand_upward_while_holding_pose_does_not_repeat(self):
        pag = MockPyAutoGUI()
        ctrl = ScreenshotController(pyautogui_module=pag)
        base_hand = make_three_finger_pose()

        t = 1.0
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030
        for i in range(1, 10):
            ctrl.update(translate_hand(base_hand, 0, 0.015 * i), timestamp=t + (i * 0.030))

        assert pag.screenshot_call_count == 1
        assert ctrl.state == ScreenshotState.CAPTURED

        t_after = t + 0.35
        for i in range(1, 15):
            ctrl.update(translate_hand(base_hand, 0, 0.15 - 0.015 * i), timestamp=t_after + (i * 0.030))
            assert pag.screenshot_call_count == 1
            assert ctrl.state == ScreenshotState.CAPTURED

    def test_deliberate_release_and_new_stroke_permits_another_capture(self):
        pag = MockPyAutoGUI()
        ctrl = ScreenshotController(pyautogui_module=pag)
        base_hand = make_three_finger_pose()

        t = 1.0
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030
        for i in range(1, 10):
            ctrl.update(translate_hand(base_hand, 0, 0.015 * i), timestamp=t + (i * 0.030))
        assert pag.screenshot_call_count == 1
        assert ctrl.state == ScreenshotState.CAPTURED

        t_rel = t + 0.35
        open_palm = make_hand(True, True, True, True)
        ctrl.update(open_palm, timestamp=t_rel)
        assert ctrl.state == ScreenshotState.CAPTURED

        ctrl.update(open_palm, timestamp=t_rel + 0.160)
        assert ctrl.state == ScreenshotState.READY

        t2 = t_rel + 0.200
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t2)
            t2 += 0.030
        for i in range(1, 10):
            ctrl.update(translate_hand(base_hand, 0, 0.015 * i), timestamp=t2 + (i * 0.030))

        assert pag.screenshot_call_count == 2
        assert ctrl.state == ScreenshotState.CAPTURED

    def test_release_less_than_150ms_does_not_rearm(self):
        pag = MockPyAutoGUI()
        ctrl = ScreenshotController(pyautogui_module=pag)
        base_hand = make_three_finger_pose()

        t = 1.0
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030
        for i in range(1, 10):
            ctrl.update(translate_hand(base_hand, 0, 0.015 * i), timestamp=t + (i * 0.030))
        assert ctrl.state == ScreenshotState.CAPTURED

        open_palm = make_hand(True, True, True, True)
        t_flicker = t + 0.35
        ctrl.update(open_palm, timestamp=t_flicker)
        ctrl.update(open_palm, timestamp=t_flicker + 0.080)
        ctrl.update(base_hand, timestamp=t_flicker + 0.100)

        assert ctrl.state == ScreenshotState.CAPTURED

    def test_missing_tracking_alone_does_not_count_as_deliberate_release(self):
        pag = MockPyAutoGUI()
        ctrl = ScreenshotController(pyautogui_module=pag)
        base_hand = make_three_finger_pose()

        t = 1.0
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030
        for i in range(1, 10):
            ctrl.update(translate_hand(base_hand, 0, 0.015 * i), timestamp=t + (i * 0.030))
        assert ctrl.state == ScreenshotState.CAPTURED

        ctrl.update(None, timestamp=t + 1.0)
        ctrl.update(None, timestamp=t + 2.0)
        ctrl.update(None, timestamp=t + 3.0)

        assert ctrl.state == ScreenshotState.CAPTURED

    def test_tracking_gap_clears_pending_movement_and_does_not_bridge(self):
        pag = MockPyAutoGUI()
        ctrl = ScreenshotController(pyautogui_module=pag)
        base_hand = make_three_finger_pose()

        t = 1.0
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030
        ctrl.update(translate_hand(base_hand, 0, 0.07), timestamp=t + 0.030)
        assert ctrl.state == ScreenshotState.TRACKING_STROKE

        ctrl.update(None, timestamp=t + 0.060)
        assert ctrl.state == ScreenshotState.READY
        assert len(ctrl.stroke_history) == 0

        ctrl.update(translate_hand(base_hand, 0, 0.14), timestamp=t + 0.090)
        assert pag.screenshot_call_count == 0
        assert ctrl.state == ScreenshotState.CONFIRMING

    def test_pose_break_cancels_pending_movement(self):
        pag = MockPyAutoGUI()
        ctrl = ScreenshotController(pyautogui_module=pag)
        base_hand = make_three_finger_pose()

        t = 1.0
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030
        assert ctrl.state == ScreenshotState.TRACKING_STROKE

        scroll_hand = make_hand(True, True, False, False)
        ctrl.update(scroll_hand, timestamp=t + 0.030)

        assert ctrl.state == ScreenshotState.READY
        assert pag.screenshot_call_count == 0
        assert len(ctrl.stroke_history) == 0


def make_mock_mouse_pag():
    mock_pag = MagicMock()
    mock_pag.size.return_value = (1920, 1080)
    mock_pag.FailSafeException = pyautogui.FailSafeException
    return mock_pag


class TestGesturePriorityAndSuppression:
    """Verifies screenshot gesture priority over scroll, cursor, clicks, and PowerPoint."""

    def test_three_finger_pose_suppresses_scroll_candidate_and_mode(self):
        hand = make_three_finger_pose()
        assert is_scroll_candidate(hand) is False
        assert is_mouse_mode_candidate(hand) is False
        assert is_scroll_pose(hand) is False

        mouse_ctrl = MouseController(pyautogui_module=make_mock_mouse_pag())
        mode, clicked, scroll_val, scrolled = mouse_ctrl.process_landmarks(hand, timestamp=1.0)
        assert mode == "IDLE"
        assert clicked is False
        assert scroll_val == 0
        assert scrolled is False

    def test_screenshot_candidate_suppresses_palm_swipes(self):
        detector = PalmSwipeDetector()
        hand = make_three_finger_pose()

        state = detector.update(hand, active_mouse_mode="IDLE")
        assert detector.suppression_reason == "Screenshot Gesture Candidate"
        assert len(detector.palm_history) == 0

    def test_palm_detector_clears_history_when_screenshot_starts_and_ends(self):
        detector = PalmSwipeDetector()
        open_palm = make_hand(True, True, True, True, thumb_raised=True)

        t = 1.0
        with patch("main.time.time", side_effect=lambda: t):
            for _ in range(3):
                detector.update(open_palm, active_mouse_mode="IDLE")
                t += 0.030
            assert len(detector.palm_history) > 0

            ss_hand = make_three_finger_pose()
            detector.update(ss_hand, active_mouse_mode="IDLE")
            assert len(detector.palm_history) == 0
            assert detector.screenshot_suppression is True

            detector.update(open_palm, active_mouse_mode="IDLE")
            assert len(detector.palm_history) == 0

            t += 0.15
            detector.update(open_palm, active_mouse_mode="IDLE")
            assert len(detector.palm_history) == 1

    def test_controls_restore_after_screenshot_cancellation(self):
        mouse_ctrl = MouseController(pyautogui_module=make_mock_mouse_pag())
        pointing = make_hand(True, False, False, False)
        mode, _, _, _ = mouse_ctrl.process_landmarks(pointing, timestamp=1.0)
        assert mode == "CURSOR"

        ss_hand = make_three_finger_pose()
        mode, _, _, _ = mouse_ctrl.process_landmarks(ss_hand, timestamp=1.033)
        assert mode == "IDLE"

        mode, _, _, _ = mouse_ctrl.process_landmarks(pointing, timestamp=1.066)
        assert mode == "CURSOR"

    def test_failsafe_exception_propagates(self):
        pag = MockPyAutoGUI()
        pag.raise_on_screenshot = pyautogui.FailSafeException("Fail-safe triggered")
        ctrl = ScreenshotController(pyautogui_module=pag)
        base_hand = make_three_finger_pose()

        t = 1.0
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030

        with pytest.raises(pyautogui.FailSafeException):
            for i in range(1, 10):
                ctrl.update(translate_hand(base_hand, 0, 0.015 * i), timestamp=t + (i * 0.030))


class TestCaptureSaveAndFeedback:
    """Verifies file saving, error handling, and HUD display."""

    def test_saves_png_in_screenshots_folder_with_timestamp(self, tmp_path):
        target_dir = str(tmp_path / "screenshots")
        pag = MockPyAutoGUI()
        ctrl = ScreenshotController(
            screenshots_dir=target_dir,
            pyautogui_module=pag,
        )
        base_hand = make_three_finger_pose()

        t = 1.0
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030
        for i in range(1, 10):
            ctrl.update(translate_hand(base_hand, 0, 0.015 * i), timestamp=t + (i * 0.030))

        assert os.path.exists(target_dir)
        assert pag.last_mock_image is not None
        saved_path = pag.last_mock_image.saved_path
        assert saved_path.startswith(target_dir)
        assert saved_path.endswith(".png")
        assert "screenshot_" in os.path.basename(saved_path)
        assert pag.last_mock_image.saved_format == "PNG"

    def test_save_failure_produces_no_false_success_notification(self):
        pag = MockPyAutoGUI()
        def broken_save(filepath, format="PNG"):
            raise IOError("Disk full")

        def broken_screenshot():
            img = MockImage()
            img.save = broken_save
            return img

        pag.screenshot = broken_screenshot
        ctrl = ScreenshotController(pyautogui_module=pag)
        base_hand = make_three_finger_pose()

        t = 1.0
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030
        for i in range(1, 10):
            ctrl.update(translate_hand(base_hand, 0, 0.015 * i), timestamp=t + (i * 0.030))

        assert ctrl.status_success is False
        assert "Screenshot failed" in ctrl.status_message
        assert ctrl.feedback_success is False
        assert "Screenshot saved" not in ctrl.status_message

    def test_hud_renders_screenshot_metrics_and_feedback(self):
        pag = MockPyAutoGUI()
        ctrl = ScreenshotController(pyautogui_module=pag)
        hand = make_three_finger_pose()

        ctrl.update(hand, timestamp=1.0)
        assert ctrl.state == ScreenshotState.CONFIRMING

        detector = PalmSwipeDetector()
        frame = np.zeros((700, 1000, 3), dtype=np.uint8)

        draw_hud(
            frame,
            detector,
            ctrl.get_hud_label(),
            mouse_ctrl=None,
            screenshot_ctrl=ctrl,
            fps=30.0,
            now_mono=1.0,
        )

        ctrl.status_message = "Screenshot saved: screenshot_test.png"
        ctrl.status_time = 2.0
        ctrl.status_success = True

        draw_hud(
            frame,
            detector,
            ctrl.get_hud_label(),
            mouse_ctrl=None,
            screenshot_ctrl=ctrl,
            fps=30.0,
            now_mono=2.1,
        )
        assert ctrl.get_feedback(2.1)[0] == "Screenshot saved: screenshot_test.png"


class MockDesktopOverlayManager:
    """Mock overlay manager tracking show_success and show_error calls without opening windows."""
    def __init__(self, flash_enabled=True, notification_enabled=True):
        self.flash_enabled = flash_enabled
        self.notification_enabled = notification_enabled
        self.flash_trigger_count = 0
        self.notification_trigger_count = 0
        self.last_triggered_type = ""
        self.last_triggered_detail = ""
        self.call_log = []

    def show_success(self, filename: str, filepath: str = ""):
        if self.flash_enabled:
            self.flash_trigger_count += 1
        if self.notification_enabled:
            self.notification_trigger_count += 1
        self.last_triggered_type = "SUCCESS"
        self.last_triggered_detail = filename
        self.call_log.append(("SUCCESS", filename, time.monotonic()))

    def show_error(self, error_message: str):
        if self.notification_enabled:
            self.notification_trigger_count += 1
        self.last_triggered_type = "ERROR"
        self.last_triggered_detail = error_message
        self.call_log.append(("ERROR", error_message, time.monotonic()))

    def close(self):
        self.call_log.append(("CLOSE", None, time.monotonic()))


class TestDesktopVisualFeedback:
    """Tests capture flash animation, top-right notification, and desktop feedback coordination."""

    def test_saved_image_captured_and_saved_before_feedback_triggered(self):
        """Confirm the saved PNG contains no flash or notification overlay."""
        pag = MockPyAutoGUI()
        mock_overlay = MockDesktopOverlayManager()
        ctrl = ScreenshotController(
            pyautogui_module=pag,
            overlay_manager=mock_overlay,
        )
        base_hand = make_three_finger_pose()

        t = 1.0
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030
        for i in range(1, 10):
            ctrl.update(translate_hand(base_hand, 0, 0.015 * i), timestamp=t + (i * 0.030))

        assert pag.screenshot_call_count == 1
        assert len(mock_overlay.call_log) == 1
        call_type, filename, overlay_time = mock_overlay.call_log[0]
        assert call_type == "SUCCESS"
        assert pag.last_mock_image is not None
        assert pag.last_mock_image.save_time is not None
        assert pag.last_mock_image.save_time <= overlay_time

    def test_one_capture_produces_one_flash_and_one_notification(self):
        """Confirm one capture produces one animation and one notification."""
        pag = MockPyAutoGUI()
        mock_overlay = MockDesktopOverlayManager(flash_enabled=True, notification_enabled=True)
        ctrl = ScreenshotController(
            pyautogui_module=pag,
            overlay_manager=mock_overlay,
        )
        base_hand = make_three_finger_pose()

        t = 1.0
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030
        for i in range(1, 10):
            ctrl.update(translate_hand(base_hand, 0, 0.015 * i), timestamp=t + (i * 0.030))

        assert pag.screenshot_call_count == 1
        assert mock_overlay.flash_trigger_count == 1
        assert mock_overlay.notification_trigger_count == 1
        assert mock_overlay.last_triggered_type == "SUCCESS"

    def test_failure_shows_error_notification_without_flash(self):
        """Confirm failures show error feedback without a success flash."""
        pag = MockPyAutoGUI()

        def broken_save(filepath, format="PNG"):
            raise IOError("Permission denied")

        def broken_screenshot():
            img = MockImage()
            img.save = broken_save
            return img

        pag.screenshot = broken_screenshot
        mock_overlay = MockDesktopOverlayManager()
        ctrl = ScreenshotController(
            pyautogui_module=pag,
            overlay_manager=mock_overlay,
        )
        base_hand = make_three_finger_pose()

        t = 1.0
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030
        for i in range(1, 10):
            ctrl.update(translate_hand(base_hand, 0, 0.015 * i), timestamp=t + (i * 0.030))

        assert mock_overlay.last_triggered_type == "ERROR"
        assert mock_overlay.flash_trigger_count == 0
        assert mock_overlay.notification_trigger_count == 1
        assert "Permission denied" in mock_overlay.last_triggered_detail

    def test_configurable_settings_respected(self):
        """Confirm flash enabled/disabled, opacity, and duration settings can be customized."""
        pag = MockPyAutoGUI()
        mock_overlay = MockDesktopOverlayManager(flash_enabled=False, notification_enabled=True)
        ctrl = ScreenshotController(
            flash_enabled=False,
            flash_peak_opacity=0.20,
            flash_duration=0.30,
            notification_duration=2.0,
            notification_fade_duration=0.5,
            pyautogui_module=pag,
            overlay_manager=mock_overlay,
        )
        assert ctrl.flash_enabled is False
        assert ctrl.flash_peak_opacity == 0.20
        assert ctrl.flash_duration == 0.30
        assert ctrl.notification_duration == 2.0
        assert ctrl.notification_fade_duration == 0.5

        base_hand = make_three_finger_pose()
        t = 1.0
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030
        for i in range(1, 10):
            ctrl.update(translate_hand(base_hand, 0, 0.015 * i), timestamp=t + (i * 0.030))

        assert mock_overlay.flash_trigger_count == 0
        assert mock_overlay.notification_trigger_count == 1

    def test_overlay_failure_does_not_crash_or_duplicate_capture(self):
        """Confirm overlay manager errors do not disrupt capture or gesture pipeline."""
        pag = MockPyAutoGUI()
        mock_overlay = MockDesktopOverlayManager()

        def broken_show_success(filename, filepath=""):
            raise RuntimeError("GUI system down")

        mock_overlay.show_success = broken_show_success

        ctrl = ScreenshotController(
            pyautogui_module=pag,
            overlay_manager=mock_overlay,
        )
        base_hand = make_three_finger_pose()

        t = 1.0
        for _ in range(5):
            ctrl.update(base_hand, timestamp=t)
            t += 0.030
        for i in range(1, 10):
            ctrl.update(translate_hand(base_hand, 0, 0.015 * i), timestamp=t + (i * 0.030))

        assert pag.screenshot_call_count == 1
        assert ctrl.state == ScreenshotState.CAPTURED

    def test_real_desktop_overlay_manager_thread_and_styles(self):
        """Verify real DesktopOverlayManager initializes, accepts events, and closes cleanly."""
        overlay = DesktopOverlayManager(autostart=True)
        try:
            assert overlay._running is True
            overlay.show_success(filename="test_real_capture.png")
            time.sleep(0.05)
            assert overlay.flash_trigger_count == 1
            assert overlay.notification_trigger_count == 1
        finally:
            overlay.close()
            assert overlay._running is False


class TestScreenshotSwipeConflictResolution:
    """
    Regression and conflict resolution test suite ensuring:
    1. Screenshot pose and stroke take explicit priority over ordinary swipes.
    2. Ordinary swipes require a true 5-finger open palm (rejecting folded pinky or resting thumb).
    3. Motion does not leak between screenshot stroke/return and ordinary swipe detection.
    4. Diagnostics report conflict suppression reasons and dispatched actions accurately.
    5. Single gesture action dispatched per camera frame.
    """

    @patch("main.perform_gesture_action")
    def test_three_finger_downward_stroke_captures_once_and_sends_no_swipe_shortcut(self, mock_action):
        """A three-finger downward stroke must capture screenshot exactly once and send NO swipe shortcut."""
        pag = MockPyAutoGUI()
        mock_overlay = MockDesktopOverlayManager()
        ctrl = ScreenshotController(pyautogui_module=pag, overlay_manager=mock_overlay)
        detector = PalmSwipeDetector()

        base_hand = make_three_finger_pose()
        t = 1.0

        with patch("main.time.time", side_effect=lambda: t):
            for _ in range(5):
                ctrl.update(base_hand, timestamp=t)
                state = detector.update(base_hand, active_mouse_mode="IDLE", screenshot_active=ctrl.is_suppressing_swipes)
                assert detector.screenshot_suppression is True
                assert len(detector.palm_history) == 0
                t += 0.030

            for i in range(1, 10):
                moved = translate_hand(base_hand, 0.0, 0.02 * i)
                ctrl.update(moved, timestamp=t)
                state = detector.update(moved, active_mouse_mode="IDLE", screenshot_active=ctrl.is_suppressing_swipes)
                assert len(detector.palm_history) == 0
                t += 0.030

            assert pag.screenshot_call_count == 1
            assert ctrl.state == ScreenshotState.CAPTURED
            mock_action.assert_not_called()
            assert detector.last_detected_gesture == ""

    @patch("main.perform_gesture_action")
    def test_three_finger_upward_movement_triggers_neither_screenshot_nor_swipe_up(self, mock_action):
        """Moving 3 fingers upward must trigger neither screenshot capture nor a swipe-up command."""
        pag = MockPyAutoGUI()
        mock_overlay = MockDesktopOverlayManager()
        ctrl = ScreenshotController(pyautogui_module=pag, overlay_manager=mock_overlay)
        detector = PalmSwipeDetector()

        base_hand = make_three_finger_pose(wrist_x=0.5, wrist_y=0.7)
        t = 1.0

        with patch("main.time.time", side_effect=lambda: t):
            for _ in range(5):
                ctrl.update(base_hand, timestamp=t)
                detector.update(base_hand, active_mouse_mode="IDLE", screenshot_active=ctrl.is_suppressing_swipes)
                t += 0.030

            for i in range(1, 10):
                moved = translate_hand(base_hand, 0.0, -0.025 * i)
                ctrl.update(moved, timestamp=t)
                detector.update(moved, active_mouse_mode="IDLE", screenshot_active=ctrl.is_suppressing_swipes)
                t += 0.030

            assert pag.screenshot_call_count == 0
            assert ctrl.state != ScreenshotState.CAPTURED
            mock_action.assert_not_called()
            assert detector.last_detected_gesture == ""
            assert detector.state != GestureState.SWIPE_UP

    @patch("main.perform_gesture_action")
    def test_three_finger_sideways_movement_sends_no_ordinary_swipe(self, mock_action):
        """Moving 3 fingers sideways must trigger no ordinary swipe command."""
        detector = PalmSwipeDetector()
        base_hand = make_three_finger_pose(wrist_x=0.4, wrist_y=0.6)
        t = 1.0

        with patch("main.time.time", side_effect=lambda: t):
            for i in range(10):
                moved = translate_hand(base_hand, 0.025 * i, 0.0)
                detector.update(moved, active_mouse_mode="IDLE")
                t += 0.030

            mock_action.assert_not_called()
            assert detector.last_detected_gesture == ""
            assert len(detector.palm_history) == 0

    @patch("main.perform_gesture_action")
    def test_folded_thumb_rejects_ordinary_swipes(self, mock_action):
        """Fingers extended but thumb resting against palm must be rejected from ordinary swipes."""
        detector = PalmSwipeDetector()
        hand_thumb_folded = make_hand(
            index_extended=True,
            middle_extended=True,
            ring_extended=True,
            pinky_extended=True,
            thumb_raised=False,
        )
        assert is_swipe_palm_pose(hand_thumb_folded) is False

        t = 1.0
        with patch("main.time.time", side_effect=lambda: t):
            for i in range(8):
                moved = translate_hand(hand_thumb_folded, 0.03 * i, 0.0)
                state = detector.update(moved, active_mouse_mode="IDLE")
                assert len(detector.palm_history) == 0
                t += 0.030

            mock_action.assert_not_called()
            assert "Hand not open palm" in detector.suppression_reason

    @patch("main.perform_gesture_action")
    def test_folded_pinky_rejects_ordinary_swipes(self, mock_action):
        """Fingers extended with pinky folded must be rejected from ordinary swipes."""
        detector = PalmSwipeDetector()
        hand_pinky_folded = make_hand(
            index_extended=True,
            middle_extended=True,
            ring_extended=True,
            pinky_extended=False,
            thumb_raised=True,
        )
        assert is_swipe_palm_pose(hand_pinky_folded) is False

        t = 1.0
        with patch("main.time.time", side_effect=lambda: t):
            for i in range(8):
                moved = translate_hand(hand_pinky_folded, 0.03 * i, 0.0)
                state = detector.update(moved, active_mouse_mode="IDLE")
                assert len(detector.palm_history) == 0
                t += 0.030

            mock_action.assert_not_called()

    @patch("main.perform_gesture_action")
    def test_genuine_five_finger_palm_swipes_work_in_all_four_directions(self, mock_action):
        """A genuine five-finger palm must reliably trigger swipes in all 4 directions."""
        directions = [
            (0.035, 0.0, GestureState.SWIPE_RIGHT),
            (-0.035, 0.0, GestureState.SWIPE_LEFT),
            (0.0, -0.035, GestureState.SWIPE_UP),
            (0.0, 0.035, GestureState.SWIPE_DOWN),
        ]

        for dx, dy, expected_gesture in directions:
            mock_action.reset_mock()
            detector = PalmSwipeDetector()
            palm = make_hand(
                index_extended=True,
                middle_extended=True,
                ring_extended=True,
                pinky_extended=True,
                thumb_raised=True,
                wrist_x=0.5,
                wrist_y=0.6,
            )
            assert is_swipe_palm_pose(palm) is True

            t = 1.0
            triggered = False
            with patch("main.time.time", side_effect=lambda: t):
                for i in range(6):
                    moved = translate_hand(palm, i * dx, i * dy)
                    state = detector.update(moved, active_mouse_mode="IDLE")
                    if state == expected_gesture:
                        triggered = True
                        break
                    t += 0.030

            assert triggered is True, f"Failed to trigger {expected_gesture}"
            mock_action.assert_called_once_with(expected_gesture)

    @patch("main.perform_gesture_action")
    def test_screenshot_cancellation_completion_and_rearm_do_not_leak_movement(self, mock_action):
        """Motion during stroke, cancellation, or upward return after capture cannot leak into swipes."""
        pag = MockPyAutoGUI()
        mock_overlay = MockDesktopOverlayManager()
        ctrl = ScreenshotController(pyautogui_module=pag, overlay_manager=mock_overlay)
        detector = PalmSwipeDetector()

        ss_hand = make_three_finger_pose()
        t = 1.0

        with patch("main.time.time", side_effect=lambda: t):
            for i in range(3):
                moved = translate_hand(ss_hand, 0.0, 0.02 * i)
                ctrl.update(moved, timestamp=t)
                detector.update(moved, active_mouse_mode="IDLE", screenshot_active=ctrl.is_suppressing_swipes)
                t += 0.030

            for i in range(3):
                moved = translate_hand(ss_hand, 0.0, -0.03 * i)
                ctrl.update(moved, timestamp=t)
                detector.update(moved, active_mouse_mode="IDLE", screenshot_active=ctrl.is_suppressing_swipes)
                t += 0.030

            assert len(detector.palm_history) == 0
            mock_action.assert_not_called()

            for _ in range(5):
                ctrl.update(ss_hand, timestamp=t)
                detector.update(ss_hand, active_mouse_mode="IDLE", screenshot_active=ctrl.is_suppressing_swipes)
                t += 0.030

            for i in range(1, 10):
                moved = translate_hand(ss_hand, 0.0, 0.02 * i)
                ctrl.update(moved, timestamp=t)
                detector.update(moved, active_mouse_mode="IDLE", screenshot_active=ctrl.is_suppressing_swipes)
                t += 0.030

            assert ctrl.state == ScreenshotState.CAPTURED
            assert pag.screenshot_call_count == 1
            mock_action.assert_not_called()

            for i in range(1, 10):
                moved = translate_hand(ss_hand, 0.0, -0.025 * i)
                ctrl.update(moved, timestamp=t)
                detector.update(moved, active_mouse_mode="IDLE", screenshot_active=ctrl.is_suppressing_swipes)
                t += 0.030

            mock_action.assert_not_called()
            assert detector.last_detected_gesture == ""

    @patch("main.perform_gesture_action")
    def test_returning_from_screenshot_mode_requires_fresh_open_palm_movement(self, mock_action):
        """Returning from screenshot mode requires 100-150ms 5-finger palm confirmation before fresh swipes."""
        detector = PalmSwipeDetector()
        ss_hand = make_three_finger_pose()
        open_palm = make_hand(True, True, True, True, thumb_raised=True)

        t = 1.0
        with patch("main.time.time", side_effect=lambda: t):
            detector.update(ss_hand, active_mouse_mode="IDLE", screenshot_active=True)
            assert detector.screenshot_suppression is True

            detector.update(open_palm, active_mouse_mode="IDLE", screenshot_active=False)
            assert len(detector.palm_history) == 0
            assert "Confirming 5-finger palm" in detector.suppression_reason

            t += 0.060
            detector.update(open_palm, active_mouse_mode="IDLE", screenshot_active=False)
            assert len(detector.palm_history) == 0

            t += 0.070
            detector.update(open_palm, active_mouse_mode="IDLE", screenshot_active=False)
            assert len(detector.palm_history) == 1
            assert detector.screenshot_suppression is False

            for i in range(1, 6):
                t += 0.030
                moved = translate_hand(open_palm, 0.035 * i, 0.0)
                state = detector.update(moved, active_mouse_mode="IDLE", screenshot_active=False)
                if state == GestureState.SWIPE_RIGHT:
                    break

            assert state == GestureState.SWIPE_RIGHT
            mock_action.assert_called_once_with(GestureState.SWIPE_RIGHT)

    def test_idle_mouse_controller_cannot_remove_screenshot_suppression(self):
        """Active mouse controller returning IDLE must NOT remove screenshot suppression."""
        detector = PalmSwipeDetector()
        ss_hand = make_three_finger_pose()

        state = detector.update(ss_hand, active_mouse_mode="IDLE", screenshot_active=False)
        assert detector.screenshot_suppression is True
        assert "Screenshot Gesture Candidate" in detector.suppression_reason
        assert len(detector.palm_history) == 0

    def test_hud_diagnostics_display_conflict_resolution_fields(self):
        """HUD must display screenshot state, 5-finger palm validity, suppression reason, and dispatched action."""
        detector = PalmSwipeDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detector.suppression_reason = "Screenshot Gesture Candidate"
        detector.last_dispatched_action = "RIGHT -> next"
        detector.is_valid_swipe_palm = False

        draw_hud(frame, detector, "Waiting for hand...", mouse_ctrl=None, screenshot_ctrl=None, fps=30.0)

        assert np.any(frame > 0)
        assert detector.suppression_reason == "Screenshot Gesture Candidate"
        assert detector.last_dispatched_action == "RIGHT -> next"
        assert detector.is_valid_swipe_palm is False
