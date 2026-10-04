"""
Automated tests for PalmSwipeDetector and detect_swipe.

Every test mocks ``perform_gesture_action`` so that no real keyboard
input is ever sent. Timing is controlled by patching ``time.time``
in the main module.

Coverage includes:
- Fast swipes completed in 5 frames
- All four cardinal directions
- Short (below threshold) and diagonal (below 1.25 dominance) movements
- Slow drift rejection by velocity filter
- Cooldown (0.6s) and display duration (0.35s) behavior
- Hand loss and tracking recovery
- Pointing pose candidate clearing old swipe labels and suppressing swipes
- Transition out of cursor mode requiring fresh movement before a swipe fires

Run with:
    .venv\\Scripts\\python.exe -m pytest test_gestures.py -v
"""

import os
import sys
from collections import deque
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    PalmSwipeDetector,
    GestureState,
    detect_swipe,
    get_palm_center,
    GESTURE_COOLDOWN,
    GESTURE_DISPLAY_DURATION,
    HISTORY_SIZE,
    MIN_HISTORY_SIZE,
    MAX_HISTORY_TIME,
    SWIPE_THRESHOLD,
    AXIS_DOMINANCE_RATIO,
    MIN_SWIPE_VELOCITY,
)
from mouse_controller import is_pointing_pose, is_pointing_candidate


class FakeLandmark:
    """Minimal stand-in for a MediaPipe NormalizedLandmark."""
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def _landmarks(cx: float, cy: float) -> list:
    """Return synthetic open-palm landmarks centered at (cx, cy)."""
    lms = [FakeLandmark(cx, cy + 0.20) for _ in range(21)]
    wrist_x, wrist_y = cx, cy + 0.20
    lms[0] = FakeLandmark(wrist_x, wrist_y)

    mcp_y = cy
    lms[5] = FakeLandmark(cx - 0.06, mcp_y)
    lms[9] = FakeLandmark(cx - 0.02, mcp_y)
    lms[13] = FakeLandmark(cx + 0.02, mcp_y)
    lms[17] = FakeLandmark(cx + 0.06, mcp_y)

    def set_finger(tip_idx, pip_idx, dip_idx, bx):
        lms[pip_idx] = FakeLandmark(bx, cy - 0.15)
        lms[dip_idx] = FakeLandmark(bx, cy - 0.25)
        lms[tip_idx] = FakeLandmark(bx, cy - 0.35)

    set_finger(8, 6, 7, cx - 0.06)
    set_finger(12, 10, 11, cx - 0.02)
    set_finger(16, 14, 15, cx + 0.02)
    set_finger(20, 18, 19, cx + 0.06)

    lms[1] = FakeLandmark(cx - 0.05, cy + 0.12)
    lms[2] = FakeLandmark(cx - 0.08, cy + 0.06)
    lms[3] = FakeLandmark(cx - 0.12, cy)
    lms[4] = FakeLandmark(cx - 0.15, cy - 0.05)

    return lms


def make_pointing_hand(wrist_x: float = 0.5, wrist_y: float = 0.8) -> list:
    """Construct synthetic landmarks representing a distinct pointing pose."""
    lms = [FakeLandmark(wrist_x, wrist_y) for _ in range(21)]
    lms[0] = FakeLandmark(wrist_x, wrist_y)

    lms[5] = FakeLandmark(wrist_x - 0.05, wrist_y - 0.20)
    lms[6] = FakeLandmark(wrist_x - 0.05, wrist_y - 0.35)
    lms[7] = FakeLandmark(wrist_x - 0.05, wrist_y - 0.45)
    lms[8] = FakeLandmark(wrist_x - 0.05, wrist_y - 0.60)

    for tip, pip, mcp, bx in [(12, 10, 9, wrist_x), (16, 14, 13, wrist_x + 0.05), (20, 18, 17, wrist_x + 0.10)]:
        lms[mcp] = FakeLandmark(bx, wrist_y - 0.20)
        lms[pip] = FakeLandmark(bx, wrist_y - 0.35)
        lms[tip - 1] = FakeLandmark(bx, wrist_y - 0.28)
        lms[tip] = FakeLandmark(bx, wrist_y - 0.22)

    return lms


def _run_swipe(detector, start_x, start_y, end_x, end_y,
               n_frames, dt, mock_time, start_t):
    """
    Feed *n_frames* of a straight-line motion into *detector*.

    Returns ``(states, t_after)`` where *t_after* is the time value that
    would follow the last frame.
    """
    states = []
    t = start_t
    for i in range(n_frames):
        frac = i / max(n_frames - 1, 1)
        x = start_x + (end_x - start_x) * frac
        y = start_y + (end_y - start_y) * frac
        mock_time.return_value = t
        state = detector.update(_landmarks(x, y))
        states.append(state)
        t += dt
    return states, t


_SWIPE_STATES = frozenset({
    GestureState.SWIPE_LEFT,
    GestureState.SWIPE_RIGHT,
    GestureState.SWIPE_UP,
    GestureState.SWIPE_DOWN,
})


class TestGetPalmCenter:
    """Verify the palm-center calculation from MCP landmarks."""

    def test_average_of_mcp_landmarks(self):
        lms = [FakeLandmark(0.0, 0.0)] * 21
        lms[5]  = FakeLandmark(0.1, 0.2)
        lms[9]  = FakeLandmark(0.3, 0.4)
        lms[13] = FakeLandmark(0.5, 0.6)
        lms[17] = FakeLandmark(0.7, 0.8)
        cx, cy = get_palm_center(lms)
        assert abs(cx - 0.4) < 1e-6
        assert abs(cy - 0.5) < 1e-6


class TestDetectSwipePure:
    """Direct tests of ``detect_swipe()`` with hand-crafted history buffers."""

    @staticmethod
    def _history(positions, dt=0.033):
        """Build a deque of (x, y, timestamp) tuples."""
        d = deque()
        t = 0.0
        for x, y in positions:
            d.append((x, y, t))
            t += dt
        return d

    def test_clear_right(self):
        h = self._history([(0.3 + i * 0.025, 0.5) for i in range(5)])
        r = detect_swipe(h, SWIPE_THRESHOLD, AXIS_DOMINANCE_RATIO, MIN_SWIPE_VELOCITY)
        assert r["direction"] == GestureState.SWIPE_RIGHT

    def test_clear_left(self):
        h = self._history([(0.7 - i * 0.03, 0.5) for i in range(5)])
        r = detect_swipe(h, SWIPE_THRESHOLD, AXIS_DOMINANCE_RATIO, MIN_SWIPE_VELOCITY)
        assert r["direction"] == GestureState.SWIPE_LEFT

    def test_clear_up(self):
        h = self._history([(0.5, 0.7 - i * 0.03) for i in range(5)])
        r = detect_swipe(h, SWIPE_THRESHOLD, AXIS_DOMINANCE_RATIO, MIN_SWIPE_VELOCITY)
        assert r["direction"] == GestureState.SWIPE_UP

    def test_clear_down(self):
        h = self._history([(0.5, 0.3 + i * 0.025) for i in range(5)])
        r = detect_swipe(h, SWIPE_THRESHOLD, AXIS_DOMINANCE_RATIO, MIN_SWIPE_VELOCITY)
        assert r["direction"] == GestureState.SWIPE_DOWN

    def test_fast_swipe_completed_in_5_frames(self):
        """A fast swipe completed in exactly 5 frames (dx=0.12 in ~0.13s) triggers."""
        h = self._history([(0.4 + i * 0.03, 0.5) for i in range(5)], dt=0.033)
        r = detect_swipe(h, SWIPE_THRESHOLD, AXIS_DOMINANCE_RATIO, MIN_SWIPE_VELOCITY)
        assert r["direction"] == GestureState.SWIPE_RIGHT
        assert r["speed"] > MIN_SWIPE_VELOCITY

    def test_no_displacement(self):
        h = self._history([(0.5, 0.5)] * 5)
        r = detect_swipe(h, SWIPE_THRESHOLD, AXIS_DOMINANCE_RATIO, MIN_SWIPE_VELOCITY)
        assert r["direction"] == ""
        assert r["ambiguous"] is False

    def test_diagonal_is_ambiguous(self):
        """When dx and dy are nearly equal (ratio < 1.25), movement is flagged ambiguous."""
        h = self._history([(0.3 + i * 0.025, 0.3 + i * 0.022) for i in range(5)])
        r = detect_swipe(h, SWIPE_THRESHOLD, AXIS_DOMINANCE_RATIO, MIN_SWIPE_VELOCITY)
        assert r["direction"] == ""
        assert r["ambiguous"] is True

    def test_short_movement_below_threshold(self):
        h = self._history([(0.5 + i * 0.015, 0.5) for i in range(5)])
        r = detect_swipe(h, SWIPE_THRESHOLD, AXIS_DOMINANCE_RATIO, MIN_SWIPE_VELOCITY)
        assert r["direction"] == ""

    def test_slow_velocity_rejected(self):
        h = self._history([(0.3 + i * 0.03, 0.5) for i in range(5)], dt=0.4)
        r = detect_swipe(h, SWIPE_THRESHOLD, AXIS_DOMINANCE_RATIO, MIN_SWIPE_VELOCITY)
        assert r["direction"] == ""


class TestSwipeDirections:
    """All four cardinal swipes should trigger in fast 5-frame movements."""

    @patch("main.time.time")
    @patch("main.perform_gesture_action", return_value="mocked")
    def test_fast_swipe_right_in_5_frames(self, mock_action, mock_time):
        d = PalmSwipeDetector()
        states, _ = _run_swipe(d, 0.3, 0.5, 0.45, 0.5, 5, 0.033, mock_time, 1000.0)
        assert GestureState.SWIPE_RIGHT in states
        mock_action.assert_called_once_with(GestureState.SWIPE_RIGHT)

    @patch("main.time.time")
    @patch("main.perform_gesture_action", return_value="mocked")
    def test_fast_swipe_left_in_5_frames(self, mock_action, mock_time):
        d = PalmSwipeDetector()
        states, _ = _run_swipe(d, 0.7, 0.5, 0.55, 0.5, 5, 0.033, mock_time, 1000.0)
        assert GestureState.SWIPE_LEFT in states
        mock_action.assert_called_once_with(GestureState.SWIPE_LEFT)

    @patch("main.time.time")
    @patch("main.perform_gesture_action", return_value="mocked")
    def test_fast_swipe_up_in_5_frames(self, mock_action, mock_time):
        d = PalmSwipeDetector()
        states, _ = _run_swipe(d, 0.5, 0.7, 0.5, 0.55, 5, 0.033, mock_time, 1000.0)
        assert GestureState.SWIPE_UP in states
        mock_action.assert_called_once_with(GestureState.SWIPE_UP)

    @patch("main.time.time")
    @patch("main.perform_gesture_action", return_value="mocked")
    def test_fast_swipe_down_in_5_frames(self, mock_action, mock_time):
        d = PalmSwipeDetector()
        states, _ = _run_swipe(d, 0.5, 0.3, 0.5, 0.45, 5, 0.033, mock_time, 1000.0)
        assert GestureState.SWIPE_DOWN in states
        mock_action.assert_called_once_with(GestureState.SWIPE_DOWN)


class TestRejection:
    """Movements that must NOT trigger any swipe action."""

    @patch("main.time.time")
    @patch("main.perform_gesture_action", return_value="mocked")
    def test_stationary_hand(self, mock_action, mock_time):
        d = PalmSwipeDetector()
        states, _ = _run_swipe(d, 0.5, 0.5, 0.5, 0.5, 8, 0.033, mock_time, 1000.0)
        assert not _SWIPE_STATES.intersection(states)
        mock_action.assert_not_called()

    @patch("main.time.time")
    @patch("main.perform_gesture_action", return_value="mocked")
    def test_diagonal_movement(self, mock_action, mock_time):
        d = PalmSwipeDetector()
        states, _ = _run_swipe(d, 0.3, 0.3, 0.45, 0.45, 6, 0.033, mock_time, 1000.0)
        assert not _SWIPE_STATES.intersection(states)
        mock_action.assert_not_called()

    @patch("main.time.time")
    @patch("main.perform_gesture_action", return_value="mocked")
    def test_short_movement(self, mock_action, mock_time):
        d = PalmSwipeDetector()
        states, _ = _run_swipe(d, 0.5, 0.5, 0.55, 0.5, 5, 0.033, mock_time, 1000.0)
        assert not _SWIPE_STATES.intersection(states)
        mock_action.assert_not_called()

    @patch("main.time.time")
    @patch("main.perform_gesture_action", return_value="mocked")
    def test_slow_drift_rejected(self, mock_action, mock_time):
        d = PalmSwipeDetector()
        states, _ = _run_swipe(d, 0.5, 0.5, 0.62, 0.5, 6, 0.24, mock_time, 1000.0)
        assert not _SWIPE_STATES.intersection(states)
        mock_action.assert_not_called()

    @patch("main.time.time")
    @patch("main.perform_gesture_action", return_value="mocked")
    def test_prunes_history_older_than_max_time(self, mock_action, mock_time):
        """Samples older than 0.45s must be pruned from history."""
        d = PalmSwipeDetector()
        mock_time.return_value = 1000.0
        d.update(_landmarks(0.3, 0.5))

        mock_time.return_value = 1000.25
        d.update(_landmarks(0.35, 0.5))

        mock_time.return_value = 1000.50
        d.update(_landmarks(0.45, 0.5))

        assert len(d.palm_history) == 2
        assert d.palm_history[0][2] == 1000.25


class TestCooldownBehavior:
    """Verifies the 0.6s cooldown and 0.35s gesture display duration."""

    @patch("main.time.time")
    @patch("main.perform_gesture_action", return_value="mocked")
    def test_gesture_display_and_cooldown_progression(self, mock_action, mock_time):
        d = PalmSwipeDetector()

        states, t_trigger = _run_swipe(d, 0.3, 0.5, 0.45, 0.5, 5, 0.033, mock_time, 1000.0)
        assert GestureState.SWIPE_RIGHT in states
        assert d.in_cooldown is True

        mock_time.return_value = 1000.0 + 4 * 0.033 + 0.20
        state = d.update(_landmarks(0.45, 0.5))
        assert state == GestureState.SWIPE_RIGHT

        mock_time.return_value = 1000.0 + 4 * 0.033 + 0.40
        state = d.update(_landmarks(0.45, 0.5))
        assert state == GestureState.COOLDOWN

        mock_time.return_value = 1000.0 + 4 * 0.033 + 0.65
        state = d.update(_landmarks(0.45, 0.5))
        assert state == GestureState.TRACKING
        assert d.in_cooldown is False

    @patch("main.time.time")
    @patch("main.perform_gesture_action", return_value="mocked")
    def test_second_swipe_blocked_within_cooldown(self, mock_action, mock_time):
        d = PalmSwipeDetector()

        _run_swipe(d, 0.3, 0.5, 0.45, 0.5, 5, 0.033, mock_time, 1000.0)
        assert mock_action.call_count == 1

        _run_swipe(d, 0.6, 0.5, 0.40, 0.5, 5, 0.033, mock_time, 1000.3)
        assert mock_action.call_count == 1, "Must be blocked by cooldown"

    @patch("main.time.time")
    @patch("main.perform_gesture_action", return_value="mocked")
    def test_second_swipe_allowed_after_0_6s_cooldown(self, mock_action, mock_time):
        d = PalmSwipeDetector()

        _run_swipe(d, 0.3, 0.5, 0.45, 0.5, 5, 0.033, mock_time, 1000.0)
        assert mock_action.call_count == 1

        states2, _ = _run_swipe(d, 0.7, 0.5, 0.55, 0.5, 5, 0.033, mock_time, 1000.7)
        assert GestureState.SWIPE_LEFT in states2
        assert mock_action.call_count == 2


class TestHandLoss:
    """Verifies state cleanup and cooldown survival on hand loss."""

    @patch("main.time.time")
    @patch("main.perform_gesture_action", return_value="mocked")
    def test_hand_loss_clears_history(self, mock_action, mock_time):
        d = PalmSwipeDetector()
        mock_time.return_value = 1000.0
        d.update(_landmarks(0.5, 0.5))
        d.update(_landmarks(0.52, 0.5))
        assert len(d.palm_history) == 2

        d.on_hand_lost()
        assert len(d.palm_history) == 0
        assert d.state == GestureState.IDLE

    @patch("main.time.time")
    @patch("main.perform_gesture_action", return_value="mocked")
    def test_hand_loss_during_cooldown_preserves_cooldown(self, mock_action, mock_time):
        d = PalmSwipeDetector()
        _run_swipe(d, 0.3, 0.5, 0.45, 0.5, 5, 0.033, mock_time, 1000.0)
        assert d.in_cooldown is True

        mock_time.return_value = 1000.35
        d.on_hand_lost()
        assert d.in_cooldown is True

        _run_swipe(d, 0.3, 0.5, 0.45, 0.5, 5, 0.033, mock_time, 1000.45)
        assert mock_action.call_count == 1, "Must not trigger duplicate action"


class TestCursorModeTransition:
    """Verifies pointing pose candidate label clearing, swipe suppression, and mode exit."""

    @patch("main.time.time")
    @patch("main.perform_gesture_action", return_value="mocked")
    def test_pointing_candidate_clears_old_swipe_label(self, mock_action, mock_time):
        d = PalmSwipeDetector()

        _run_swipe(d, 0.3, 0.5, 0.45, 0.5, 5, 0.033, mock_time, 1000.0)
        assert d.last_detected_gesture == GestureState.SWIPE_RIGHT

        pointing = make_pointing_hand()
        mock_time.return_value = 1000.2
        state = d.update(pointing)

        assert d.last_detected_gesture == ""
        assert state == GestureState.CURSOR_CONTROL

    @patch("main.time.time")
    @patch("main.perform_gesture_action", return_value="mocked")
    def test_pointing_movement_does_not_trigger_swipe(self, mock_action, mock_time):
        d = PalmSwipeDetector()
        pointing = make_pointing_hand()

        t = 1000.0
        for i in range(10):
            mock_time.return_value = t
            pointing[0] = FakeLandmark(0.2 + i * 0.05, 0.8)
            pointing[5] = FakeLandmark(0.2 + i * 0.05, 0.6)
            pointing[8] = FakeLandmark(0.2 + i * 0.05, 0.2)
            state = d.update(pointing)
            assert state == GestureState.CURSOR_CONTROL
            t += 0.033

        mock_action.assert_not_called()

    @patch("main.time.time")
    @patch("main.perform_gesture_action", return_value="mocked")
    def test_transition_out_of_cursor_mode_requires_fresh_movement(self, mock_action, mock_time):
        d = PalmSwipeDetector()
        pointing = make_pointing_hand()

        t = 1000.0
        for _ in range(4):
            mock_time.return_value = t
            d.update(pointing)
            t += 0.033

        mock_time.return_value = t
        state1 = d.update(_landmarks(0.5, 0.5))
        assert state1 == GestureState.TRACKING
        assert len(d.palm_history) == 1
        mock_action.assert_not_called()

        states, _ = _run_swipe(d, 0.5, 0.5, 0.65, 0.5, 5, 0.033, mock_time, t + 0.033)
        assert GestureState.SWIPE_RIGHT in states
        mock_action.assert_called_once_with(GestureState.SWIPE_RIGHT)


# =============================================================================
# PowerPoint Targeting, Navigation, and Full Verification Tests
# =============================================================================

import ctypes
from powerpoint_target import PowerPointTarget
from main import perform_gesture_action, TARGET_MODE_POWERPOINT
import pyautogui


class MockUser32:
    """Mock for Windows user32 library providing full control over window state."""

    def __init__(self, windows=None, fg_hwnd=0):
        # windows is a list of tuples: (hwnd, class_name, title, is_visible)
        self.windows = windows or []
        self._fg_hwnd = fg_hwnd
        self.attached_threads = []
        self.calls = []

    def IsWindowVisible(self, hwnd):
        for h, c, t, vis in self.windows:
            if h == hwnd:
                return vis
        return True

    def GetWindowTextLengthW(self, hwnd):
        for h, c, t, vis in self.windows:
            if h == hwnd:
                return len(t)
        return 0

    def GetWindowTextW(self, hwnd, buf, max_len):
        for h, c, t, vis in self.windows:
            if h == hwnd:
                ctypes.memmove(buf, ctypes.create_unicode_buffer(t), min(len(t) + 1, max_len) * 2)
                return len(t)
        return 0

    def GetClassNameW(self, hwnd, buf, max_len):
        for h, c, t, vis in self.windows:
            if h == hwnd:
                ctypes.memmove(buf, ctypes.create_unicode_buffer(c), min(len(c) + 1, max_len) * 2)
                return len(c)
        return 0

    def EnumWindows(self, cb, lparam):
        for h, c, t, vis in self.windows:
            if not cb(h, lparam):
                break
        return 1

    def GetForegroundWindow(self):
        return self._fg_hwnd

    def SetForegroundWindow(self, hwnd):
        self._fg_hwnd = hwnd
        self.calls.append(("SetForegroundWindow", hwnd))
        return 1

    def BringWindowToTop(self, hwnd):
        self.calls.append(("BringWindowToTop", hwnd))
        return 1

    def ShowWindow(self, hwnd, cmd):
        self.calls.append(("ShowWindow", hwnd, cmd))
        return 1

    def IsIconic(self, hwnd):
        return 0

    def GetWindowThreadProcessId(self, hwnd, pid_ptr):
        return hwnd * 10

    def AttachThreadInput(self, id_attach, id_attach_to, f_attach):
        self.attached_threads.append((id_attach, id_attach_to, f_attach))
        return 1

    def keybd_event(self, b_vk, b_scan, dw_flags, dw_extra_info):
        self.calls.append(("keybd_event", b_vk, dw_flags))
        return 0


class TestPowerPointTargeting:
    """Verifies Slide Show and Presenter View identification, focus acquisition, and navigation."""

    def test_find_presentation_window_slide_show(self):
        user32 = MockUser32(
            windows=[
                (101, "screenClass", "PowerPoint Slide Show - Presentation1", True),
                (102, "PPTFrameClass", "Presentation1 - PowerPoint", True),
            ]
        )
        target = PowerPointTarget(user32_module=user32)
        hwnd, ttype = target.find_presentation_window()
        assert hwnd == 101
        assert ttype == "Slide Show"

    def test_find_presentation_window_presenter_view(self):
        user32 = MockUser32(
            windows=[
                (201, "screenClass", "PowerPoint Presenter View - Presentation1", True),
                (202, "screenClass", "PowerPoint Slide Show - Presentation1", True),
            ]
        )
        target = PowerPointTarget(user32_module=user32)
        hwnd, ttype = target.find_presentation_window()
        assert hwnd == 201
        assert ttype == "Presenter View"

    def test_find_presentation_window_normal_window(self):
        user32 = MockUser32(
            windows=[
                (301, "PPTFrameClass", "Financial Deck - PowerPoint", True),
            ]
        )
        target = PowerPointTarget(user32_module=user32)
        hwnd, ttype = target.find_presentation_window()
        assert hwnd == 301
        assert ttype == "PowerPoint Window"

    def test_find_presentation_window_skips_camera_preview(self):
        user32 = MockUser32(
            windows=[
                (401, "OpenCVWindowClass", "Palm Gesture Controller", True),
            ]
        )
        target = PowerPointTarget(user32_module=user32)
        hwnd, ttype = target.find_presentation_window()
        assert hwnd is None
        assert ttype == "None"

    def test_focus_presentation_when_preview_had_focus(self):
        preview_hwnd = 500
        ppt_hwnd = 501
        user32 = MockUser32(
            windows=[
                (preview_hwnd, "OpenCVWindowClass", "Palm Gesture Controller", True),
                (ppt_hwnd, "screenClass", "PowerPoint Slide Show - Deck", True),
            ],
            fg_hwnd=preview_hwnd,
        )
        target = PowerPointTarget(user32_module=user32)
        focused = target.focus_presentation(ppt_hwnd)
        assert focused is True
        assert user32.GetForegroundWindow() == ppt_hwnd
        assert ("SetForegroundWindow", ppt_hwnd) in user32.calls

    def test_focus_presentation_failure_reported(self):
        ppt_hwnd = 601
        user32 = MockUser32(
            windows=[(ppt_hwnd, "screenClass", "PowerPoint Slide Show - Deck", True)],
            fg_hwnd=999,
        )
        # Force SetForegroundWindow to fail
        user32.SetForegroundWindow = MagicMock(return_value=0)
        target = PowerPointTarget(user32_module=user32)
        focused = target.focus_presentation(ppt_hwnd)
        assert focused is False
        assert user32.GetForegroundWindow() != ppt_hwnd

    def test_navigate_swipe_right_presses_pagedown(self):
        mock_pag = MagicMock()
        mock_pag.FailSafeException = pyautogui.FailSafeException
        user32 = MockUser32(
            windows=[(100, "screenClass", "PowerPoint Slide Show - Deck", True)],
            fg_hwnd=100,
        )
        target = PowerPointTarget(pyautogui_module=mock_pag, user32_module=user32)
        success, msg = target.navigate("RIGHT")
        assert success is True
        assert "Next slide (PageDown)" in msg
        mock_pag.press.assert_called_once()
        assert mock_pag.press.call_args[0][0] == "pagedown"

    def test_navigate_swipe_left_presses_pageup(self):
        mock_pag = MagicMock()
        mock_pag.FailSafeException = pyautogui.FailSafeException
        user32 = MockUser32(
            windows=[(200, "screenClass", "PowerPoint Presenter View - Deck", True)],
            fg_hwnd=200,
        )
        target = PowerPointTarget(pyautogui_module=mock_pag, user32_module=user32)
        success, msg = target.navigate("LEFT")
        assert success is True
        assert "Previous slide (PageUp)" in msg
        mock_pag.press.assert_called_once()
        assert mock_pag.press.call_args[0][0] == "pageup"

    def test_navigate_missing_presentation_issues_no_keys(self):
        mock_pag = MagicMock()
        mock_pag.FailSafeException = pyautogui.FailSafeException
        user32 = MockUser32(windows=[])
        target = PowerPointTarget(pyautogui_module=mock_pag, user32_module=user32)
        success, msg = target.navigate("RIGHT")
        assert success is False
        assert "PowerPoint presentation not found" in msg
        mock_pag.press.assert_not_called()

    def test_navigate_failed_focus_issues_no_keys(self):
        mock_pag = MagicMock()
        mock_pag.FailSafeException = pyautogui.FailSafeException
        user32 = MockUser32(
            windows=[(100, "screenClass", "PowerPoint Slide Show - Deck", True)],
            fg_hwnd=999,
        )
        user32.SetForegroundWindow = MagicMock(return_value=0)
        target = PowerPointTarget(pyautogui_module=mock_pag, user32_module=user32)
        success, msg = target.navigate("RIGHT")
        assert success is False
        assert "Failed to bring PowerPoint" in msg
        mock_pag.press.assert_not_called()

    def test_navigate_failsafe_propagates(self):
        mock_pag = MagicMock()
        mock_pag.FailSafeException = pyautogui.FailSafeException
        mock_pag.press.side_effect = pyautogui.FailSafeException("Fail-safe triggered")
        user32 = MockUser32(
            windows=[(100, "screenClass", "PowerPoint Slide Show - Deck", True)],
            fg_hwnd=100,
        )
        target = PowerPointTarget(pyautogui_module=mock_pag, user32_module=user32)
        with pytest.raises(pyautogui.FailSafeException):
            target.navigate("RIGHT")


class TestPresentationModeAndGestureSuppression:
    """Verifies end-to-end swipe recognition, presentation navigation, and gesture suppression."""

    @patch("main.time.time")
    def test_end_to_end_right_swipe_navigates_powerpoint(self, mock_time):
        mock_pag = MagicMock()
        mock_pag.FailSafeException = pyautogui.FailSafeException
        user32 = MockUser32(
            windows=[(100, "screenClass", "PowerPoint Slide Show - Deck", True)],
            fg_hwnd=100,
        )
        ppt = PowerPointTarget(pyautogui_module=mock_pag, user32_module=user32)
        detector = PalmSwipeDetector(powerpoint_target=ppt, target_mode_powerpoint=True)

        states, _ = _run_swipe(detector, 0.3, 0.5, 0.45, 0.5, 5, 0.033, mock_time, 1000.0)
        assert GestureState.SWIPE_RIGHT in states
        assert "Next slide (PageDown)" in detector.last_dispatched_action
        assert mock_pag.press.call_count == 1
        assert mock_pag.press.call_args[0][0] == "pagedown"

    @patch("main.time.time")
    def test_end_to_end_left_swipe_navigates_powerpoint(self, mock_time):
        mock_pag = MagicMock()
        mock_pag.FailSafeException = pyautogui.FailSafeException
        user32 = MockUser32(
            windows=[(200, "screenClass", "PowerPoint Presenter View - Deck", True)],
            fg_hwnd=200,
        )
        ppt = PowerPointTarget(pyautogui_module=mock_pag, user32_module=user32)
        detector = PalmSwipeDetector(powerpoint_target=ppt, target_mode_powerpoint=True)

        states, _ = _run_swipe(detector, 0.7, 0.5, 0.55, 0.5, 5, 0.033, mock_time, 1000.0)
        assert GestureState.SWIPE_LEFT in states
        assert "Previous slide (PageUp)" in detector.last_dispatched_action
        assert mock_pag.press.call_count == 1
        assert mock_pag.press.call_args[0][0] == "pageup"

    @patch("main.time.time")
    def test_missing_powerpoint_produces_no_keyboard_input(self, mock_time):
        mock_pag = MagicMock()
        mock_pag.FailSafeException = pyautogui.FailSafeException
        user32 = MockUser32(windows=[])
        ppt = PowerPointTarget(pyautogui_module=mock_pag, user32_module=user32)
        detector = PalmSwipeDetector(powerpoint_target=ppt, target_mode_powerpoint=True)

        _run_swipe(detector, 0.3, 0.5, 0.45, 0.5, 5, 0.033, mock_time, 1000.0)
        mock_pag.press.assert_not_called()
        assert detector.last_detected_gesture == ""
        assert detector.last_dispatched_action == ""
        assert "Dispatch failed" in detector.suppression_reason

    @patch("main.time.time")
    def test_generic_mode_outside_powerpoint(self, mock_time):
        mock_pag = MagicMock()
        mock_pag.FailSafeException = pyautogui.FailSafeException
        with patch("main.pyautogui", mock_pag):
            detector = PalmSwipeDetector(target_mode_powerpoint=False)
            states, _ = _run_swipe(detector, 0.3, 0.5, 0.45, 0.5, 5, 0.033, mock_time, 1000.0)
            assert GestureState.SWIPE_RIGHT in states
            assert mock_pag.press.call_count == 1
            assert mock_pag.press.call_args[0][0] == "right"

    @patch("main.time.time")
    def test_cursor_mode_suppresses_slide_navigation(self, mock_time):
        mock_pag = MagicMock()
        mock_pag.FailSafeException = pyautogui.FailSafeException
        user32 = MockUser32(
            windows=[(100, "screenClass", "PowerPoint Slide Show - Deck", True)],
            fg_hwnd=100,
        )
        ppt = PowerPointTarget(pyautogui_module=mock_pag, user32_module=user32)
        detector = PalmSwipeDetector(powerpoint_target=ppt, target_mode_powerpoint=True)

        pointing = make_pointing_hand()
        mock_time.return_value = 1000.0
        state = detector.update(pointing, active_mouse_mode="CURSOR")
        assert state == GestureState.CURSOR_CONTROL
        assert "Active Mouse Mode (CURSOR)" in detector.suppression_reason
        mock_pag.press.assert_not_called()

    @patch("main.time.time")
    def test_scroll_mode_suppresses_slide_navigation(self, mock_time):
        mock_pag = MagicMock()
        mock_pag.FailSafeException = pyautogui.FailSafeException
        user32 = MockUser32(
            windows=[(100, "screenClass", "PowerPoint Slide Show - Deck", True)],
            fg_hwnd=100,
        )
        ppt = PowerPointTarget(pyautogui_module=mock_pag, user32_module=user32)
        detector = PalmSwipeDetector(powerpoint_target=ppt, target_mode_powerpoint=True)

        pointing = make_pointing_hand()
        mock_time.return_value = 1000.0
        state = detector.update(pointing, active_mouse_mode="SCROLL")
        assert state == GestureState.SCROLL_CONTROL
        assert "Active Mouse Mode (SCROLL)" in detector.suppression_reason
        mock_pag.press.assert_not_called()

    @patch("main.time.time")
    def test_single_swipe_produces_exactly_one_action(self, mock_time):
        mock_pag = MagicMock()
        mock_pag.FailSafeException = pyautogui.FailSafeException
        user32 = MockUser32(
            windows=[(100, "screenClass", "PowerPoint Slide Show - Deck", True)],
            fg_hwnd=100,
        )
        ppt = PowerPointTarget(pyautogui_module=mock_pag, user32_module=user32)
        detector = PalmSwipeDetector(powerpoint_target=ppt, target_mode_powerpoint=True)

        # 10 continuous frames of rightward movement
        _run_swipe(detector, 0.2, 0.5, 0.8, 0.5, 10, 0.033, mock_time, 1000.0)
        assert mock_pag.press.call_count == 1

    @patch("main.time.time")
    def test_opposite_return_stroke_suppressed(self, mock_time):
        mock_pag = MagicMock()
        mock_pag.FailSafeException = pyautogui.FailSafeException
        user32 = MockUser32(
            windows=[(100, "screenClass", "PowerPoint Slide Show - Deck", True)],
            fg_hwnd=100,
        )
        ppt = PowerPointTarget(pyautogui_module=mock_pag, user32_module=user32)
        detector = PalmSwipeDetector(powerpoint_target=ppt, target_mode_powerpoint=True)

        # 1. Swipe Right
        _run_swipe(detector, 0.3, 0.5, 0.45, 0.5, 5, 0.033, mock_time, 1000.0)
        assert mock_pag.press.call_count == 1

        # 2. Return stroke moving Left at t=1000.75 (post-cooldown continuous return)
        states_return, _ = _run_swipe(detector, 0.45, 0.5, 0.3, 0.5, 5, 0.033, mock_time, 1000.75)
        # Return stroke must be rejected
        assert mock_pag.press.call_count == 1
        assert "Opposite return stroke suppressed" in detector.suppression_reason

