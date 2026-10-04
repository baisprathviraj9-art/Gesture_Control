import cv2
import mediapipe as mp
import time
from mediapipe.tasks.python import vision
from collections import deque
import os
import platform
import pyautogui
from mouse_controller import (
    MouseController,
    is_pointing_pose,
    is_pointing_candidate,
    is_scroll_pose,
    is_scroll_candidate,
    is_mouse_mode_candidate,
    is_open_palm_pose,
    is_swipe_palm_pose,
    is_three_finger_screenshot_pose,
    is_thumb_raised,
)
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
    get_three_fingertip_center,
)
from powerpoint_target import PowerPointTarget

if platform.system() == "Windows":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("gesturecontrol.palmrecognizer.1.0")
    except Exception:
        pass

WINDOW_NAME = "Palm Gesture Controller"
WINDOW_INITIAL_WIDTH = 1000
WINDOW_INITIAL_HEIGHT = 700


HISTORY_SIZE = 8
MIN_HISTORY_SIZE = 3
MAX_HISTORY_TIME = 0.45
SWIPE_THRESHOLD = 0.10
AXIS_DOMINANCE_RATIO = 1.25
GESTURE_COOLDOWN = 0.6
GESTURE_DISPLAY_DURATION = 0.35
ACTION_DISPLAY_DURATION = 0.6
MIN_SWIPE_VELOCITY = 0.15

COLOR_GREEN = (0, 255, 0)
COLOR_CYAN = (255, 255, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_DARK_BG = (20, 20, 20)


pyautogui.PAUSE = 0.08
pyautogui.FAILSAFE = True


TARGET_MODE_POWERPOINT: bool = True


def perform_gesture_action(
    gesture: str,
    powerpoint_target: PowerPointTarget | None = None,
    target_mode_powerpoint: bool = TARGET_MODE_POWERPOINT,
):
    """Execute the laptop action for one detected palm swipe."""
    system = platform.system()

    try:
        if gesture in (GestureState.SWIPE_RIGHT, GestureState.SWIPE_LEFT):
            if target_mode_powerpoint:
                target = powerpoint_target or PowerPointTarget(pyautogui_module=pyautogui)
                direction = "RIGHT" if gesture == GestureState.SWIPE_RIGHT else "LEFT"
                success, msg = target.navigate(direction)
                if success:
                    return msg
                else:
                    print(f"[PowerPoint] Navigation failed: {msg}")
                    return None
            else:
                if gesture == GestureState.SWIPE_RIGHT:
                    pyautogui.press("right")
                    return "RIGHT -> next"
                else:
                    pyautogui.press("left")
                    return "LEFT -> previous"

        if gesture == GestureState.SWIPE_UP:
            if system == "Windows":
                pyautogui.hotkey("win", "tab")
            elif system == "Darwin":
                pyautogui.hotkey("ctrl", "up")
            else:
                pyautogui.press("win")
            return "UP -> desktop overview"

        if gesture == GestureState.SWIPE_DOWN:
            if system == "Windows":
                pyautogui.hotkey("win", "d")
            elif system == "Darwin":
                pyautogui.press("f11")
            else:
                pyautogui.hotkey("win", "d")
            return "DOWN -> show desktop"

    except pyautogui.FailSafeException:
        raise
    except Exception as exc:
        print(f"Gesture action failed: {exc}")

    return None


class GestureState:
    IDLE = "Waiting for hand..."
    TRACKING = "Tracking palm..."
    SWIPE_LEFT = "Swipe Left"
    SWIPE_RIGHT = "Swipe Right"
    SWIPE_UP = "Swipe Up"
    SWIPE_DOWN = "Swipe Down"
    AMBIGUOUS = "Ambiguous movement"
    COOLDOWN = "Cooldown"
    CURSOR_CONTROL = "Cursor control"
    SCROLL_CONTROL = "Scroll control"


class PalmSwipeDetector:
    """Tracks palm-center movement over time and classifies directional swipes."""

    def __init__(
        self,
        history_size: int = HISTORY_SIZE,
        min_history_size: int = MIN_HISTORY_SIZE,
        max_history_time: float = MAX_HISTORY_TIME,
        swipe_threshold: float = SWIPE_THRESHOLD,
        axis_dominance_ratio: float = AXIS_DOMINANCE_RATIO,
        cooldown: float = GESTURE_COOLDOWN,
        gesture_display_duration: float = GESTURE_DISPLAY_DURATION,
        min_velocity: float = MIN_SWIPE_VELOCITY,
        powerpoint_target: PowerPointTarget | None = None,
        target_mode_powerpoint: bool = TARGET_MODE_POWERPOINT,
    ):
        self.history_size = history_size
        self.min_history_size = min_history_size
        self.max_history_time = max_history_time
        self.swipe_threshold = swipe_threshold
        self.axis_dominance_ratio = axis_dominance_ratio
        self.cooldown = cooldown
        self.gesture_display_duration = gesture_display_duration
        self.min_velocity = min_velocity
        self.powerpoint_target = powerpoint_target
        self.target_mode_powerpoint = target_mode_powerpoint

        self.palm_history: deque = deque(maxlen=history_size)

        self.state: str = GestureState.IDLE
        self.last_detected_gesture: str = ""
        self.gesture_trigger_time: float = 0.0
        self.in_cooldown: bool = False
        self.pointing_suppression: bool = False

        self.debug_palm_x: float = 0.0
        self.debug_palm_y: float = 0.0
        self.debug_dx: float = 0.0
        self.debug_dy: float = 0.0
        self.debug_direction: str = ""
        self.debug_speed: float = 0.0

        # Diagnostics & Return-stroke suppression tracking
        self.suppression_reason: str = ""
        self.intended_target: str = ""
        self.last_dispatched_action: str = ""
        self.last_dispatched_time: float = 0.0
        self.last_dispatched_direction: str = ""
        self.last_swipe_start: tuple[float, float] = (0.0, 0.0)
        self.last_swipe_end: tuple[float, float] = (0.0, 0.0)
        self.has_settled_after_swipe: bool = True
        self.last_frame_time: float = 0.0

        # Screenshot coordination & 5-finger palm confirmation
        self.swipe_resume_confirm_time: float = 0.12  # ~120 ms required before resuming swipes
        self.screenshot_suppression: bool = False
        self.resume_confirm_start_time: float | None = None
        self.is_valid_swipe_palm: bool = False

    def update(
        self,
        hand_landmarks,
        active_mouse_mode: str = "IDLE",
        screenshot_active: bool = False,
    ) -> str:
        """Processes current hand landmarks and returns the active state label."""
        now = time.time()
        dt_frame = now - self.last_frame_time if self.last_frame_time > 0 else 0.0
        self.last_frame_time = now

        if hand_landmarks is None or len(hand_landmarks) < 21:
            self.on_hand_lost()
            return self.state

        is_valid_palm = is_swipe_palm_pose(hand_landmarks)
        self.is_valid_swipe_palm = is_valid_palm

        ss_pose = is_three_finger_screenshot_pose(hand_landmarks)
        is_ss = screenshot_active or ss_pose

        # Priority 1: Screenshot Recognition Explicit Priority
        if is_ss:
            self.screenshot_suppression = True
            self.resume_confirm_start_time = None
            self.palm_history.clear()
            self.last_detected_gesture = ""
            self.pointing_suppression = True
            self.suppression_reason = "Screenshot Gesture Candidate" if ss_pose else "Screenshot Gesture Active"
            self.state = GestureState.TRACKING
            return self.state

        # Priority 1b: Resuming from screenshot mode requires 100-150ms continuous 5-finger palm confirmation
        if self.screenshot_suppression:
            if not is_valid_palm:
                self.resume_confirm_start_time = None
                self.palm_history.clear()
                self.suppression_reason = "Screenshot Release: Awaiting valid 5-finger palm"
                self.state = GestureState.TRACKING
                return self.state

            if self.resume_confirm_start_time is None:
                self.resume_confirm_start_time = now
                self.palm_history.clear()
                self.suppression_reason = "Confirming 5-finger palm after screenshot"
                self.state = GestureState.TRACKING
                return self.state

            elapsed = now - self.resume_confirm_start_time
            if elapsed < (self.swipe_resume_confirm_time - 1e-5):
                self.palm_history.clear()
                pct = int((elapsed / self.swipe_resume_confirm_time) * 100)
                self.suppression_reason = f"Confirming 5-finger palm after screenshot ({pct}%)"
                self.state = GestureState.TRACKING
                return self.state

            # Confirmed! Reset screenshot suppression and begin fresh history
            self.screenshot_suppression = False
            self.resume_confirm_start_time = None
            self.palm_history.clear()
            self.suppression_reason = ""

        # Trace gesture suppression for active mouse controller
        mouse_suppressed = False
        if active_mouse_mode in ("CURSOR", "SCROLL"):
            mouse_suppressed = True
            self.suppression_reason = f"Active Mouse Mode ({active_mouse_mode})"
        elif not is_valid_palm and is_mouse_mode_candidate(hand_landmarks):
            mouse_suppressed = True
            if active_mouse_mode == "SCROLL" or is_scroll_pose(hand_landmarks) or is_scroll_candidate(hand_landmarks):
                self.suppression_reason = "Scroll Gesture"
            elif active_mouse_mode == "CURSOR" or is_pointing_pose(hand_landmarks) or is_pointing_candidate(hand_landmarks):
                self.suppression_reason = "Cursor / Pointing Gesture"
            else:
                self.suppression_reason = "Mouse Mode Candidate"

        if mouse_suppressed:
            self.last_detected_gesture = ""
            if not self.pointing_suppression:
                self.reset()
                self.pointing_suppression = True
            if active_mouse_mode == "SCROLL" or is_scroll_pose(hand_landmarks):
                self.state = GestureState.SCROLL_CONTROL
            elif active_mouse_mode == "CURSOR" or is_pointing_pose(hand_landmarks):
                self.state = GestureState.CURSOR_CONTROL
            else:
                self.state = GestureState.TRACKING
            return self.state
        elif self.pointing_suppression:
            self.reset()
            self.pointing_suppression = False

        if self.in_cooldown:
            elapsed = now - self.gesture_trigger_time
            if elapsed < self.cooldown:
                self.suppression_reason = "In Cooldown"
                if elapsed < self.gesture_display_duration:
                    self.state = self.last_detected_gesture
                else:
                    self.state = GestureState.COOLDOWN
                return self.state
            else:
                self.in_cooldown = False
                self.last_detected_gesture = ""
                self.reset()
                self.state = GestureState.TRACKING

        # Requirement 2: Require a true open palm for ordinary swipes
        if not is_valid_palm:
            self.reset()
            self.suppression_reason = "Hand not open palm (5 fingers required)"
            self.state = GestureState.TRACKING
            return self.state

        palm_x, palm_y = get_palm_center(hand_landmarks)

        self.debug_palm_x = palm_x
        self.debug_palm_y = palm_y

        self.palm_history.append((palm_x, palm_y, now))
        self.state = GestureState.TRACKING

        while self.palm_history and (now - self.palm_history[0][2] > self.max_history_time):
            self.palm_history.popleft()

        if len(self.palm_history) < self.min_history_size:
            return self.state

        result = detect_swipe(
            self.palm_history,
            self.swipe_threshold,
            self.axis_dominance_ratio,
            self.min_velocity,
        )

        self.debug_dx = result["dx"]
        self.debug_dy = result["dy"]
        self.debug_direction = result["direction"]
        self.debug_speed = result["speed"]

        # Track hand settling (stationary or low velocity)
        if result["speed"] < self.min_velocity or (
            abs(result["dx"]) < self.swipe_threshold / 2 and abs(result["dy"]) < self.swipe_threshold / 2
        ):
            self.has_settled_after_swipe = True

        if result["direction"]:
            swipe_name = result["direction"]

            # Re-check 5-finger palm pose before dispatching action:
            # "Apply this check before collecting swipe movement history and again before dispatching an action."
            if not is_swipe_palm_pose(hand_landmarks) or is_three_finger_screenshot_pose(hand_landmarks) or screenshot_active:
                self.suppression_reason = "Pose invalidated before swipe dispatch"
                self.palm_history.clear()
                self.state = GestureState.TRACKING
                return self.state

            hist_start_x, hist_start_y, _ = self.palm_history[0]
            hist_end_x, hist_end_y, _ = self.palm_history[-1]

            # Return-stroke check:
            # If the swipe is opposite to the last dispatched swipe, and occurred within
            # 1.2s originating from where the last swipe ended without the hand settling,
            # reject it as an unintended return stroke.
            is_opposite = (
                (self.last_dispatched_direction == GestureState.SWIPE_RIGHT and swipe_name == GestureState.SWIPE_LEFT)
                or (self.last_dispatched_direction == GestureState.SWIPE_LEFT and swipe_name == GestureState.SWIPE_RIGHT)
                or (self.last_dispatched_direction == GestureState.SWIPE_UP and swipe_name == GestureState.SWIPE_DOWN)
                or (self.last_dispatched_direction == GestureState.SWIPE_DOWN and swipe_name == GestureState.SWIPE_UP)
            )

            dist_from_last_end = (
                (hist_start_x - self.last_swipe_end[0]) ** 2 + (hist_start_y - self.last_swipe_end[1]) ** 2
            ) ** 0.5

            if (
                is_opposite
                and (dist_from_last_end < 0.18)
                and not self.has_settled_after_swipe
                and (now - self.last_dispatched_time < 1.2)
            ):
                self.suppression_reason = "Opposite return stroke suppressed"
                self.palm_history.clear()
                self.state = GestureState.TRACKING
                return self.state

            # Query intended target
            if self.target_mode_powerpoint:
                ppt = self.powerpoint_target or PowerPointTarget()
                hwnd, ttype = ppt.find_presentation_window()
                self.intended_target = f"PowerPoint ({ttype})" if hwnd else "PowerPoint (None)"
            else:
                self.intended_target = "Generic Keypress"

            if self.powerpoint_target is not None or self.target_mode_powerpoint != TARGET_MODE_POWERPOINT:
                action_result = perform_gesture_action(
                    swipe_name,
                    powerpoint_target=self.powerpoint_target,
                    target_mode_powerpoint=self.target_mode_powerpoint,
                )
            else:
                action_result = perform_gesture_action(swipe_name)

            if action_result:
                self.state = swipe_name
                self.last_detected_gesture = swipe_name
                self.last_dispatched_action = action_result
                self.last_dispatched_time = now
                self.last_dispatched_direction = swipe_name
                self.last_swipe_start = (hist_start_x, hist_start_y)
                self.last_swipe_end = (hist_end_x, hist_end_y)
                self.has_settled_after_swipe = False
                self.gesture_trigger_time = now
                self.in_cooldown = True
                self.suppression_reason = ""
                print(f"[Gesture Dispatch] {swipe_name} -> {action_result} (Target: {self.intended_target})")
            else:
                self.suppression_reason = f"Dispatch failed for {self.intended_target}"
                self.last_detected_gesture = ""
                self.state = GestureState.TRACKING
                print(f"[Gesture Dispatch] {swipe_name} failed -> {self.intended_target}")

            self.palm_history.clear()
            return self.state
        elif result["ambiguous"]:
            self.state = GestureState.AMBIGUOUS
            self.suppression_reason = "Ambiguous movement"

        return self.state

    def on_hand_lost(self):
        """Called when no hand is present in the frame."""
        self.reset()
        self.pointing_suppression = False
        self.resume_confirm_start_time = None
        self.is_valid_swipe_palm = False
        self.suppression_reason = "No hand detected"
        if self.in_cooldown:
            now = time.time()
            elapsed = now - self.gesture_trigger_time
            if elapsed >= self.cooldown:
                self.in_cooldown = False
                self.last_detected_gesture = ""
                self.state = GestureState.IDLE
            elif elapsed < self.gesture_display_duration:
                self.state = self.last_detected_gesture
            else:
                self.state = GestureState.COOLDOWN
        else:
            self.state = GestureState.IDLE
            self.last_detected_gesture = ""

    def reset(self):
        """Clears movement history and resets displacement values."""
        self.palm_history.clear()
        self.resume_confirm_start_time = None
        self.debug_dx = 0.0
        self.debug_dy = 0.0
        self.debug_direction = ""
        self.debug_speed = 0.0


def get_palm_center(hand_landmarks):
    """Calculates representative palm center as the average of MCP landmarks (5, 9, 13, 17)."""
    palm_x = (
        hand_landmarks[5].x
        + hand_landmarks[9].x
        + hand_landmarks[13].x
        + hand_landmarks[17].x
    ) / 4.0

    palm_y = (
        hand_landmarks[5].y
        + hand_landmarks[9].y
        + hand_landmarks[13].y
        + hand_landmarks[17].y
    ) / 4.0

    return palm_x, palm_y


def detect_swipe(
    palm_history,
    threshold: float,
    dominance_ratio: float,
    min_velocity: float,
):
    """Evaluates displacement across palm history and classifies the dominant swipe."""
    start_x, start_y, start_t = palm_history[0]
    end_x, end_y, end_t = palm_history[-1]

    dx = end_x - start_x
    dy = end_y - start_y
    dt = end_t - start_t if end_t > start_t else 1e-5

    abs_dx = abs(dx)
    abs_dy = abs(dy)
    speed = (abs_dx**2 + abs_dy**2) ** 0.5 / dt

    result = {
        "dx": dx,
        "dy": dy,
        "speed": speed,
        "direction": "",
        "ambiguous": False,
    }

    if abs_dx < threshold and abs_dy < threshold:
        return result

    if min_velocity > 0 and speed < min_velocity:
        return result

    if abs_dx > abs_dy * dominance_ratio:
        if abs_dx >= threshold:
            result["direction"] = GestureState.SWIPE_RIGHT if dx > 0 else GestureState.SWIPE_LEFT
    elif abs_dy > abs_dx * dominance_ratio:
        if abs_dy >= threshold:
            result["direction"] = GestureState.SWIPE_DOWN if dy > 0 else GestureState.SWIPE_UP
    else:
        result["ambiguous"] = True

    return result


def draw_palm_center(frame, palm_x, palm_y, w, h):
    """Draws a prominent dual-ring marker at the computed palm center."""
    cx, cy = int(palm_x * w), int(palm_y * h)
    cv2.circle(frame, (cx, cy), 11, COLOR_MAGENTA, -1)
    cv2.circle(frame, (cx, cy), 13, COLOR_WHITE, 2)
    cv2.circle(frame, (cx, cy), 3, COLOR_YELLOW, -1)


def draw_palm_trajectory(frame, palm_history, w, h):
    """Draws a motion trail connecting recent palm positions."""
    if len(palm_history) < 2:
        return
    points = [(int(x * w), int(y * h)) for x, y, _ in palm_history]
    for i in range(1, len(points)):
        alpha = i / len(points)
        color = (
            int(COLOR_CYAN[0] * alpha),
            int(COLOR_CYAN[1] * alpha),
            int(COLOR_CYAN[2] * alpha),
        )
        cv2.line(frame, points[i - 1], points[i], color, 2)
    cv2.circle(frame, points[0], 4, COLOR_YELLOW, -1)


def draw_hud(
    frame,
    detector,
    state_label,
    mouse_ctrl=None,
    screenshot_ctrl=None,
    fps: float = 0.0,
    now_mono: float | None = None,
):
    """Renders gesture state banner, active mode, live FPS, and compact diagnostic values."""
    cv2.rectangle(frame, (20, 15), (460, 65), COLOR_DARK_BG, -1)
    cv2.rectangle(frame, (20, 15), (460, 65), COLOR_WHITE, 1)

    if screenshot_ctrl and (screenshot_ctrl.is_candidate or screenshot_ctrl.state == ScreenshotState.CAPTURED):
        badge_color = COLOR_GREEN if screenshot_ctrl.state == ScreenshotState.CAPTURED else COLOR_CYAN
    elif state_label == GestureState.CURSOR_CONTROL:
        badge_color = COLOR_CYAN
    elif state_label == GestureState.SCROLL_CONTROL:
        badge_color = COLOR_MAGENTA
    elif "Swipe" in state_label:
        badge_color = COLOR_YELLOW
    elif state_label == GestureState.COOLDOWN:
        badge_color = COLOR_ORANGE
    elif state_label == GestureState.AMBIGUOUS:
        badge_color = COLOR_RED
    else:
        badge_color = COLOR_GREEN

    cv2.putText(
        frame,
        f"Mode: {state_label}",
        (30, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        badge_color,
        2,
    )

    if fps > 0:
        cv2.putText(
            frame,
            f"{fps:.1f} FPS",
            (365, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_WHITE,
            1,
        )

    if screenshot_ctrl and (screenshot_ctrl.is_candidate or screenshot_ctrl.state == ScreenshotState.CAPTURED):
        hud_height = 210
        cv2.rectangle(frame, (20, 75), (380, hud_height), COLOR_DARK_BG, -1)
        cv2.rectangle(frame, (20, 75), (380, hud_height), (80, 80, 80), 1)

        disp = screenshot_ctrl.downward_displacement
        min_disp = SCREENSHOT_MIN_DOWNWARD_DISPLACEMENT
        vel = screenshot_ctrl.current_velocity
        state_str = str(screenshot_ctrl.state)
        ready_str = (
            "CAPTURED (Release to rearm)"
            if screenshot_ctrl.state == ScreenshotState.CAPTURED
            else ("CONFIRMING" if screenshot_ctrl.state == ScreenshotState.CONFIRMING else "STROKE TRACKING")
        )

        hud_lines = [
            f"Screenshot: 3-Finger Swipe Down",
            f"State: {state_str} ({ready_str})",
            f"Down Displacement: {disp:.3f} / {min_disp:.3f}",
            f"Velocity: {vel:.2f} / {SCREENSHOT_MIN_VELOCITY:.2f} u/s",
            f"FPS: {fps:.1f}",
        ]
        pct = min(1.0, max(0.0, disp / min_disp)) if min_disp > 0 else 0.0
        bar_x = 28
        bar_y = 188
        bar_w = 220
        bar_h = 10
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        bar_color = COLOR_GREEN if pct >= 1.0 else COLOR_YELLOW
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * pct), bar_y + bar_h), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_WHITE, 1)

        y_pos = 95
        for line in hud_lines:
            cv2.putText(
                frame,
                line,
                (28, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                COLOR_WHITE,
                1,
            )
            y_pos += 18
    elif mouse_ctrl and mouse_ctrl.is_active:
        hud_height = 250 if mouse_ctrl.mode == "CURSOR" else (198 if mouse_ctrl.mode == "SCROLL" else 185)
        cv2.rectangle(frame, (20, 75), (380, hud_height), COLOR_DARK_BG, -1)
        cv2.rectangle(frame, (20, 75), (380, hud_height), (80, 80, 80), 1)

        hud_lines = [
            f"Raw Tip: ({mouse_ctrl.raw_tip_x:.2f}, {mouse_ctrl.raw_tip_y:.2f})",
            f"Target: ({mouse_ctrl.target_screen_x}, {mouse_ctrl.target_screen_y})",
            f"Thumb: {mouse_ctrl.thumb_state.capitalize()} ({mouse_ctrl.thumb_ratio:.2f})",
            f"Active Margin: {int(mouse_ctrl.margin_x * 100)}%",
            f"FPS: {fps:.1f}",
        ]
        if mouse_ctrl.mode == "SCROLL":
            sc = mouse_ctrl.scroll_ctrl
            dir_label = "UP" if sc.direction > 0 else ("DOWN" if sc.direction < 0 else "---")
            hud_lines.append(f"Scroll: {dir_label}  Rate: {sc.current_rate:.1f}/s")
        elif mouse_ctrl.mode == "CURSOR":
            rc = mouse_ctrl.right_click_ctrl
            rc_state = rc.state.upper()
            hud_lines.append(
                f"Pinch: {rc.pinch_ratio:.2f} (raw:{rc.raw_pinch_ratio:.2f}) [Contact:<={rc.contact_threshold:.2f} Rel:>={rc.release_threshold:.2f}]"
            )
            pose_status = "Valid" if rc.is_pose_valid else f"Blocked ({rc.pose_rejection_reason})"
            hud_lines.append(f"RC Pose: {pose_status}")
            if rc.state == "confirming":
                prog_pct = int(rc.confirmation_progress * 100)
                hud_lines.append(
                    f"RC State: CONFIRMING ({prog_pct}% {rc.contact_elapsed:.2f}/{rc.confirmation_duration:.2f}s)"
                )
            elif rc.state in ("clicked", "clicked_held"):
                hud_lines.append(f"RC State: CLICKED (Latched)")
            elif not rc.armed:
                sep_pct = int(rc.separation_progress * 100)
                sep_info = f" ({sep_pct}% {rc.separation_elapsed:.2f}/{rc.separation_duration:.2f}s)" if rc.separation_start_time is not None else ""
                hud_lines.append(f"RC State: DISARMED{sep_info}")
            elif rc.state == "armed":
                hud_lines.append(f"RC State: ARMED")
            else:
                hud_lines.append(f"RC State: {rc_state}")

            if not rc.is_pose_valid:
                diag = f"Blocked (Pose: {rc.pose_rejection_reason})"
            elif not rc.armed:
                if rc.separation_start_time is not None:
                    diag = f"Arming ({int(rc.separation_progress * 100)}% {rc.separation_elapsed:.2f}/{rc.separation_duration:.2f}s)"
                else:
                    diag = f"Blocked (Separation: Need >={rc.release_threshold:.2f})"
            elif rc.state in ("clicked", "clicked_held"):
                diag = "Latched (Release to rearm)"
            elif rc.state == "confirming":
                diag = f"Confirming ({int(rc.confirmation_progress * 100)}% contact time)"
            elif rc.state == "approaching":
                diag = f"Approaching (Need contact <={rc.contact_threshold:.2f})"
            else:
                diag = f"Armed (Need contact <={rc.contact_threshold:.2f})"
            hud_lines.append(f"RC Status: {diag}")

        y_pos = 93
        for line in hud_lines:
            cv2.putText(
                frame,
                line,
                (28, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                COLOR_WHITE,
                1,
            )
            y_pos += 17
    else:
        cv2.rectangle(frame, (20, 75), (395, 258), COLOR_DARK_BG, -1)
        cv2.rectangle(frame, (20, 75), (395, 258), (80, 80, 80), 1)

        target_label = (
            detector.intended_target
            if detector.intended_target
            else ("PowerPoint Mode" if detector.target_mode_powerpoint else "Generic Mode")
        )
        supp_label = detector.suppression_reason if detector.suppression_reason else "None"
        act_label = detector.last_dispatched_action if detector.last_dispatched_action else "None"
        ss_status = (
            screenshot_ctrl.state if screenshot_ctrl
            else ("Active" if detector.screenshot_suppression else "Ready")
        )
        palm_valid_str = "Valid" if detector.is_valid_swipe_palm else "Invalid"

        hud_lines = [
            f"Target: {target_label}",
            f"Screenshot: {ss_status}",
            f"5-Finger Palm: {palm_valid_str}",
            f"Suppressed: {supp_label}",
            f"Dispatched: {act_label}",
            f"Swipe: {detector.debug_direction or 'None'}",
            f"Palm: ({detector.debug_palm_x:.2f}, {detector.debug_palm_y:.2f})  Speed: {detector.debug_speed:.2f}",
            f"DX: {detector.debug_dx:+.3f}  DY: {detector.debug_dy:+.3f}",
            f"FPS: {fps:.1f}",
        ]
        y_pos = 93
        for line in hud_lines:
            cv2.putText(
                frame,
                line,
                (28, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                COLOR_WHITE,
                1,
            )
            y_pos += 17

    hud_now = now_mono if now_mono is not None else time.monotonic()
    ss_msg, ss_success = screenshot_ctrl.get_feedback(hud_now) if screenshot_ctrl else ("", False)
    action_age = hud_now - mouse_ctrl.last_action_time if mouse_ctrl else 0.0
    palm_action_age = hud_now - detector.last_dispatched_time if detector else 0.0

    if ss_msg:
        action_color = COLOR_GREEN if ss_success else COLOR_RED
        cv2.rectangle(frame, (20, 265), (420, 299), COLOR_DARK_BG, -1)
        cv2.rectangle(frame, (20, 265), (420, 299), action_color, 1)
        cv2.putText(
            frame,
            ss_msg,
            (28, 287),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            action_color,
            2,
        )
    elif mouse_ctrl and mouse_ctrl.last_action and 0.0 <= action_age < ACTION_DISPLAY_DURATION:
        action_text = f"Action: {mouse_ctrl.last_action}"
        action_color = COLOR_GREEN if "Click" in mouse_ctrl.last_action else COLOR_CYAN
        cv2.rectangle(frame, (20, 265), (380, 299), COLOR_DARK_BG, -1)
        cv2.rectangle(frame, (20, 265), (380, 299), action_color, 1)
        cv2.putText(
            frame,
            action_text,
            (28, 287),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            action_color,
            2,
        )
    elif detector and detector.last_dispatched_action and 0.0 <= palm_action_age < ACTION_DISPLAY_DURATION:
        action_name = (
            "Next"
            if ("Next" in detector.last_dispatched_action or "RIGHT" in detector.last_dispatched_action)
            else (
                "Previous"
                if ("Previous" in detector.last_dispatched_action or "LEFT" in detector.last_dispatched_action)
                else detector.last_dispatched_action
            )
        )
        action_text = f"Action: {action_name}"
        cv2.rectangle(frame, (20, 226), (380, 260), COLOR_DARK_BG, -1)
        cv2.rectangle(frame, (20, 226), (380, 260), COLOR_GREEN, 1)
        cv2.putText(
            frame,
            action_text,
            (28, 248),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_GREEN,
            2,
        )
    elif screenshot_ctrl and (screenshot_ctrl.is_candidate or screenshot_ctrl.state == ScreenshotState.CAPTURED):
        cv2.rectangle(frame, (20, 226), (355, 260), COLOR_DARK_BG, -1)
        cv2.rectangle(frame, (20, 226), (355, 260), COLOR_CYAN, 1)
        cv2.putText(
            frame,
            "Swipe Down: Capture Screen",
            (28, 248),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_CYAN,
            1,
        )
    elif state_label == GestureState.CURSOR_CONTROL:
        cv2.rectangle(frame, (20, 226), (355, 260), COLOR_DARK_BG, -1)
        cv2.rectangle(frame, (20, 226), (355, 260), COLOR_CYAN, 1)
        if mouse_ctrl and mouse_ctrl.right_click_ctrl.is_approaching:
            rc_state = mouse_ctrl.right_click_ctrl.state
            if rc_state in ("confirming", "clicked_held"):
                rc_text = "Right Click: Contact"
                rc_color = COLOR_GREEN
            elif rc_state == "clicked":
                rc_text = "Right Click: Clicked"
                rc_color = COLOR_GREEN
            else:
                rc_text = "Right Click: Approach"
                rc_color = COLOR_YELLOW
            cv2.putText(
                frame,
                rc_text,
                (28, 248),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                rc_color,
                1,
            )
        else:
            cv2.putText(
                frame,
                "Index: Move | Thumb: Click",
                (28, 248),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_CYAN,
                1,
            )
    elif state_label == GestureState.SCROLL_CONTROL:
        cv2.rectangle(frame, (20, 226), (355, 260), COLOR_DARK_BG, -1)
        cv2.rectangle(frame, (20, 226), (355, 260), COLOR_MAGENTA, 1)
        cv2.putText(
            frame,
            "2-Finger Vertical Scroll",
            (28, 248),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_MAGENTA,
            1,
        )

    cv2.putText(
        frame,
        "Press 'Q' to quit (or click X)",
        (20, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )


def draw_skeleton(frame, hand_landmarks, w, h):
    """Draws hand landmark joints and connections."""
    for lm in hand_landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, COLOR_GREEN, 1)

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20),
        (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    ]
    for s_idx, e_idx in connections:
        s = hand_landmarks[s_idx]
        e = hand_landmarks[e_idx]
        cv2.line(
            frame,
            (int(s.x * w), int(s.y * h)),
            (int(e.x * w), int(e.y * h)),
            COLOR_GREEN,
            1,
        )


def main():
    model_path = os.path.join(
        os.path.dirname(__file__), "models", "hand_landmarker.task"
    )

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run: python setup_models.py")
        return

    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = PalmSwipeDetector()
    mouse_ctrl = MouseController()
    screenshot_ctrl = ScreenshotController()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_INITIAL_WIDTH, WINDOW_INITIAL_HEIGHT)

    print("=" * 60)
    print("Palm Gesture Controller started (VIDEO mode).")
    print("Press Q to quit.")
    print("=" * 60)

    last_timestamp_ms = 0
    fps_history = deque(maxlen=20)
    last_frame_time = time.monotonic()
    consecutive_empty_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_empty_frames += 1
                if consecutive_empty_frames > 30:
                    print("Failed to read frame from webcam repeatedly.")
                    break
                time.sleep(0.02)
                continue
            consecutive_empty_frames = 0

            now_mono = time.monotonic()
            dt = now_mono - last_frame_time
            last_frame_time = now_mono
            if dt > 0:
                fps_history.append(1.0 / dt)
            current_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0

            timestamp_ms = max(last_timestamp_ms + 1, int(now_mono * 1000))
            last_timestamp_ms = timestamp_ms

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            if detection_result.hand_landmarks:
                hand_landmarks = detection_result.hand_landmarks[0]
                draw_skeleton(frame, hand_landmarks, w, h)

                ss_state = screenshot_ctrl.update(hand_landmarks, timestamp=now_mono)
                ss_suppressing = (
                    screenshot_ctrl.is_suppressing_swipes
                    or screenshot_ctrl.is_candidate
                    or (ss_state == ScreenshotState.CAPTURED)
                    or is_three_finger_screenshot_pose(hand_landmarks)
                )

                if ss_suppressing:
                    mouse_ctrl.on_pose_lost()
                    detector.reset()
                    detector.update(
                        hand_landmarks,
                        active_mouse_mode="IDLE",
                        screenshot_active=True,
                    )
                    mode = "IDLE"
                    clicked = False
                    scroll_val = 0
                    scrolled = False
                    state_label = screenshot_ctrl.get_hud_label()

                    # Highlight the three tracked fingertips (Index 8, Middle 12, Ring 16)
                    tip8 = hand_landmarks[8]
                    tip12 = hand_landmarks[12]
                    tip16 = hand_landmarks[16]
                    t8x, t8y = int(tip8.x * w), int(tip8.y * h)
                    t12x, t12y = int(tip12.x * w), int(tip12.y * h)
                    t16x, t16y = int(tip16.x * w), int(tip16.y * h)

                    ss_color = COLOR_GREEN if ss_state == ScreenshotState.CAPTURED else COLOR_CYAN
                    cv2.circle(frame, (t8x, t8y), 11, ss_color, 2)
                    cv2.circle(frame, (t12x, t12y), 11, ss_color, 2)
                    cv2.circle(frame, (t16x, t16y), 11, ss_color, 2)
                    cv2.line(frame, (t8x, t8y), (t12x, t12y), ss_color, 2)
                    cv2.line(frame, (t12x, t12y), (t16x, t16y), ss_color, 2)

                    cx, cy = int(screenshot_ctrl.last_tip_pos[0] * w), int(screenshot_ctrl.last_tip_pos[1] * h)
                    cv2.circle(frame, (cx, cy), 5, COLOR_YELLOW, -1)
                else:
                    mode, clicked, scroll_val, scrolled = mouse_ctrl.process_landmarks(
                        hand_landmarks, timestamp=now_mono
                    )
                    if clicked or scrolled:
                        # Ensure only one gesture action is dispatched in this camera frame
                        detector.reset()
                    state_label = detector.update(
                        hand_landmarks,
                        active_mouse_mode=mode,
                        screenshot_active=False,
                    )

                    if mode == "SCROLL":
                        tip8 = hand_landmarks[8]
                        tip12 = hand_landmarks[12]
                        f8x, f8y = int(tip8.x * w), int(tip8.y * h)
                        f12x, f12y = int(tip12.x * w), int(tip12.y * h)
                        cv2.circle(frame, (f8x, f8y), 11, COLOR_MAGENTA, 2)
                        cv2.circle(frame, (f12x, f12y), 11, COLOR_MAGENTA, 2)
                        cv2.line(frame, (f8x, f8y), (f12x, f12y), COLOR_MAGENTA, 2)

                        neutral_y = mouse_ctrl.scroll_ctrl.neutral_y
                        if neutral_y is not None:
                            ny_px = int(neutral_y * h)
                            cv2.line(frame, (0, ny_px), (w, ny_px), COLOR_YELLOW, 1)
                    elif mode == "CURSOR":
                        mx, my = int(mouse_ctrl.margin_x * w), int(mouse_ctrl.margin_y * h)
                        cv2.rectangle(frame, (mx, my), (w - mx, h - my), (50, 50, 50), 1)

                        tip = hand_landmarks[8]
                        fx, fy = int(tip.x * w), int(tip.y * h)
                        cv2.circle(frame, (fx, fy), 12, COLOR_CYAN, 2)
                        cv2.circle(frame, (fx, fy), 4, COLOR_YELLOW, -1)

                        thumb_tip = hand_landmarks[4]
                        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                        if is_thumb_raised(hand_landmarks):
                            cv2.circle(frame, (tx, ty), 9, COLOR_GREEN if clicked else COLOR_YELLOW, -1)
                            cv2.circle(frame, (tx, ty), 12, COLOR_WHITE, 2)
                        else:
                            cv2.circle(frame, (tx, ty), 5, COLOR_CYAN, 1)

                        rc = mouse_ctrl.right_click_ctrl
                        if rc.is_approaching:
                            mid_tip = hand_landmarks[12]
                            mx, my = int(mid_tip.x * w), int(mid_tip.y * h)
                            pinch_color = (
                                COLOR_GREEN
                                if rc.state in ("confirming", "clicked", "clicked_held")
                                else COLOR_YELLOW
                            )
                            cv2.circle(frame, (tx, ty), 9, pinch_color, -1)
                            cv2.circle(frame, (tx, ty), 12, COLOR_WHITE, 2)
                            cv2.circle(frame, (mx, my), 9, pinch_color, -1)
                            cv2.circle(frame, (mx, my), 12, COLOR_WHITE, 2)
                            cv2.line(frame, (tx, ty), (mx, my), pinch_color, 2)
                    else:
                        draw_palm_center(frame, detector.debug_palm_x, detector.debug_palm_y, w, h)
                        draw_palm_trajectory(frame, detector.palm_history, w, h)
            else:
                screenshot_ctrl.update(None, timestamp=now_mono)
                mode, clicked, scroll_val, scrolled = mouse_ctrl.process_landmarks(
                    None, timestamp=now_mono
                )
                detector.on_hand_lost()
                if screenshot_ctrl.state == ScreenshotState.CAPTURED:
                    state_label = screenshot_ctrl.get_hud_label()
                elif mode in ("CURSOR", "SCROLL"):
                    state_label = (
                        GestureState.CURSOR_CONTROL if mode == "CURSOR"
                        else GestureState.SCROLL_CONTROL
                    )
                else:
                    state_label = detector.state

            draw_hud(
                frame,
                detector,
                state_label,
                mouse_ctrl=mouse_ctrl,
                screenshot_ctrl=screenshot_ctrl,
                fps=current_fps,
                now_mono=now_mono,
            )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("Quit requested.")
                break

            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
                    print("Window closed by user.")
                    break
            except cv2.error:
                print("Window closed by user.")
                break

    except pyautogui.FailSafeException:
        print("\n" + "=" * 60)
        print("EMERGENCY STOP: PyAutoGUI fail-safe triggered.")
        print("Mouse was moved to screen corner. Exiting safely.")
        print("=" * 60)
    finally:
        hand_landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        if "screenshot_ctrl" in locals() and screenshot_ctrl is not None:
            screenshot_ctrl.close()
        print("Application stopped cleanly.")


if __name__ == "__main__":
    main()
