"""
Screenshot Controller - 3-Finger Downward Swipe.

Implements the three-finger downward swipe gesture for screen capture:
- Pose: Index (8), Middle (12), Ring (16) extended; Pinky (20) folded into palm.
        Thumb (4) unrestricted.
- Confirmation: Confirmed for ~120 ms before stroke recognition begins.
- Downward Stroke: Track average position of fingertips 8, 12, 16.
                   Downward movement is increasing Y in mirrored feed.
                   Requires dy >= 0.12, dy >= 1.5 * abs(dx), velocity >= 0.25/s within 0.8s.
- Single capture per stroke.
- Rearming: Requires leaving the 3-finger pose for ~150 ms before rearming to READY.
- Saves PNG in screenshots/ directory relative to project root.
- Propagates PyAutoGUI FailSafeException emergency stop.
"""

import os
import sys
import math
import time
import datetime
from collections import deque
import pyautogui

from mouse_controller import (
    is_three_finger_screenshot_pose,
)
from desktop_overlay import (
    DesktopOverlayManager,
    DEFAULT_FLASH_ENABLED,
    DEFAULT_FLASH_PEAK_OPACITY,
    DEFAULT_FLASH_DURATION,
    DEFAULT_NOTIFICATION_ENABLED,
    DEFAULT_NOTIFICATION_DURATION,
    DEFAULT_NOTIFICATION_FADE_DURATION,
)

SCREENSHOT_POSE_CONFIRM_TIME: float = 0.12
SCREENSHOT_MIN_DOWNWARD_DISPLACEMENT: float = 0.12
SCREENSHOT_VERTICAL_DOMINANCE_RATIO: float = 1.5
SCREENSHOT_MIN_VELOCITY: float = 0.25
SCREENSHOT_MAX_STROKE_DURATION: float = 0.80
SCREENSHOT_RELEASE_TIME: float = 0.15
SCREENSHOT_FEEDBACK_DURATION: float = 2.0

SCREENSHOT_FLASH_ENABLED: bool = DEFAULT_FLASH_ENABLED
SCREENSHOT_FLASH_PEAK_OPACITY: float = DEFAULT_FLASH_PEAK_OPACITY
SCREENSHOT_FLASH_DURATION: float = DEFAULT_FLASH_DURATION
SCREENSHOT_NOTIFICATION_ENABLED: bool = DEFAULT_NOTIFICATION_ENABLED
SCREENSHOT_NOTIFICATION_DURATION: float = DEFAULT_NOTIFICATION_DURATION
SCREENSHOT_NOTIFICATION_FADE_DURATION: float = DEFAULT_NOTIFICATION_FADE_DURATION


class ScreenshotState:
    READY = "READY"
    CONFIRMING = "CONFIRMING"
    TRACKING_STROKE = "TRACKING_STROKE"
    CAPTURED = "CAPTURED"


def get_three_fingertip_center(hand_landmarks) -> tuple[float, float]:
    """Calculate average position of fingertips 8 (index), 12 (middle), and 16 (ring)."""
    tx = (hand_landmarks[8].x + hand_landmarks[12].x + hand_landmarks[16].x) / 3.0
    ty = (hand_landmarks[8].y + hand_landmarks[12].y + hand_landmarks[16].y) / 3.0
    return tx, ty


class ScreenshotController:
    """State machine and stroke tracker recognizing 3-finger downward swipe for screenshot."""

    def __init__(
        self,
        pose_confirm_time: float = SCREENSHOT_POSE_CONFIRM_TIME,
        min_downward_displacement: float = SCREENSHOT_MIN_DOWNWARD_DISPLACEMENT,
        vertical_dominance_ratio: float = SCREENSHOT_VERTICAL_DOMINANCE_RATIO,
        min_velocity: float = SCREENSHOT_MIN_VELOCITY,
        max_stroke_duration: float = SCREENSHOT_MAX_STROKE_DURATION,
        release_time: float = SCREENSHOT_RELEASE_TIME,
        flash_enabled: bool = SCREENSHOT_FLASH_ENABLED,
        flash_peak_opacity: float = SCREENSHOT_FLASH_PEAK_OPACITY,
        flash_duration: float = SCREENSHOT_FLASH_DURATION,
        notification_enabled: bool = SCREENSHOT_NOTIFICATION_ENABLED,
        notification_duration: float = SCREENSHOT_NOTIFICATION_DURATION,
        notification_fade_duration: float = SCREENSHOT_NOTIFICATION_FADE_DURATION,
        screenshots_dir: str | None = None,
        pyautogui_module=pyautogui,
        overlay_manager: DesktopOverlayManager | None = None,
    ):
        self.pose_confirm_time = pose_confirm_time
        self.min_downward_displacement = min_downward_displacement
        self.vertical_dominance_ratio = vertical_dominance_ratio
        self.min_velocity = min_velocity
        self.max_stroke_duration = max_stroke_duration
        self.release_time = release_time
        self.flash_enabled = flash_enabled
        self.flash_peak_opacity = flash_peak_opacity
        self.flash_duration = flash_duration
        self.notification_enabled = notification_enabled
        self.notification_duration = notification_duration
        self.notification_fade_duration = notification_fade_duration
        self.pag = pyautogui_module

        if overlay_manager is not None:
            self.overlay_manager = overlay_manager
        else:
            try:
                self.overlay_manager = DesktopOverlayManager(
                    flash_enabled=flash_enabled,
                    flash_peak_opacity=flash_peak_opacity,
                    flash_duration=flash_duration,
                    notification_enabled=notification_enabled,
                    notification_duration=notification_duration,
                    notification_fade_duration=notification_fade_duration,
                )
            except Exception as e:
                print(f"[Screenshot] Desktop overlay init error: {e}")
                self.overlay_manager = None

        if screenshots_dir is not None:
            self.screenshots_dir = screenshots_dir
        else:
            self.screenshots_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "screenshots"
            )

        self.state: str = ScreenshotState.READY
        self.pose_start_time: float | None = None
        self.release_start_time: float | None = None

        self.stroke_history: deque = deque()

        self.current_dy: float = 0.0
        self.current_dx: float = 0.0
        self.current_velocity: float = 0.0
        self.last_tip_pos: tuple[float, float] = (0.0, 0.0)

        self.last_saved_path: str = ""
        self.status_message: str = ""
        self.status_time: float = 0.0
        self.status_success: bool = False
        self.last_capture_time: float = 0.0

    @property
    def is_candidate(self) -> bool:
        """Return True when 3-finger pose is present (even during confirmation) or stroke is active."""
        return self.state in (ScreenshotState.CONFIRMING, ScreenshotState.TRACKING_STROKE)

    @property
    def is_active(self) -> bool:
        """Return True while tracking stroke or confirming pose."""
        return self.state in (ScreenshotState.CONFIRMING, ScreenshotState.TRACKING_STROKE)

    @property
    def is_suppressing_swipes(self) -> bool:
        """
        Return True if ordinary swipes should be suppressed:
        - 3-finger pose is present or confirming (is_candidate).
        - In stroke tracking or capture state (CAPTURED).
        - Releasing / rearming stage (release_start_time is not None).
        - Within settle window (0.35s) following capture.
        """
        now = time.monotonic()
        if self.is_candidate or self.state == ScreenshotState.CAPTURED:
            return True
        if self.release_start_time is not None:
            return True
        if (now - self.last_capture_time) < 0.35:
            return True
        return False

    @property
    def confirmation_progress(self) -> float:
        """Return progress (0.0 to 1.0) towards 120 ms pose confirmation."""
        if self.state == ScreenshotState.CONFIRMING and self.pose_start_time is not None:
            now = time.monotonic()
            elapsed = now - self.pose_start_time
            return min(1.0, max(0.0, elapsed / self.pose_confirm_time))
        if self.state in (ScreenshotState.TRACKING_STROKE, ScreenshotState.CAPTURED):
            return 1.0
        return 0.0

    @property
    def displacement_progress(self) -> float:
        """Return progress (0.0 to 1.0) towards minimum downward displacement."""
        if self.min_downward_displacement <= 0:
            return 1.0
        return min(1.0, max(0.0, self.current_dy / self.min_downward_displacement))

    @property
    def downward_displacement(self) -> float:
        """Return current downward displacement dy measured from apex."""
        return self.current_dy

    def get_feedback(self, now: float | None = None) -> tuple[str, bool]:
        """Return (message, is_success) if feedback duration has not elapsed."""
        t = now if now is not None else time.monotonic()
        if self.status_message and (t - self.status_time) < SCREENSHOT_FEEDBACK_DURATION:
            return self.status_message, self.status_success
        return "", False

    @property
    def feedback_message(self) -> str:
        """Return temporary status feedback message if active."""
        return self.get_feedback()[0]

    @property
    def feedback_success(self) -> bool:
        """Return True if last feedback was success."""
        return self.get_feedback()[1]

    def reset_stroke(self):
        """Clear stroke history and instantaneous displacement metrics."""
        self.stroke_history.clear()
        self.current_dy = 0.0
        self.current_dx = 0.0
        self.current_velocity = 0.0

    def get_hud_label(self) -> str:
        """Return human-readable state label for preview HUD."""
        if self.state == ScreenshotState.CONFIRMING:
            pct = int(self.confirmation_progress * 100)
            return f"Screenshot: Confirming ({pct}%)"
        if self.state == ScreenshotState.TRACKING_STROKE:
            disp_pct = int(self.displacement_progress * 100)
            return f"Screenshot: Downward Stroke ({disp_pct}%) [dy:{self.current_dy:+.2f}]"
        if self.state == ScreenshotState.CAPTURED:
            return "Screenshot: Captured (Release pose to rearm)"
        return "Screenshot: Ready"

    def _capture_and_save(self, now: float) -> bool:
        """Capture primary display, save timestamped PNG, and record status."""
        failsafe_cls = getattr(self.pag, "FailSafeException", pyautogui.FailSafeException)
        if not (isinstance(failsafe_cls, type) and issubclass(failsafe_cls, BaseException)):
            failsafe_cls = pyautogui.FailSafeException

        try:
            os.makedirs(self.screenshots_dir, exist_ok=True)
            img = self.pag.screenshot()
            if img is None:
                raise RuntimeError("Screenshot capture returned None")

            now_dt = datetime.datetime.now()
            filename = f"screenshot_{now_dt.strftime('%Y%m%d_%H%M%S_%f')}.png"
            filepath = os.path.join(self.screenshots_dir, filename)
            img.save(filepath, "PNG")

            self.last_saved_path = filepath
            self.status_message = f"Screenshot saved: {filename}"
            self.status_time = now
            self.status_success = True
            self.last_capture_time = now
            print(f"[Screenshot] Saved: {filepath}")

            if self.overlay_manager is not None:
                try:
                    self.overlay_manager.show_success(filename=filename, filepath=filepath)
                except Exception as anim_exc:
                    print(f"[Screenshot] Overlay notification warning: {anim_exc}")

            return True

        except failsafe_cls:
            raise
        except Exception as exc:
            self.last_saved_path = ""
            self.status_message = f"Screenshot failed: {exc}"
            self.status_time = now
            self.status_success = False
            print(f"[Screenshot] Error during capture/save: {exc}")

            if self.overlay_manager is not None:
                try:
                    self.overlay_manager.show_error(error_message=str(exc))
                except Exception as anim_exc:
                    print(f"[Screenshot] Overlay notification warning: {anim_exc}")

            return False

    def close(self):
        """Clean up overlay manager and background thread."""
        if self.overlay_manager is not None:
            try:
                self.overlay_manager.close()
            except Exception:
                pass

    def update(self, hand_landmarks, timestamp: float | None = None) -> str:
        """
        Process current frame landmarks and evaluate 3-finger downward stroke.
        Returns the active state label.
        """
        now = timestamp if timestamp is not None else time.monotonic()

        if hand_landmarks is None or len(hand_landmarks) < 21:
            self.reset_stroke()
            self.pose_start_time = None
            if self.state != ScreenshotState.CAPTURED:
                self.state = ScreenshotState.READY
            self.release_start_time = None
            return self.state

        is_pose = is_three_finger_screenshot_pose(hand_landmarks)
        tip_x, tip_y = get_three_fingertip_center(hand_landmarks)
        self.last_tip_pos = (tip_x, tip_y)

        if self.state == ScreenshotState.CAPTURED:
            if not is_pose:
                if self.release_start_time is None:
                    self.release_start_time = now
                elif (now - self.release_start_time) >= (self.release_time - 1e-5):
                    self.state = ScreenshotState.READY
                    self.release_start_time = None
                    self.pose_start_time = None
                    self.reset_stroke()
            else:
                self.release_start_time = None
            return self.state

        if not is_pose:
            self.reset_stroke()
            self.pose_start_time = None
            self.state = ScreenshotState.READY
            return self.state

        if self.pose_start_time is None:
            self.pose_start_time = now
            self.reset_stroke()
            self.stroke_history.append((tip_x, tip_y, now))
            self.state = ScreenshotState.CONFIRMING
            return self.state

        self.stroke_history.append((tip_x, tip_y, now))

        while self.stroke_history and (now - self.stroke_history[0][2] > self.max_stroke_duration):
            self.stroke_history.popleft()

        elapsed_pose = now - self.pose_start_time
        if elapsed_pose < (self.pose_confirm_time - 1e-5):
            self.state = ScreenshotState.CONFIRMING
            return self.state

        self.state = ScreenshotState.TRACKING_STROKE

        if len(self.stroke_history) < 2:
            return self.state

        min_idx = min(range(len(self.stroke_history)), key=lambda i: self.stroke_history[i][1])
        start_x, start_y, start_t = self.stroke_history[min_idx]
        current_x, current_y, current_t = self.stroke_history[-1]

        dt = current_t - start_t
        dy = current_y - start_y
        dx = current_x - start_x
        dist = math.hypot(dx, dy)
        velocity = (dist / dt) if dt > 1e-5 else 0.0

        self.current_dy = max(0.0, dy)
        self.current_dx = dx
        self.current_velocity = velocity

        is_downward_enough = dy >= self.min_downward_displacement
        is_vertically_dominant = dy >= (self.vertical_dominance_ratio * abs(dx))
        is_fast_enough = velocity >= self.min_velocity
        is_within_time = (0.0 < dt <= self.max_stroke_duration)

        if is_downward_enough and is_vertically_dominant and is_fast_enough and is_within_time:
            self._capture_and_save(now)
            self.state = ScreenshotState.CAPTURED
            self.reset_stroke()
            self.pose_start_time = None
            self.release_start_time = None
            return self.state

        return self.state
