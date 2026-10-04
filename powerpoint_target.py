"""
PowerPoint presentation targeting and navigation module.

Handles:
- Identification of running PowerPoint presentations (Slide Show mode, Presenter View, or Active Window).
- Reliable foreground window activation on Windows bypassing OS focus locks.
- Dispatching slide navigation keys (PageDown for Next, PageUp for Previous) exclusively to PowerPoint.
- Graceful error reporting when no presentation is running, avoiding sending stray keys to other applications.
- PyAutoGUI fail-safe propagation for emergency stop.
"""

import os
import platform
import time
import pyautogui


class PowerPointTarget:
    """Finds, focuses, and sends slide navigation commands to PowerPoint presentations."""

    def __init__(
        self,
        pyautogui_module=pyautogui,
        user32_module=None,
        preview_window_title: str = "Palm Gesture Controller",
    ):
        self.pag = pyautogui_module
        self.preview_window_title = preview_window_title

        if user32_module is not None:
            self.user32 = user32_module
        elif platform.system() == "Windows":
            import ctypes
            self.user32 = ctypes.windll.user32
        else:
            self.user32 = None

        self.last_target_type: str = "None"
        self.last_target_hwnd: int | None = None
        self.last_error: str = ""

    def find_presentation_window(self) -> tuple[int | None, str]:
        """
        Scan top-level windows for PowerPoint presentation targets.
        Priority:
          1. Slide Show mode (class "screenClass" or title containing "Slide Show")
          2. Presenter View (title containing "Presenter View")
          3. Main PowerPoint window (class "PPTFrameClass" or title containing "PowerPoint")
        Returns:
          (hwnd, target_type) e.g. (12345, "Slide Show") or (None, "None")
        """
        if self.user32 is None:
            return None, "None"

        import ctypes
        from ctypes import wintypes

        candidates = []

        def enum_cb(hwnd, _):
            try:
                if not self.user32.IsWindowVisible(hwnd):
                    return True

                length = self.user32.GetWindowTextLengthW(hwnd)
                if length <= 0:
                    return True

                buf = ctypes.create_unicode_buffer(length + 1)
                self.user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value.strip()

                if not title or title == self.preview_window_title:
                    return True

                cls_buf = ctypes.create_unicode_buffer(256)
                self.user32.GetClassNameW(hwnd, cls_buf, 256)
                cls_name = cls_buf.value.strip()

                title_lower = title.lower()
                cls_lower = cls_name.lower()

                # Priority 1: Presenter View
                if "presenter view" in title_lower or "mode présentateur" in title_lower:
                    candidates.append((1, hwnd, "Presenter View", title))
                # Priority 2: Slide Show mode
                elif "screenclass" in cls_lower or "slide show" in title_lower or "diaporama" in title_lower:
                    candidates.append((2, hwnd, "Slide Show", title))
                # Priority 3: Active PowerPoint presentation window
                elif "pptframeclass" in cls_lower or ("powerpoint" in title_lower and not "gesture" in title_lower):
                    candidates.append((3, hwnd, "PowerPoint Window", title))

            except Exception:
                pass
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        cb = WNDENUMPROC(enum_cb)
        try:
            self.user32.EnumWindows(cb, 0)
        except Exception:
            pass

        if not candidates:
            self.last_target_type = "None"
            self.last_target_hwnd = None
            return None, "None"

        # Sort by priority rank (1 is highest)
        candidates.sort(key=lambda c: c[0])
        best = candidates[0]
        self.last_target_hwnd = best[1]
        self.last_target_type = best[2]
        return best[1], best[2]

    def focus_presentation(self, hwnd: int) -> bool:
        """
        Brings the specified PowerPoint window to the foreground.
        Uses thread input attachment and key event unlock to reliably bypass Windows foreground lock.
        """
        if self.user32 is None or not hwnd:
            return False

        try:
            fg_hwnd = self.user32.GetForegroundWindow()
            if fg_hwnd == hwnd:
                return True

            # If minimized, restore it
            if self.user32.IsIconic(hwnd):
                self.user32.ShowWindow(hwnd, 9)  # 9 = SW_RESTORE

            # Attach thread input to bypass foreground lock if needed
            cur_thread = self.user32.GetWindowThreadProcessId(fg_hwnd, None) if fg_hwnd else 0
            target_thread = self.user32.GetWindowThreadProcessId(hwnd, None)

            attached = False
            if cur_thread and target_thread and cur_thread != target_thread:
                attached = bool(self.user32.AttachThreadInput(cur_thread, target_thread, True))

            try:
                # Unlock SetForegroundWindow if needed using ALT keystroke simulation
                try:
                    self.user32.keybd_event(0x12, 0, 0, 0)      # ALT down
                    self.user32.keybd_event(0x12, 0, 2, 0)      # ALT up
                except Exception:
                    pass

                self.user32.BringWindowToTop(hwnd)
                self.user32.SetForegroundWindow(hwnd)
                self.user32.ShowWindow(hwnd, 5)  # 5 = SW_SHOW
            finally:
                if attached:
                    self.user32.AttachThreadInput(cur_thread, target_thread, False)

            time.sleep(0.03)
            new_fg = self.user32.GetForegroundWindow()
            if new_fg == hwnd:
                return True

            time.sleep(0.02)
            return self.user32.GetForegroundWindow() == hwnd
        except Exception as exc:
            self.last_error = f"Focus error: {exc}"
            return False

    def navigate(self, direction: str) -> tuple[bool, str]:
        """
        Brings PowerPoint presentation into focus and dispatches navigation key.
        - direction == "RIGHT": PageDown (Next slide/animation)
        - direction == "LEFT": PageUp (Previous slide/animation)
        Returns:
          (success: bool, message: str)
        """
        hwnd, target_type = self.find_presentation_window()
        if not hwnd:
            msg = "PowerPoint presentation not found (neither Slide Show nor Presenter View active)"
            self.last_error = msg
            return False, msg

        if not self.focus_presentation(hwnd):
            msg = f"Failed to bring PowerPoint {target_type} (HWND {hwnd}) to foreground"
            self.last_error = msg
            return False, msg

        # Ensure PyAutoGUI fail-safe propagates directly
        try:
            if direction.upper() in ("RIGHT", "SWIPE RIGHT"):
                try:
                    self.pag.press("pagedown", _pause=False)
                except TypeError:
                    self.pag.press("pagedown")
                return True, f"Next slide (PageDown) -> {target_type}"
            elif direction.upper() in ("LEFT", "SWIPE LEFT"):
                try:
                    self.pag.press("pageup", _pause=False)
                except TypeError:
                    self.pag.press("pageup")
                return True, f"Previous slide (PageUp) -> {target_type}"
            else:
                return False, f"Unsupported navigation direction: {direction}"
        except self.pag.FailSafeException:
            raise
        except Exception as exc:
            msg = f"Key dispatch error: {exc}"
            self.last_error = msg
            return False, msg
