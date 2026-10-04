"""
Desktop Overlay Manager for Screenshot Feedback.

Provides non-intrusive, click-through desktop visual feedback:
1. Brief screen flash animation (white translucent overlay with ~18% peak opacity fading out over ~250ms).
2. Compact top-right notification card showing "Screenshot saved" + filename (or error message).
3. Non-interactive, click-through window styles:
   WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
4. Zero focus stealing, no taskbar presence, no interruption to PowerPoint or active applications.
5. Operates in a dedicated daemon UI thread with non-blocking elapsed-time animation steps.
"""

import ctypes
import os
import platform
import queue
import sys
import threading
import time
import tkinter as tk

DEFAULT_FLASH_ENABLED: bool = True
DEFAULT_FLASH_PEAK_OPACITY: float = 0.18
DEFAULT_FLASH_DURATION: float = 0.25
DEFAULT_NOTIFICATION_ENABLED: bool = True
DEFAULT_NOTIFICATION_DURATION: float = 1.5
DEFAULT_NOTIFICATION_FADE_DURATION: float = 0.3


class DesktopOverlayManager:
    """
    Manages desktop-level visual feedback overlays for screenshot captures.
    Runs a lightweight, non-blocking Tkinter event loop in a dedicated background thread.
    """

    def __init__(
        self,
        flash_enabled: bool = DEFAULT_FLASH_ENABLED,
        flash_peak_opacity: float = DEFAULT_FLASH_PEAK_OPACITY,
        flash_duration: float = DEFAULT_FLASH_DURATION,
        notification_enabled: bool = DEFAULT_NOTIFICATION_ENABLED,
        notification_duration: float = DEFAULT_NOTIFICATION_DURATION,
        notification_fade_duration: float = DEFAULT_NOTIFICATION_FADE_DURATION,
        autostart: bool = True,
    ):
        self.flash_enabled: bool = flash_enabled
        self.flash_peak_opacity: float = float(flash_peak_opacity)
        self.flash_duration: float = float(flash_duration)
        self.notification_enabled: bool = notification_enabled
        self.notification_duration: float = float(notification_duration)
        self.notification_fade_duration: float = float(notification_fade_duration)

        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running: bool = False
        self._ready_event: threading.Event = threading.Event()
        self._root: tk.Tk | None = None
        self._flash_win: tk.Toplevel | None = None
        self._notify_win: tk.Toplevel | None = None

        self.flash_trigger_count: int = 0
        self.notification_trigger_count: int = 0
        self.last_triggered_type: str = ""
        self.last_triggered_detail: str = ""

        if autostart:
            self.start()

    def start(self):
        """Starts the background overlay thread."""
        if self._running:
            return
        self._running = True
        self._ready_event.clear()
        self._thread = threading.Thread(
            target=self._ui_thread_main,
            name="DesktopOverlayThread",
            daemon=True,
        )
        self._thread.start()
        self._ready_event.wait(timeout=1.0)

    def show_success(self, filename: str, filepath: str = ""):
        """Post a success notification and flash request to the overlay thread."""
        self.flash_trigger_count += 1 if self.flash_enabled else 0
        self.notification_trigger_count += 1 if self.notification_enabled else 0
        self.last_triggered_type = "SUCCESS"
        self.last_triggered_detail = filename
        if self._running:
            self._queue.put(("SUCCESS", filename))

    def show_error(self, error_message: str):
        """Post an error notification request (no flash) to the overlay thread."""
        self.notification_trigger_count += 1 if self.notification_enabled else 0
        self.last_triggered_type = "ERROR"
        self.last_triggered_detail = error_message
        if self._running:
            self._queue.put(("ERROR", error_message))

    def close(self):
        """Clean up all overlay windows and stop the background thread."""
        if not self._running:
            return
        self._running = False
        try:
            self._queue.put(("STOP", None))
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.6)

    def _make_click_through(self, window: tk.Toplevel):
        """
        Applies Win32 extended window styles to make the overlay completely
        click-through, non-activating, and excluded from taskbar / Alt-Tab.
        """
        if platform.system() != "Windows":
            return

        try:
            user32 = ctypes.windll.user32
            window.update_idletasks()
            hwnd = user32.GetParent(window.winfo_id()) or window.winfo_id()
            if not hwnd:
                return

            GWL_EXSTYLE = -20
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_LAYERED = 0x00080000
            WS_EX_NOACTIVATE = 0x08000000
            WS_EX_TOOLWINDOW = 0x00000080
            SW_SHOWNOACTIVATE = 4

            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(
                hwnd,
                GWL_EXSTYLE,
                style | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
            )
            user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
        except Exception:
            pass

    def _ui_thread_main(self):
        try:
            root = tk.Tk()
            root.withdraw()
            self._root = root
            self._ready_event.set()

            def poll_queue():
                if not self._running:
                    self._cleanup_windows()
                    try:
                        root.quit()
                    except Exception:
                        pass
                    return

                try:
                    while not self._queue.empty():
                        msg_type, data = self._queue.get_nowait()
                        if msg_type == "SUCCESS":
                            if self.flash_enabled:
                                self._trigger_flash()
                            if self.notification_enabled:
                                self._trigger_notification(
                                    is_success=True,
                                    title="Screenshot saved",
                                    detail=data,
                                )
                        elif msg_type == "ERROR":
                            if self.notification_enabled:
                                self._trigger_notification(
                                    is_success=False,
                                    title="Screenshot failed",
                                    detail=data,
                                )
                        elif msg_type == "STOP":
                            self._cleanup_windows()
                            try:
                                root.quit()
                            except Exception:
                                pass
                            return
                except Exception:
                    pass

                if self._running:
                    root.after(16, poll_queue)

            root.after(16, poll_queue)
            root.mainloop()
            self._cleanup_windows()
        except Exception:
            self._ready_event.set()
        finally:
            self._running = False
            self._root = None

    def _cleanup_windows(self):
        """Destroy any active flash or notification window."""
        if self._flash_win:
            try:
                self._flash_win.destroy()
            except Exception:
                pass
            self._flash_win = None

        if self._notify_win:
            try:
                self._notify_win.destroy()
            except Exception:
                pass
            self._notify_win = None

    def _trigger_flash(self):
        """Spawns fullscreen translucent white flash overlay and fades it out."""
        if not self._root:
            return

        if self._flash_win:
            try:
                self._flash_win.destroy()
            except Exception:
                pass
            self._flash_win = None

        try:
            sw = self._root.winfo_screenwidth()
            sh = self._root.winfo_screenheight()
            if platform.system() == "Windows":
                try:
                    user32 = ctypes.windll.user32
                    sw = user32.GetSystemMetrics(0) or sw
                    sh = user32.GetSystemMetrics(1) or sh
                except Exception:
                    pass

            win = tk.Toplevel(self._root)
            win.overrideredirect(True)
            win.attributes("-topmost", True)
            win.attributes("-alpha", self.flash_peak_opacity)
            win.geometry(f"{sw}x{sh}+0+0")
            win.configure(bg="#FFFFFF")

            self._make_click_through(win)
            self._flash_win = win

            start_time = time.monotonic()
            duration = max(0.05, self.flash_duration)
            peak_alpha = self.flash_peak_opacity

            def animate():
                if not self._running or self._flash_win != win:
                    try:
                        win.destroy()
                    except Exception:
                        pass
                    return

                elapsed = time.monotonic() - start_time
                if elapsed >= duration:
                    try:
                        win.destroy()
                    except Exception:
                        pass
                    if self._flash_win == win:
                        self._flash_win = None
                else:
                    progress = elapsed / duration
                    current_alpha = peak_alpha * (1.0 - progress)
                    try:
                        win.attributes("-alpha", max(0.001, current_alpha))
                        self._root.after(16, animate)
                    except Exception:
                        if self._flash_win == win:
                            self._flash_win = None

            self._root.after(16, animate)
        except Exception as exc:
            print(f"[DesktopOverlay] Flash creation failed: {exc}")

    def _trigger_notification(self, is_success: bool, title: str, detail: str):
        """
        Spawns a sleek, compact desktop notification near the top-right corner.
        Holds for notification_duration, then smoothly fades out over fade_duration.
        """
        if not self._root:
            return

        if self._notify_win:
            try:
                self._notify_win.destroy()
            except Exception:
                pass
            self._notify_win = None

        try:
            card_w = 330
            card_h = 68
            sw = self._root.winfo_screenwidth()
            if platform.system() == "Windows":
                try:
                    user32 = ctypes.windll.user32
                    sw = user32.GetSystemMetrics(0) or sw
                except Exception:
                    pass

            pad_x = 24
            pad_y = 24
            pos_x = max(0, sw - card_w - pad_x)
            pos_y = pad_y

            win = tk.Toplevel(self._root)
            win.overrideredirect(True)
            win.attributes("-topmost", True)
            win.attributes("-alpha", 0.95)
            win.geometry(f"{card_w}x{card_h}+{pos_x}+{pos_y}")
            win.configure(bg="#1c1d22")

            border_color = "#388e3c" if is_success else "#d32f2f"
            icon_symbol = "✓" if is_success else "✕"
            icon_color = "#4caf50" if is_success else "#f44336"

            card = tk.Frame(
                win,
                bg="#1c1d22",
                highlightbackground=border_color,
                highlightthickness=1,
            )
            card.pack(fill="both", expand=True, padx=1, pady=1)

            icon_lbl = tk.Label(
                card,
                text=icon_symbol,
                font=("Segoe UI", 16, "bold"),
                fg=icon_color,
                bg="#1c1d22",
                width=2,
            )
            icon_lbl.pack(side="left", padx=(10, 6))

            text_frame = tk.Frame(card, bg="#1c1d22")
            text_frame.pack(side="left", fill="both", expand=True, pady=8)

            title_color = "#FFFFFF" if is_success else "#FF6B6B"
            title_lbl = tk.Label(
                text_frame,
                text=title,
                font=("Segoe UI", 10, "bold"),
                fg=title_color,
                bg="#1c1d22",
                anchor="w",
            )
            title_lbl.pack(fill="x")

            detail_str = os.path.basename(detail) if detail else ""
            if len(detail_str) > 42:
                detail_str = detail_str[:39] + "..."

            detail_lbl = tk.Label(
                text_frame,
                text=detail_str,
                font=("Segoe UI", 8),
                fg="#9da5b4",
                bg="#1c1d22",
                anchor="w",
            )
            detail_lbl.pack(fill="x")

            self._make_click_through(win)
            self._notify_win = win

            start_time = time.monotonic()
            hold_time = max(0.1, self.notification_duration)
            fade_time = max(0.05, self.notification_fade_duration)
            total_time = hold_time + fade_time

            def animate():
                if not self._running or self._notify_win != win:
                    try:
                        win.destroy()
                    except Exception:
                        pass
                    return

                elapsed = time.monotonic() - start_time
                if elapsed >= total_time:
                    try:
                        win.destroy()
                    except Exception:
                        pass
                    if self._notify_win == win:
                        self._notify_win = None
                elif elapsed > hold_time:
                    fade_progress = (elapsed - hold_time) / fade_time
                    current_alpha = 0.95 * (1.0 - fade_progress)
                    try:
                        win.attributes("-alpha", max(0.001, current_alpha))
                        self._root.after(20, animate)
                    except Exception:
                        if self._notify_win == win:
                            self._notify_win = None
                else:
                    try:
                        self._root.after(20, animate)
                    except Exception:
                        if self._notify_win == win:
                            self._notify_win = None

            self._root.after(20, animate)
        except Exception as exc:
            print(f"[DesktopOverlay] Notification creation failed: {exc}")
