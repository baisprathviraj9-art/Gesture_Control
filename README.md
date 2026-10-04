# Whole-PC Gesture Controller 🎮🖐️

Control your entire PC hands-free using real-time hand gestures and computer vision powered by **MediaPipe**, **OpenCV**, and **PyAutoGUI**.

This system supports seamless switching between whole-desktop swipe commands, smooth index-finger cursor navigation, thumb-triggered left clicking, and two-finger vertical scrolling.

---

## 🧠 Gesture Controls & Actions

### 1. Palm Swipes & PowerPoint Presentation Navigation (Open Palm)
Perform a clear, deliberate swipe with an open palm:

| Gesture | PowerPoint Mode (Default) | Generic Mode | Shortcut (Windows) | Shortcut (macOS) |
| :--- | :--- | :--- | :--- | :--- |
| **Swipe Right** | **Next Slide / Animation** | Next / Forward | `PageDown` (PPT) / `Right Arrow` | `Right Arrow` |
| **Swipe Left** | **Previous Slide / Animation** | Previous / Backward | `PageUp` (PPT) / `Left Arrow` | `Left Arrow` |
| **Swipe Up** | Task View / Desktop Overview | Task View / Desktop Overview | `Win + Tab` | `Ctrl + Up` |
| **Swipe Down** | Show / Minimize to Desktop | Show / Minimize to Desktop | `Win + D` | `F11` |

- **Intelligent PowerPoint Targeting**:
  - Automatically identifies active PowerPoint presentations by scanning for **Presenter View** (Priority 1) or full-screen **Slide Show** mode (`screenClass`, Priority 2).
  - Uses Windows Win32 API (`AttachThreadInput`, `BringWindowToTop`, `SetForegroundWindow`) to ensure the presentation window receives keystrokes even if the camera preview window was previously focused.
  - The camera preview window never steals focus back.
  - If PowerPoint is not running or focus acquisition fails, reports the failure gracefully without sending unintended keystrokes to any other application.
  - Does not start, restart, or close presentations automatically.
- **Filtering & Duplicate Prevention**:
  - Requires at least 0.10 normalized displacement, a 1.25x dominant axis ratio (rejects diagonals), and minimum swipe velocity.
  - **Single Action Per Swipe**: 0.6s cooldown and immediate history clearing guarantee one navigation command per physical swipe.
  - **Opposite Return-Stroke Suppression**: Tracks the originating coordinates of subsequent strokes; rapid return movement back towards the starting position within 1.2s without settling is strictly rejected to prevent unintended slide reversals.
  - **Camera Feed Mirroring**: The camera view is mirrored (`cv2.flip`), ensuring your physical right swipe advances to the NEXT slide and left swipe moves to the PREVIOUS slide.

---

### 2. Mouse Cursor Control (One Finger)
- **Pose**: Index finger extended, middle/ring/pinky folded into palm, thumb resting beside palm.
- **Action**: Moves the desktop mouse cursor tracking the index fingertip (landmark 8) in the mirrored camera feed.
- **Adaptive Smoothing**: Time-aware and speed-adaptive exponential moving average (EMA) filter. When stationary, low alpha (0.20) and a deadzone eliminate hand tremor completely; on fast hand motions, alpha scales up to 0.85 and deadzone disappears, delivering snappy, zero-lag target acquisition across variable webcam frame rates.
- **Configurable Active Margin**: Maps an active zone (default 12% margin) so the cursor reaches every screen edge and corner comfortably without having to push your hand to the extreme physical borders of the camera view.
- **Transition Tolerance (Hysteresis)**: Noisy intermediate frames or 1-frame tracking dropouts do not reset smoother state or cause cursor jumps.

---

### 3. Thumb Left Click (Pointing Mode)
- **Pose**: While in one-finger cursor mode, raise the thumb outward away from the palm (like a hitchhiker or finger gun).
- **Behavior**:
  - **Deliberate Transition**: Requires a clear lowered-to-raised transition. Entering cursor mode with an already raised thumb will **neither** click **nor** permanently prevent cursor movement.
  - **Single Click**: Dispatches exactly one left click per thumb raise.
  - **Rearming**: The thumb must return to its lowered position beside the palm before another click is permitted.
  - **Cursor Freeze During Click**: The cursor position is frozen at its last stable position during the deliberate click gesture so natural index-finger twitching cannot shift the click target.

---

### 4. Thumb-to-Middle Finger Pinch Right Click
- **Pose**: While index finger remains extended, touch the thumb tip (landmark 4) to the middle finger tip (landmark 12), with ring and pinky fingers folded into palm (middle finger free to bend naturally). Arbitrary hand poses with fingertip contact are strictly rejected.
- **Behavior**:
  - **Deliberate Contact State Machine**: Evaluates contact (`<= 0.21`), approach (`<= 0.28`), and release (`>= 0.35`) relative to palm scale with hysteresis.
  - **Continuous Separation to Arm**: Arming requires fingertips to be continuously separated beyond the release threshold (`>= 0.35`) for approximately 150 ms. Entering tracking with fingertips already close remains disarmed.
  - **Deliberate Contact Confirmation**: Requires continuous valid contact (`<= 0.21`) in the valid pose for approximately 200 ms before dispatching exactly one `pyautogui.click(button="right", _pause=False)`.
  - **Incidental Proximity Rejected**: Tightened contact threshold (0.21) prevents resting thumb proximity or pointing poses (0.22–0.25) from triggering accidental clicks.
  - **No Repeat on Hold & Latched Click State**: Dispatches exactly one click per deliberate gesture. The clicked state remains latched until separation beyond the release threshold is confirmed for approximately 150 ms.
  - **Tracking Loss Resilience**: Latched clicked state is preserved through brief tracking dropouts to prevent duplicate clicks upon recovery; pending contact confirmation resets if tracking drops or contact breaks.
  - **Conflict Prevention**: Approaching or confirming right-click freezes cursor movement and suppresses left-click, scrolling, and swipes. Entering right-click with thumb raised will not trigger a left-click; after right-click, the thumb must return to its lowered position before left-click is rearmed.

---

### 5. Two-Finger Vertical Scrolling
- **Pose**: Index and middle fingers extended, ring and pinky folded into palm.
- **Behavior**:
  - **Scroll Up**: Move both fingers upward in the camera view (decreasing Y).
  - **Scroll Down**: Move both fingers downward in the camera view (increasing Y).
  - **High-Speed 10x Scrolling**: Scaled by 10.0x from baseline (min 40.0 steps/s to max 220.0 steps/s with a 30 steps/frame limit) for ultra-fast, smooth desktop navigation.
  - **Instant Neutral Stop**: Returning fingers to the latched neutral position stops scrolling immediately.
  - **Jitter Rejection**: Uses average fingertip position with a movement deadzone and quadratic distance acceleration curve.
  - **Burst Clamping**: Tracking loss gaps are clamped to prevent runaway catch-up bursts on recovery.
  - **Cursor Freeze**: Cursor movement is completely frozen during scrolling.

---

### 6. Three-Finger Downward Swipe Screenshot 📸
- **Pose**: Extend **INDEX (8) + MIDDLE (12) + RING (16)** fingers, keep **PINKY (20)** folded into palm. Thumb is unrestricted (may be resting or raised).
- **Movement**: Slide the hand downward in the camera view (increasing Y in mirrored feed).
- **Behavior**:
  - **Pose Confirmation**: Requires holding the three-finger pose stable for ~120 ms (`SCREENSHOT_POSE_CONFIRM_TIME = 0.12s`) before stroke tracking begins. Holding the pose still does not capture.
  - **Deliberate Downward Stroke**: Evaluates fingertip center (average of landmarks 8, 12, 16) measured from stroke apex (raising hand before swiping does not corrupt stroke):
    - Minimum downward displacement: `0.12` frame height (`SCREENSHOT_MIN_DOWNWARD_DISPLACEMENT = 0.12`)
    - Vertical dominance ratio: `1.5x` horizontal movement (`SCREENSHOT_VERTICAL_DOMINANCE_RATIO = 1.5`)
    - Minimum velocity: `0.25` units/s (`SCREENSHOT_MIN_VELOCITY = 0.25`)
    - Maximum stroke duration: `0.80` s (`SCREENSHOT_MAX_STROKE_DURATION = 0.80`)
  - **Rejection of Unintended Motions**: Stationary holding, slow drift, upward strokes, sideways swipes, and strongly diagonal strokes are strictly rejected.
  - **Single Capture & Rearming**: Captures exactly one screenshot per stroke, latching in `CAPTURED` state. Holding the pose or returning the hand upward will never repeat. Rearming requires a confirmed release (leaving the three-finger pose for ~150 ms, `SCREENSHOT_RELEASE_TIME = 0.15s`). Missing tracking alone never counts as a release.
  - **Gesture Priority & Suppression**:
    - Evaluated before scrolling, cursor movement, and palm swipes.
    - Three extended fingers with folded pinky are explicitly blocked from triggering scroll mode (`is_scroll_candidate`).
    - Cursor position freezes and all clicks/swipes are suppressed during confirmation and stroke.
    - Stale swipe history is cleared when screenshot gesture starts and ends.
    - Full controls restore cleanly after cancellation or deliberate release.
  - **Storage & Desktop Feedback**:
    - Captures primary display using `pyautogui.screenshot()`.
    - Automatically creates and saves PNG files to `screenshots/` directory relative to project root (`screenshots/screenshot_%Y%m%d_%H%M%S_%f.png`).
    - Capture and save execute **before** displaying any feedback, guaranteeing that overlays never appear in saved PNGs.
    - **Desktop Capture Flash Animation**: Shows a subtle translucent white overlay (~18% peak opacity) across the captured display, fading out over ~250 ms. Only flashes on successful captures; never flashes on failure.
    - **Compact Desktop Notification Card**: Displays a sleek top-right desktop notification ("Screenshot saved" + filename with a green `✓` icon, or red error banner on failure). Remains visible for ~1.5s, then fades out smoothly over ~300 ms.
    - **Click-Through & Non-Activating**: Windows use `WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW` to remain completely click-through, non-interactive, and invisible to taskbar/Alt-Tab. Focus is preserved on target applications (including PowerPoint) and the camera preview is never brought forward.
    - Displays `"Screenshot saved: <filename>"` in camera HUD only after file save succeeds.
    - PyAutoGUI fail-safe (`pyautogui.FailSafeException`) propagates directly to the emergency-stop handler.

---

## ⚡ Gesture Priority & Rules

1. **Screenshot Gesture takes top priority**: Recognized 3-finger candidate freezes cursor, suppresses left/right clicks, scrolling, and slide swipes.
2. **Right-Click Approach & Contact takes secondary priority**: Freezes cursor movement and suppresses left-click, scrolling, and swipes.
3. **Two-Finger Scrolling takes priority** over thumb left-clicking when not in a screenshot or right-click pinch.
4. **Thumb Left-Clicking is available only** in one-finger cursor mode when the thumb raises from a lowered state.
5. **Swipe Suppression**: Palm swipe detection is suppressed throughout screenshot recognition, cursor control, thumb left-clicking, right-clicking, and two-finger scrolling.
6. **State Cleanup**: Stale movement history is cleared on mode transitions. Brief landmark dropouts or tracking losses will never produce an accidental click, cursor jump, scroll burst, swipe, or screenshot.
7. **Fail-Safe**: PyAutoGUI fail-safe is enabled (`FAILSAFE = True`). Push the physical mouse to any screen corner to trigger an immediate emergency stop.

---

## 🖥️ On-Screen Camera Preview HUD

The live OpenCV window displays real-time diagnostics:
- **Active Mode Banner & Live FPS**: Shows current state (`Cursor control`, `Scroll control`, `Screenshot: ...`, `Swipe Left/Right/Up/Down`, `Tracking palm...`, `Cooldown`) and real-time FPS.
- **Action & Screenshot Badges**: Highlights recognized actions only after successful dispatch (`Action: Next`, `Action: Previous`, `Action: Left Click`, `Action: Right Click`, `Action: Scroll Up`, `Action: Scroll Down`, or green `Screenshot saved: screenshot_...png` / red `Screenshot failed: ...`).
- **Compact Diagnostic Box**:
  - **In Screenshot Mode**:
    - State: `CONFIRMING`, `TRACKING_STROKE`, or `CAPTURED (Release to rearm)`
    - Down Displacement: Live displacement `dy` vs threshold `0.12` with animated visual progress bar
    - Velocity: Live velocity vs `0.25` u/s
    - FPS
  - **In Cursor Mode**:
    - Raw Tip `(x, y)` & Target Screen `(px, py)`
    - Thumb state & spread ratio (`Lowered` / `Raised` `(0.24)`)
    - Active Margin & FPS
    - **Pinch Distance & Thresholds**: `Pinch: 0.20 [Contact:0.21 Rel:0.35]`
    - **Right-Click State Machine**: `RC State: DISARMED`, `ARMED`, `APPROACHING`, `CONFIRMING (65% 0.13/0.20s)`, or `CLICKED (Latched)`
  - **In Palm Mode**:
    - **Target**: `PowerPoint (Slide Show)`, `PowerPoint (Presenter View)`, or `Generic Mode`
    - **Swipe Dir**: Real-time classified swipe direction (`Right`, `Left`, `Up`, `Down`, or `None`)
    - **Suppressed**: Explicit suppression reason (`Screenshot Gesture Candidate`, `Active Mouse Mode`, `Cursor Gesture`, `Scroll Gesture`, `In Cooldown`, `Opposite return stroke suppressed`, `None`)
    - **Dispatched**: Dispatched command (`Next slide (PageDown)`, `Previous slide (PageUp)`, or `None`)
    - Palm center `(x, y)`, Speed, Displacement `(dx, dy)`, and FPS.
- **Visual Overlays**:
  - Connected cyan/green reticles across fingertips 8, 12, and 16 with center yellow dot during screenshot pose.
  - Active reach boundary rectangle.
  - Cyan reticle on the index fingertip during cursor movement.
  - Color-coded thumb indicator (Yellow when raised, Green on click).
  - Green/Yellow connection line between thumb and middle fingertips during right-click approach/contact.
  - Magenta bridge between index and middle fingertips during scrolling.
  - Dual-ring markers on palm center during swipe tracking.

---

## 📦 Installation & Setup

### 1. Requirements
Ensure Python 3.10–3.13 is installed:
```bash
python --version
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Hand Landmarker Model
Downloads Google's MediaPipe Hand Landmarker model securely with validation:
```bash
python setup_models.py
```

### 4. Run the Application
```bash
python main.py
```
- Press **Q** in the camera window or click the window's close button (**X**) to exit.

---

## 🧪 Automated Testing

The project includes an extensive test suite verifying swipe detection, cursor movement, clicking logic, scrolling, jitter rejection, and mode transitions:

```bash
pytest -v
```

> [!NOTE]
> All automated tests mock PyAutoGUI to ensure the real mouse and keyboard are never manipulated during testing.