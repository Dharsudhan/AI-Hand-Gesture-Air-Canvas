"""
Air Canvas Main Application
Interactive touch-free hand gesture digital drawing using OpenCV and MediaPipe.
"""

import argparse
import sys
import time
import cv2
import numpy as np

# Ensure Windows terminal doesn't crash on character encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from hand_tracker import HandDetector, INDEX_TIP, MIDDLE_TIP
from canvas import AirCanvas


def parse_args():
    parser = argparse.ArgumentParser(description="Touch-Free Hand Gesture Air Canvas")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Capture height (default: 720)")
    parser.add_argument("--debug", action="store_true", help="Show hand skeleton landmarks")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize Webcam
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fallback to default backend
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"[Error] Could not open video device {args.camera}.")
        print("Please check your webcam connection or verify the camera index.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Read actual dimensions
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_w == 0 or actual_h == 0:
        actual_w, actual_h = args.width, args.height

    print("=" * 60)
    print(" [*] Hand Gesture Recognition Air-Canvas")
    print(f" Camera Resolution: {actual_w}x{actual_h}")
    print(" Controls:")
    print("   - Index finger UP: Draw / Paint")
    print("   - Index + Middle fingers UP: Selection & Hover")
    print("   - Open Palm / Fist: Standby / Pause")
    print(" Hotkeys:")
    print("   - 'c' : Clear canvas")
    print("   - 's' : Save drawing")
    print("   - 'w' : Toggle Whiteboard / Camera view")
    print("   - 'e' : Toggle Eraser mode")
    print("   - '+' / '-' : Adjust brush size")
    print("   - 'd' : Toggle skeleton debug view")
    print("   - 'q' or ESC : Exit")
    print("=" * 60)

    # Initialize Modules
    detector = HandDetector(num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    canvas = AirCanvas(width=actual_w, height=actual_h)

    show_debug = args.debug
    prev_time = time.time()
    fps = 0.0

    window_name = "Air Canvas - Hand Gesture Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Warning] Failed to grab webcam frame. Retrying...")
                time.sleep(0.05)
                continue

            # Mirror frame horizontally for intuitive interaction
            frame = cv2.flip(frame, 1)

            # Resize if webcam resolution differs from canvas buffer
            if frame.shape[1] != actual_w or frame.shape[0] != actual_h:
                frame = cv2.resize(frame, (actual_w, actual_h))

            # Calculate FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Detect Hand Landmarks
            hands = detector.find_hands(frame)

            cursor_pt = None
            hovered_btn = None

            if hands:
                landmarks = hands[0]
                fingers = detector.get_fingers_up(landmarks)
                # fingers: [Thumb, Index, Middle, Ring, Pinky]

                index_tip = landmarks[INDEX_TIP]
                middle_tip = landmarks[MIDDLE_TIP]

                # Draw skeleton if debug mode is active
                if show_debug:
                    frame = detector.draw_hand(frame, landmarks)

                # GESTURE 1: Selection / Hover Mode (Index + Middle fingers UP)
                if fingers[1] == 1 and fingers[2] == 1:
                    canvas.mode = "Selection"
                    canvas.reset_stroke()

                    # Cursor point is centered between index and middle fingertips
                    cx = (index_tip[0] + middle_tip[0]) // 2
                    cy = (index_tip[1] + middle_tip[1]) // 2
                    cursor_pt = (cx, cy)

                    # Check for button selection in top header
                    hovered_btn = canvas.check_ui_click(cursor_pt)

                # GESTURE 2: Drawing Mode (Index finger UP, others DOWN)
                elif fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
                    cursor_pt = index_tip

                    if canvas.mode == "Selection":
                        canvas.mode = "Draw"

                    canvas.draw_on_canvas(index_tip)

                # GESTURE 3: Standby / Idle (Open Palm or Fist)
                else:
                    canvas.mode = "Standby"
                    canvas.reset_stroke()
            else:
                # No hand detected in frame
                canvas.mode = "Standby"
                canvas.reset_stroke()

            # Blend canvas with camera frame (or whiteboard)
            blended_display = canvas.blend(frame)

            # Render UI Navigation Bar and Cursor Feedback
            final_display = canvas.render_ui(blended_display, cursor_pt, hovered_btn)

            # Render HUD Info (FPS, Mode Badge, Brush Size)
            _render_hud(final_display, fps, canvas.mode, canvas.brush_size, canvas.current_color, canvas.is_whiteboard)

            # Display Output Window
            cv2.imshow(window_name, final_display)

            # Handle Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):  # ESC or 'q'
                break
            elif key in (ord('c'), ord('C')):
                canvas.clear_canvas()
                canvas.set_notification("Canvas Cleared")
            elif key in (ord('s'), ord('S')):
                saved_path = canvas.save_drawing()
                canvas.set_notification(f"Saved to {saved_path.split('/')[-1].split(chr(92))[-1]}")
            elif key in (ord('w'), ord('W')):
                canvas.is_whiteboard = not canvas.is_whiteboard
                status = "Whiteboard Mode" if canvas.is_whiteboard else "Camera Overlay Mode"
                canvas.set_notification(status)
            elif key in (ord('e'), ord('E')):
                if canvas.mode == "Eraser":
                    canvas.mode = "Draw"
                    canvas.set_notification("Brush Mode")
                else:
                    canvas.mode = "Eraser"
                    canvas.set_notification("Eraser Mode")
            elif key in (ord('+'), ord('=')):
                canvas.brush_size = min(40, canvas.brush_size + 2)
                canvas.set_notification(f"Brush Size: {canvas.brush_size}px")
            elif key in (ord('-'), ord('_')):
                canvas.brush_size = max(2, canvas.brush_size - 2)
                canvas.set_notification(f"Brush Size: {canvas.brush_size}px")
            elif key in (ord('d'), ord('D')):
                show_debug = not show_debug
                canvas.set_notification("Skeleton Debug: ON" if show_debug else "Skeleton Debug: OFF")

    except KeyboardInterrupt:
        print("\n[AirCanvas] Stopped by user.")
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[AirCanvas] Cleaned up resources and exited successfully.")


def _render_hud(frame, fps, mode, brush_size, current_color, is_whiteboard):
    """Draws sleek HUD badges for FPS, Mode, and Tool stats."""
    h, w, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 1. Mode Status Badge (Bottom Left)
    mode_text = f"MODE: {mode.upper()}"
    badge_color = (60, 60, 60)
    text_color = (255, 255, 255)

    if mode == "Draw":
        mode_text += " (DRAWING)"
        badge_color = (0, 150, 0)
    elif mode == "Selection":
        mode_text += " (NAVIGATE/SELECT)"
        badge_color = (180, 100, 0)
    elif mode == "Eraser":
        mode_text += " (ERASING)"
        badge_color = (0, 0, 180)
    elif mode == "Standby":
        mode_text += " (IDLE)"
        badge_color = (40, 40, 40)

    # Background pill for mode
    (tw, th), _ = cv2.getTextSize(mode_text, font, 0.55, 2)
    by = h - 25
    cv2.rectangle(frame, (15, by - th - 8), (25 + tw + 10, by + 8), (20, 20, 20), -1)
    cv2.rectangle(frame, (15, by - th - 8), (25 + tw + 10, by + 8), badge_color, 2)
    cv2.putText(frame, mode_text, (20, by), font, 0.55, text_color, 2, cv2.LINE_AA)

    # 2. Controls Help Bar (Bottom Center/Right)
    help_text = "[c] Clear | [s] Save | [w] Whiteboard | [e] Eraser | [+/-] Size | [q] Quit"
    (hw, hh), _ = cv2.getTextSize(help_text, font, 0.42, 1)
    hx = w - hw - 20
    cv2.rectangle(frame, (hx - 10, by - hh - 6), (w - 10, by + 8), (20, 20, 20), -1)
    cv2.putText(frame, help_text, (hx, by), font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    # 3. FPS Indicator (Top Left under header)
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(frame, fps_text, (20, 115), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


if __name__ == "__main__":
    main()
