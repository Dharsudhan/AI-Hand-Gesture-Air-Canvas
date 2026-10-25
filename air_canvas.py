"""
Hand Gesture Recognition Air-Canvas (Single-File Complete Edition)
Interactive touch-free drawing using Python, OpenCV, and Google MediaPipe Tasks API.
"""

import os
import sys
import time
import math
import argparse
import urllib.request
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

# Ensure Windows terminal doesn't crash on character encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions

# ==============================================================================
# MODEL CONFIGURATION & AUTO-DOWNLOAD
# ==============================================================================
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# 21 Hand Landmark Indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

HAND_CONNECTIONS = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (MIDDLE_MCP, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (RING_MCP, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    (PINKY_MCP, WRIST)
]


def ensure_model_exists():
    """Downloads the official MediaPipe HandLandmarker model asset if missing."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"[*] Downloading MediaPipe HandLandmarker model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[*] Model downloaded successfully.")


# ==============================================================================
# HAND TRACKER ENGINE
# ==============================================================================
class HandDetector:
    """Wrapper around MediaPipe Tasks HandLandmarker for real-time video."""

    def __init__(self, num_hands: int = 1, min_conf: float = 0.6):
        ensure_model_exists()
        self.options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self.detector = HandLandmarker.create_from_options(self.options)
        self.last_timestamp_ms = 0

    def find_hands(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> List[List[Tuple[int, int]]]:
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        hands_landmarks = []
        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                hands_landmarks.append(pts)

        return hands_landmarks

    def get_fingers_up(self, lm: List[Tuple[int, int]]) -> List[int]:
        """Returns [Thumb, Index, Middle, Ring, Pinky] as 1 (extended) or 0 (folded)."""
        if not lm or len(lm) < 21:
            return [0, 0, 0, 0, 0]

        fingers = [0, 0, 0, 0, 0]

        # Thumb
        d_tip = math.hypot(lm[THUMB_TIP][0] - lm[PINKY_MCP][0], lm[THUMB_TIP][1] - lm[PINKY_MCP][1])
        d_ip = math.hypot(lm[THUMB_IP][0] - lm[PINKY_MCP][0], lm[THUMB_IP][1] - lm[PINKY_MCP][1])
        if d_tip > d_ip * 1.1:
            fingers[0] = 1

        # Fingers (Index, Middle, Ring, Pinky): Compare TIP y with PIP y
        if lm[INDEX_TIP][1] < lm[INDEX_PIP][1]:
            fingers[1] = 1
        if lm[MIDDLE_TIP][1] < lm[MIDDLE_PIP][1]:
            fingers[2] = 1
        if lm[RING_TIP][1] < lm[RING_PIP][1]:
            fingers[3] = 1
        if lm[PINKY_TIP][1] < lm[PINKY_PIP][1]:
            fingers[4] = 1

        return fingers

    def draw_skeleton(self, frame: np.ndarray, lm: List[Tuple[int, int]]) -> np.ndarray:
        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(frame, lm[p1], lm[p2], (180, 180, 180), 2, cv2.LINE_AA)
        for i, pt in enumerate(lm):
            c = (0, 0, 255) if i == INDEX_TIP else ((255, 200, 0) if i == MIDDLE_TIP else (0, 255, 255))
            cv2.circle(frame, pt, 5, c, -1, cv2.LINE_AA)
        return frame

    def close(self):
        if self.detector:
            self.detector.close()


# ==============================================================================
# AIR CANVAS DRAWING & UI ENGINE
# ==============================================================================
class AirCanvas:
    """Manages drawing surfaces, virtual palettes, smoothing filters, and exports."""

    PALETTE = {
        "BLUE": (255, 140, 0),     # Sky Blue
        "GREEN": (50, 215, 60),    # Vibrant Green
        "RED": (50, 60, 255),      # Crimson Red
        "YELLOW": (30, 220, 255),  # Sunshine Yellow
        "PURPLE": (230, 50, 180),  # Neon Magenta
    }

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self.current_color = self.PALETTE["BLUE"]
        self.brush_size = 8
        self.eraser_size = 50
        self.mode = "Selection"
        self.is_whiteboard = False

        self.prev_pt: Optional[Tuple[int, int]] = None
        self.smooth_x: Optional[float] = None
        self.smooth_y: Optional[float] = None
        self.smoothing_factor = 0.55

        self.header_height = 85
        self.buttons: List[Dict[str, Any]] = []
        self._init_buttons()

        self.notif_text = ""
        self.notif_expiry = 0.0
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_drawings")
        os.makedirs(self.save_dir, exist_ok=True)

    def _init_buttons(self):
        defs = [
            {"id": "clear", "label": "CLEAR", "type": "action", "color": (40, 40, 40), "text_color": (255, 255, 255)},
            {"id": "color_BLUE", "label": "BLUE", "type": "color", "color": self.PALETTE["BLUE"], "text_color": (0, 0, 0)},
            {"id": "color_GREEN", "label": "GREEN", "type": "color", "color": self.PALETTE["GREEN"], "text_color": (0, 0, 0)},
            {"id": "color_RED", "label": "RED", "type": "color", "color": self.PALETTE["RED"], "text_color": (255, 255, 255)},
            {"id": "color_YELLOW", "label": "YELLOW", "type": "color", "color": self.PALETTE["YELLOW"], "text_color": (0, 0, 0)},
            {"id": "color_PURPLE", "label": "PURPLE", "type": "color", "color": self.PALETTE["PURPLE"], "text_color": (255, 255, 255)},
            {"id": "eraser", "label": "ERASER", "type": "eraser", "color": (60, 60, 60), "text_color": (255, 255, 255)},
            {"id": "save", "label": "SAVE", "type": "action", "color": (30, 110, 40), "text_color": (255, 255, 255)},
        ]
        n = len(defs)
        margin = 12
        btn_w = (self.width - margin * (n + 1)) // n
        self.buttons = []
        for i, d in enumerate(defs):
            x1 = margin + i * (btn_w + margin)
            b = d.copy()
            b["bounds"] = (x1, 10, x1 + btn_w, self.header_height - 10)
            self.buttons.append(b)

    def smooth(self, pt: Tuple[int, int]) -> Tuple[int, int]:
        x, y = pt
        if self.smooth_x is None:
            self.smooth_x, self.smooth_y = float(x), float(y)
        else:
            self.smooth_x = self.smoothing_factor * x + (1.0 - self.smoothing_factor) * self.smooth_x
            self.smooth_y = self.smoothing_factor * y + (1.0 - self.smoothing_factor) * self.smooth_y
        return int(self.smooth_x), int(self.smooth_y)

    def reset_stroke(self):
        self.prev_pt = None
        self.smooth_x = None
        self.smooth_y = None

    def draw(self, raw_pt: Tuple[int, int]):
        if raw_pt[1] <= self.header_height:
            self.reset_stroke()
            return

        cur = self.smooth(raw_pt)
        if self.prev_pt is None:
            self.prev_pt = cur
            return

        if self.mode == "Eraser":
            cv2.line(self.canvas, self.prev_pt, cur, (0, 0, 0), self.eraser_size, cv2.LINE_AA)
            cv2.circle(self.canvas, cur, self.eraser_size // 2, (0, 0, 0), -1, cv2.LINE_AA)
        else:
            cv2.line(self.canvas, self.prev_pt, cur, self.current_color, self.brush_size, cv2.LINE_AA)
            cv2.circle(self.canvas, cur, self.brush_size // 2, self.current_color, -1, cv2.LINE_AA)

        self.prev_pt = cur

    def check_click(self, pt: Tuple[int, int]) -> Optional[str]:
        if pt[1] > self.header_height:
            return None
        for b in self.buttons:
            x1, y1, x2, y2 = b["bounds"]
            if x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2:
                bid = b["id"]
                if bid == "clear":
                    self.canvas[:] = 0
                    self.reset_stroke()
                    self.notify("Canvas Cleared")
                elif bid == "eraser":
                    self.mode = "Eraser"
                    self.notify("Eraser Tool Active")
                elif bid == "save":
                    p = self.save()
                    self.notify(f"Saved: {os.path.basename(p)}")
                elif bid.startswith("color_"):
                    cname = bid.replace("color_", "")
                    self.current_color = self.PALETTE[cname]
                    self.mode = "Draw"
                    self.notify(f"{cname} Color Selected")
                return bid
        return None

    def notify(self, text: str, sec: float = 2.0):
        self.notif_text = text
        self.notif_expiry = time.time() + sec

    def save(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.save_dir, f"canvas_{ts}.png")
        white = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        bg = cv2.bitwise_and(white, white, mask=cv2.bitwise_not(mask))
        fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        cv2.imwrite(filepath, cv2.add(bg, fg))
        return filepath

    def blend(self, frame: np.ndarray) -> np.ndarray:
        base = np.ones_like(frame) * 255 if self.is_whiteboard else frame
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        frame_bg = cv2.bitwise_and(base, base, mask=cv2.bitwise_not(mask))
        drawing_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        return cv2.add(frame_bg, drawing_fg)

    def render_ui(self, frame: np.ndarray, cursor_pt: Optional[Tuple[int, int]], hovered: Optional[str]) -> np.ndarray:
        bar = frame.copy()
        cv2.rectangle(bar, (0, 0), (self.width, self.header_height), (20, 20, 20), -1)
        cv2.addWeighted(bar, 0.75, frame, 0.25, 0, frame)

        for b in self.buttons:
            x1, y1, x2, y2 = b["bounds"]
            bid = b["id"]
            active = (b["type"] == "color" and self.mode != "Eraser" and self.current_color == b["color"]) or \
                     (b["type"] == "eraser" and self.mode == "Eraser")
            cv2.rectangle(frame, (x1, y1), (x2, y2), b["color"], -1, cv2.LINE_AA)
            border = (255, 255, 255) if (active or bid == hovered) else (80, 80, 80)
            thick = 3 if active else (2 if bid == hovered else 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border, thick, cv2.LINE_AA)

            (tw, th), _ = cv2.getTextSize(b["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.putText(frame, b["label"], (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, b["text_color"], 2, cv2.LINE_AA)

        if time.time() < self.notif_expiry and self.notif_text:
            (nw, nh), _ = cv2.getTextSize(self.notif_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            nx = (self.width - nw) // 2
            cv2.rectangle(frame, (nx - 15, self.header_height + 15), (nx + nw + 15, self.header_height + 55), (30, 30, 30), -1)
            cv2.rectangle(frame, (nx - 15, self.header_height + 15), (nx + nw + 15, self.header_height + 55), (0, 255, 200), 2)
            cv2.putText(frame, self.notif_text, (nx, self.header_height + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2, cv2.LINE_AA)

        if cursor_pt:
            cx, cy = cursor_pt
            if self.mode == "Selection":
                cv2.circle(frame, (cx, cy), 14, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1, cv2.LINE_AA)
            elif self.mode == "Eraser":
                cv2.circle(frame, (cx, cy), self.eraser_size // 2, (100, 100, 100), 2, cv2.LINE_AA)
            elif self.mode == "Draw":
                cv2.circle(frame, (cx, cy), self.brush_size // 2 + 2, self.current_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), self.brush_size // 2 + 4, (255, 255, 255), 1, cv2.LINE_AA)

        return frame


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Touch-Free Hand Gesture Air Canvas")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Capture height (default: 720)")
    parser.add_argument("--debug", action="store_true", help="Show hand skeleton landmarks")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"[Error] Could not open camera {args.camera}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height

    print("=" * 60)
    print(" [*] Hand Gesture Recognition Air-Canvas")
    print(f" Resolution: {actual_w}x{actual_h}")
    print(" Gestures: [Index UP] = Draw | [Index+Middle UP] = Hover/Select | [Palm] = Pause")
    print(" Keys: [c] Clear | [s] Save | [w] Whiteboard | [e] Eraser | [+/-] Size | [q] Quit")
    print("=" * 60)

    detector = HandDetector(num_hands=1, min_conf=0.6)
    canvas = AirCanvas(width=actual_w, height=actual_h)

    show_debug = args.debug
    prev_time = time.time()
    fps = 0.0

    cv2.namedWindow("Air Canvas - Hand Gesture Recognition", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            if frame.shape[1] != actual_w or frame.shape[0] != actual_h:
                frame = cv2.resize(frame, (actual_w, actual_h))

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            hands = detector.find_hands(frame)
            cursor_pt, hovered_btn = None, None

            if hands:
                lm = hands[0]
                fingers = detector.get_fingers_up(lm)
                index_tip, middle_tip = lm[INDEX_TIP], lm[MIDDLE_TIP]

                if show_debug:
                    frame = detector.draw_skeleton(frame, lm)

                # Selection Mode (Index + Middle UP)
                if fingers[1] == 1 and fingers[2] == 1:
                    canvas.mode = "Selection"
                    canvas.reset_stroke()
                    cursor_pt = ((index_tip[0] + middle_tip[0]) // 2, (index_tip[1] + middle_tip[1]) // 2)
                    hovered_btn = canvas.check_click(cursor_pt)

                # Draw Mode (Index UP only)
                elif fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
                    cursor_pt = index_tip
                    if canvas.mode == "Selection":
                        canvas.mode = "Draw"
                    canvas.draw(index_tip)

                # Standby Mode
                else:
                    canvas.mode = "Standby"
                    canvas.reset_stroke()
            else:
                canvas.mode = "Standby"
                canvas.reset_stroke()

            # Blend & render UI
            display = canvas.blend(frame)
            display = canvas.render_ui(display, cursor_pt, hovered_btn)

            # HUD
            cv2.putText(display, f"FPS: {int(fps)}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            badge_color = (0, 150, 0) if canvas.mode == "Draw" else ((180, 100, 0) if canvas.mode == "Selection" else (40, 40, 40))
            cv2.rectangle(display, (15, actual_h - 45), (260, actual_h - 15), (20, 20, 20), -1)
            cv2.rectangle(display, (15, actual_h - 45), (260, actual_h - 15), badge_color, 2)
            cv2.putText(display, f"MODE: {canvas.mode.upper()}", (25, actual_h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Air Canvas - Hand Gesture Recognition", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('c'), ord('C')):
                canvas.canvas[:] = 0
                canvas.reset_stroke()
                canvas.notify("Canvas Cleared")
            elif key in (ord('s'), ord('S')):
                p = canvas.save()
                canvas.notify(f"Saved to {os.path.basename(p)}")
            elif key in (ord('w'), ord('W')):
                canvas.is_whiteboard = not canvas.is_whiteboard
                canvas.notify("Whiteboard Mode" if canvas.is_whiteboard else "Camera View")
            elif key in (ord('e'), ord('E')):
                canvas.mode = "Draw" if canvas.mode == "Eraser" else "Eraser"
                canvas.notify(f"{canvas.mode} Mode")
            elif key in (ord('+'), ord('=')):
                canvas.brush_size = min(40, canvas.brush_size + 2)
                canvas.notify(f"Brush Size: {canvas.brush_size}px")
            elif key in (ord('-'), ord('_')):
                canvas.brush_size = max(2, canvas.brush_size - 2)
                canvas.notify(f"Brush Size: {canvas.brush_size}px")
            elif key in (ord('d'), ord('D')):
                show_debug = not show_debug
                canvas.notify(f"Debug View: {'ON' if show_debug else 'OFF'}")

    except KeyboardInterrupt:
        pass
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
