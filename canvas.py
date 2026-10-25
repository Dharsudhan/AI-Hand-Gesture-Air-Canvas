"""
Air Canvas Engine
Manages the drawing buffer, UI header buttons, color palette, smoothing filters,
and composite blending with the webcam video stream.
"""

import os
import time
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List

import cv2
import numpy as np


class AirCanvas:
    """Manages digital drawing, color palettes, virtual UI buttons, and frame blending."""

    # Pre-defined vibrant BGR colors
    PALETTE = {
        "BLUE": (255, 140, 0),     # Vibrant Sky Blue
        "GREEN": (50, 215, 60),    # Emerald Green
        "RED": (50, 60, 255),      # Crimson Red
        "YELLOW": (30, 220, 255),  # Sunshine Yellow
        "PURPLE": (230, 50, 180),  # Neon Magenta/Purple
        "WHITE": (255, 255, 255),  # Pure White
    }

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

        # Drawing layers:
        # canvas holds the active color lines on a black background
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Drawing states
        self.current_color = self.PALETTE["BLUE"]
        self.brush_size = 8
        self.eraser_size = 50
        self.mode = "Selection"  # "Selection", "Draw", "Eraser", "Standby"
        self.is_whiteboard = False  # Toggle drawing on pure white background

        # Smoothing and stroke continuity
        self.prev_pt: Optional[Tuple[int, int]] = None
        self.smooth_x: Optional[float] = None
        self.smooth_y: Optional[float] = None
        self.smoothing_factor = 0.55  # Exponential Moving Average factor (0 = rigid, 1 = immediate)

        # UI Header Configuration
        self.header_height = 85
        self.buttons: List[Dict[str, Any]] = []
        self._init_ui_buttons()

        # Temporary banner notifications (e.g. "Saved to saved_drawings/...")
        self.notification_text = ""
        self.notification_expiry = 0.0

        # Save directory
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_drawings")
        os.makedirs(self.save_dir, exist_ok=True)

    def _init_ui_buttons(self):
        """Builds interactive UI buttons across the top header."""
        # Top bar layout:
        # [ CLEAR ALL ] [ BLUE ] [ GREEN ] [ RED ] [ YELLOW ] [ PURPLE ] [ ERASER ] [ SAVE ]
        button_defs = [
            {"id": "clear", "label": "CLEAR", "type": "action", "color": (40, 40, 40), "text_color": (255, 255, 255)},
            {"id": "color_BLUE", "label": "BLUE", "type": "color", "color": self.PALETTE["BLUE"], "text_color": (0, 0, 0)},
            {"id": "color_GREEN", "label": "GREEN", "type": "color", "color": self.PALETTE["GREEN"], "text_color": (0, 0, 0)},
            {"id": "color_RED", "label": "RED", "type": "color", "color": self.PALETTE["RED"], "text_color": (255, 255, 255)},
            {"id": "color_YELLOW", "label": "YELLOW", "type": "color", "color": self.PALETTE["YELLOW"], "text_color": (0, 0, 0)},
            {"id": "color_PURPLE", "label": "PURPLE", "type": "color", "color": self.PALETTE["PURPLE"], "text_color": (255, 255, 255)},
            {"id": "eraser", "label": "ERASER", "type": "eraser", "color": (60, 60, 60), "text_color": (255, 255, 255)},
            {"id": "save", "label": "SAVE", "type": "action", "color": (30, 110, 40), "text_color": (255, 255, 255)},
        ]

        total_buttons = len(button_defs)
        margin = 12
        padding_y = 10
        btn_y1 = padding_y
        btn_y2 = self.header_height - padding_y

        total_width = self.width - (margin * (total_buttons + 1))
        btn_width = total_width // total_buttons

        self.buttons = []
        for i, b in enumerate(button_defs):
            x1 = margin + i * (btn_width + margin)
            x2 = x1 + btn_width
            b_info = b.copy()
            b_info["bounds"] = (x1, btn_y1, x2, btn_y2)
            self.buttons.append(b_info)

    def smooth_point(self, pt: Tuple[int, int]) -> Tuple[int, int]:
        """Applies Exponential Moving Average (EMA) filter to coordinate stream."""
        x, y = pt
        if self.smooth_x is None or self.smooth_y is None:
            self.smooth_x = float(x)
            self.smooth_y = float(y)
        else:
            self.smooth_x = self.smoothing_factor * x + (1.0 - self.smoothing_factor) * self.smooth_x
            self.smooth_y = self.smoothing_factor * y + (1.0 - self.smoothing_factor) * self.smooth_y

        return int(self.smooth_x), int(self.smooth_y)

    def reset_stroke(self):
        """Clears previous coordinate to avoid connecting disjoint strokes."""
        self.prev_pt = None
        self.smooth_x = None
        self.smooth_y = None

    def draw_on_canvas(self, raw_pt: Tuple[int, int]):
        """Renders continuous line strokes or eraser on the canvas buffer."""
        # Avoid drawing over the top header area
        if raw_pt[1] <= self.header_height:
            self.reset_stroke()
            return

        current_pt = self.smooth_point(raw_pt)

        if self.prev_pt is None:
            self.prev_pt = current_pt
            return

        if self.mode == "Eraser":
            # Eraser draws black (0, 0, 0) onto the canvas mask with larger radius
            cv2.line(self.canvas, self.prev_pt, current_pt, (0, 0, 0), self.eraser_size, cv2.LINE_AA)
            cv2.circle(self.canvas, current_pt, self.eraser_size // 2, (0, 0, 0), -1, cv2.LINE_AA)
        else:
            # Active brush draw
            cv2.line(self.canvas, self.prev_pt, current_pt, self.current_color, self.brush_size, cv2.LINE_AA)
            cv2.circle(self.canvas, current_pt, self.brush_size // 2, self.current_color, -1, cv2.LINE_AA)

        self.prev_pt = current_pt

    def check_ui_click(self, cursor_pt: Tuple[int, int]) -> Optional[str]:
        """
        Tests if cursor is hovering or clicking a header button.
        Returns the button ID clicked, or None.
        """
        cx, cy = cursor_pt
        if cy > self.header_height:
            return None

        for btn in self.buttons:
            x1, y1, x2, y2 = btn["bounds"]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                btn_id = btn["id"]
                if btn_id == "clear":
                    self.clear_canvas()
                    self.set_notification("Canvas Cleared!")
                elif btn_id == "eraser":
                    self.mode = "Eraser"
                    self.set_notification("Eraser Selected")
                elif btn_id == "save":
                    path = self.save_drawing()
                    self.set_notification(f"Saved: {os.path.basename(path)}")
                elif btn_id.startswith("color_"):
                    color_name = btn_id.replace("color_", "")
                    self.current_color = self.PALETTE[color_name]
                    self.mode = "Draw"
                    self.set_notification(f"{color_name} Color Selected")
                return btn_id

        return None

    def clear_canvas(self):
        """Wipes the drawing canvas completely."""
        self.canvas[:] = 0
        self.reset_stroke()

    def set_notification(self, text: str, duration: float = 2.5):
        """Displays a short visual banner on the HUD."""
        self.notification_text = text
        self.notification_expiry = time.time() + duration

    def save_drawing(self) -> str:
        """
        Saves high-resolution PNG copies of:
        1. Pure artwork on crisp white background
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"canvas_{timestamp}.png"
        filepath = os.path.join(self.save_dir, filename)

        # Create white background version of canvas
        white_bg = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask)

        # Blend on white background
        bg_masked = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
        canvas_masked = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        result = cv2.add(bg_masked, canvas_masked)

        cv2.imwrite(filepath, result)
        print(f"[AirCanvas] Drawing saved to {filepath}")
        return filepath

    def blend(self, camera_frame: np.ndarray) -> np.ndarray:
        """
        Blends drawing canvas onto video frame or pure whiteboard.
        """
        if self.is_whiteboard:
            base = np.ones_like(camera_frame) * 255
        else:
            base = camera_frame

        # Find non-black pixels from drawing canvas
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask)

        # Composite frame and drawing
        frame_bg = cv2.bitwise_and(base, base, mask=inv_mask)
        drawing_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        blended = cv2.add(frame_bg, drawing_fg)

        return blended

    def render_ui(
        self,
        frame: np.ndarray,
        cursor_pt: Optional[Tuple[int, int]] = None,
        hovered_btn: Optional[str] = None,
    ) -> np.ndarray:
        """Renders the top navigation toolbar, active states, and HUD notifications."""
        # 1. Dark semi-transparent header bar overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.header_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # 2. Draw Header Buttons
        for btn in self.buttons:
            x1, y1, x2, y2 = btn["bounds"]
            btn_id = btn["id"]
            color = btn["color"]
            label = btn["label"]
            text_color = btn["text_color"]

            # Determine button state (Active or Hovered)
            is_active = False
            if btn["type"] == "color" and self.mode != "Eraser" and self.current_color == color:
                is_active = True
            elif btn["type"] == "eraser" and self.mode == "Eraser":
                is_active = True

            is_hovered = (btn_id == hovered_btn)

            # Draw button background
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)

            # Draw border (highlight if active or hovered)
            border_color = (255, 255, 255) if (is_active or is_hovered) else (80, 80, 80)
            thickness = 3 if is_active else (2 if is_hovered else 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, thickness, cv2.LINE_AA)

            # Label text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            text_thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
            tx = x1 + (x2 - x1 - tw) // 2
            ty = y1 + (y2 - y1 + th) // 2
            cv2.putText(frame, label, (tx, ty), font, font_scale, text_color, text_thickness, cv2.LINE_AA)

        # 3. Notification message banner
        if time.time() < self.notification_expiry and self.notification_text:
            notif_y = self.header_height + 35
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            (nw, nh), _ = cv2.getTextSize(self.notification_text, font, font_scale, 2)
            nx1 = (self.width - nw) // 2 - 20
            nx2 = (self.width + nw) // 2 + 20
            cv2.rectangle(frame, (nx1, notif_y - nh - 10), (nx2, notif_y + 10), (30, 30, 30), -1)
            cv2.rectangle(frame, (nx1, notif_y - nh - 10), (nx2, notif_y + 10), (0, 255, 200), 2)
            cv2.putText(
                frame,
                self.notification_text,
                ((self.width - nw) // 2, notif_y),
                font,
                font_scale,
                (0, 255, 200),
                2,
                cv2.LINE_AA,
            )

        # 4. Cursor Feedback when in Selection or Drawing mode
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
