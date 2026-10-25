"""
Automated Pipeline Test
Simulates camera frames and hand landmarks to verify drawing, blending, and saving.
"""

import os
import cv2
import numpy as np

from hand_tracker import HandDetector, INDEX_TIP, MIDDLE_TIP
from canvas import AirCanvas


def run_pipeline_test():
    print("[Test] Initializing AirCanvas...")
    w, h = 1280, 720
    canvas = AirCanvas(width=w, height=h)

    print("[Test] Testing button clicks...")
    # Click CLEAR button (bounds around x=12..160, y=10..75)
    clear_click = canvas.check_ui_click((50, 40))
    assert clear_click == "clear", f"Expected 'clear', got {clear_click}"

    # Click RED color button
    red_click = canvas.check_ui_click((550, 40))
    print(f"[Test] Button click detected: {red_click}")

    print("[Test] Simulating draw strokes...")
    canvas.mode = "Draw"
    # Stroke across screen
    for i in range(100, 300, 20):
        canvas.draw_on_canvas((i, 250))

    # Reset stroke
    canvas.reset_stroke()

    # Stroke for eraser
    canvas.mode = "Eraser"
    for i in range(120, 200, 20):
        canvas.draw_on_canvas((i, 250))

    canvas.reset_stroke()

    print("[Test] Testing frame blending...")
    dummy_frame = np.full((h, w, 3), 120, dtype=np.uint8)
    blended = canvas.blend(dummy_frame)
    assert blended.shape == (h, w, 3), f"Invalid blend shape: {blended.shape}"

    ui_frame = canvas.render_ui(blended, cursor_pt=(200, 200), hovered_btn=None)
    assert ui_frame.shape == (h, w, 3), f"Invalid UI shape: {ui_frame.shape}"

    print("[Test] Testing canvas save...")
    saved_path = canvas.save_drawing()
    assert os.path.exists(saved_path), f"Saved file not found: {saved_path}"
    print(f"[Test] Saved image verified at: {saved_path}")

    print("[Test] Pipeline test completed successfully!")


if __name__ == "__main__":
    run_pipeline_test()
