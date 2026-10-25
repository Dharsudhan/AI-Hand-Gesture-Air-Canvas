"""
Hand Tracker Module for Air Canvas
Utilizes Google MediaPipe Tasks HandLandmarker for real-time, low-latency 21-landmark 3D hand tracking.
"""

import os
import time
import math
import urllib.request
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# MediaPipe Hand Landmark Indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Standard hand skeleton connections
HAND_CONNECTIONS = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (MIDDLE_MCP, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (RING_MCP, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    (PINKY_MCP, WRIST)
]


def ensure_model_exists():
    """Downloads the official MediaPipe Hand Landmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"[HandTracker] Downloading MediaPipe HandLandmarker model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[HandTracker] Model downloaded successfully.")


class HandDetector:
    """Wrapper for MediaPipe Hand Landmarker supporting real-time video streams."""

    def __init__(
        self,
        num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        ensure_model_exists()
        self.options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = HandLandmarker.create_from_options(self.options)
        self.last_timestamp_ms = 0

    def find_hands(
        self, frame: np.ndarray, timestamp_ms: Optional[int] = None
    ) -> List[List[Tuple[int, int]]]:
        """
        Processes a BGR image frame and returns detected hand landmarks in pixel coordinates.
        Returns:
            List of hands, where each hand is a list of 21 (x, y) tuples.
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        # Monotonic timestamp safeguard for MediaPipe video mode
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        hands_landmarks = []
        if result.hand_landmarks:
            for hand_landmark_list in result.hand_landmarks:
                points = []
                for lm in hand_landmark_list:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    points.append((cx, cy))
                hands_landmarks.append(points)

        return hands_landmarks

    def get_fingers_up(self, landmarks: List[Tuple[int, int]]) -> List[int]:
        """
        Determines which fingers are currently extended upwards.
        Returns a list of 5 binary flags: [Thumb, Index, Middle, Ring, Pinky]
        (1 = extended/up, 0 = folded/down)
        """
        if not landmarks or len(landmarks) < 21:
            return [0, 0, 0, 0, 0]

        fingers = [0, 0, 0, 0, 0]

        # 1. Thumb: Check distance of tip from pinky MCP vs IP from pinky MCP
        # Or compare thumb tip with thumb IP in horizontal/vertical span
        wrist = landmarks[WRIST]
        thumb_tip = landmarks[THUMB_TIP]
        thumb_ip = landmarks[THUMB_IP]
        index_mcp = landmarks[INDEX_MCP]

        # Thumb extension metric: Euclidean distance from thumb tip to pinky MCP vs IP to pinky MCP
        dist_tip_pinky = self.calculate_distance(thumb_tip, landmarks[PINKY_MCP])
        dist_ip_pinky = self.calculate_distance(thumb_ip, landmarks[PINKY_MCP])
        if dist_tip_pinky > dist_ip_pinky * 1.1:
            fingers[0] = 1

        # 2. Index finger (Tip vs PIP)
        if landmarks[INDEX_TIP][1] < landmarks[INDEX_PIP][1]:
            fingers[1] = 1

        # 3. Middle finger (Tip vs PIP)
        if landmarks[MIDDLE_TIP][1] < landmarks[MIDDLE_PIP][1]:
            fingers[2] = 1

        # 4. Ring finger (Tip vs PIP)
        if landmarks[RING_TIP][1] < landmarks[RING_PIP][1]:
            fingers[3] = 1

        # 5. Pinky finger (Tip vs PIP)
        if landmarks[PINKY_TIP][1] < landmarks[PINKY_PIP][1]:
            fingers[4] = 1

        return fingers

    @staticmethod
    def calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Computes Euclidean distance between two 2D points."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def draw_hand(
        self,
        frame: np.ndarray,
        landmarks: List[Tuple[int, int]],
        color_connections: Tuple[int, int, int] = (180, 180, 180),
        color_points: Tuple[int, int, int] = (0, 255, 255),
    ) -> np.ndarray:
        """Renders hand skeleton connections and landmark points onto the frame."""
        if not landmarks or len(landmarks) < 21:
            return frame

        # Draw bones / connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, landmarks[start_idx], landmarks[end_idx], color_connections, 2, cv2.LINE_AA)

        # Draw joints / landmark circles
        for i, point in enumerate(landmarks):
            # Highlight index tip and middle tip
            if i == INDEX_TIP:
                cv2.circle(frame, point, 8, (0, 0, 255), -1, cv2.LINE_AA)
            elif i == MIDDLE_TIP:
                cv2.circle(frame, point, 7, (255, 200, 0), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, point, 4, color_points, -1, cv2.LINE_AA)

        return frame

    def close(self):
        """Releases the detector resources."""
        if self.detector:
            self.detector.close()
