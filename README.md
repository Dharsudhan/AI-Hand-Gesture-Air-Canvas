# Hand Gesture Recognition Air-Canvas ✋💻🎨

An interactive, touch-free digital drawing application that tracks 21 real-time 3D hand landmarks via a standard consumer webcam. Built using Python, OpenCV, and Google MediaPipe.

---

## ✨ Features

- **Real-Time 21-Landmark Hand Tracking**: Powered by Google MediaPipe Tasks API for sub-millisecond gesture detection without dedicated hardware.
- **Buttery-Smooth Drawing**: Integrates Exponential Moving Average (EMA) coordinate filtering to eliminate webcam frame jitter and sudden coordinate jumps.
- **Touch-Free Virtual Toolbar**: Hover and select colors, clear the canvas, or activate the eraser by raising your index and middle fingers.
- **Dynamic Whiteboard Toggle**: Switch seamlessly between drawing directly over your live camera feed or a crisp digital whiteboard.
- **High-Resolution Export**: Save high-quality PNG snapshots of your artwork with automatic timestamps.
- **Modular Architecture**: Clean separation between computer vision tracking (`hand_tracker.py`), canvas drawing engine (`canvas.py`), and the application loop (`main.py`).

---

## ✋ Gesture Control Guide

| Gesture | Fingertip State | Mode | Action |
| :--- | :--- | :--- | :--- |
| **Index Finger UP** 👆 | Index: UP<br>Others: DOWN | **Draw Mode** | Draws continuous brushstrokes at the index fingertip. |
| **Index + Middle UP** ✌️ | Index: UP<br>Middle: UP<br>Others: DOWN | **Selection Mode** | Free cursor movement. Hover over top buttons to switch color or tool. |
| **Eraser Active + Index UP** 👆 | Eraser Selected<br>Index: UP | **Eraser Mode** | Erases drawings under the fingertip. |
| **Open Palm / Fist** ✋ ✊ | All Fingers UP / Folded | **Standby Mode** | Input locked / paused. Avoids accidental strokes while repositioning. |

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
| :---: | :--- |
| **`c`** | **Clear Canvas**: Instantly wipe the canvas clean. |
| **`s`** | **Save Drawing**: Save high-res PNG to `saved_drawings/`. |
| **`w`** | **Toggle Mode**: Switch between Camera Overlay and Pure Whiteboard. |
| **`e`** | **Toggle Eraser**: Switch between Brush and Eraser mode. |
| **`+` / `-`** | **Adjust Size**: Increase or decrease brush stroke thickness. |
| **`d`** | **Debug View**: Toggle 21-landmark hand skeleton visualization. |
| **`q`** / **`ESC`** | **Quit**: Safely release camera resources and exit. |

---

## 📁 Project Structure

```
air_canvas/
├── hand_tracker.py       # MediaPipe HandLandmarker wrapper & finger state detection
├── canvas.py             # Drawing buffer, virtual UI buttons, smoothing & blending
├── main.py               # Video capture loop, HUD rendering, and hotkey controls
├── requirements.txt      # Python dependencies
├── models/               # Auto-downloaded MediaPipe HandLandmarker model (.task)
└── saved_drawings/       # Destination folder for exported artwork
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ (tested up to Python 3.14)
- A connected USB or built-in webcam

### 2. Environment Setup

Clone or navigate to the project directory:
```bash
cd air_canvas
```

Create and activate a virtual environment:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
python main.py
```

#### Optional CLI Arguments:
```bash
# Specify camera ID (default: 0) and custom resolution:
python main.py --camera 0 --width 1280 --height 720

# Run with skeleton landmark debugging enabled on startup:
python main.py --debug
```

---

## 🛠️ Tech Stack
- **Python**: Core programming language.
- **OpenCV (cv2)**: Frame capture, image processing, coordinate mapping, and HUD rendering.
- **MediaPipe (Google)**: 21-landmark 3D hand tracking and palm detection.
- **NumPy**: Matrix transformations, alpha masking, and high-performance pixel blending.

---

## 💡 Troubleshooting
- **Webcam not opening**: If you have multiple cameras (or an external webcam), try running `python main.py --camera 1`.
- **Lighting and Jitter**: Ensure your hand is evenly illuminated against the background for optimal landmark confidence scores.
