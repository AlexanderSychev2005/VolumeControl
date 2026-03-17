# 🖐️ AI Hand Gesture Volume Control
A real-time system that controls system audio volume using hand gestures and monitors user presence via facial detection.

## Table of Contents
- [🖐️ AI Hand Gesture Volume Control](#️-ai-hand-gesture-volume-control)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Technical Architecture](#technical-architecture)
  - [Mathematical Basis](#mathematical-basis)
    - [1. Euclidean Distance](#1-euclidean-distance)
    - [2. Linear Interpolation](#2-linear-interpolation)
    - [3. Signal Smoothing](#3-signal-smoothing)
  - [Installation](#installation)
  - [How it works](#how-it-works)

## Overview
This project leverages computer vision to provide a hands-free volume control experience. It uses facial detection to ensure the user is present and hand landmark tracking to map finger proximity to system volume levels.

## Technical Architecture
The system pipeline consists of three main components:
* **Detection Layer:** Uses Google's `MediaPipe` to identify facial presence and hand landmarks (specifically points 4 and 8).
* **Processing Layer:** Computes geometric distances and applies a `deque`-based rolling average filter to ensure smooth volume transitions.
* **Control Layer:** Interfaces with the Windows Core Audio API via `pycaw` to set the master volume.

## Mathematical Basis
The core functionality relies on calculating the Euclidean distance between the thumb and index finger.


### 1. Euclidean Distance
The distance $d$ between the thumb $(x_1, y_1)$ and index finger $(x_2, y_2)$ is calculated as:
```math
d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
```

### 2. Linear Interpolation
To map the pixel distance $d$ to the system's decibel range $[v_{min}, v_{max}]$, we use linear interpolation. Given a measured distance $d \in [d_{min}, d_{max}]$, the mapped volume $v$ is:
```math
v = v_{min} + \frac{(d - d_{min}) \cdot (v_{max} - v_{min})}{d_{max} - d_{min}}
```


### 3. Signal Smoothing
To prevent "jitter" caused by noisy sensor data, we use a moving average filter:
```math
\bar{v}_t = \frac{1}{N} \sum_{i=0}^{N-1} v_{t-i}
```
Where $N$ is the smoothing window size (set to 8 in this implementation).

## Installation
1. Ensure you have Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.
2. Install the required dependencies:
```bash
uv sync
```

## How it works
1. **User Presence**: The face detector runs continuously. If no face is detected, the `minVol` is set, and the program enters a muted state.
2. **Gesture Interaction**: When a face is detected, the hand landmarker processes the video frame.
3. **Feedback**: The system draws the landmark skeleton and a visual progress bar, providing immediate UI feedback.