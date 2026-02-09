🖐️ AI Hand Gesture Volume Control
A computer vision project that allows you to control system volume using hand gestures. Built with Python, OpenCV, and Google's latest MediaPipe Hand Landmarker.

📝 Description
This project uses a webcam to detect hand landmarks in real-time. By calculating the distance between the Thumb and the Index Finger, the application adjusts the system's master volume.

It features a smoothing algorithm (moving average) to prevent volume flickering caused by hand tremors or detection jitter, ensuring a smooth user experience.

🚀 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AlexanderSychev2005/VolumeControl
   cd VolumeControl
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the application:
    ```bash
    python main.py
    ```