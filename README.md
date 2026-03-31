Hand Gesture Volume Controller

Control your system volume using hand gestures captured through a webcam. This project uses computer vision to detect hand movements and maps finger distance to volume levels in real time.

Overview

This application uses OpenCV for video processing and MediaPipe for hand landmark detection. By measuring the distance between the thumb tip and index fingertip, the system dynamically adjusts volume without any physical interaction.

It demonstrates a simple but effective human-computer interaction system using gesture recognition.

Features
Real-time hand tracking using webcam
Gesture-based volume control (pinch to decrease, spread to increase)
Smooth volume transitions using exponential smoothing
Visual feedback with:
Hand landmarks
Finger distance indicator
Dynamic volume bar
Lightweight and fast execution
How It Works
Webcam captures live video feed
MediaPipe detects 21 hand landmarks
Thumb tip (landmark 4) and index tip (landmark 8) are tracked
Distance between these points is calculated
Distance is mapped to a volume range (0–100%)
Smoothing is applied to reduce jitter
Volume bar and UI update in real time
Tech Stack
Python
OpenCV
MediaPipe
NumPy
Installation
pip install opencv-python mediapipe numpy
Usage
python main.py
Use your thumb and index finger to control volume
Bring fingers closer → decrease volume
Move fingers apart → increase volume
Press Q to quit
Project Structure
├── main.py
├── README.md
Future Improvements
Integrate with system-level audio APIs for actual volume control
Add support for multiple gestures (mute, next track, etc.)
Improve robustness under low lighting conditions
Multi-hand support
Applications
Touchless control systems
Smart home interfaces
Accessibility tools for physically challenged users
AR/VR interaction systems
Limitations
Requires good lighting for accurate detection
Works best with a single hand in frame
Volume control is simulated unless integrated with OS audio APIs
License

This project is open-source and available for educational and research purposes.
