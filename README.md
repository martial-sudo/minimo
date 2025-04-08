# Minimo AI Assistant
#  MINIATURE OPERATIONAL MODEL FOR PERSONAL ASSISTANCE AND SECURITY

## Overview

Minimo is an AI-powered assistant for Raspberry Pi with facial recognition, voice interaction, emotion detection, and security features.
The system integrates several componentsto create a smart assistant that can:

- Recognize registered users by face
- Respond to voice commands
- Detect user emotions
- Display reactive facial expressions
- Switch between day and night modes automatically
- Provide intruder detection during night hours

## Components

### 1. Facial Recognition System
- Uses OpenCV and DeepFace for face detection and recognition
- Maintains a database of registered users
- Performs real-time facial recognition
- Detects user emotions (happy, sad, neutral)

### 2. Voice Assistant
- Voice recognition using Vosk
- Text-to-speech using pyttsx3
- Processes basic commands like greeting, time queries, jokes

### 3. Face Display
- Animated face with different emotions (happy, sad, thinking, etc.)
- Dynamic facial expressions with blinking eyes
- Mouth animation during speech
- Night mode clock display

### 4. Intruder Detection
- Motion detection during night hours
- Triggers alerts when sustained motion is detected
- Background subtraction for reliable motion detection

### 5. Day/Night Mode Manager
- Automatically switches between day and night modes
- Day mode: Interactive assistant with facial recognition
- Night mode: Security monitoring and intruder detection

## Requirements

### Hardware
- Raspberry Pi (recommended: Pi 4 with at least 2GB RAM)
- Camera module (USB webcam or Raspberry Pi Camera)
- Display screen (for facial expressions)
- Microphone
- Speakers

### Software Dependencies
- Python 3.7+
- OpenCV
- MediaPipe
- DeepFace
- Pygame
- SQLite3
- Vosk (with English model)[Download the model from vosk website and mention the path in code]
- pyttsx3
- PyAudio
- NumPy

## Installation

1. **Clone the repository**
   ```
   git clone https://github.com/yourusername/Minimo-ai.git
   cd Minimo-ai
   ```

2. **Install dependencies**
   ```
   pip install opencv-python mediapipe deepface pygame vosk pyttsx3 pyaudio numpy
   ```

3. **Download Vosk speech recognition model**
   ```
   wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
   unzip vosk-model-small-en-us-0.15.zip
   ```

4. **Install SQLite (if not already installed)**
   ```
   sudo apt-get update
   sudo apt-get install sqlite3
   ```

5. **Optional: Prepare alarm sound**
   - Place an alarm sound file named `alarm.mp3` in the project directory for intruder alerts

## Usage

1. **Start the application**
   ```
   python main.py
   ```

2. **Register your face**
   - When the system starts, say "hello" to initiate facial recognition
   - If not recognized, the system will prompt you to register your face and name

3. **Voice Commands**
   - Say "hello" to initiate interaction
   - Ask for the time: "What time is it?"
   - Ask for the assistant's name: "What's your name?"
   - Request a joke: "Tell me a joke"
   - End conversation: "exit"

4. **Night Mode**
   - The system automatically switches to night mode between 10 PM and 6 AM
   - The display shows a digital clock
   - Motion detection is activated for security

## Code Structure

- `main.py` - Main application entry point
- `Face_recognition.py` - Face recognition and emotion detection
- `voice_assistant.py` - Voice recognition and speech synthesis
- `face_display.py` - Animated facial display
- `intruder_detection.py` - Motion detection and security alerts
- `dayornight.py` - Day/night mode detection
- `camm.py` - Camera testing utility

## Customization

### Adjusting Face Recognition Sensitivity
In `Face_recognition.py`, you can modify the recognition threshold:
```python
# Lower values = more strict matching
return best_match if best_similarity < 15 else "Unknown"
```

### Motion Detection Sensitivity
In `intruder_detection.py`, adjust these parameters:
```python
intruder_system = IntruderDetectionSystem(
    sensitivity=500,  # Lower = more sensitive
    threshold_duration=3  # Duration in seconds
)
```

### Adding New Voice Commands
Extend the `process_command` method in `voice_assistant.py`:
```python
def process_command(self, command):
    # Existing commands...
    elif "your_new_command" in command:
        # Your custom action here
```

## Troubleshooting

### Camera Issues
- If the camera doesn't initialize, check the camera index in `main.py` (default is 2)
- Try different indices (0, 1, 2) depending on your setup
- Ensure camera permissions are set correctly

### Voice Recognition Problems
- Check microphone connections
- Ensure the Vosk model is downloaded and in the correct path
- Speak clearly and not too fast

### Display Issues
- Verify pygame is correctly installed
- Adjust screen resolution in `face_display.py` if needed

## Future Enhancements

- Web interface for remote monitoring
- Cloud backup of face database
- Advanced conversation capabilities
- Smart home integration
- Custom wake word detection
- Weather information
- Calendar integration

## License

This project was created for educational purposes as part of academic coursework at APJKTU. 

Copyright (c) 2003 martial-sudo

This code is submitted as part of an academic assignment and is not licensed for commercial use. All rights reserved.

Note: The project uses several open-source libraries and components, each with their own licenses.

## Acknowledgments

- [Vosk](https://alphacephei.com/vosk/) for offline speech recognition
- [DeepFace](https://github.com/serengil/deepface) for facial analysis
- [Pygame](https://www.pygame.org/) for display graphics
- [OpenCV](https://opencv.org/) for computer vision
