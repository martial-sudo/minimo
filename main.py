import threading
import datetime
import time 
import queue
#import openai  # Import OpenAI API
import os
import webbrowser
import cv2

from Face_recognition import FaceRecognitionSystem
from voice_assistant import VoiceAssistant
from intruder_detection import IntruderDetectionSystem
from face_display import FaceDisplay
from dayornight import DayNightDetector


class AIBot:
    def __init__(self):
        self.response_queue = queue.Queue()
        
        # Initialize shared camera first
        print("Initializing shared camera...")
        self.shared_camera = cv2.VideoCapture(2)
        
        if not self.shared_camera.isOpened():
            print("⚠️ WARNING: Failed to open shared camera. Will try individual initialization.")
            self.shared_camera = None
        else:
            print("✓ Successfully opened shared camera!")
            
        # Initialize components with shared camera
        self.face_recognition = FaceRecognitionSystem(shared_camera=self.shared_camera)
        self.voice_assistant = VoiceAssistant()
        self.intruder_detection = IntruderDetectionSystem(shared_camera=self.shared_camera)
        self.face_display = FaceDisplay(self.response_queue)
        self.shutdown_event = threading.Event()
        self.face_display_thread = threading.Thread(target=self.face_display.run, args=(self.shutdown_event,), daemon=True)
        self.face_display_thread.start()
        self.day_night_detector = DayNightDetector()

    def day_mode(self):
        print("Entering Day Mode")
        self.voice_assistant.speak("Entering day mode")

        while True:
            try:
                command = self.voice_assistant.listen()
                print(command)

                if "hello" in command:
                    self.face_display.update_emotion("attentive")
                    names, frame = self.face_recognition.capture_and_recognize_face()

                    if names and names[0] != "Unknown" and names[0] != "Error":
                        greeting = f"Hello {names[0]}!"
                        self.voice_assistant.speak(greeting)
                        self.response_queue.put(greeting)
                        self.face_display.update_emotion("happy")

                        # 🟢 New Feature: Detect user's emotion
                        _, frame = self.face_recognition.capture_and_recognize_face()
                        if frame is not None:
                            emotion = self.face_recognition.detect_emotion(frame)
                            print(emotion)  # Add this function in Face_recognition.py

                            if emotion == "happy":
                                self.voice_assistant.speak(f"You look happy, {names[0]}! Let's have a conversation.")
                                self.start_conversation()
                            elif emotion == "sad":
                                self.voice_assistant.speak(f"{names[0]}, you look sad. What's wrong?")
                                user_response = self.voice_assistant.listen()
                                self.voice_assistant.speak(f"I understand, {names[0]}. I'm here to help. Let me know if you want to talk.")
                                self.face_display.update_emotion("caring")
                                self.start_conversation()
                            elif  emotion == "neutral":
                                self.start_conversation()


                    elif names and names[0] == "Error":
                        self.voice_assistant.speak("I'm having trouble with the camera. Please check the connections.")
                        self.face_display.update_emotion("confused")
                    else:
                        response = "I don't recognize you. Let me help you register."
                        self.voice_assistant.speak(response)
                        self.response_queue.put(response)
                        self.face_display.update_emotion("confused")

                        # Capture face again for embedding
                        _, frame = self.face_recognition.capture_and_recognize_face()
                        if frame is not None:
                            embedding = self.face_recognition.get_embedding(frame)

                            if embedding is not None:
                                self.voice_assistant.speak("Tell me your name")
                                name = self.voice_assistant.listen()
                                self.face_recognition.add_new_user(name, embedding)
                                self.voice_assistant.speak(f"Registered {name} successfully!")
                                self.face_display.update_emotion("happy")
                            else:
                                self.voice_assistant.speak("Face capture failed. Please try again.")
                                self.face_display.update_emotion("confused")
                        else:
                            self.voice_assistant.speak("Camera error. Please check connections.")
                            self.face_display.update_emotion("confused")

                time.sleep(0.3)
            except Exception as e:
                print(f"Error in day_mode: {e}")
                self.voice_assistant.speak("An error occurred. Restarting day mode.")


    def night_mode(self):
        """Intruder detection mode"""
        print("Entering Night Mode")
        while self.day_night_detector.is_night_time():
            try:
                self.intruder_detection.run_night_mode()
                
                # Short sleep to prevent high CPU usage
                time.sleep(0.1)
            
            except Exception as e:
                print(f"Error in night mode: {e}")
                time.sleep(1)
    def start_conversation(self):
        """Start an interactive conversation without OpenAI API."""
        self.voice_assistant.speak("How can I assist you today?")
        while True:
            user_input = self.voice_assistant.listen()
            if not user_input:
                continue
            if "exit" in user_input.lower():
                self.voice_assistant.speak("Goodbye!")
                break
            self.voice_assistant.process_command(user_input)

    def run(self):
        try:
            if self.day_night_detector.is_night_time():
                self.night_mode()
            else:
                self.day_mode()
        except KeyboardInterrupt:
            print("AI Bot shutting down...")
            self.shutdown_event.set()
        finally:
            # Clean up resources
            self.face_recognition.close_connection()
            self.voice_assistant.close()
            self.intruder_detection.close()
            
            # Release shared camera if it exists
            if self.shared_camera is not None and self.shared_camera.isOpened():
                self.shared_camera.release()
                print("Released shared camera")

if __name__ == "__main__":
    bot = AIBot()
    bot.run()