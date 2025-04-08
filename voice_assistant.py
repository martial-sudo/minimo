import vosk
import json
import pyaudio
import pyttsx3
import datetime
import os
import random
class VoiceAssistant:
    def __init__(self):
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 150)  # Adjust speech speed
        
        # Initialize speech recognition
        model_path = "vosk-model-small-en-us-0.15"
        if not os.path.exists(model_path):
            print(f"Please download Vosk model to {model_path}")
            exit(1)
        
        self.model = vosk.Model(model_path)
        self.rec = vosk.KaldiRecognizer(self.model, 16000)
        
        # Initialize PyAudio stream
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=4096,  # Optimized buffer size
        )
    
    def listen(self):
        """Listen for voice command and return recognized text"""
        try:
            while True:
                data = self.stream.read(4096, exception_on_overflow=False)  # Prevent overflow error
                
                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    return result.get("text", "").strip().lower()  # Avoid KeyError if "text" is missing
        except Exception as e:
            print(f"Error in listen(): {e}")
            return ""  # Return empty string instead of crashing

    def speak(self, text):
        """Convert text to speech"""
        self.engine.stop()  # Stop previous speech to prevent overlap
        self.engine.say(text)
        self.engine.runAndWait()

    def get_audio_input(self, prompt):
        """Prompt user for voice input and return recognized text"""
        self.speak(prompt)
        return self.listen()

    def process_command(self, command):
        """Process recognized voice commands"""
        if "hello" in command or "hi" in command:
            self.speak("Hello! How can I help you today?")
        elif "time" in command:
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            self.speak(f"The current time is {current_time}")
        elif "name" in command:
            self.speak("My name is Minimo")
        elif "joke" in command:
            jokes = [
                "Why don’t skeletons fight each other? Because they don’t have the guts!",
                "What do you call fake spaghetti? An impasta!",
                "Why did the scarecrow win an award? Because he was outstanding in his field!"
            ]
            self.speak(random.choice(jokes))
        else:
            self.speak("Sorry, I didn't understand that command.")

    def close(self):
        """Close audio streams and PyAudio instance"""
        try:
            if self.stream.is_active():
                self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
        except Exception as e:
            print(f"Error closing PyAudio: {e}")

# Example usage
if __name__ == "__main__":
    assistant = VoiceAssistant()
    try:
        while True:
            print("Listening...")
            command = assistant.listen()
            if command:
                print("Heard:", command)
                assistant.process_command(command)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        assistant.close()
