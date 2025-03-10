import threading
import time
import datetime
import logging
import json
from queue import Queue
from vosk import Model, KaldiRecognizer

class AssistantMode:
    def __init__(self, audio_manager, camera_manager, face_display, user_manager, power_manager, command_queue=None, response_queue=None):
        """Initialize the day-time personal assistant mode."""
        self.audio_manager = audio_manager
        self.camera_manager = camera_manager
        self.face_display = face_display
        self.user_manager = user_manager
        self.power_manager = power_manager
        self.command_queue = command_queue or Queue()
        self.response_queue = response_queue
        self.running = False
        self.thread = None
        self.logger = logging.getLogger('assistant_mode')
        
        # Initialize Vosk model
        try:
            self.model = Model(r"F:\desktop\minimo\modules\vosk-model-small-en-us-0.15")
            self.logger.info("Vosk model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Vosk model: {e}")
            raise
        
        # Voice commands with confidence thresholds
        self.commands = {
            "time": 0.7,
            "weather": 0.7,
            "who am i": 0.75,
            "schedule": 0.7,
            "calendar": 0.7,
            "help": 0.6,
            "look": 0.7,
            "see": 0.7,
            "show": 0.7,
            "recognize": 0.8,
            "identify": 0.8,
            "scan": 0.7
        }
        
    def start(self):
        """Start the assistant mode thread."""
        if self.thread is None or not self.thread.is_alive():
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            self.logger.info("Assistant mode started")
            self.face_display.set_emotion("happy")
            
    def stop(self):
        """Stop the assistant mode thread."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
            self.logger.info("Assistant mode stopped")
            
    def _run(self):
        """Main loop for the assistant mode."""
        while self.running:
            if not self.command_queue.empty():
                command = self.command_queue.get()
                self._process_command(command)
            else:
                # Sleep to reduce CPU usage
                time.sleep(0.1)
    
    def process_command(self, command_text):
        """Process a voice command (for compatibility with main.py)."""
        self.enqueue_command(command_text)
                
    def process_audio_data(self, audio_data, sample_rate=16000):
        """Process audio data using Vosk model and detect commands."""
        # Create recognizer for this audio stream
        recognizer = KaldiRecognizer(self.model, sample_rate)
        
        if recognizer.AcceptWaveform(audio_data):
            result = json.loads(recognizer.Result())
            if 'text' in result and result['text']:
                command_text = result['text'].lower()
                self.logger.debug(f"Recognized text: {command_text}")
                
                # Check for wake word before processing the command
                if "hey assistant" in command_text or "hey ai" in command_text:
                    # Extract the actual command (everything after the wake word)
                    if "hey assistant" in command_text:
                        actual_command = command_text.split("hey assistant", 1)[1].strip()
                    else:
                        actual_command = command_text.split("hey ai", 1)[1].strip()
                        
                    # If there's a command after the wake word, process it
                    if actual_command:
                        self.enqueue_command(actual_command)
                        return True
                    else:
                        # Just the wake word was detected
                        self.face_display.set_emotion("attentive")
                        self.audio_manager.speak("I'm listening")
                        return True
                
        return False
                
    def enqueue_command(self, command):
        """Add a command to the processing queue."""
        self.command_queue.put(command)
        
    def _process_command(self, command):
        """Process a voice command from the user."""
        self.logger.info(f"Processing command: {command}")
        self.face_display.set_emotion("thinking")
        
        # Parse command and determine response
        response = self._generate_response(command)
        
        # If camera is needed for this command, activate it
        if self._needs_camera(command):
            self.power_manager.request_resource('camera')
            self.camera_manager.activate()
            
            # Process with camera based on command type
            if "who" in command or "recognize" in command or "identify" in command:
                user = self.user_manager.identify_user()
                if user:
                    response = f"I recognize you as {user['name']}"
                else:
                    response = "I don't recognize you. Would you like to register as a new user?"
            elif "scan" in command:
                response = "I've scanned the area and everything looks normal."
                
            self.camera_manager.deactivate()
            self.power_manager.release_resource('camera')
        
        # Respond to the user
        self.face_display.set_emotion("talking")
        if self.response_queue:
            self.response_queue.put(response)
        else:
            self.audio_manager.speak(response)
        self.face_display.set_emotion("neutral")
        
    def _generate_response(self, command):
        """Generate a response based on the command."""
        command = command.lower()
        
        # Time related queries
        if "time" in command:
            return f"The time is {datetime.datetime.now().strftime('%H:%M')}"
        
        # Weather related queries (just an example, would need an API)
        elif "weather" in command:
            return "I'm sorry, I don't have weather information available right now."
        
        # User identification
        elif "who am i" in command:
            if self.user_manager.current_user:
                return f"You are {self.user_manager.current_user['name']}"
            else:
                return "I don't recognize you yet. Would you like to register?"
        
        # Calendar/schedule queries (would need integration)
        elif "schedule" in command or "calendar" in command:
            return "You don't have any upcoming appointments today."
        
        # Help command
        elif "help" in command:
            return "I can help with time, weather, reminders, and more. Just ask!"
        
        # Default response
        else:
            return "I'm not sure how to help with that. Could you try a different command?"
    
    def _needs_camera(self, command):
        """Determine if a command requires camera activation."""
        camera_keywords = ["look", "see", "show", "recognize", "identify", "scan"]
        return any(keyword in command.lower() for keyword in camera_keywords)
        
    def handle_partial_vosk_result(self, partial_result):
        """Handle partial recognition results from Vosk."""
        if partial_result and 'partial' in partial_result:
            partial_text = partial_result['partial'].lower()
            
            # Check for wake words in partial results
            if "hey assistant" in partial_text or "hey ai" in partial_text:
                # Light feedback that wake word is being processed
                self.face_display.set_emotion("listening")
                return True
                
        return False