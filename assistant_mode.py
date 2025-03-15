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
        self.response_queue = response_queue or Queue()
        self.running = False
        self.thread = None
        self.logger = logging.getLogger('assistant_mode')
        
        # Initialize Vosk model - Use try/except with better error handling
        try:
            # Use a relative path for better portability
            model_path = "./modules/vosk-model-small-en-us-0.15"
            self.model = Model(model_path)
            self.logger.info(f"Vosk model loaded successfully from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load Vosk model: {e}")
            # Set a flag to indicate model loading failure
            self.model_loaded = False
            # Don't raise here - allow graceful degradation
            self.model = None
        else:
            self.model_loaded = True
        
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
            "scan": 0.7,
            "exit": 0.7,
            "goodbye": 0.7,
            "stop listening": 0.8,
            "stop": 0.8  # Added "stop" as a command to end the conversation
        }
        
        # Active listening state
        self.active_listening = False
        self.last_interaction_time = time.time()
        self.inactive_timeout = 600  # 10 minutes in seconds
        
        # Memory optimization - add a command cache to avoid duplicate processing
        self.command_cache = {}
        self.cache_expiry = 3  # seconds
        
    def start(self):
        """Start the assistant mode thread."""
        if self.thread is None or not self.thread.is_alive():
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            self.logger.info("Assistant mode started")
            self.face_display.update_emotion("happy")
            
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
            try:
                # Check for timeout if in active listening mode
                if self.active_listening and (time.time() - self.last_interaction_time > self.inactive_timeout):
                    self.logger.info("Inactive timeout reached, disabling active listening")
                    self.active_listening = False
                    self.face_display.set_emotion("neutral")
                    self.audio_manager.speak("I'm going back to sleep due to inactivity. Say 'Hey Assistant' to wake me up again.")
                
                # Clean expired command cache entries
                self._clean_command_cache()
                
                if not self.command_queue.empty():
                    command = self.command_queue.get()
                    self._process_command(command)
                else:
                    # Sleep to reduce CPU usage - increased to reduce CPU load
                    time.sleep(0.2)
            except Exception as e:
                self.logger.error(f"Error in assistant mode main loop: {e}")
                # Continue running despite errors
                time.sleep(1.0)
    
    def _clean_command_cache(self):
        """Remove expired entries from command cache to prevent memory leaks."""
        current_time = time.time()
        expired_keys = [k for k, v in self.command_cache.items() if current_time - v["timestamp"] > self.cache_expiry]
        for k in expired_keys:
            del self.command_cache[k]
    
    def process_command(self, command_text):
        """Process a voice command (for compatibility with main.py)."""
        # Clean up command text - remove duplications and extra whitespace
        clean_command = self._clean_command_text(command_text)
        self.enqueue_command(clean_command)
    
    def _clean_command_text(self, command_text):
        """Clean up command text to remove duplications and normalize whitespace."""
        # Split into words and remove duplicates while preserving order
        if not command_text:
            return ""
            
        words = command_text.lower().split()
        seen = set()
        unique_words = []
        for word in words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)
        
        # Check for repeated phrases (common with speech recognition)
        clean_text = " ".join(unique_words)
        
        # Further check for repeated phrases
        phrases = clean_text.split()
        if len(phrases) >= 4:
            half = len(phrases) // 2
            if phrases[:half] == phrases[half:2*half]:
                clean_text = " ".join(phrases[:half])
        
        return clean_text
                
    def process_audio_data(self, audio_data, sample_rate=16000):
        """Process audio data using Vosk model and detect commands."""
        # Check if model is loaded before processing
        if not self.model_loaded:
            self.logger.warning("Vosk model not loaded, cannot process audio")
            return False
            
        # Create recognizer for this audio stream
        try:
            recognizer = KaldiRecognizer(self.model, sample_rate)
            
            if recognizer.AcceptWaveform(audio_data):
                result = json.loads(recognizer.Result())
                if 'text' in result and result['text']:
                    command_text = result['text'].lower()
                    self.logger.debug(f"Recognized text: {command_text}")
                    
                    # If we're already in active listening mode, process any command directly
                    if self.active_listening:
                        # Update the last interaction time
                        self.last_interaction_time = time.time()
                        
                        # Check for stop command (added "stop" as a standalone command)
                        if any(stop_cmd in command_text for stop_cmd in ["exit", "goodbye", "stop listening", "stop"]):
                            self.active_listening = False
                            self.face_display.set_emotion("neutral")
                            self.audio_manager.speak("Goodbye! Say 'Hey Assistant' when you need me again.")
                            return True
                        
                        # Process the command directly, but first clean it up
                        if command_text.strip():
                            clean_command = self._clean_command_text(command_text)
                            
                            # Check for duplicate commands (within cache expiry time)
                            command_hash = hash(clean_command)
                            if command_hash in self.command_cache:
                                self.logger.debug(f"Ignoring duplicate command: {clean_command}")
                                return True
                                
                            # Add to cache and process
                            self.command_cache[command_hash] = {
                                "command": clean_command,
                                "timestamp": time.time()
                            }
                            self.enqueue_command(clean_command)
                            return True
                        
                    # Check for wake word if not in active listening mode
                    elif "hey assistant" in command_text or "hey ai" in command_text:
                        # Enter active listening mode
                        self.active_listening = True
                        self.last_interaction_time = time.time()
                        self.face_display.set_emotion("attentive")
                        self.audio_manager.speak("I'm listening")
                        
                        # Extract the actual command (everything after the wake word)
                        if "hey assistant" in command_text:
                            actual_command = command_text.split("hey assistant", 1)[1].strip()
                        else:
                            actual_command = command_text.split("hey ai", 1)[1].strip()
                            
                        # If there's a command after the wake word, process it
                        if actual_command:
                            clean_command = self._clean_command_text(actual_command)
                            self.enqueue_command(clean_command)
                        
                        return True
        except Exception as e:
            self.logger.error(f"Error processing audio data: {e}")
            return False
                
        return False
                
    def enqueue_command(self, command):
        """Add a command to the processing queue."""
        if command and command.strip():
            self.command_queue.put(command)
        
    def _process_command(self, command):
        """Process a voice command from the user."""
        # Handle both string commands and dictionary commands
        if isinstance(command, dict):
            command_text = command.get("text", "")
        else:
            command_text = command
            
        self.logger.info(f"Processing command: {command_text}")
        
        # Skip empty commands
        if not command_text or not command_text.strip():
            self.logger.warning("Received empty command, skipping")
            return
            
        self.face_display.set_emotion("thinking")
        
        # Check if we should use the camera, activate it before processing
        needs_camera = self._needs_camera(command)
        camera_activated = False
        user = None
        
        try:
            # If camera is needed for this command, activate it
            if needs_camera:
                camera_activated = self._activate_camera()
                
                # Process with camera based on command type
                if camera_activated and ("who" in command or "recognize" in command or "identify" in command):
                    user = self.user_manager.identify_user()
            
            # Parse command and determine response based on camera results
            response = self._generate_response(command, user=user, camera_active=camera_activated)
            
            # Respond to the user
            self.face_display.set_emotion("talking")
            if self.response_queue:
                self.response_queue.put(response)
            else:
                self.audio_manager.speak(response)
            
        except Exception as e:
            self.logger.error(f"Error processing command '{command}': {e}")
            response = "I'm sorry, I encountered an error processing your request."
            self.response_queue.put(response)
        finally:
            # Always deactivate camera if we activated it
            if camera_activated:
                self._deactivate_camera()
            
            # Keep the "attentive" emotion if in active listening mode
            if self.active_listening:
                self.face_display.set_emotion("attentive")
            else:
                self.face_display.set_emotion("neutral")
    
    def _activate_camera(self):
        """Safely activate the camera with resource management."""
        try:
            self.power_manager.request_resource('camera')
            self.camera_manager.activate()
            return True
        except Exception as e:
            self.logger.error(f"Failed to activate camera: {e}")
            return False
            
    def _deactivate_camera(self):
        """Safely deactivate the camera and release resources."""
        try:
            self.camera_manager.deactivate()
            self.power_manager.release_resource('camera')
        except Exception as e:
            self.logger.error(f"Error deactivating camera: {e}")
        
    def _generate_response(self, command, user=None, camera_active=False):
        """Generate a response based on the command and available resources."""
        command = command.lower()
        
        # User identification responses (camera-dependent)
        if ("who am i" in command or "recognize" in command or "identify" in command) and camera_active:
            if user:
                return f"I recognize you as {user['name']}"
            else:
                return "I don't recognize you. Would you like to register as a new user?"
                
        # Camera-based responses when camera is active
        elif "scan" in command and camera_active:
            return "I've scanned the area and everything looks normal."
            
        # Handle case where camera was needed but not available
        elif self._needs_camera(command) and not camera_active:
            return "I'm sorry, I couldn't access the camera to help with that request."
        
        # Time related queries
        elif "time" in command:
            return f"The time is {datetime.datetime.now().strftime('%H:%M')}"
        
        # Weather related queries (just an example, would need an API)
        elif "weather" in command:
            return "I'm sorry, I don't have weather information available right now."
        
        # User identification without camera
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
            return "I can help with time, weather, reminders, and more. Just ask! Say 'stop' when you're done and I'll wait for the wake word again."
        
        # Default response
        else:
            return "I'm not sure how to help with that. Could you try a different command?"
    
    def _needs_camera(self, command):
        """Determine if a command requires camera activation."""
        camera_keywords = ["look", "see", "show", "recognize", "identify", "scan"]
        return any(keyword in command.lower() for keyword in camera_keywords)
        
    def handle_partial_vosk_result(self, partial_result):
        """Handle partial recognition results from Vosk."""
        if not partial_result:
            return False
            
        try:
            if 'partial' in partial_result:
                partial_text = partial_result['partial'].lower()
                
                # Already in active listening mode, show feedback
                if self.active_listening:
                    self.face_display.set_emotion("listening")
                    return True
                    
                # Check for wake words in partial results
                elif "hey assistant" in partial_text or "hey ai" in partial_text:
                    # Light feedback that wake word is being processed
                    self.face_display.set_emotion("listening")
                    return True
                    
        except Exception as e:
            self.logger.error(f"Error handling partial result: {e}")
            
        return False