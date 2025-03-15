import threading
import time
import datetime
import logging
import json
import requests
import webbrowser
import os
from queue import Queue
from vosk import Model, KaldiRecognizer

class AssistantMode:
    def __init__(self, audio_manager, camera_manager, face_display, user_manager, power_manager, command_queue=None, response_queue=None):
        """Initialize the day-time personal assistant mode."""
        self.audio_manager = audio_manager
        self.camera_manager = camera_manager  # Uses Raspberry Pi Camera
        self.face_display = face_display
        self.user_manager = user_manager
        self.power_manager = power_manager
        self.command_queue = command_queue or Queue()
        self.response_queue = response_queue or Queue()
        self.running = False
        self.thread = None
        self.logger = logging.getLogger('assistant_mode')
        
        # Initialize Vosk model
        try:
            model_path = r"./modules/vosk-model-small-en-us-0.15"
            self.model = Model(model_path)
            self.model_loaded = True
        except Exception as e:
            self.logger.error(f"Failed to load Vosk model: {e}")
            self.model_loaded = False
            self.model = None
        
        # API keys from environment variables
        self.weather_api_key = os.environ.get('WEATHER_API_KEY', '')
        self.search_api_key = os.environ.get('SEARCH_API_KEY', '')
        
        # Command types and state tracking
        self.active_listening = False
        self.last_interaction_time = time.time()
        self.inactive_timeout = 600  # 10 minutes
        self.command_cache = {}
        self.cache_expiry = 3  # seconds
        self.user_location = "New York"  # Default location
        
    def start(self):
        """Start the assistant mode thread."""
        if self.thread is None or not self.thread.is_alive():
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            self.face_display.update_emotion("happy")
            
    def stop(self):
        """Stop the assistant mode thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
            
    def _run(self):
        """Main thread loop for processing commands."""
        while self.running:
            try:
                # Check for inactivity timeout
                if self.active_listening and (time.time() - self.last_interaction_time > self.inactive_timeout):
                    self.active_listening = False
                    self.face_display.set_emotion("neutral")
                    self.audio_manager.speak("Going to sleep mode. Say 'Hey Assistant' to wake me up.")
                
                # Process commands in queue
                if not self.command_queue.empty():
                    command = self.command_queue.get()
                    self._process_command(command)
                else:
                    time.sleep(0.2)  # Reduced CPU usage
                    
                # Clean expired cache entries
                self._clean_command_cache()
                
            except Exception as e:
                self.logger.error(f"Error in assistant loop: {e}")
                time.sleep(1.0)
    
    def _clean_command_cache(self):
        """Remove expired command cache entries."""
        current_time = time.time()
        expired_keys = [k for k, v in self.command_cache.items() 
                        if current_time - v["timestamp"] > self.cache_expiry]
        for k in expired_keys:
            del self.command_cache[k]
                
    def process_audio_data(self, audio_data, sample_rate=16000):
        """Process audio data with Vosk for command recognition."""
        if not self.model_loaded:
            return False
            
        try:
            recognizer = KaldiRecognizer(self.model, sample_rate)
            
            if recognizer.AcceptWaveform(audio_data):
                result = json.loads(recognizer.Result())
                if 'text' in result and result['text']:
                    command_text = result['text'].lower()
                    
                    # Active listening mode - process any command
                    if self.active_listening:
                        self.last_interaction_time = time.time()
                        
                        # Check for stop commands
                        if any(cmd in command_text for cmd in ["exit", "goodbye", "stop"]):
                            self.active_listening = False
                            self.face_display.set_emotion("neutral")
                            self.audio_manager.speak("Goodbye! Say 'Hey Assistant' when you need me.")
                            return True
                        
                        # Process valid commands
                        if command_text.strip():
                            clean_command = self._clean_text(command_text)
                            cmd_hash = hash(clean_command)
                            
                            # Skip duplicates
                            if cmd_hash not in self.command_cache:
                                self.command_cache[cmd_hash] = {
                                    "command": clean_command,
                                    "timestamp": time.time()
                                }
                                self.command_queue.put(clean_command)
                            return True
                    
                    # Wake word detection
                    elif "hey assistant" in command_text or "hey ai" in command_text:
                        self.active_listening = True
                        self.last_interaction_time = time.time()
                        self.face_display.set_emotion("attentive")
                        self.audio_manager.speak("I'm listening")
                        
                        # Extract command after wake word
                        if "hey assistant" in command_text:
                            command = command_text.split("hey assistant", 1)[1].strip()
                        else:
                            command = command_text.split("hey ai", 1)[1].strip()
                            
                        if command:
                            self.command_queue.put(self._clean_text(command))
                        
                        return True
                        
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            
        return False
    
    def _clean_text(self, text):
        """Clean up recognition text."""
        if not text:
            return ""
            
        # Remove duplicated words
        words = text.lower().split()
        unique_words = []
        seen = set()
        
        for word in words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)
                
        return " ".join(unique_words)
        
    def _process_command(self, command):
        """Process a command and generate a response."""
        # Normalize command format
        if isinstance(command, dict):
            command = command.get("text", "")
            
        if not command or not command.strip():
            return
            
        self.face_display.set_emotion("thinking")
        
        try:
            # Process the command and generate response
            command = command.lower()
            response = ""
            
            # Time commands
            if "time" in command:
                now = datetime.datetime.now()
                response = f"It's {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d')}"
                
            # Weather commands
            elif "weather" in command:
                location = self._extract_location(command) or self.user_location
                response = self._get_weather(location)
                
            # Set location
            elif "set location" in command or "my location is" in command:
                location = self._extract_location(command)
                if location:
                    self.user_location = location
                    response = f"Location set to {location}"
                else:
                    response = "Please specify a location"
                    
            # Search commands
            elif "search" in command or "google" in command:
                query = self._extract_query(command, ["search for", "google"])
                if query:
                    response = f"Searching for: {query}"
                    webbrowser.open(f"https://www.google.com/search?q={query.replace(' ', '+')}")
                else:
                    response = "What would you like to search for?"
                    
            # YouTube commands
            elif "youtube" in command or ("play" in command and "video" in command):
                query = self._extract_query(command, ["youtube", "play video", "video of", "video for"])
                if query:
                    response = f"Opening YouTube video for: {query}"
                    webbrowser.open(f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}")
                else:
                    response = "What video would you like to watch?"
                    
            # Help command
            elif "help" in command:
                response = "I can tell time, check weather, search Google, play YouTube videos, and more."
                
            # Camera commands handled by existing code
            elif self._needs_camera(command):
                camera_active = self._activate_camera()
                if camera_active:
                    if "who am i" in command:
                        user = self.user_manager.identify_user()
                        response = f"I recognize you as {user['name']}" if user else "I don't recognize you"
                    elif "scan" in command:
                        response = "Area scanned, everything looks normal"
                    self._deactivate_camera()
                else:
                    response = "Camera is not available"
                    
            # Default response
            else:
                response = "I'm not sure how to help with that"
                
            # Send response
            self.face_display.set_emotion("talking")
            if self.response_queue:
                self.response_queue.put(response)
            else:
                self.audio_manager.speak(response)
                
        except Exception as e:
            self.logger.error(f"Error processing command: {e}")
            response = "I encountered an error with that request"
            if self.response_queue:
                self.response_queue.put(response)
                
        finally:
            # Reset facial expression
            if self.active_listening:
                self.face_display.set_emotion("attentive")
            else:
                self.face_display.set_emotion("neutral")
    
    def _needs_camera(self, command):
        """Check if command requires camera."""
        camera_keywords = ["look", "see", "show", "recognize", "identify", "scan", "who am i"]
        return any(keyword in command for keyword in camera_keywords)
        
    def _activate_camera(self):
        """Activate Raspberry Pi camera."""
        try:
            self.power_manager.request_resource('camera')
            self.camera_manager.activate_rpi_camera()
            return True
        except Exception as e:
            self.logger.error(f"Failed to activate Raspberry Pi camera: {e}")
            return False
        
            
    def _deactivate_camera(self):
        """Deactivate Raspberry Pi camera."""
        try:
            self.camera_manager.deactivate_rpi_camera()
            self.power_manager.release_resource('camera')
        except Exception as e:
            self.logger.error(f"Error deactivating Raspberry Pi camera: {e}")
    
    def _extract_location(self, command):
        """Extract location from command."""
        location_indicators = ["in ", "for ", "at "]
        command = command.lower()
        
        # Remove command words
        for word in ["weather", "forecast", "temperature", "set location to", "my location is"]:
            command = command.replace(word, "")
            
        # Extract location after indicators
        for indicator in location_indicators:
            if indicator in command:
                return command.split(indicator, 1)[1].strip()
                
        return None
        
    def _extract_query(self, command, prefixes):
        """Extract search query after removing prefixes."""
        command = command.lower()
        
        for prefix in prefixes:
            if prefix in command:
                return command.split(prefix, 1)[1].strip()
                
        # Extract everything after first few words if no prefix found
        words = command.split()
        if len(words) > 2:
            return " ".join(words[2:])
            
        return None
        
    def _get_weather(self, location):
        """Get weather information for location."""
        try:
            if not self.weather_api_key:
                return f"Weather feature requires API key configuration. Weather lookup for {location} not available."
                
            url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={self.weather_api_key}&units=imperial"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if response.status_code == 200:
                temp = data["main"]["temp"]
                condition = data["weather"][0]["description"]
                humidity = data["main"]["humidity"]
                return f"In {location}: {condition}, {temp}°F with {humidity}% humidity"
            else:
                return f"Couldn't get weather for {location}"
                
        except Exception as e:
            self.logger.error(f"Weather API error: {e}")
            return f"Weather service unavailable for {location}"
