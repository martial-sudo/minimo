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
            model_path = r"F:/desktop/minimo/modules/vosk-model-small-en-us-0.15"
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
        
        # User tracking
        self.current_user = None
        self.last_user_check = 0
        self.user_check_interval = 10  # Check for user every 10 seconds
        
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
                
                # Periodically check for users in view
                current_time = time.time()
                if (current_time - self.last_user_check) > self.user_check_interval:
                    self._check_for_users()
                    self.last_user_check = current_time
                
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
    
    def _check_for_users(self):
        """Check if there are any recognized users in view of the camera."""
        try:
            # Only check if camera is available and we're not actively listening
            if not self.active_listening and not self.camera_manager.capturing:
                camera_active = self._activate_camera()
                if not camera_active:
                    return
                    
                try:
                    recognized_user = self.user_manager.identify_user()
                    if recognized_user:
                        # If this is a new user or different from the previous user
                        if self.current_user is None or self.current_user['id'] != recognized_user['id']:
                            self.current_user = recognized_user
                            self._greet_user(recognized_user)
                finally:
                    self._deactivate_camera()
                    
        except Exception as e:
            self.logger.error(f"Error checking for users: {e}")
    
    def _greet_user(self, user):
        """Greet a recognized user."""
        # Calculate time since last seen
        current_time = time.time()
        time_since_last_seen = current_time - user['last_seen']
        
        # Get the hour of the day for time-appropriate greeting
        current_hour = datetime.datetime.now().hour
        
        # Choose greeting based on time since last seen and time of day
        greeting = ""
        
        # Time of day greeting
        if current_hour < 12:
            greeting = "Good morning"
        elif current_hour < 18:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"
            
        # Add personalized part based on how recently seen
        if time_since_last_seen < 3600:  # Less than an hour
            message = f"{greeting}, {user['name']}. Welcome back."
        elif time_since_last_seen < 86400:  # Less than a day
            message = f"{greeting}, {user['name']}. Nice to see you again today."
        else:
            # Get days since last seen
            days_since = int(time_since_last_seen / 86400)
            if days_since == 1:
                message = f"{greeting}, {user['name']}. I haven't seen you since yesterday."
            else:
                message = f"{greeting}, {user['name']}. It's been {days_since} days since I last saw you."
        
        # Activate and greet
        self.face_display.set_emotion("happy")
        self.audio_manager.speak(message)
        
        # Wait a moment before returning to neutral
        time.sleep(2)
        self.face_display.set_emotion("neutral")
    
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
                        
                        # Check for user
                        if not self.current_user:
                            camera_active = self._activate_camera()
                            if camera_active:
                                try:
                                    self.current_user = self.user_manager.identify_user()
                                finally:
                                    self._deactivate_camera()
                        
                        # Personalized greeting if user is recognized
                        if self.current_user:
                            self.audio_manager.speak(f"I'm listening, {self.current_user['name']}")
                        else:
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
            
            # User management commands
            elif "register me" in command or "add me as user" in command or "remember me" in command:
                camera_active = self._activate_camera()
                if camera_active:
                    name = self._extract_name(command)
                    if not name:
                        self.audio_manager.speak("What should I call you?")
                        # Ideally we'd wait for a response here, but for simplicity
                        name = "Unknown User"
                    
                    success = self.user_manager.add_user(name)
                    if success:
                        response = f"Nice to meet you, {name}. I'll remember you."
                        # Update current user
                        self.current_user = self.user_manager.identify_user()
                    else:
                        response = "I'm sorry, I couldn't register you. Please try again."
                    self._deactivate_camera()
                else:
                    response = "Camera is not available for registration"
                
            # Who am I command
            elif "who am i" in command:
                camera_active = self._activate_camera()
                if camera_active:
                    user = self.user_manager.identify_user()
                    if user:
                        self.current_user = user
                        response = f"I recognize you as {user['name']}"
                    else:
                        response = "I don't recognize you. Would you like me to remember you?"
                    self._deactivate_camera()
                else:
                    response = "Camera is not available"
                    
            # Forget me command
            elif "forget me" in command and self.current_user:
                response = f"Are you sure you want me to forget you, {self.current_user['name']}? Please say 'yes, forget me' to confirm."
            
            # Confirmation for forget me
            elif "yes, forget me" in command and self.current_user:
                success = self.user_manager.remove_user(self.current_user['id'])
                if success:
                    response = f"I've removed your profile, {self.current_user['name']}. I won't remember you anymore."
                    self.current_user = None
                else:
                    response = "I'm sorry, I couldn't remove your profile."
                    
            # Update my face command
            elif "update my face" in command and self.current_user:
                camera_active = self._activate_camera()
                if camera_active:
                    success = self.user_manager.update_face_data(self.current_user['id'])
                    if success:
                        response = f"I've updated your face data, {self.current_user['name']}."
                    else:
                        response = "I couldn't update your face data. Please try again."
                    self._deactivate_camera()
                else:
                    response = "Camera is not available"
                    
            # List all users command
            elif "list users" in command or "who do you know" in command:
                users = self.user_manager.list_users()
                if users:
                    names = [user['name'] for user in users]
                    response = f"I know {len(names)} people: {', '.join(names)}"
                else:
                    response = "I don't know anyone yet"
                    
            # Power options
            elif "shutdown" in command or "power off" in command:
                response = "Shutting down system"
                # Queue shutdown after response
                threading.Timer(3.0, self.power_manager.shutdown).start()
                
            # Restart options
            elif "restart" in command or "reboot" in command:
                response = "Restarting system"
                # Queue restart after response
                threading.Timer(3.0, self.power_manager.restart).start()
                
            # Default response
            else:
                response = "I'm not sure how to help with that"
                
            # Log the conversation if we have a recognized user
            if self.current_user:
                self.user_manager.log_conversation(self.current_user['id'], command, response)
                
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
    
    def _extract_name(self, command):
        """Extract name from command."""
        name_indicators = ["call me ", "my name is ", "i am "]
        command = command.lower()
        
        for indicator in name_indicators:
            if indicator in command:
                name = command.split(indicator, 1)[1].strip()
                # Remove any trailing commands or punctuation
                for word in ["and", "but", "please", ".", "?"]:
                    if word in name:
                        name = name.split(word, 1)[0].strip()
                return name
                
        return None
                
    def _needs_camera(self, command):
        """Check if command requires camera."""
        camera_keywords = ["look", "see", "show", "recognize", "identify", "scan", "who am i", "register me", "remember me"]
        return any(keyword in command for keyword in camera_keywords)
        
    def _activate_camera(self):
        """Activate camera."""
        try:
            self.power_manager.request_resource('camera')
            self.camera_manager.start_capture()
            return True
        except Exception as e:
            self.logger.error(f"Failed to activate camera: {e}")
            return False
            
    def _deactivate_camera(self):
        """Deactivate camera."""
        try:
            self.camera_manager.stop_capture()
            self.power_manager.release_resource('camera')
        except Exception as e:
            self.logger.error(f"Error deactivating camera: {e}")
    
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