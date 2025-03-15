#!/usr/bin/env python3
import threading
import time
import datetime
import logging
from enum import Enum
import queue

# Import our custom modules
import audio_manager
import camera_manager
import face_display
import user_manager
import assistant_mode
import security_mode
import power_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("assistant.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class SystemState(Enum):
    IDLE = 0
    LISTENING = 1
    PROCESSING = 2
    RESPONDING = 3
    SECURITY = 4

class EventManager:
    def __init__(self):
        self.listeners = {}
        self.event_queue = queue.Queue()
        self._running = True
        
    def subscribe(self, event_type, callback):
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)
        
    def publish(self, event_type, event_data=None):
        self.event_queue.put((event_type, event_data))
        
    def start(self):
        threading.Thread(target=self._process_events, daemon=True).start()
        
    def _process_events(self):
        while self._running:
            try:
                event_type, event_data = self.event_queue.get(timeout=0.1)
                if event_type in self.listeners:
                    for callback in self.listeners[event_type]:
                        try:
                            callback(event_data)
                        except Exception as e:
                            logging.error(f"Error in event handler: {e}")
            except queue.Empty:
                pass
                
    def run_async(self, func, *args, **kwargs):
        """Run a function asynchronously and return a Future"""
        return threading.Thread(target=func, args=args, kwargs=kwargs).start()
        
    def stop(self):
        """Stop the event processing"""
        self._running = False

class ModeManager:
    """Manages transitions between assistant and security modes"""
    
    def __init__(self, time_check_func, event_manager):
        self.current_mode = "assistant"
        self.time_check_func = time_check_func
        self.event_manager = event_manager
        self.mode_lock = threading.Lock()
        
    def start_monitoring(self):
        """Start mode monitoring thread"""
        threading.Thread(target=self._monitor_mode, daemon=True).start()
        
    def _monitor_mode(self):
        """Continuously check if mode should change"""
        while True:
            is_night = self.time_check_func()
            self._update_mode_if_needed(is_night)
            time.sleep(60)  # Only check once per minute to save power
            
    def _update_mode_if_needed(self, is_night):
        with self.mode_lock:
            if is_night and self.current_mode != "security":
                self.current_mode = "security"
                self.event_manager.publish("mode_changed", {"mode": "security"})
            elif not is_night and self.current_mode != "assistant":
                self.current_mode = "assistant"
                self.event_manager.publish("mode_changed", {"mode": "assistant"})

class AssistantSystem:
    def __init__(self):
        logger.info("Initializing Assistant System")
        
        # Create a centralized event manager
        self.event_manager = EventManager()
        
        # Create shared queues for backward compatibility
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # System state
        self.state = SystemState.IDLE
        self.state_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
        # Initialize core managers first
        self.power_manager = power_manager.PowerManager()
        self.mode_manager = ModeManager(self.is_night_time, self.event_manager)
        
        # Initialize I/O subsystems
        self.audio_manager = audio_manager.AudioManager(self.command_queue)
        self.camera_manager = camera_manager.CameraManager(power_save_mode=True)
        self.face_display = face_display.FaceDisplay(self.response_queue)
        
        # Initialize functional subsystems
        self.user_manager = user_manager.UserManager(self.camera_manager)
        
        # Initialize mode handlers last (after dependencies)
        self.assistant = assistant_mode.AssistantMode(
            self.audio_manager,
            self.camera_manager,
            self.face_display,
            self.user_manager,
            self.power_manager,
            self.command_queue,
            self.response_queue
        )
        self.security = security_mode.SecurityMode(
            self.audio_manager,
            self.camera_manager,
            self.face_display,
            self.user_manager,
            self.power_manager
                ip_camera_url="http://192.168.1.100:8080/video"  # ipcameraurl
        )
        
        # Set up event handlers
        self._setup_event_handlers()
        
        # Threads
        self.threads = []
        
        # Initialize the active assistant mode flag
        self.assistant_active = False
        
        # Command processing state to prevent race conditions
        self.processing_lock = threading.Lock()
        self.currently_processing = False
        
    def _setup_event_handlers(self):
        """Set up all event subscriptions"""
        self.event_manager.subscribe("mode_changed", self._handle_mode_change)
        self.event_manager.subscribe("user_identified", self._handle_user_identified)
        self.event_manager.subscribe("system_error", self._handle_system_error)
        self.event_manager.subscribe("command_processed", self._handle_command_processed)
        
    def _handle_mode_change(self, event_data):
        """Handle mode change events"""
        mode = event_data["mode"]
        if mode == "security":
            self.set_state(SystemState.SECURITY)
            self.security.start()
            self.face_display.update_emotion("vigilant")
            self.response_queue.put("Switching to security mode for the night. Camera active, other functions disabled.")
        elif mode == "assistant":
            self.set_state(SystemState.IDLE)
            self.security.stop()
            self.face_display.update_emotion("neutral")
            self.response_queue.put("Good morning! Switching to assistant mode. All functions enabled.")
    
    def _handle_user_identified(self, event_data):
        """Handle user identification events"""
        user = event_data.get("user")
        if user:
            self.response_queue.put(f"Hello {user['name']}! How can I help you?")
        else:
            self.response_queue.put("Hello! How can I help you?")
    
    def _handle_system_error(self, event_data):
        """Handle system error events"""
        error = event_data.get("error", "Unknown error")
        logger.error(f"System error: {error}")
        self.face_display.update_emotion("error")
        
        # Attempt recovery based on severity
        if "critical" in event_data.get("tags", []):
            self._recover_from_critical_error()
    
    def _handle_command_processed(self, event_data):
        """Handle command processed events"""
        result = event_data.get("result", {})
        if result.get("success", False):
            # Stay in listening mode if assistant is in active listening mode
            if self.assistant.active_listening:
                self.set_state(SystemState.LISTENING)
                self.face_display.update_emotion("attentive")
            else:
                self.set_state(SystemState.IDLE)
                self.face_display.update_emotion("neutral")
        else:
            # Handle error
            self.set_state(SystemState.IDLE)
            self.face_display.update_emotion("sad")
            
    def _recover_from_critical_error(self):
        """Attempt to recover from a critical error"""
        logger.info("Attempting recovery from critical error")
        
        # Reset all subsystems
        try:
            self.camera_manager.stop_capture()
            time.sleep(1)
            
            # Reset to idle state
            self.set_state(SystemState.IDLE)
            self.face_display.update_emotion("neutral")
            
            logger.info("Recovery complete")
        except Exception as e:
            logger.error(f"Error during recovery: {e}")
        
    def set_state(self, new_state):
        with self.state_lock:
            self.state = new_state
            logger.info(f"System state changed to: {new_state.name}")
    
    def get_state(self):
        with self.state_lock:
            return self.state
            
    def is_night_time(self):
        """Determine if it's night time based on current hour"""
        current_hour = datetime.datetime.now().hour
        # Consider 10PM to 6AM as night time
        return current_hour >= 22 or current_hour < 6
        
    def start(self):
        """Start all system threads"""
        logger.info("Starting Assistant System")
        
        # Start the event manager
        self.event_manager.start()
        
        # Start the mode manager
        self.mode_manager.start_monitoring()
        
        # Start the face display
        display_thread = threading.Thread(target=self.face_display.run, args=(self.shutdown_event,))
        display_thread.daemon = True
        display_thread.start()
        self.threads.append(display_thread)
        
        # Start always-on audio monitoring
        audio_thread = threading.Thread(target=self.audio_manager.run, args=(self.shutdown_event,))
        audio_thread.daemon = True
        audio_thread.start()
        self.threads.append(audio_thread)
        
        # Start the power manager
        self.power_manager.start()
        
        # Start the assistant mode
        self.assistant.start()
        
        # Start the main loop in a separate thread
        main_thread = threading.Thread(target=self.main_loop)
        main_thread.daemon = True
        main_thread.start()
        self.threads.append(main_thread)
        
        # Start a response processing thread
        response_thread = threading.Thread(target=self.process_responses)
        response_thread.daemon = True
        response_thread.start()
        self.threads.append(response_thread)
        
        # Wait for shutdown
        try:
            while not self.shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down")
            self.shutdown()
    
    def process_responses(self):
        """Process responses from the assistant"""
        while not self.shutdown_event.is_set():
            try:
                if not self.response_queue.empty():
                    response = self.response_queue.get()
                    # Set face to talking when speaking
                    self.face_display.update_emotion("talking")
                    self.audio_manager.speak(response)
                    
                    # After speaking, return to appropriate state
                    with self.processing_lock:
                        if not self.currently_processing:
                            if self.assistant.active_listening:
                                self.set_state(SystemState.LISTENING)
                                self.face_display.update_emotion("attentive")
                            else:
                                self.set_state(SystemState.IDLE)
                                self.face_display.update_emotion("neutral")
                
                # Brief pause to prevent CPU hogging
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in response processing: {e}")
    
    def main_loop(self):
        """Main processing loop to handle commands and manage system state"""
        while not self.shutdown_event.is_set():
            try:
                # Process the command queue
                if not self.command_queue.empty():
                    # Use the processing lock to prevent race conditions
                    with self.processing_lock:
                        if self.currently_processing:
                            # Skip this cycle if already processing a command
                            time.sleep(0.1)
                            continue
                    
                    # Get the command
                    command = self.command_queue.get()
                    current_state = self.get_state()
                    
                    # Handle case where command is a string (backward compatibility)
                    if isinstance(command, str):
                        logger.info(f"Received legacy string command: {command}")
                        # Clean up command text - remove duplications (like "what the time what the time")
                        clean_command = self.assistant._clean_command_text(command) if hasattr(self.assistant, '_clean_command_text') else command
                        command = {"type": "voice_command", "text": clean_command}
                    
                    # Now proceed with dictionary-based command processing
                    # Handle commands based on current state
                    if current_state == SystemState.SECURITY:
                        if command.get("type") == "wake_word":
                            # Handle wake word during security mode
                            self.security.handle_wake_word_during_security(command)
                    else:
                        if command.get("type") == "wake_word":
                            # Wake word detected
                            self.set_state(SystemState.LISTENING)
                            self.face_display.update_emotion("attentive")
                            # Process wake word in a separate thread
                            threading.Thread(target=self.process_wake, args=(command,)).start()
                        
                        elif command.get("type") == "voice_command" or isinstance(command, str):
                            # Get the command text for logging
                            command_text = command.get("text", command) if isinstance(command, dict) else command
                            
                            # Process the command if we're in listening mode or assistant is actively listening
                            if current_state == SystemState.LISTENING or self.assistant.active_listening:
                                # Process command in a separate thread
                                threading.Thread(target=self.process_command, args=(command,)).start()
                            else:
                                # We received a command but weren't in listening mode
                                logger.warning(f"Received command '{command_text}' but not in listening mode")
                
                # Brief pause to prevent CPU hogging
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.event_manager.publish("system_error", {
                    "error": str(e),
                    "location": "main_loop"
                })
    
    def process_wake(self, command):
        """Process wake word detection by identifying the user"""
        try:
            # Activate camera with timeout
            self.camera_manager.start_capture(timeout_seconds=10)
            
            # Identify user
            user = self.user_manager.identify_user()
            
            # Publish user identified event
            self.event_manager.publish("user_identified", {"user": user})
                
            # Enable active listening in the assistant
            self.assistant.active_listening = True
            self.assistant.last_interaction_time = time.time()
            
            # Check if there's a command after the wake word
            if "text" in command and command["text"]:
                wake_words = ["hey assistant", "hey ai"]
                command_text = command["text"].lower()
                
                for wake_word in wake_words:
                    if wake_word in command_text:
                        # Extract the actual command after the wake word
                        actual_command = command_text.split(wake_word, 1)[1].strip()
                        if actual_command:
                            # Create a new command and queue it for processing
                            # Don't process immediately to maintain proper flow
                            self.command_queue.put({"type": "voice_command", "text": actual_command})
                            break
            
            # Keep in listening mode
            self.set_state(SystemState.LISTENING)
                
        except Exception as e:
            error_msg = f"Error in process_wake: {e}"
            logger.error(error_msg)
            self.event_manager.publish("system_error", {
                "error": error_msg,
                "location": "process_wake"
            })
            self.set_state(SystemState.IDLE)
            self.camera_manager.stop_capture()
    
    def process_command(self, command):
        """Process a voice command using the assistant module"""
        try:
            with self.processing_lock:
                self.currently_processing = True
                
            # Extract the text if it's a command object
            command_text = command["text"] if isinstance(command, dict) else command
            logger.info(f"Processing command: '{command_text}'")
            
            # Set state to processing and update face
            self.set_state(SystemState.PROCESSING)
            self.face_display.update_emotion("thinking")
            
            # Let assistant process the command
            self.assistant.process_command(command_text)
            
            # Stay in processing state until response is generated
            # The response processing thread will handle state transition
            
        except Exception as e:
            error_msg = f"Error processing command: {e}"
            logger.error(error_msg)
            self.event_manager.publish("system_error", {
                "error": error_msg,
                "location": "process_command"
            })
            self.response_queue.put("Sorry, I encountered an error processing your request.")
            
            # Reset state
            if self.assistant.active_listening:
                self.set_state(SystemState.LISTENING)
                self.face_display.update_emotion("attentive")
            else:
                self.set_state(SystemState.IDLE)
                self.face_display.update_emotion("sad")
                self.camera_manager.stop_capture()
        finally:
            # Always release the processing lock
            with self.processing_lock:
                self.currently_processing = False
    
    def shutdown(self):
        """Gracefully shutdown the system"""
        logger.info("Shutting down Assistant System")
        self.shutdown_event.set()
        
        # Signal modules to stop
        self.camera_manager.stop_capture()
        self.event_manager.stop()
        self.assistant.stop()
        
        # Stop security mode if it's running
        if self.get_state() == SystemState.SECURITY:
            self.security.stop()
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2)
        
        logger.info("Assistant System shutdown complete")

if __name__ == "__main__":
    system = AssistantSystem()
    system.start()
