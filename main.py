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

class AssistantSystem:
    def __init__(self):
        logger.info("Initializing Assistant System")
        
        # Create shared queues for inter-thread communication
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # System state
        self.state = SystemState.IDLE
        self.state_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
        # Initialize modules
        self.face_display = face_display.FaceDisplay(self.response_queue)
        self.audio_manager = audio_manager.AudioManager(self.command_queue)
        self.camera_manager = camera_manager.CameraManager()
        self.user_manager = user_manager.UserManager(self.camera_manager)
        self.power_manager = power_manager.PowerManager()
        
        # Mode modules
        self.assistant = assistant_mode.AssistantMode(
            self.command_queue, 
            self.response_queue, 
            self.camera_manager,
            self.user_manager
        )
        self.security = security_mode.SecurityMode(
            self.camera_manager, 
            self.response_queue,
            self.user_manager
        )
        
        # Threads
        self.threads = []
        
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
        power_thread = threading.Thread(target=self.power_manager.run, args=(self.shutdown_event,))
        power_thread.daemon = True
        power_thread.start()
        self.threads.append(power_thread)
        
        # Start the main loop in a separate thread
        main_thread = threading.Thread(target=self.main_loop)
        main_thread.daemon = True
        main_thread.start()
        self.threads.append(main_thread)
        
        # Wait for shutdown
        try:
            while not self.shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down")
            self.shutdown()
    
    def main_loop(self):
        """Main processing loop to handle commands and manage system state"""
        while not self.shutdown_event.is_set():
            try:
                # Check if in security mode
                if self.is_night_time() and self.get_state() != SystemState.SECURITY:
                    self.set_state(SystemState.SECURITY)
                    security_thread = threading.Thread(target=self.security.run, args=(self.shutdown_event,))
                    security_thread.daemon = True
                    security_thread.start()
                    self.threads.append(security_thread)
                    self.face_display.update_emotion("vigilant")
                    self.response_queue.put("Switching to security mode for the night.")
                
                # Check if should exit security mode
                if not self.is_night_time() and self.get_state() == SystemState.SECURITY:
                    self.set_state(SystemState.IDLE)
                    self.face_display.update_emotion("neutral")
                    self.response_queue.put("Good morning! Switching to assistant mode.")
                
                # Process commands from audio manager
                if not self.command_queue.empty():
                    command = self.command_queue.get()
                    
                    if command.get("type") == "wake_word":
                        # Wake word detected
                        if self.get_state() != SystemState.SECURITY:
                            self.set_state(SystemState.LISTENING)
                            self.face_display.update_emotion("attentive")
                            # Only now activate the camera for face recognition
                            self.camera_manager.start_capture()
                            # Start face recognition in a separate thread
                            threading.Thread(target=self.process_wake, args=(command,)).start()
                    
                    elif command.get("type") == "voice_command" and self.get_state() == SystemState.LISTENING:
                        self.set_state(SystemState.PROCESSING)
                        self.face_display.update_emotion("thinking")
                        # Start command processing in a separate thread
                        threading.Thread(target=self.process_command, args=(command,)).start()
                
                # Brief pause to prevent CPU hogging
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.face_display.update_emotion("error")
    
    def process_wake(self, command):
        """Process wake word detection by identifying the user"""
        try:
            user = self.user_manager.identify_user()
            if user:
                self.response_queue.put(f"Hello {user['name']}! How can I help you?")
            else:
                self.response_queue.put("Hello! How can I help you?")
                
            # Keep camera on for a short time awaiting command
            time.sleep(5)
            
            # If no command received, go back to idle
            if self.get_state() == SystemState.LISTENING:
                self.set_state(SystemState.IDLE)
                self.face_display.update_emotion("neutral")
                self.camera_manager.stop_capture()
                
        except Exception as e:
            logger.error(f"Error in process_wake: {e}")
            self.set_state(SystemState.IDLE)
            self.camera_manager.stop_capture()
    
    def process_command(self, command):
        """Process a voice command using the assistant module"""
        try:
            # Let assistant process the command
            self.assistant.process_command(command["text"])
            
            # After processing, return to idle and stop camera
            self.set_state(SystemState.IDLE)
            self.face_display.update_emotion("neutral")
            self.camera_manager.stop_capture()
            
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            self.response_queue.put("Sorry, I encountered an error processing your request.")
            self.set_state(SystemState.IDLE)
            self.face_display.update_emotion("sad")
            self.camera_manager.stop_capture()
    
    def shutdown(self):
        """Gracefully shutdown the system"""
        logger.info("Shutting down Assistant System")
        self.shutdown_event.set()
        
        # Signal modules to stop
        self.camera_manager.stop_capture()
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2)
        
        logger.info("Assistant System shutdown complete")

if __name__ == "__main__":
    system = AssistantSystem()
    system.start()