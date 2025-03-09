import threading
import time
import logging
import datetime
from queue import Queue
import os
import cv2
import numpy as np

class SecurityMode:
    def __init__(self, audio_manager, camera_manager, face_display, user_manager, power_manager, video_source=0):
        """Initialize the night-time security mode with intruder detection."""
        self.audio_manager = audio_manager
        self.camera_manager = camera_manager
        self.face_display = face_display
        self.user_manager = user_manager
        self.power_manager = power_manager
        self.alert_queue = Queue()
        self.running = False
        self.thread = None
        self.security_level = "medium"  # low, medium, high
        self.motion_detected = False
        self.last_check = time.time()
        self.check_interval = 10  # seconds between periodic checks
        self.logger = logging.getLogger('security_mode')
        self.recordings_dir = "security_recordings"
        
        # Intruder detection specific attributes
        self.video_source = video_source
        self.cap = None
        self.background_subtractor = None
        self.MIN_AREA = 5000  # Minimum area for motion to be considered an intruder
        self.intruder_detection_active = False
        self.intruder_thread = None
        
        # Create recordings directory if it doesn't exist
        if not os.path.exists(self.recordings_dir):
            os.makedirs(self.recordings_dir)
            
    def start(self):
        """Start the security mode thread."""
        if self.thread is None or not self.thread.is_alive():
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            self.logger.info("Security mode started")
            self.face_display.set_emotion("vigilant")
            self.audio_manager.speak("Security mode activated")
            
    def stop(self):
        """Stop the security mode thread."""
        self.running = False
        
        # Stop intruder detection if active
        self.stop_intruder_detection()
        
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
            self.logger.info("Security mode stopped")
            self.face_display.set_emotion("neutral")
            
    def set_security_level(self, level):
        """Set the security sensitivity level."""
        if level in ["low", "medium", "high"]:
            self.security_level = level
            self.logger.info(f"Security level set to {level}")
            
            # Adjust check interval based on security level
            if level == "low":
                self.check_interval = 30
                self.MIN_AREA = 8000  # Less sensitive
            elif level == "medium":
                self.check_interval = 10
                self.MIN_AREA = 5000  # Medium sensitivity
            else:  # high
                self.check_interval = 5
                self.MIN_AREA = 3000  # More sensitive
                
    def _run(self):
        """Main loop for the security mode."""
        while self.running:
            current_time = time.time()
            
            # Periodic camera check based on security level
            if current_time - self.last_check >= self.check_interval:
                self._security_check()
                self.last_check = current_time
                
            # Process any security alerts in the queue
            if not self.alert_queue.empty():
                alert = self.alert_queue.get()
                self._handle_alert(alert)
                
            # Sleep to reduce CPU usage
            time.sleep(0.5)
            
    def _security_check(self):
        """Perform a routine security check."""
        self.logger.debug("Performing security check")
        
        # Request camera resource
        self.power_manager.request_resource('camera')
        self.camera_manager.activate()
        
        # Check for motion/intruders
        motion_detected = self.camera_manager.detect_motion()
        
        if motion_detected:
            self.logger.info("Motion detected during security check")
            
            # Try to identify if this is a known person
            face_detected = self.camera_manager.detect_face()
            if face_detected:
                user = self.user_manager.identify_user()
                
                if user:
                    self.logger.info(f"Identified known user: {user['name']}")
                    # Known user, just log and continue
                    self.face_display.set_emotion("happy")
                    self.audio_manager.speak(f"Good evening, {user['name']}")
                else:
                    # Unknown person detected, start continuous intruder detection
                    self._create_alert("unknown_person")
                    self.start_intruder_detection()
            else:
                # Motion but no face, could be an animal or object
                self._create_alert("unidentified_motion")
                
                # Start intruder detection if security level is high
                if self.security_level == "high":
                    self.start_intruder_detection()
                
        # Release camera resource
        self.camera_manager.deactivate()
        self.power_manager.release_resource('camera')
            
    def _create_alert(self, alert_type):
        """Create a security alert and add it to the queue."""
        alert = {
            'type': alert_type,
            'timestamp': datetime.datetime.now(),
            'image_path': self._capture_evidence()
        }
        self.alert_queue.put(alert)
        
    def _capture_evidence(self):
        """Capture image/video evidence of the security event."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{self.recordings_dir}/security_{timestamp}.jpg"
        
        # Capture image
        self.camera_manager.capture_image(filepath)
        self.logger.info(f"Evidence captured: {filepath}")
        
        return filepath
        
    def _handle_alert(self, alert):
        """Handle a security alert based on its type."""
        self.logger.warning(f"Security alert: {alert['type']} at {alert['timestamp']}")
        
        if alert['type'] == "unknown_person":
            # High priority alert - unknown person detected
            self.face_display.set_emotion("alarmed")
            self.audio_manager.speak("Alert! Unidentified person detected!")
            
            # Could send notification to owner's phone here
            # self._send_notification(alert)
            
        elif alert['type'] == "unidentified_motion":
            # Medium priority alert - motion but no person identified
            if self.security_level == "high":
                self.face_display.set_emotion("concerned")
                self.audio_manager.speak("Motion detected. Investigating.")
                
                # Request camera again for a follow-up check
                time.sleep(2)  # Wait a moment before rechecking
                self._security_check()
        
        elif alert['type'] == "intruder":
            # Intruder alert from continuous monitoring
            self.face_display.set_emotion("alarmed")
            self.audio_manager.speak("Warning! Intruder detected! Security measures activated!")
            
            # Save evidence with timestamp
            cv2.imwrite(f"{self.recordings_dir}/intruder_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", alert.get('frame', None))
            
            # Could trigger alarm system, call security service, etc.
            # self._trigger_alarm()
        
    def handle_wake_word_during_security(self, command):
        """Handle wake word detection during security mode."""
        self.face_display.set_emotion("alert")
        
        # Check if it's the owner giving a command
        self.power_manager.request_resource('camera')
        self.camera_manager.activate()
        
        user = self.user_manager.identify_user()
        
        if user and user.get('is_owner', False):
            self.audio_manager.speak("Security mode paused. How can I help you?")
            # Process the command here or pass to assistant mode temporarily
            
            # Temporarily stop intruder detection if it's running
            if self.intruder_detection_active:
                self.stop_intruder_detection()
                
        else:
            # Unknown person or not owner
            self.audio_manager.speak("Unauthorized access attempt recorded")
            self._create_alert("unauthorized_access_attempt")
            
            # Start intruder detection if not already running
            if not self.intruder_detection_active:
                self.start_intruder_detection()
            
        self.camera_manager.deactivate()
        self.power_manager.release_resource('camera')
    
    # Intruder detection specific methods
    def start_intruder_detection(self):
        """Start continuous intruder detection in a separate thread."""
        if not self.intruder_detection_active:
            self.intruder_detection_active = True
            self.intruder_thread = threading.Thread(target=self._run_intruder_detection, daemon=True)
            self.intruder_thread.start()
            self.logger.info("Continuous intruder detection started")
    
    def stop_intruder_detection(self):
        """Stop the intruder detection thread."""
        if self.intruder_detection_active:
            self.intruder_detection_active = False
            if self.intruder_thread:
                self.intruder_thread.join(timeout=1.0)
                self.intruder_thread = None
                
            # Release video capture if it exists
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
                
            self.logger.info("Intruder detection stopped")
            
    def _run_intruder_detection(self):
        """Run the intruder detection algorithm in a continuous loop."""
        try:
            # Initialize video capture
            self.cap = cv2.VideoCapture(self.video_source)
            
            if not self.cap.isOpened():
                self.logger.error("Error: Unable to access the video source.")
                self.intruder_detection_active = False
                return
                
            # Initialize background subtractor
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
            
            self.logger.info(f"Intruder detection running with min area: {self.MIN_AREA}")
            
            while self.intruder_detection_active:
                # Request camera resource
                self.power_manager.request_resource('camera')
                
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Video feed ended or cannot be read.")
                    break
                    
                # Resize frame for faster processing
                frame_resized = cv2.resize(frame, (640, 480))
                
                # Apply background subtraction
                mask = self.background_subtractor.apply(frame_resized)
                
                # Remove noise (morphological operations)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                mask = cv2.dilate(mask, None, iterations=2)
                
                # Find contours of the motion
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                intruder_detected = False
                
                for contour in contours:
                    if cv2.contourArea(contour) < self.MIN_AREA:
                        continue
                        
                    # Draw bounding box around the motion
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    intruder_detected = True
                    
                # If intruder detected, create an alert
                if intruder_detected:
                    alert_text = "Intruder Detected!"
                    cv2.putText(frame_resized, alert_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    self.logger.warning(f"{datetime.datetime.now()}: Intruder Alert!")
                    
                    # Create an intruder alert
                    alert = {
                        'type': 'intruder',
                        'timestamp': datetime.datetime.now(),
                        'frame': frame_resized
                    }
                    self.alert_queue.put(alert)
                    
                    # Capture evidence
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = f"{self.recordings_dir}/intruder_{timestamp}.jpg"
                    cv2.imwrite(filepath, frame_resized)
                    
                # Release camera resource temporarily to allow other processes to use it
                self.power_manager.release_resource('camera')
                
                # Small sleep to reduce CPU usage
                time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Error in intruder detection: {str(e)}")
        finally:
            # Clean up resources
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
                
            self.intruder_detection_active = False
            self.power_manager.release_resource('camera')