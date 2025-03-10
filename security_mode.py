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
        self.check_interval = 5  # shorter interval for more frequent checks
        self.logger = logging.getLogger('security_mode')
        self.recordings_dir = "security_recordings"
        self.shutdown_event = threading.Event()
        
        # Intruder detection specific attributes
        self.video_source = video_source
        self.cap = None
        self.background_subtractor = None
        self.MIN_AREA = 5000  # Minimum area for motion to be considered an intruder
        self.intruder_detection_active = False
        self.intruder_thread = None
        self.display_video = False  # Option to display video feed for debugging
        
        # Create recordings directory if it doesn't exist
        if not os.path.exists(self.recordings_dir):
            os.makedirs(self.recordings_dir)
    
    def run(self, shutdown_event):
        """Compatible method for main.py to start security mode"""
        self.shutdown_event = shutdown_event
        self.start()
        
        # Run until shutdown is signaled
        while not shutdown_event.is_set() and self.running:
            time.sleep(1)
            
        # Stop if still running when shutdown is signaled
        if self.running:
            self.stop()
            
    def start(self):
        """Start the security mode thread."""
        if self.thread is None or not self.thread.is_alive():
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            self.logger.info("Security mode started")
            self.face_display.update_emotion("vigilant")
            self.audio_manager.speak("Security mode activated. Camera is now active. All other functions are disabled.")
            
            # Enable security mode in camera manager and KEEP CAMERA ON
            self.camera_manager.set_security_mode(True)
            self.camera_manager.start_capture()  # Always keep camera on during security mode
            
            # Disable other functions by not requesting them in power manager
            self._disable_non_security_functions()
            
            # Start continuous intruder detection immediately
            self.start_intruder_detection()
            
    def stop(self):
        """Stop the security mode thread."""
        self.running = False
        
        # Disable security mode in camera manager
        self.camera_manager.set_security_mode(False)
        self.camera_manager.stop_capture()  # Stop the camera when exiting security mode
        
        # Stop intruder detection if active
        self.stop_intruder_detection()
        
        # Re-enable all functions
        self._enable_all_functions()
        
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
            self.logger.info("Security mode stopped")
            self.face_display.update_emotion("neutral")
        
        # Ensure all OpenCV windows are closed
        cv2.destroyAllWindows()
            
    def _disable_non_security_functions(self):
        """Disable all non-security functions to save power and focus on security."""
        self.logger.info("Disabling non-security functions")
        # Keep only minimal services needed for security
        self.power_manager.disable_non_essential_services()
        
    def _enable_all_functions(self):
        """Re-enable all functions when exiting security mode."""
        self.logger.info("Re-enabling all functions")
        # Re-enable all services
        self.power_manager.enable_all_services()
            
    def set_security_level(self, level):
        """Set the security sensitivity level."""
        if level in ["low", "medium", "high"]:
            self.security_level = level
            self.logger.info(f"Security level set to {level}")
            
            # Adjust check interval based on security level
            if level == "low":
                self.check_interval = 15
                self.MIN_AREA = 8000  # Less sensitive
                self.camera_manager.motion_threshold = 40  # Less sensitive
            elif level == "medium":
                self.check_interval = 5
                self.MIN_AREA = 5000  # Medium sensitivity
                self.camera_manager.motion_threshold = 30  # Default sensitivity
            else:  # high
                self.check_interval = 2
                self.MIN_AREA = 3000  # More sensitive
                self.camera_manager.motion_threshold = 20  # More sensitive
            
            # Update intruder detection parameters if active
            if self.intruder_detection_active:
                self.restart_intruder_detection()
    
    def restart_intruder_detection(self):
        """Helper method to restart intruder detection with new parameters"""
        # Added new method to avoid duplicated code
        self.stop_intruder_detection()
        self.start_intruder_detection()
                
    def _run(self):
        """Main loop for the security mode."""
        while self.running:
            try:
                current_time = time.time()
                
                # Periodic camera check based on security level
                if current_time - self.last_check >= self.check_interval:
                    self._security_check()
                    self.last_check = current_time
                    
                # Process any security alerts in the queue
                while not self.alert_queue.empty() and self.running:
                    # Process all pending alerts
                    alert = self.alert_queue.get()
                    self._handle_alert(alert)
                    
                # Sleep to reduce CPU usage
                time.sleep(0.1)
            except Exception as e:
                # Added exception handling to prevent thread crashes
                self.logger.error(f"Error in security mode main loop: {str(e)}")
                time.sleep(1)  # Pause briefly before continuing
            
    def _security_check(self):
        """Perform a routine security check."""
        self.logger.debug("Performing security check")
        
        try:
            # Check for motion by looking at the current frame and previous frames
            frame = self.camera_manager.get_frame()
            
            # Check if motion was detected
            try:
                motion_detected = hasattr(self.camera_manager, '_simulate_random_event') and \
                                self.camera_manager._simulate_random_event(0.1)  # 10% chance
            except AttributeError:
                # If _simulate_random_event is not available, use random event
                motion_detected = np.random.random() < 0.1
            
            if motion_detected:
                self.logger.info("Motion detected during security check")
                
                # Try to identify if this is a known person using face detection
                faces = self.camera_manager.detect_faces()
                
                if faces:
                    user = self.user_manager.identify_user()
                    
                    if user:
                        self.logger.info(f"Identified known user: {user['name']}")
                        # Known user, just log and continue
                        self.face_display.update_emotion("happy")
                        self.audio_manager.speak(f"Good evening, {user['name']}")
                    else:
                        # Unknown person detected, escalate alert
                        self._create_alert("unknown_person", frame)
                else:
                    # Motion but no face, could be an animal or object
                    self._create_alert("unidentified_motion", frame)
        except Exception as e:
            # Added error handling for security check
            self.logger.error(f"Error during security check: {str(e)}")
                
    def _create_alert(self, alert_type, frame=None):
        """Create a security alert and add it to the queue."""
        try:
            image_path = self._capture_evidence(frame)
            
            # Create a standardized alert dictionary
            alert = {
                'type': alert_type,
                'timestamp': datetime.datetime.now(),
                'image_path': image_path
            }
            
            # Only add frame data if provided and valid
            if frame is not None and isinstance(frame, np.ndarray):
                alert['frame'] = frame
                
            self.alert_queue.put(alert)
            self.logger.info(f"Created {alert_type} alert")
        except Exception as e:
            self.logger.error(f"Error creating alert: {str(e)}")
        
    def _capture_evidence(self, frame=None):
        """Capture image/video evidence of the security event."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{self.recordings_dir}/security_{timestamp}.txt"
        
        try:
            # If no frame provided, try to get one from the camera manager
            if frame is None:
                frame = self.camera_manager.get_frame()
            
            # Save frame as evidence if possible
            if frame is not None and isinstance(frame, np.ndarray):
                # In a real system, we would save the actual image
                # cv2.imwrite(f"{self.recordings_dir}/security_{timestamp}.jpg", frame)
                
                # For this demo, just write frame info to a text file
                with open(filepath, "w") as f:
                    f.write(f"Security event capture at {timestamp}\n")
                    f.write(f"Frame shape: {frame.shape if hasattr(frame, 'shape') else 'Unknown'}\n")
                    f.write(f"Frame type: {type(frame)}")
            else:
                with open(filepath, "w") as f:
                    f.write(f"Security event capture at {timestamp} - no valid frame available")
                    
            self.logger.info(f"Evidence captured: {filepath}")
        except Exception as e:
            self.logger.error(f"Error capturing evidence: {str(e)}")
            with open(filepath, "w") as f:
                f.write(f"Error capturing evidence at {timestamp}: {str(e)}")
        
        return filepath
        
    def _handle_alert(self, alert):
        """Handle a security alert based on its type."""
        if not isinstance(alert, dict) or 'type' not in alert or 'timestamp' not in alert:
            self.logger.error(f"Invalid alert format: {alert}")
            return
            
        self.logger.warning(f"Security alert: {alert['type']} at {alert['timestamp']}")
        
        try:
            if alert['type'] == "unknown_person":
                # High priority alert - unknown person detected
                self.face_display.update_emotion("alarmed")
                self.audio_manager.speak("Alert! Unidentified person detected!")
                
                # Could send notification to owner's phone here
                # self._send_notification(alert)
                
            elif alert['type'] == "unidentified_motion":
                # Medium priority alert - motion but no person identified
                if self.security_level == "high":
                    self.face_display.update_emotion("concerned")
                    self.audio_manager.speak("Motion detected. Investigating.")
                    
                    # Already have camera on, just do a follow-up check
                    time.sleep(2)  # Wait a moment before rechecking
                    self._security_check()
            
            elif alert['type'] == "intruder":
                # Intruder alert from continuous monitoring
                self.face_display.update_emotion("alarmed")
                self.audio_manager.speak("Warning! Intruder detected! Security measures activated!")
                
                # Save evidence with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"{self.recordings_dir}/intruder_{timestamp}.txt"
                
                with open(file_path, "w") as f:
                    f.write(f"Intruder detected at {timestamp}\n")
                    if 'frame' in alert:
                        f.write(f"Frame data available: {bool(alert['frame'])}")
                    if 'image_path' in alert:
                        f.write(f"\nEvidence stored at: {alert['image_path']}")
                        
                self.logger.info(f"Intruder evidence saved: {file_path}")
                
                # Could trigger alarm system, call security service, etc.
                # self._trigger_alarm()
                
            elif alert['type'] == "unauthorized_access_attempt":
                # Handle unauthorized access attempts
                self.logger.warning("Unauthorized access attempt detected")
                # Implement additional security measures
        except Exception as e:
            self.logger.error(f"Error handling alert: {str(e)}")
        
    def handle_wake_word_during_security(self, command):
        """Handle wake word detection during security mode."""
        self.face_display.update_emotion("alert")
        
        try:
            # Check if it's the owner giving a command
            user = self.user_manager.identify_user()
            
            if user and user.get('is_owner', False):
                self.audio_manager.speak("Security mode paused. How can I help you?")
                # Process the command here or pass to assistant mode temporarily
                
                # Temporarily stop intruder detection if it's running
                if self.intruder_detection_active:
                    self.stop_intruder_detection()
                    
                # Resume intruder detection after 1 minute
                def resume_security():
                    try:
                        time.sleep(60)
                        if self.running and not self.intruder_detection_active:
                            self.start_intruder_detection()
                            self.audio_manager.speak("Resuming full security mode")
                    except Exception as e:
                        self.logger.error(f"Error resuming security: {str(e)}")
                        
                threading.Thread(target=resume_security, daemon=True).start()
                    
            else:
                # Unknown person or not owner
                self.audio_manager.speak("Unauthorized access attempt recorded")
                self._create_alert("unauthorized_access_attempt")
        except Exception as e:
            self.logger.error(f"Error handling wake word: {str(e)}")
    
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
            
            # Allow intruder thread to close properly
            if self.intruder_thread and self.intruder_thread.is_alive():
                # Wait for thread to terminate, with timeout
                self.intruder_thread.join(timeout=2.0)
                self.intruder_thread = None
                
            # Release video capture if it exists
            self._release_camera_resources()
                
            self.logger.info("Intruder detection stopped")
    
    def _release_camera_resources(self):
        """Helper method to release camera resources safely"""
        try:
            # Release video capture if it exists
            if self.cap and hasattr(self.cap, 'isOpened') and self.cap.isOpened():
                self.cap.release()
                self.cap = None
                
            # Close any open windows
            cv2.destroyAllWindows()
        except Exception as e:
            self.logger.error(f"Error releasing camera resources: {str(e)}")
            
    def _run_intruder_detection(self):
        """Run intruder detection using OpenCV."""
        try:
            # Initialize video capture
            if isinstance(self.video_source, str):
                # Use a video file if specified
                self.cap = cv2.VideoCapture(self.video_source)
            else:
                # Get frames directly from camera_manager
                self.cap = None

            # Initialize background subtractor
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=25, detectShadows=True)
            
            self.logger.info(f"Intruder detection running with min area: {self.MIN_AREA}")
            
            consecutive_errors = 0
            max_consecutive_errors = 5
            
            while self.intruder_detection_active:
                try:
                    # Get frame either from OpenCV capture or camera_manager
                    if self.cap and hasattr(self.cap, 'isOpened') and self.cap.isOpened():
                        ret, frame = self.cap.read()
                        if not ret:
                            self.logger.warning("Failed to read from video source")
                            time.sleep(0.5)
                            consecutive_errors += 1
                            if consecutive_errors >= max_consecutive_errors:
                                self.logger.error("Too many consecutive errors reading from video source")
                                break
                            continue
                    else:
                        # Get frame from camera_manager
                        frame = self.camera_manager.get_frame()
                        if frame is None:
                            self.logger.warning("Could not get valid frame from camera_manager")
                            time.sleep(0.5)
                            consecutive_errors += 1
                            if consecutive_errors >= max_consecutive_errors:
                                self.logger.error("Too many consecutive errors getting frames")
                                break
                            continue
                        
                        # Handle the case where camera_manager returns metadata instead of actual frame
                        if not isinstance(frame, np.ndarray):
                            # Create a dummy frame for testing
                            frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    
                    # Reset error counter on successful frame acquisition
                    consecutive_errors = 0

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
                        # Add alert text to frame
                        alert_text = "Intruder Detected!"
                        cv2.putText(frame_resized, alert_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        self.logger.warning(f"{datetime.datetime.now()}: Intruder Alert!")
                        
                        # Create an intruder alert with standardized format
                        self._create_alert("intruder", frame_resized)
                        
                        # Only display video when intruder is detected
                        if self.display_video:
                            cv2.imshow("CCTV Intruder Detection", frame_resized)

                    # Break loop on 'q' key press
                    if self.display_video and cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    # Small sleep to reduce CPU usage
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Error in intruder detection loop: {str(e)}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error("Too many consecutive errors in intruder detection, stopping")
                        break
                    time.sleep(1)  # Wait before retrying
                
        except Exception as e:
            self.logger.error(f"Critical error in intruder detection: {str(e)}")
        finally:
            # Release resources
            self._release_camera_resources()
            self.intruder_detection_active = False
            self.logger.info("Intruder detection thread terminated")