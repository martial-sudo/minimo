import cv2
import numpy as np
import datetime
import pygame
import time

class IntruderDetectionSystem:
    def __init__(self, sensitivity=500, threshold_duration=3, shared_camera=None):
        # Use shared camera or initialize our own
        self.owns_camera = shared_camera is None
        
        if shared_camera:
            # Use the shared camera instance
            self.camera = shared_camera
            print("IntruderDetectionSystem: Using shared camera instance")
        else:
            # Initialize video capture
            self.camera = cv2.VideoCapture(2)
            print("IntruderDetectionSystem: Opening camera at index 2")
        
        # Background subtractor
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            history=100,  # Increased history for better background modeling
            varThreshold=40  # Adjusted threshold for shadow detection
        )
        
        # Sensitivity for motion detection
        self.sensitivity = sensitivity
        
        # Threshold duration for sustained motion
        self.threshold_duration = threshold_duration
        
        # Motion tracking variables
        self.motion_start_time = None
        self.is_motion_detected = False
        
        # Initialize sound
        pygame.mixer.init()
        try:
            # Load a beep sound
            self.alert_sound = pygame.mixer.Sound('alarm.mp3')
        except Exception as e:
            print(f"Could not load sound file: {e}")
            self.alert_sound = None
    
    def detect_motion(self, frame):
        """Advanced motion detection method"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(gray)
        
        # Threshold to remove shadows and noise
        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        
        # Dilate the image to fill in holes
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Check for significant motion
        significant_motion = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.sensitivity:
                significant_motion = True
                break
        
        return significant_motion
    
    def trigger_alert(self):
        """Trigger sound alert"""
        if self.alert_sound:
            # Play sound alert
            self.alert_sound.play()
            print("Intruder Alert! Sustained Motion Detected!")
        else:
            # Fallback to console alert and system beep
            print("\a")  # System beep
            print("Intruder Alert! Sustained Motion Detected!")
    
    def run_night_mode(self):
        """Continuous monitoring during night with improved motion detection"""
        print("Night mode activated. Monitoring for intruders...")
        
        if not self.camera.isOpened():
            print("Camera not open, attempting to reopen...")
            if self.owns_camera:  # Only try to reopen if we own the camera
                self.camera = cv2.VideoCapture(2)
            if not self.camera.isOpened():
                print("Failed to open camera. Night mode cannot run.")
                return
        
        # Allow background subtractor to stabilize
        for _ in range(30):
            ret, _ = self.camera.read()
            if not ret:
                print("Failed to capture frame during stabilization")
                return
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to capture frame during detection")
                break
            
            # Detect motion
            motion_detected = self.detect_motion(frame)
            
            current_time = time.time()
            
            if motion_detected:
                if not self.is_motion_detected:
                    # First time motion is detected
                    self.motion_start_time = current_time
                    self.is_motion_detected = True
                
                # Check if motion has been sustained
                if (current_time - self.motion_start_time) >= self.threshold_duration:
                    self.trigger_alert()
                    # Wait before checking for motion again to prevent continuous alerts
                    time.sleep(10)
                    # Reset motion tracking
                    self.is_motion_detected = False
                    self.motion_start_time = None
            else:
                # Reset motion tracking if no motion
                self.is_motion_detected = False
                self.motion_start_time = None
            
            # Short sleep to prevent high CPU usage
            time.sleep(0.1)
    
    def close(self):
        """Release camera resources"""
        # Only release the camera if we own it
        if self.owns_camera and self.camera is not None and self.camera.isOpened():
            self.camera.release()
        pygame.mixer.quit()

# Example usage
if __name__ == "__main__":
    intruder_system = IntruderDetectionSystem(
        sensitivity=500,  # Adjust this value based on your environment
        threshold_duration=3  # Motion must be sustained for 3 seconds
    )
    try:
        intruder_system.run_night_mode()
    except KeyboardInterrupt:
        intruder_system.close()