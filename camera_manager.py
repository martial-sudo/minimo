#!/usr/bin/env python3
import logging
import threading
import time
import numpy as np
import os
import cv2
import mediapipe as mp
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self, resolution=(640, 480), rotation=0, power_save_mode=True):
        self.resolution = resolution
        self.rotation = rotation
        self.power_save_mode = power_save_mode
        
        # Camera state
        self.capturing = False
        self.camera_thread = None
        self.shutdown_requested = threading.Event()
        self.capture_timeout = None
        self.last_activity_time = None
        
        # Frame management
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.last_frame = None
        self.last_frame_time = None
        
        # Security features
        self.security_mode = False
        self.motion_threshold = 30
        self.recordings_dir = "security_recordings"
        self.video_writer = None
        self.recording_active = False
        self.recording_start_time = None
        
        # Camera hardware
        self.webcam = None
        self.rpi_camera = None
        self.ip_camera = None
        self.active_camera_type = None  # 'webcam', 'rpi', or 'ip'
        self.ip_camera_url = None
        self.brightness = 50
        self.night_mode_enabled = False
        
        # Create recordings directory if it doesn't exist
        if not os.path.exists(self.recordings_dir):
            os.makedirs(self.recordings_dir)
        
        # Load face recognition models
        self.load_face_models()
        
        # Initialize face database
        self.init_face_database()
        
        # Event handlers dictionary
        self.event_handlers = {}
        
    def register_event_handler(self, event_type, callback):
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(callback)
        
    def trigger_event(self, event_type, event_data=None):
        if event_type in self.event_handlers:
            for callback in self.event_handlers[event_type]:
                try:
                    callback(event_data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
        
    def init_face_database(self):
        """Initialize SQLite database for face recognition data"""
        try:
            self.face_db_conn = sqlite3.connect('face_recognition.db', check_same_thread=False)
            cursor = self.face_db_conn.cursor()
            
            # Create table for face data if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_data (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    face_encodings BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create table for face recognition logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recognition_logs (
                    id INTEGER PRIMARY KEY,
                    face_id INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL,
                    FOREIGN KEY (face_id) REFERENCES face_data(id)
                )
            ''')
            
            self.face_db_conn.commit()
            logger.info("Face recognition database initialized")
        except Exception as e:
            logger.error(f"Failed to initialize face database: {e}")
            self.face_db_conn = None
        
    def load_face_models(self):
        """Load MediaPipe face detection and mesh models"""
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5,
            model_selection=1
        )
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=3,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        logger.info("MediaPipe face models loaded")
    
    def activate_rpi_camera(self):
        """Activates the Raspberry Pi Camera."""
        self._release_current_camera()
        
        try:
            self.rpi_camera = cv2.VideoCapture(0)  # 0 is typically the RPi camera
            self.rpi_camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.rpi_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            if not self.rpi_camera.isOpened():
                logger.error("Failed to open Raspberry Pi Camera")
                self.trigger_event("camera_error", {"error": "Failed to open Raspberry Pi Camera"})
                self.rpi_camera = None
                return False
            
            self.active_camera_type = "rpi"
            logger.info("Raspberry Pi Camera activated")
            self.trigger_event("camera_activated", {"type": "rpi"})
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate Raspberry Pi Camera: {e}")
            self.trigger_event("camera_error", {"error": f"RPi Camera activation failed: {e}"})
            self.rpi_camera = None
            return False
    
    def deactivate_rpi_camera(self):
        """Deactivates the Raspberry Pi Camera."""
        if self.rpi_camera:
            self.rpi_camera.release()
            self.rpi_camera = None
            self.active_camera_type = None
            logger.info("Raspberry Pi Camera deactivated")
            self.trigger_event("camera_deactivated", {"type": "rpi"})
    
    def activate_ip_camera(self, ip_url):
        """Activates an IP Camera using the given URL."""
        self._release_current_camera()
        
        try:
            self.ip_camera = cv2.VideoCapture(ip_url)
            
            if not self.ip_camera.isOpened():
                logger.error(f"Failed to open IP Camera at {ip_url}")
                self.trigger_event("camera_error", {"error": f"Failed to open IP Camera at {ip_url}"})
                self.ip_camera = None
                return False
            
            self.ip_camera_url = ip_url
            self.active_camera_type = "ip"
            logger.info(f"IP Camera activated at {ip_url}")
            self.trigger_event("camera_activated", {"type": "ip", "url": ip_url})
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate IP Camera: {e}")
            self.trigger_event("camera_error", {"error": f"IP Camera activation failed: {e}"})
            self.ip_camera = None
            return False
    
    def deactivate_ip_camera(self):
        """Deactivates the IP Camera."""
        if self.ip_camera:
            self.ip_camera.release()
            self.ip_camera = None
            self.ip_camera_url = None
            self.active_camera_type = None
            logger.info("IP Camera deactivated")
            self.trigger_event("camera_deactivated", {"type": "ip"})
    
    def _release_current_camera(self):
        """Release any currently active camera before switching to another."""
        if self.webcam:
            self.webcam.release()
            self.webcam = None
        
        if self.rpi_camera:
            self.rpi_camera.release()
            self.rpi_camera = None
        
        if self.ip_camera:
            self.ip_camera.release()
            self.ip_camera = None
        
        self.active_camera_type = None
    
    def _initialize_webcam(self, camera_id=1):
        self._release_current_camera()
        
        try:
            self.webcam = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            if not self.webcam.isOpened():
                logger.error("Failed to open webcam")
                self.trigger_event("camera_error", {"error": "Failed to open webcam"})
                return False
            
            self.active_camera_type = "webcam"
            logger.info(f"Webcam initialized with resolution {self.resolution}")
            self.trigger_event("camera_initialized", {"resolution": self.resolution})
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize webcam: {e}")
            self.trigger_event("camera_error", {"error": f"Initialization failed: {e}"})
            return False
    
    def _get_active_camera(self):
        """Returns the currently active camera object."""
        if self.active_camera_type == "webcam":
            return self.webcam
        elif self.active_camera_type == "rpi":
            return self.rpi_camera
        elif self.active_camera_type == "ip":
            return self.ip_camera
        return None
    
    def start_capture(self, timeout_seconds=None, camera_type=None):
        if self.capturing:
            logger.info("Camera already running")
            return
        
        # If camera_type is specified, activate that camera
        if camera_type:
            if camera_type == "webcam":
                if not self._initialize_webcam():
                    return
            elif camera_type == "rpi":
                if not self.activate_rpi_camera():
                    return
            elif camera_type == "ip" and self.ip_camera_url:
                if not self.activate_ip_camera(self.ip_camera_url):
                    return
            else:
                logger.error(f"Invalid camera type: {camera_type}")
                return
        
        # If no camera is active, try to initialize a webcam
        if not self._get_active_camera():
            if not self._initialize_webcam():
                return
        
        logger.info(f"Starting camera capture with {self.active_camera_type} camera")
        self.capturing = True
        self.shutdown_requested.clear()
        
        if timeout_seconds:
            self.capture_timeout = timeout_seconds
            self.last_activity_time = time.time()
        else:
            self.capture_timeout = None
            
        self.camera_thread = threading.Thread(target=self._capture_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
        self.trigger_event("camera_started", {"type": self.active_camera_type})
    
    def stop_capture(self):
        if not self.capturing:
            return
            
        logger.info("Stopping camera capture")
        self.capturing = False
        self.shutdown_requested.set()
        
        self._stop_recording()
        
        if self.camera_thread:
            self.camera_thread.join(timeout=2)
            self.camera_thread = None
            
        self.trigger_event("camera_stopped", {})
    
    def _capture_loop(self):
        logger.info("Camera capture thread started")
        
        active_camera = self._get_active_camera()
        if not active_camera:
            logger.error("No active camera found in capture loop")
            self.capturing = False
            self.trigger_event("camera_error", {"error": "No active camera available"})
            return
        
        try:
            while self.capturing and not self.shutdown_requested.is_set():
                if self.capture_timeout and self.last_activity_time:
                    if time.time() - self.last_activity_time > self.capture_timeout:
                        logger.info(f"Camera timeout after {self.capture_timeout} seconds")
                        self.capturing = False
                        self.trigger_event("camera_timeout", {})
                        break
                
                active_camera = self._get_active_camera()
                if not active_camera:
                    logger.error("Active camera lost during capture")
                    self.capturing = False
                    self.trigger_event("camera_error", {"error": "Active camera lost"})
                    break
                
                ret, frame = active_camera.read()
                
                if not ret or frame is None:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                if self.rotation != 0:
                    if self.rotation == 90:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif self.rotation == 180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif self.rotation == 270:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                if self.night_mode_enabled:
                    frame = self._apply_night_mode_effect(frame)
                    
                frame = self._adjust_brightness_effect(frame)
                    
                with self.frame_lock:
                    self.current_frame = frame.copy()
                    self.last_frame_time = time.time()
                    if self.capture_timeout:
                        self.last_activity_time = time.time()
                
                if self.security_mode:
                    self._process_security_frame()
                
                if self.recording_active and self.video_writer is not None:
                    self.video_writer.write(frame)
                    if time.time() - self.recording_start_time > 30:
                        self._stop_recording()
                
                sleep_time = 0.05 if self.power_save_mode and not self.security_mode else 0.01
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Error in camera capture: {e}")
            self.trigger_event("camera_error", {"error": f"Capture error: {e}"})
        finally:
            self._release_current_camera()
            logger.info("Camera released and closed")
        
        logger.info("Camera capture thread stopped")
    
    def _apply_night_mode_effect(self, frame):
        if frame is None:
            return None
            
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        result[:,:,0] = result[:,:,0] * 0.8  # Reduce blue
        result[:,:,2] = result[:,:,2] * 0.8  # Reduce red
        result = cv2.convertScaleAbs(result, alpha=1.2, beta=10)
        
        return result
    
    def _adjust_brightness_effect(self, frame):
        if frame is None:
            return None
            
        alpha = 0.5 + (self.brightness / 100.0)
        beta = (self.brightness - 50)
        adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        return adjusted
    def _process_security_frame(self):
        if self.last_frame is None:
            with self.frame_lock:
                if self.current_frame is not None:
                    self.last_frame = self.current_frame.copy()
            return
        
        with self.frame_lock:
            if self.current_frame is None:
                return
            
            current_gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            last_gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
            
            frame_diff = cv2.absdiff(current_gray, last_gray)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=2)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            significant_motion = False
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.motion_threshold:
                    significant_motion = True
                    break
            
            if significant_motion:
                logger.info("Motion detected in security mode")
                self.trigger_event("motion_detected", {})
                self._record_security_event()
            
            self.last_frame = self.current_frame.copy()

def _record_security_event(self):
    event_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not self.recording_active:
        self._start_recording(event_time)
    
    faces = self.detect_faces()
    face_detected = len(faces) > 0
    
    recognized_faces = []
    for face in faces:
        if "recognized_id" in face:
            recognized_faces.append(face["recognized_id"])
    
    img_filename = f"{self.recordings_dir}/security_event_{event_time}.jpg"
    if self.current_frame is not None:
        cv2.imwrite(img_filename, self.current_frame)
        
    log_filename = f"{self.recordings_dir}/security_event_{event_time}.txt"
    with open(log_filename, "w") as f:
        event_details = {
            "timestamp": datetime.now().isoformat(),
            "face_detected": face_detected,
            "num_faces": len(faces) if face_detected else 0,
            "recognized_faces": recognized_faces,
            "camera_type": self.active_camera_type,
            "image_file": img_filename,
            "recording_filename": f"security_video_{event_time}.mp4" if self.recording_active else None
        }
        f.write(str(event_details))
        
    self.trigger_event("security_event", {
        "timestamp": datetime.now().isoformat(),
        "face_detected": face_detected,
        "num_faces": len(faces) if face_detected else 0,
        "recognized_faces": recognized_faces,
        "camera_type": self.active_camera_type,
        "image_file": img_filename,
        "log_file": log_filename
    })
    
    return log_filename

def _start_recording(self, timestamp):
    if self.recording_active:
        return
        
    video_filename = f"{self.recordings_dir}/security_video_{timestamp}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    if self.current_frame is not None:
        h, w = self.current_frame.shape[:2]
        self.video_writer = cv2.VideoWriter(
            video_filename, 
            fourcc, 
            10.0,
            (w, h)
        )
        
        self.recording_active = True
        self.recording_start_time = time.time()
        logger.info(f"Started security recording: {video_filename}")
        
        self.trigger_event("recording_started", {
            "filename": video_filename,
            "camera_type": self.active_camera_type
        })

def _stop_recording(self):
    if not self.recording_active:
        return
        
    if self.video_writer:
        self.video_writer.release()
        self.video_writer = None
        
    duration = time.time() - self.recording_start_time
    logger.info(f"Stopped security recording after {duration:.2f} seconds")
    self.recording_active = False
    
    self.trigger_event("recording_stopped", {"duration": duration})
    
def get_frame(self):
    with self.frame_lock:
        if self.current_frame is not None:
            if self.capture_timeout:
                self.last_activity_time = time.time()
            return self.current_frame.copy()
        return None

def take_photo(self, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        camera_type = self.active_camera_type or "cam"
        filename = f"photo_{camera_type}_{timestamp}.jpg"
    
    frame = self.get_frame()
    
    if frame is not None:
        cv2.imwrite(filename, frame)
        logger.info(f"Photo saved: {filename}")
        self.trigger_event("photo_taken", {
            "filename": filename,
            "camera_type": self.active_camera_type
        })
        return filename
    
    logger.error("Failed to take photo: No frame available")
    self.trigger_event("camera_error", {"error": "Failed to take photo"})
    return None

def set_security_mode(self, enabled, camera_type=None):
    # If camera_type is specified and different from current active camera,
    # switch to that camera for security mode
    if camera_type and camera_type != self.active_camera_type:
        if camera_type == "webcam":
            if not self._initialize_webcam():
                logger.error("Failed to initialize webcam for security mode")
                return False
        elif camera_type == "rpi":
            if not self.activate_rpi_camera():
                logger.error("Failed to initialize RPi camera for security mode")
                return False
        elif camera_type == "ip" and self.ip_camera_url:
            if not self.activate_ip_camera(self.ip_camera_url):
                logger.error("Failed to initialize IP camera for security mode")
                return False
        else:
            logger.error(f"Invalid camera type for security mode: {camera_type}")
            return False
    
    self.security_mode = enabled
    logger.info(f"Security mode {'enabled' if enabled else 'disabled'} using {self.active_camera_type} camera")
    
    if enabled and self.current_frame is not None:
        with self.frame_lock:
            self.last_frame = self.current_frame.copy()
            
    if enabled:
        self.set_night_mode(True)
        
        # Make sure we're capturing if not already
        if not self.capturing:
            self.start_capture()
            
    self.trigger_event("security_mode_changed", {
        "enabled": enabled,
        "camera_type": self.active_camera_type
    })
    
    return True

def set_ip_camera_url(self, url):
    """Set IP camera URL for later activation"""
    self.ip_camera_url = url
    logger.info(f"IP camera URL set to: {url}")
    return True

def get_camera_status(self):
    return {
        "active": self.capturing,
        "camera_type": self.active_camera_type,
        "security_mode": self.security_mode,
        "night_mode": self.night_mode_enabled,
        "brightness": self.brightness,
        "recording": self.recording_active,
        "last_frame_time": self.last_frame_time,
        "resolution": self.resolution,
        "rotation": self.rotation,
        "ip_camera_url": self.ip_camera_url if self.active_camera_type == "ip" else None
    }

def detect_faces(self, frame=None):
    if frame is None:
        frame = self.get_frame()
        
    if frame is None:
        return []
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = self.face_detection.process(rgb_frame)
    
    face_detections = []
    
    if results.detections:
        h, w, _ = frame.shape
        
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            landmarks = {}
            for idx, landmark in enumerate(detection.location_data.relative_keypoints):
                landmarks[f"point_{idx}"] = {
                    "x": int(landmark.x * w),
                    "y": int(landmark.y * h)
                }
            
            face_detections.append({
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "confidence": detection.score[0],
                "landmarks": landmarks
            })
            
        if face_detections:
            self.trigger_event("faces_detected", {
                "count": len(face_detections),
                "camera_type": self.active_camera_type
            })
            
            # For each detected face, attempt to recognize it
            for face in face_detections:
                face_id = self.recognize_face(frame, face)
                if face_id:
                    face["recognized_id"] = face_id
                
    return face_detections

def __del__(self):
    self.stop_capture()
    
    if hasattr(self, 'face_db_conn') and self.face_db_conn:
        self.face_db_conn.close()

"""# Example usage function
def example_usage():
    """
   """ Example of how to use the CameraManager class with different camera types"""
    """
    # Initialize the camera manager with default settings
    camera = CameraManager(resolution=(640, 480), rotation=0, power_save_mode=True)
    
    # Define a simple callback for motion detection
    def motion_callback(event_data):
        print(f"Motion detected! Event data: {event_data}")
    
    # Register event handlers
    camera.register_event_handler("motion_detected", motion_callback)
    camera.register_event_handler("security_event", lambda data: print(f"Security event: {data}"))
    camera.register_event_handler("face_recognized", lambda data: print(f"Face recognized: {data['name']}"))
    
    # Example 1: Use default webcam
    camera.start_capture(camera_type="webcam")
    time.sleep(5)  # Run for 5 seconds
    
    # Example 2: Switch to Raspberry Pi camera for security mode
    camera.stop_capture()
    camera.set_security_mode(True, camera_type="rpi")
    time.sleep(10)  # Run security mode for 10 seconds
    
    # Example 3: Use IP camera with custom URL
    camera.stop_capture()
    camera.set_ip_camera_url("rtsp://username:password@192.168.1.100:554/stream")
    camera.start_capture(camera_type="ip")
    time.sleep(5)  # Run for 5 seconds
    
    # Clean up
    camera.stop_capture()"""

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the example
    example_usage()