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
    def __init__(self, resolution=(640, 480), rotation=0, camera_id=1, power_save_mode=True):
        self.resolution = resolution
        self.rotation = rotation
        self.camera_id = camera_id
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
    
    def _initialize_webcam(self):
        try:
            self.webcam = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            if not self.webcam.isOpened():
                logger.error("Failed to open webcam")
                self.trigger_event("camera_error", {"error": "Failed to open webcam"})
                return False
                
            logger.info(f"Webcam initialized with resolution {self.resolution}")
            self.trigger_event("camera_initialized", {"resolution": self.resolution})
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize webcam: {e}")
            self.trigger_event("camera_error", {"error": f"Initialization failed: {e}"})
            return False
    
    def start_capture(self, timeout_seconds=None):
        if self.capturing:
            logger.info("Camera already running")
            return
            
        logger.info("Starting camera capture")
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
        
        self.trigger_event("camera_started", {})
    
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
        
        if not self._initialize_webcam():
            self.capturing = False
            self.trigger_event("camera_error", {"error": "Failed to initialize webcam in capture loop"})
            return
        
        try:
            while self.capturing and not self.shutdown_requested.is_set():
                if self.capture_timeout and self.last_activity_time:
                    if time.time() - self.last_activity_time > self.capture_timeout:
                        logger.info(f"Camera timeout after {self.capture_timeout} seconds")
                        self.capturing = False
                        self.trigger_event("camera_timeout", {})
                        break
                
                ret, frame = self.webcam.read()
                
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
            if self.webcam:
                self.webcam.release()
                self.webcam = None
            logger.info("Webcam released and closed")
        
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
    
    def get_frame(self):
        with self.frame_lock:
            if self.current_frame is not None:
                if self.capture_timeout:
                    self.last_activity_time = time.time()
                return self.current_frame.copy()
            return None
    
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
                self.trigger_event("faces_detected", {"count": len(face_detections)})
                
                # For each detected face, attempt to recognize it
                for face in face_detections:
                    face_id = self.recognize_face(frame, face)
                    if face_id:
                        face["recognized_id"] = face_id
                    
        return face_detections
    
    def extract_face_encoding(self, frame, face_data):
        """Extract face encoding features for recognition"""
        try:
            x, y, w, h = face_data["x"], face_data["y"], face_data["width"], face_data["height"]
            
            # Extract face region with some margin
            margin = int(0.1 * max(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            face_region = frame[y1:y2, x1:x2]
            
            # Convert to RGB for mesh analysis
            rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Get detailed face mesh
            mesh_results = self.mp_face_mesh.process(rgb_face)
            
            if not mesh_results.multi_face_landmarks:
                return None
            
            # Extract landmark coordinates as face encoding
            landmarks = mesh_results.multi_face_landmarks[0].landmark
            encoding = []
            
            # Use a subset of landmarks for efficiency
            landmark_indices = list(range(0, 468, 10))  # Use every 10th landmark
            
            for idx in landmark_indices:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    encoding.extend([lm.x, lm.y, lm.z])
            
            return np.array(encoding)
            
        except Exception as e:
            logger.error(f"Error extracting face encoding: {e}")
            return None
    
    def register_face(self, frame, face_data, name):
        """Register a new face in the database"""
        if self.face_db_conn is None:
            logger.error("Face database not initialized")
            return False
        
        encoding = self.extract_face_encoding(frame, face_data)
        if encoding is None:
            logger.error("Failed to extract face encoding")
            return False
        
        try:
            cursor = self.face_db_conn.cursor()
            
            # Convert numpy array to bytes for storage
            encoding_bytes = encoding.tobytes()
            
            cursor.execute(
                "INSERT INTO face_data (name, face_encodings) VALUES (?, ?)",
                (name, encoding_bytes)
            )
            
            self.face_db_conn.commit()
            face_id = cursor.lastrowid
            
            logger.info(f"Registered new face for {name} with ID {face_id}")
            self.trigger_event("face_registered", {"name": name, "id": face_id})
            
            return face_id
            
        except Exception as e:
            logger.error(f"Failed to register face: {e}")
            return False
    
    def recognize_face(self, frame, face_data, threshold=0.7):
        """Recognize a face against the database"""
        if self.face_db_conn is None:
            return None
        
        encoding = self.extract_face_encoding(frame, face_data)
        if encoding is None:
            return None
        
        try:
            cursor = self.face_db_conn.cursor()
            cursor.execute("SELECT id, name, face_encodings FROM face_data")
            
            best_match = None
            best_similarity = -1
            
            for row in cursor.fetchall():
                face_id, name, stored_encoding_bytes = row
                
                # Convert bytes back to numpy array
                stored_encoding = np.frombuffer(stored_encoding_bytes, dtype=np.float64)
                
                # Ensure same length for comparison
                min_len = min(len(encoding), len(stored_encoding))
                encoding_a = encoding[:min_len]
                encoding_b = stored_encoding[:min_len]
                
                # Calculate similarity (cosine similarity)
                similarity = np.dot(encoding_a, encoding_b) / (np.linalg.norm(encoding_a) * np.linalg.norm(encoding_b))
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (face_id, name, similarity)
            
            # If similarity exceeds threshold, consider it a match
            if best_match and best_similarity > threshold:
                face_id, name, confidence = best_match
                
                # Log recognition
                cursor.execute(
                    "INSERT INTO recognition_logs (face_id, confidence) VALUES (?, ?)",
                    (face_id, confidence)
                )
                self.face_db_conn.commit()
                
                logger.info(f"Recognized face: {name} (ID: {face_id}, confidence: {confidence:.2f})")
                self.trigger_event("face_recognized", {
                    "id": face_id, 
                    "name": name, 
                    "confidence": confidence
                })
                
                return {"id": face_id, "name": name, "confidence": confidence}
            
            return None
            
        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            return None
    
    def detect_face_mesh(self, frame=None):
        if frame is None:
            frame = self.get_frame()
            
        if frame is None:
            return []
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        mesh_results = []
        
        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for idx, landmark in enumerate(face_landmarks.landmark):
                    landmarks.append({
                        "idx": idx,
                        "x": int(landmark.x * w),
                        "y": int(landmark.y * h),
                        "z": landmark.z
                    })
                
                x_coords = [lm["x"] for lm in landmarks]
                y_coords = [lm["y"] for lm in landmarks]
                
                x = min(x_coords)
                y = min(y_coords)
                width = max(x_coords) - x
                height = max(y_coords) - y
                
                mesh_results.append({
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "landmarks": landmarks
                })
        
            if mesh_results:
                self.trigger_event("face_mesh_detected", {"count": len(mesh_results)})
                
        return mesh_results
    
    def take_photo(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"photo_{timestamp}.jpg"
        
        frame = self.get_frame()
        
        if frame is not None:
            cv2.imwrite(filename, frame)
            logger.info(f"Photo saved: {filename}")
            self.trigger_event("photo_taken", {"filename": filename})
            return filename
        
        logger.error("Failed to take photo: No frame available")
        self.trigger_event("camera_error", {"error": "Failed to take photo"})
        return None
    
    def set_security_mode(self, enabled):
        self.security_mode = enabled
        logger.info(f"Security mode {'enabled' if enabled else 'disabled'}")
        
        if enabled and self.current_frame is not None:
            with self.frame_lock:
                self.last_frame = self.current_frame.copy()
                
        if enabled:
            self.set_night_mode(True)
            
        self.trigger_event("security_mode_changed", {"enabled": enabled})
    
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
                "image_file": img_filename,
                "recording_filename": f"security_video_{event_time}.mp4" if self.recording_active else None
            }
            f.write(str(event_details))
            
        self.trigger_event("security_event", {
            "timestamp": datetime.now().isoformat(),
            "face_detected": face_detected,
            "num_faces": len(faces) if face_detected else 0,
            "recognized_faces": recognized_faces,
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
            
            self.trigger_event("recording_started", {"filename": video_filename})
    
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
        
    def adjust_brightness(self, brightness):
        self.brightness = max(0, min(100, brightness))
        logger.info(f"Camera brightness set to {self.brightness}")
        self.trigger_event("brightness_changed", {"value": self.brightness})
    
    def set_night_mode(self, enabled):
        self.night_mode_enabled = enabled
        logger.info(f"Night mode {'enabled' if enabled else 'disabled'}")
        self.trigger_event("night_mode_changed", {"enabled": enabled})
        
    def set_motion_threshold(self, threshold):
        self.motion_threshold = max(5, min(100, threshold))
        logger.info(f"Motion threshold set to {self.motion_threshold}")
        
    def get_camera_status(self):
        return {
            "active": self.capturing,
            "security_mode": self.security_mode,
            "night_mode": self.night_mode_enabled,
            "brightness": self.brightness,
            "recording": self.recording_active,
            "last_frame_time": self.last_frame_time
        }
    
    def __del__(self):
        if hasattr(self, 'face_db_conn') and self.face_db_conn:
            self.face_db_conn.close()