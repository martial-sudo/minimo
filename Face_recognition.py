import cv2
import mediapipe as mp
import numpy as np
import sqlite3
from deepface import DeepFace
import time

class FaceRecognitionSystem:
    def __init__(self, ip_camera_url=None, camera_index=2, shared_camera=None):
        """Initialize Face Recognition System with either an IP Camera or USB Camera
        
        Args:
            ip_camera_url: URL for IP camera if used
            camera_index: Camera index to use (default 2)
            shared_camera: Optional shared camera instance from another component
        """
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

        # Database connection
        self.conn = sqlite3.connect("face_database.db", check_same_thread=False)
        self.cursor = self.conn.cursor()

        # Create table if it doesn't exist
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            embedding BLOB
        )
        ''')
        self.conn.commit()

        # Use shared camera or open new camera
        self.owns_camera = shared_camera is None
        self.ip_camera_url = ip_camera_url
        
        if shared_camera:
            # Use the shared camera instance
            self.cap = shared_camera
            print("Using shared camera instance")
        elif ip_camera_url:
            # Use IP camera
            self.cap = cv2.VideoCapture(ip_camera_url)
            print(f"Opening IP camera at {ip_camera_url}")
        else:
            # Try multiple ways to open the camera
            print(f"Attempting to open camera at index {camera_index}")
            
            # Release any existing camera instances at this index
            temp_cap = cv2.VideoCapture(camera_index)
            temp_cap.release()
            
            # Try with default backend
            self.cap = cv2.VideoCapture(camera_index)
            
            # If that fails, try with explicit backends
            if not self.cap.isOpened():
                print(f"Failed with default backend, trying V4L2...")
                self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
            
            # If that still fails, try with absolute path
            if not self.cap.isOpened():
                print(f"Failed with V4L2 backend, trying absolute path...")
                self.cap = cv2.VideoCapture(f"/dev/video{camera_index}")

        # Final check
        if not self.cap.isOpened():
            print("Could not access the camera feed. Please check connections.")
            # Don't raise an exception to allow the program to continue
            # Instead, we'll handle errors during capture

    def get_embedding(self, image):
        """Extract embeddings using DeepFace (Facenet)"""
        try:
            embedding = DeepFace.represent(image, model_name="Facenet")[0]["embedding"]
            return np.array(embedding)
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
            
    def recognize_face(self, embedding):
        """Recognize a face by comparing embeddings"""
        self.cursor.execute("SELECT name, embedding FROM faces")
        records = self.cursor.fetchall()

        if not records:
            print("Database is empty.")
            return "Unknown"

        embedding = np.array(embedding[:128], dtype=np.float32)
        embedding /= np.linalg.norm(embedding)  # Normalize input embedding

        best_match = None
        best_similarity = float("inf")  # Lower is better

        print("Stored faces in database:")
        for name, db_embedding in records:
            db_embedding = np.frombuffer(db_embedding, dtype=np.float32)[:128].copy()  # Ensure writable

            if np.linalg.norm(db_embedding) == 0:
                print(f"Skipping {name} due to invalid embedding.")
                continue

            db_embedding /= np.linalg.norm(db_embedding)  # Normalize stored embedding
            similarity = np.linalg.norm(db_embedding - embedding)

            print(f"Comparing with {name}: Similarity = {similarity}")

            if similarity < best_similarity:  
                best_similarity = similarity
                best_match = name

        print(f"Best match: {best_match}, Best similarity: {best_similarity}")

        return best_match if best_similarity < 15 else "Unknown"

    def add_new_user(self, name, face_embedding):
        """Save a new user's face embedding to the database"""
        face_embedding = np.array(face_embedding[:128], dtype=np.float32)
        
        if np.linalg.norm(face_embedding) == 0:
            print(f"Invalid embedding for {name}. Not saving.")
            return

        face_embedding /= np.linalg.norm(face_embedding)  # Normalize before storing
        embedding_blob = face_embedding.tobytes()

        self.cursor.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, embedding_blob))
        self.conn.commit()

        print(f"Registered {name} successfully!")
        print(f"Stored embedding: {face_embedding[:5]}... (only showing first 5 values)")

    def capture_and_recognize_face(self):
        """Capture an image from the camera and recognize the face"""
        if not self.cap.isOpened():
            print("Camera not open, attempting to reopen...")
            if self.owns_camera:  # Only try to reopen if we own the camera
                self.cap = cv2.VideoCapture(2)
            if not self.cap.isOpened():
                return ["Error"], None
        
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read frame from camera")
            return ["Error"], None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # Ensure bounding box is within the image
                x, y = max(0, x), max(0, y)
                width, height = min(w - x, width), min(h - y, height)

                face_crop = frame[y:y+height, x:x+width]

                if face_crop.size > 0:
                    embedding = self.get_embedding(face_crop)
                    if embedding is not None:
                        recognized_name = self.recognize_face(embedding)
                        return [recognized_name], frame

        return ["Unknown"], frame
        
    def detect_emotion(self, frame):
        """Detects the emotion of a person from a captured frame"""
        try:
            result = DeepFace.analyze(frame, actions=['emotion'])[0]
            emotion = result['dominant_emotion']
            if emotion in ['happy', 'smile']:
                return "happy"
            elif emotion in ['sad', 'disgust']:
                return "sad"
            else:
                return "neutral"
        except Exception as e:
            print(f"Error detecting emotion: {e}")
            return "unknown"

    def close_connection(self):
        """Close the database connection and release resources"""
        self.conn.close()
        # Only release the camera if we own it
        if self.owns_camera and self.cap is not None and self.cap.isOpened():
            self.cap.release()