#!/usr/bin/env python3
import logging
import json
import os
import time
import threading
import cv2
import numpy as np
import face_recognition
import sqlite3
import pickle

logger = logging.getLogger(__name__)

class UserManager:
    def __init__(self, camera_manager):
        logger.info("Initializing User Manager")
        self.camera_manager = camera_manager
        self.face_data_dir = "face_data"
        self.db_path = "assistant_data.db"
        self.user_lock = threading.Lock()
        
        # Ensure face data directory exists
        if not os.path.exists(self.face_data_dir):
            os.makedirs(self.face_data_dir)
            
        # Initialize database
        self._init_database()
        
        # Face recognition parameters
        self.face_match_threshold = 0.6  # Lower is more strict
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created REAL NOT NULL,
                last_seen REAL NOT NULL,
                face_encoding BLOB
            )
            ''')
            
            # Create conversation_logs table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp REAL NOT NULL,
                command TEXT NOT NULL,
                response TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def get_users(self):
        """Load user data from database"""
        users = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, name, created, last_seen, face_encoding FROM users")
            rows = cursor.fetchall()
            
            for row in rows:
                user_id, name, created, last_seen, face_encoding_blob = row
                
                user = {
                    "id": user_id,
                    "name": name,
                    "created": created,
                    "last_seen": last_seen,
                }
                
                if face_encoding_blob:
                    user['face_encoding'] = pickle.loads(face_encoding_blob)
                
                users.append(user)
            
            conn.close()
            logger.info(f"Loaded {len(users)} users from database")
        except Exception as e:
            logger.error(f"Error loading users from database: {e}")
        
        return users
    
    def identify_user(self):
        """Identify a user from camera input using facial recognition"""
        logger.info("Attempting to identify user")
        
        # Ensure camera is capturing
        if not self.camera_manager.capturing:
            logger.warning("Camera not active, cannot identify user")
            return None
        
        # Get current frame from camera
        frame = self.camera_manager.get_frame()
        if frame is None:
            logger.warning("Failed to get frame from camera")
            return None
        
        # Process the image for face recognition (resize for faster processing)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        
        if not face_locations:
            logger.info("No faces detected")
            return None
        
        # Get face encodings for any faces in the frame
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # Get all users from database
        users = self.get_users()
        
        # Try to match with known users
        for face_encoding in face_encodings:
            matches = []
            for user in users:
                if 'face_encoding' in user:
                    # Compare face with stored user face data
                    match = face_recognition.compare_faces(
                        [user['face_encoding']], 
                        face_encoding, 
                        tolerance=self.face_match_threshold
                    )[0]
                    
                    if match:
                        # Calculate face distance for confidence score
                        face_distance = face_recognition.face_distance([user['face_encoding']], face_encoding)[0]
                        confidence = 1 - face_distance  # Convert distance to confidence score
                        matches.append((user, confidence))
            
            if matches:
                # Return the user with the highest confidence
                matches.sort(key=lambda x: x[1], reverse=True)
                recognized_user = matches[0][0]
                confidence = matches[0][1]
                logger.info(f"Recognized user: {recognized_user['name']} with confidence: {confidence:.2f}")
                
                # Update last seen timestamp in database
                try:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE users SET last_seen = ? WHERE id = ?", 
                        (time.time(), recognized_user['id'])
                    )
                    conn.commit()
                    conn.close()
                except Exception as e:
                    logger.error(f"Error updating last_seen timestamp: {e}")
                
                return recognized_user
        
        logger.info("No user recognized from detected faces")
        return None
    
    def add_user(self, name):
        """Add a new user with the given name using facial recognition"""
        logger.info(f"Adding new user: {name}")
        
        # Ensure camera is capturing
        if not self.camera_manager.capturing:
            logger.warning("Camera not active, cannot add user")
            return False
        
        # Get current frame from camera
        frame = self.camera_manager.get_frame()
        if frame is None:
            logger.warning("Failed to get frame from camera")
            return False
        
        # Process the image for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all face locations in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            logger.warning("No face detected, cannot add user")
            return False
        
        if len(face_locations) > 1:
            logger.warning("Multiple faces detected, please ensure only one face is visible")
            return False
        
        # Extract face encoding for the detected face
        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
        
        # Check for duplicate names
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE name = ?", (name,))
            if cursor.fetchone():
                logger.warning(f"User with name '{name}' already exists")
                conn.close()
                return False
            
            # Insert new user into database
            current_time = time.time()
            cursor.execute(
                "INSERT INTO users (name, created, last_seen, face_encoding) VALUES (?, ?, ?, ?)",
                (name, current_time, current_time, pickle.dumps(face_encoding))
            )
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            # Save user image for reference (optional)
            face_top, face_right, face_bottom, face_left = face_locations[0]
            face_image = frame[face_top:face_bottom, face_left:face_right]
            cv2.imwrite(os.path.join(self.face_data_dir, f"user_{user_id}.jpg"), face_image)
            
            logger.info(f"User {name} added successfully with ID {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding new user: {e}")
            return False
    
    def remove_user(self, user_id):
        """Remove a user by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get user name first for logging
            cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
            user = cursor.fetchone()
            
            if not user:
                logger.warning(f"No user found with ID {user_id}")
                conn.close()
                return False
            
            user_name = user[0]
            
            # Delete conversation logs for this user
            cursor.execute("DELETE FROM conversation_logs WHERE user_id = ?", (user_id,))
            
            # Delete the user
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            
            conn.commit()
            conn.close()
            
            # Remove user image if it exists
            user_image = os.path.join(self.face_data_dir, f"user_{user_id}.jpg")
            if os.path.exists(user_image):
                os.remove(user_image)
            
            logger.info(f"User {user_name} removed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error removing user: {e}")
            return False
    
    def list_users(self):
        """Return a list of all users without face encoding data"""
        user_list = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, name, created, last_seen FROM users")
            rows = cursor.fetchall()
            
            for row in rows:
                user_id, name, created, last_seen = row
                user = {
                    "id": user_id,
                    "name": name,
                    "created": created,
                    "last_seen": last_seen
                }
                user_list.append(user)
            
            conn.close()
        except Exception as e:
            logger.error(f"Error listing users: {e}")
        
        return user_list
    
    def update_user(self, user_id, new_name=None):
        """Update user information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
            if not cursor.fetchone():
                logger.warning(f"No user found with ID {user_id}")
                conn.close()
                return False
            
            if new_name:
                cursor.execute("UPDATE users SET name = ? WHERE id = ?", (new_name, user_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"User {user_id} updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            return False
    
    def update_face_data(self, user_id):
        """Update the face data for an existing user"""
        logger.info(f"Updating face data for user ID: {user_id}")
        
        # Check if user exists
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
            user = cursor.fetchone()
            conn.close()
            
            if not user:
                logger.warning(f"No user found with ID {user_id}")
                return False
                
            user_name = user[0]
        except Exception as e:
            logger.error(f"Error checking user existence: {e}")
            return False
        
        # Ensure camera is capturing
        if not self.camera_manager.capturing:
            logger.warning("Camera not active, cannot update user face data")
            return False
        
        # Get current frame from camera
        frame = self.camera_manager.get_frame()
        if frame is None:
            logger.warning("Failed to get frame from camera")
            return False
        
        # Process the image for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all face locations in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            logger.warning("No face detected, cannot update user face data")
            return False
        
        if len(face_locations) > 1:
            logger.warning("Multiple faces detected, please ensure only one face is visible")
            return False
        
        # Extract face encoding for the detected face
        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
        
        # Update the user's face encoding in database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE users SET face_encoding = ? WHERE id = ?", 
                (pickle.dumps(face_encoding), user_id)
            )
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error updating face data in database: {e}")
            return False
        
        # Update user image for reference
        face_top, face_right, face_bottom, face_left = face_locations[0]
        face_image = frame[face_top:face_bottom, face_left:face_right]
        cv2.imwrite(os.path.join(self.face_data_dir, f"user_{user_id}.jpg"), face_image)
        
        logger.info(f"Face data updated for user {user_name}")
        return True
    
    def log_conversation(self, user_id, command, response):
        """Log a conversation entry for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO conversation_logs (user_id, timestamp, command, response) VALUES (?, ?, ?, ?)",
                (user_id, time.time(), command, response)
            )
            
            conn.commit()
            conn.close()
            logger.debug(f"Logged conversation for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error logging conversation: {e}")
            return False
    
    def get_conversation_history(self, user_id, limit=10):
        """Get recent conversation history for a user"""
        history = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT timestamp, command, response FROM conversation_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", 
                (user_id, limit)
            )
            
            rows = cursor.fetchall()
            for row in rows:
                timestamp, command, response = row
                history.append({
                    "timestamp": timestamp,
                    "command": command,
                    "response": response
                })
            
            conn.close()
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
        
        return history