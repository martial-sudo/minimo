#!/usr/bin/env python3
import logging
import json
import os
import time
import threading
import cv2
import numpy as np
import face_recognition

logger = logging.getLogger(__name__)

class UserManager:
    def __init__(self, camera_manager):
        logger.info("Initializing User Manager")
        self.camera_manager = camera_manager
        self.users_file = "users.json"
        self.users = []
        self.user_lock = threading.Lock()
        self.face_data_dir = "face_data"
        
        # Ensure face data directory exists
        if not os.path.exists(self.face_data_dir):
            os.makedirs(self.face_data_dir)
            
        # Load existing users
        self.load_users()
        
        # Face recognition parameters
        self.face_match_threshold = 0.6  # Lower is more strict
    
    def load_users(self):
        """Load user data from storage"""
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    self.users = json.load(f)
                
                # Load face encodings for each user
                for user in self.users:
                    encoding_file = os.path.join(self.face_data_dir, f"user_{user['id']}_encoding.npy")
                    if os.path.exists(encoding_file):
                        user['face_encoding'] = np.load(encoding_file)
                    else:
                        logger.warning(f"Face encoding not found for user {user['name']}")
                
                logger.info(f"Loaded {len(self.users)} users")
            else:
                self.users = []
                logger.info("No existing users found")
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            self.users = []
    
    def save_users(self):
        """Save user data to storage"""
        try:
            with self.user_lock:
                # Create a copy of users without the face_encoding field for JSON serialization
                users_to_save = []
                for user in self.users:
                    user_copy = user.copy()
                    if 'face_encoding' in user_copy:
                        del user_copy['face_encoding']
                    users_to_save.append(user_copy)
                
                with open(self.users_file, 'w') as f:
                    json.dump(users_to_save, f)
            logger.info(f"Saved {len(self.users)} users")
        except Exception as e:
            logger.error(f"Error saving users: {e}")
    
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
        
        # Try to match with known users
        for face_encoding in face_encodings:
            matches = []
            for user in self.users:
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
                
                # Update last seen timestamp
                recognized_user['last_seen'] = time.time()
                self.save_users()
                
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
        
        # Create a new user with the face encoding
        with self.user_lock:
            user_id = len(self.users) + 1 if self.users else 1
            
            # Check for duplicate names
            for user in self.users:
                if user["name"].lower() == name.lower():
                    logger.warning(f"User with name '{name}' already exists")
                    return False
            
            # Create the new user
            new_user = {
                "id": user_id,
                "name": name,
                "created": time.time(),
                "last_seen": time.time(),
                "face_encoding": face_encoding
            }
            self.users.append(new_user)
        
        # Save face encoding to file
        encoding_file = os.path.join(self.face_data_dir, f"user_{user_id}_encoding.npy")
        np.save(encoding_file, face_encoding)
        
        # Save user image for reference (optional)
        face_top, face_right, face_bottom, face_left = face_locations[0]
        face_image = frame[face_top:face_bottom, face_left:face_right]
        cv2.imwrite(os.path.join(self.face_data_dir, f"user_{user_id}.jpg"), face_image)
        
        # Save updated user list
        self.save_users()
        logger.info(f"User {name} added successfully with ID {user_id}")
        return True
    
    def remove_user(self, user_id):
        """Remove a user by ID"""
        with self.user_lock:
            for i, user in enumerate(self.users):
                if user["id"] == user_id:
                    removed_user = self.users.pop(i)
                    
                    # Remove face encoding file
                    encoding_file = os.path.join(self.face_data_dir, f"user_{user_id}_encoding.npy")
                    if os.path.exists(encoding_file):
                        os.remove(encoding_file)
                    
                    # Remove user image if it exists
                    user_image = os.path.join(self.face_data_dir, f"user_{user_id}.jpg")
                    if os.path.exists(user_image):
                        os.remove(user_image)
                    
                    self.save_users()
                    logger.info(f"User {removed_user['name']} removed successfully")
                    return True
        
        logger.warning(f"No user found with ID {user_id}")
        return False
    
    def list_users(self):
        """Return a list of all users"""
        # Create a copy of users without the face_encoding field
        user_list = []
        for user in self.users:
            user_copy = user.copy()
            if 'face_encoding' in user_copy:
                del user_copy['face_encoding']
            user_list.append(user_copy)
        
        return user_list
    
    def update_user(self, user_id, new_name=None):
        """Update user information"""
        with self.user_lock:
            for user in self.users:
                if user["id"] == user_id:
                    if new_name:
                        user["name"] = new_name
                    self.save_users()
                    logger.info(f"User {user_id} updated successfully")
                    return True
        
        logger.warning(f"No user found with ID {user_id}")
        return False
    
    def update_face_data(self, user_id):
        """Update the face data for an existing user"""
        logger.info(f"Updating face data for user ID: {user_id}")
        
        # Find the user
        target_user = None
        for user in self.users:
            if user["id"] == user_id:
                target_user = user
                break
        
        if not target_user:
            logger.warning(f"No user found with ID {user_id}")
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
        
        # Update the user's face encoding
        with self.user_lock:
            target_user['face_encoding'] = face_encoding
            target_user['updated'] = time.time()
        
        # Save face encoding to file
        encoding_file = os.path.join(self.face_data_dir, f"user_{user_id}_encoding.npy")
        np.save(encoding_file, face_encoding)
        
        # Update user image for reference
        face_top, face_right, face_bottom, face_left = face_locations[0]
        face_image = frame[face_top:face_bottom, face_left:face_right]
        cv2.imwrite(os.path.join(self.face_data_dir, f"user_{user_id}.jpg"), face_image)
        
        # Save updated user list
        self.save_users()
        logger.info(f"Face data updated for user {target_user['name']}")
        return True