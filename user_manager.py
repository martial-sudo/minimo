#!/usr/bin/env python3
import logging
import json
import os
import time
import threading

logger = logging.getLogger(__name__)

class UserManager:
    def __init__(self, camera_manager):
        logger.info("Initializing User Manager")
        self.camera_manager = camera_manager
        self.users_file = "users.json"
        self.users = []
        self.user_lock = threading.Lock()
        
        # Load existing users
        self.load_users()
    
    def load_users(self):
        """Load user data from storage"""
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    self.users = json.load(f)
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
                with open(self.users_file, 'w') as f:
                    json.dump(self.users, f)
            logger.info(f"Saved {len(self.users)} users")
        except Exception as e:
            logger.error(f"Error saving users: {e}")
    
    def identify_user(self):
        """Identify a user from camera input"""
        logger.info("Attempting to identify user")
        
        # Ensure camera is capturing
        if not self.camera_manager.capturing:
            logger.warning("Camera not active, cannot identify user")
            return None
        
        # Get current frame and detect faces
        faces = self.camera_manager.detect_faces()
        
        if not faces:
            logger.info("No faces detected")
            return None
        
        # In a real implementation, this would extract face features and
        # compare with stored user face data to identify the user
        
        # For demo, randomly determine if user is recognized
        # and if so, which user (if any exist)
        if self.users and self._simulate_random_event(0.8):  # 80% chance to recognize if users exist
            user_index = int(self._simulate_random_event(len(self.users)))
            recognized_user = self.users[user_index]
            logger.info(f"Recognized user: {recognized_user['name']}")
            return recognized_user
        else:
            logger.info("User not recognized")
            return None
    
    def add_user(self, name):
        """Add a new user with the given name"""
        logger.info(f"Adding new user: {name}")
        
        # Ensure camera is capturing
        if not self.camera_manager.capturing:
            logger.warning("Camera not active, cannot add user")
            return False
        
        # Get current frame and detect faces
        faces = self.camera_manager.detect_faces()
        
        if not faces:
            logger.warning("No face detected, cannot add user")
            return False
        
        # In a real implementation, this would extract face features
        # from the detected face to use for future recognition
        
        # For demo, just create a user entry with the name
        with self.user_lock:
            user_id = len(self.users) + 1
            new_user = {
                "id": user_id,
                "name": name,
                "created": time.time(),
                "face_data": f"simulated_face_data_{user_id}"
                # In a real implementation, this would be the actual face features
            }
            self.users.append(new_user)
        
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
                    self.save_users()
                    logger.info(f"User {removed_user['name']} removed successfully")
                    return True
        
        logger.warning(f"No user found with ID {user_id}")
        return False
    
    def list_users(self):
        """Return a list of all users"""
        return self.users
    
    def _simulate_random_event(self, probability):
        """Helper to simulate random events with given probability"""
        import random
        if isinstance(probability, (int, float)):
            return random.random() < probability
        else:
            return random.randint(0, probability-1)