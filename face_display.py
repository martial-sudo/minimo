#!/usr/bin/env python3
import logging
import threading
import time
import json
import os
from datetime import datetime
import random
import pygame
import math

logger = logging.getLogger(__name__)

class FaceDisplay:
    def __init__(self, response_queue):
        logger.info("Initializing Face Display")
        self.response_queue = response_queue
        self.current_emotion = "neutral"
        self.emotion_lock = threading.Lock()
        self.display_active = False
        self.display_thread = None
        
        # Screen settings
        self.screen_width = 800
        self.screen_height = 480  # Common for Raspberry Pi displays
        self.screen = None
        
        # Emotion definition with facial elements
        self.emotions = {
            "neutral": {
                "eyes": "normal",
                "mouth": "slight_smile",
                "color": "#3498db"  # Blue
            },
            "happy": {
                "eyes": "wide",
                "mouth": "big_smile",
                "color": "#2ecc71"  # Green
            },
            "sad": {
                "eyes": "lowered",
                "mouth": "frown",
                "color": "#9b59b6"  # Purple
            },
            "thinking": {
                "eyes": "looking_up",
                "mouth": "straight",
                "color": "#f39c12"  # Orange
            },
            "attentive": {
                "eyes": "wide",
                "mouth": "slight_smile",
                "color": "#1abc9c"  # Teal
            },
            "confused": {
                "eyes": "squint",
                "mouth": "zigzag",
                "color": "#e74c3c"  # Red
            },
            "error": {
                "eyes": "x_eyes",
                "mouth": "frown",
                "color": "#e74c3c"  # Red
            },
            "vigilant": {
                "eyes": "scanning",
                "mouth": "straight",
                "color": "#d35400"  # Dark Orange
            }
        }
        
        # Face asset variables
        self.face_assets = {}
        self.blink_state = False
        self.blink_counter = 0
        self.mouth_position = 0
        self.mouth_direction = 1
        self.speaking = False
        
        # Load face assets
        self.load_face_assets()
    
    def load_face_assets(self):
        """Load face display assets"""
        logger.info("Loading face display assets")
        
        # Initialize asset containers
        self.face_assets = {
            "eyes": {},
            "mouths": {}
        }
        
        # Define eye configurations
        self.face_assets["eyes"] = {
            "normal": {
                "draw": self._draw_normal_eyes
            },
            "wide": {
                "draw": self._draw_wide_eyes
            },
            "lowered": {
                "draw": self._draw_lowered_eyes
            },
            "looking_up": {
                "draw": self._draw_looking_up_eyes
            },
            "squint": {
                "draw": self._draw_squint_eyes
            },
            "x_eyes": {
                "draw": self._draw_x_eyes
            },
            "scanning": {
                "draw": self._draw_scanning_eyes
            },
            "blinking": {
                "draw": self._draw_blinking_eyes
            }
        }
        
        # Define mouth configurations
        self.face_assets["mouths"] = {
            "slight_smile": {
                "draw": self._draw_slight_smile
            },
            "big_smile": {
                "draw": self._draw_big_smile
            },
            "frown": {
                "draw": self._draw_frown
            },
            "straight": {
                "draw": self._draw_straight_mouth
            },
            "zigzag": {
                "draw": self._draw_zigzag_mouth
            },
            "speaking": {
                "draw": self._draw_speaking_mouth
            }
        }
    
    def update_emotion(self, emotion):
        """Update the current emotion displayed on the face"""
        if emotion not in self.emotions:
            logger.warning(f"Unknown emotion: {emotion}, defaulting to neutral")
            emotion = "neutral"
            
        with self.emotion_lock:
            logger.info(f"Changing face emotion to: {emotion}")
            self.current_emotion = emotion
    
    def get_current_emotion(self):
        """Get the current emotion"""
        with self.emotion_lock:
            return self.current_emotion
    
    def run(self, shutdown_event):
        """Main face display loop"""
        logger.info("Starting face display")
        self.display_active = True
        
        # Initialize the display
        self._init_display()
        
        # Display loop
        while not shutdown_event.is_set():
            try:
                # Check for pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        shutdown_event.set()
                
                # Update the display based on current emotion
                self._update_display()
                
                # Check for responses to speak
                if not self.response_queue.empty():
                    response = self.response_queue.get()
                    self._animate_speaking(response)
                
                # Update the screen
                pygame.display.flip()
                
                # Animation frame rate
                time.sleep(0.05)  # 20 FPS
                
            except Exception as e:
                logger.error(f"Error in face display: {e}")
                time.sleep(1)  # Pause before retry
        
        # Clean up
        self._close_display()
        logger.info("Face display stopped")
    
    def _init_display(self):
        """Initialize the display system"""
        logger.info("Initializing display system")
        
        # Initialize pygame
        pygame.init()
        
        # Set up the display
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Assistant Face")
        
        # Hide the mouse cursor for full-screen applications
        pygame.mouse.set_visible(False)
        
        logger.info("Display system initialized")
    
    def _update_display(self):
        """Update the display based on current emotion"""
        # Get current emotion configuration
        emotion = self.get_current_emotion()
        emotion_config = self.emotions.get(emotion, self.emotions["neutral"])
        
        # Clear the screen with the emotion background color
        background_color = pygame.Color(emotion_config["color"])
        self.screen.fill(background_color)
        
        # Draw face components
        self._draw_face(emotion_config)
        
        # Simulate occasional blinking
        self._handle_blinking()
    
    def _draw_face(self, emotion_config):
        """Draw the face with the specified emotion"""
        # Draw eyes
        eye_type = "blinking" if self.blink_state else emotion_config["eyes"]
        self.face_assets["eyes"][eye_type]["draw"]()
        
        # Draw mouth
        mouth_type = "speaking" if self.speaking else emotion_config["mouth"]
        self.face_assets["mouths"][mouth_type]["draw"]()
    
    def _handle_blinking(self):
        """Handle blinking animation"""
        # Increment blink counter
        self.blink_counter += 1
        
        # Check if it's time to blink
        if not self.blink_state and random.random() < 0.02:  # 2% chance per frame
            self.blink_state = True
            self.blink_counter = 0
        
        # Blink for 5 frames
        if self.blink_state and self.blink_counter > 5:
            self.blink_state = False
    
    def _simulate_blink(self):
        """Simulate the face blinking"""
        self.blink_state = True
        self.blink_counter = 0
        logger.debug("Face blinked")

    
    def _animate_speaking(self, text):
        """Animate the face speaking the given text"""
        logger.info(f"Animating speaking: {text}")
        
        # Change to a speaking expression
        original_emotion = self.get_current_emotion()
        self.update_emotion("attentive")
        
        # Set speaking state
        self.speaking = True
        
        # Calculate speaking duration based on text length
        # Roughly 5 chars per second for speaking rate
        duration = len(text) / 5
        
        # Minimum duration
        duration = max(duration, 1.5)
        
        # Simulate speaking animation
        start_time = time.time()
        while time.time() - start_time < duration:
            # Update the mouth position for speaking animation
            self.mouth_position += 0.2 * self.mouth_direction
            if self.mouth_position > 1 or self.mouth_position < 0:
                self.mouth_direction *= -1
            
            # Update the screen
            self._update_display()
            pygame.display.flip()
            
            time.sleep(0.05)  # Animation frame rate
        
        # Reset speaking state
        self.speaking = False
        
        # Revert to original emotion
        self.update_emotion(original_emotion)
    
    def _close_display(self):
        """Close the display and clean up resources"""
        logger.info("Closing display")
        pygame.quit()
    
    # Eye drawing functions
    def _draw_normal_eyes(self):
        """Draw normal eyes"""
        left_eye_pos = (self.screen_width // 3, self.screen_height // 3)
        right_eye_pos = (2 * self.screen_width // 3, self.screen_height // 3)
        eye_radius = 40
        
        # Draw eye whites
        pygame.draw.circle(self.screen, pygame.Color("white"), left_eye_pos, eye_radius)
        pygame.draw.circle(self.screen, pygame.Color("white"), right_eye_pos, eye_radius)
        
        # Draw pupils
        pupil_radius = 15
        pygame.draw.circle(self.screen, pygame.Color("black"), left_eye_pos, pupil_radius)
        pygame.draw.circle(self.screen, pygame.Color("black"), right_eye_pos, pupil_radius)
    
    def _draw_wide_eyes(self):
        """Draw wide, surprised eyes"""
        left_eye_pos = (self.screen_width // 3, self.screen_height // 3)
        right_eye_pos = (2 * self.screen_width // 3, self.screen_height // 3)
        eye_radius = 50  # Larger for wide eyes
        
        # Draw eye whites
        pygame.draw.circle(self.screen, pygame.Color("white"), left_eye_pos, eye_radius)
        pygame.draw.circle(self.screen, pygame.Color("white"), right_eye_pos, eye_radius)
        
        # Draw pupils
        pupil_radius = 18
        pygame.draw.circle(self.screen, pygame.Color("black"), left_eye_pos, pupil_radius)
        pygame.draw.circle(self.screen, pygame.Color("black"), right_eye_pos, pupil_radius)
    
    def _draw_lowered_eyes(self):
        """Draw lowered, sad eyes"""
        left_eye_center = (self.screen_width // 3, self.screen_height // 3)
        right_eye_center = (2 * self.screen_width // 3, self.screen_height // 3)
        eye_width = 80
        eye_height = 30  # Flatter for sad eyes
        
        # Draw eye whites
        pygame.draw.ellipse(self.screen, pygame.Color("white"), 
                           [left_eye_center[0] - eye_width//2, 
                            left_eye_center[1] - eye_height//2, 
                            eye_width, eye_height])
        pygame.draw.ellipse(self.screen, pygame.Color("white"), 
                           [right_eye_center[0] - eye_width//2, 
                            right_eye_center[1] - eye_height//2, 
                            eye_width, eye_height])
        
        # Draw pupils (slightly lower)
        pupil_radius = 12
        pygame.draw.circle(self.screen, pygame.Color("black"), 
                          (left_eye_center[0], left_eye_center[1] + 5), pupil_radius)
        pygame.draw.circle(self.screen, pygame.Color("black"), 
                          (right_eye_center[0], right_eye_center[1] + 5), pupil_radius)
    
    def _draw_looking_up_eyes(self):
        """Draw eyes looking upward for thinking"""
        left_eye_center = (self.screen_width // 3, self.screen_height // 3)
        right_eye_center = (2 * self.screen_width // 3, self.screen_height // 3)
        eye_radius = 40
        
        # Draw eye whites
        pygame.draw.circle(self.screen, pygame.Color("white"), left_eye_center, eye_radius)
        pygame.draw.circle(self.screen, pygame.Color("white"), right_eye_center, eye_radius)
        
        # Draw pupils (higher position for looking up)
        pupil_radius = 15
        pygame.draw.circle(self.screen, pygame.Color("black"), 
                          (left_eye_center[0], left_eye_center[1] - 12), pupil_radius)
        pygame.draw.circle(self.screen, pygame.Color("black"), 
                          (right_eye_center[0], right_eye_center[1] - 12), pupil_radius)
    
    def _draw_squint_eyes(self):
        """Draw squinting eyes for confusion/scrutiny"""
        left_eye_center = (self.screen_width // 3, self.screen_height // 3)
        right_eye_center = (2 * self.screen_width // 3, self.screen_height // 3)
        eye_width = 80
        eye_height = 20  # Very flat for squinting
        
        # Draw eye whites
        pygame.draw.ellipse(self.screen, pygame.Color("white"), 
                           [left_eye_center[0] - eye_width//2, 
                            left_eye_center[1] - eye_height//2, 
                            eye_width, eye_height])
        pygame.draw.ellipse(self.screen, pygame.Color("white"), 
                           [right_eye_center[0] - eye_width//2, 
                            right_eye_center[1] - eye_height//2, 
                            eye_width, eye_height])
        
        # Draw pupils
        pupil_radius = 10
        pygame.draw.circle(self.screen, pygame.Color("black"), left_eye_center, pupil_radius)
        pygame.draw.circle(self.screen, pygame.Color("black"), right_eye_center, pupil_radius)
    
    def _draw_x_eyes(self):
        """Draw X eyes for error"""
        left_eye_center = (self.screen_width // 3, self.screen_height // 3)
        right_eye_center = (2 * self.screen_width // 3, self.screen_height // 3)
        eye_radius = 40
        
        # Draw eye whites
        pygame.draw.circle(self.screen, pygame.Color("white"), left_eye_center, eye_radius)
        pygame.draw.circle(self.screen, pygame.Color("white"), right_eye_center, eye_radius)
        
        # Draw X in each eye
        line_length = 25
        
        # Left eye X
        pygame.draw.line(self.screen, pygame.Color("red"), 
                        (left_eye_center[0] - line_length, left_eye_center[1] - line_length),
                        (left_eye_center[0] + line_length, left_eye_center[1] + line_length), 5)
        pygame.draw.line(self.screen, pygame.Color("red"), 
                        (left_eye_center[0] - line_length, left_eye_center[1] + line_length),
                        (left_eye_center[0] + line_length, left_eye_center[1] - line_length), 5)
        
        # Right eye X
        pygame.draw.line(self.screen, pygame.Color("red"), 
                        (right_eye_center[0] - line_length, right_eye_center[1] - line_length),
                        (right_eye_center[0] + line_length, right_eye_center[1] + line_length), 5)
        pygame.draw.line(self.screen, pygame.Color("red"), 
                        (right_eye_center[0] - line_length, right_eye_center[1] + line_length),
                        (right_eye_center[0] + line_length, right_eye_center[1] - line_length), 5)
    
    def _draw_scanning_eyes(self):
        """Draw scanning eyes for vigilant mode"""
        left_eye_center = (self.screen_width // 3, self.screen_height // 3)
        right_eye_center = (2 * self.screen_width // 3, self.screen_height // 3)
        eye_radius = 40
        
        # Scanning eye movement
        scan_offset = 15 * math.sin(time.time() * 3)
        
        # Draw eye whites
        pygame.draw.circle(self.screen, pygame.Color("white"), left_eye_center, eye_radius)
        pygame.draw.circle(self.screen, pygame.Color("white"), right_eye_center, eye_radius)
        
        # Draw pupils with scanning motion
        pupil_radius = 15
        pygame.draw.circle(self.screen, pygame.Color("black"), 
                          (left_eye_center[0] + scan_offset, left_eye_center[1]), pupil_radius)
        pygame.draw.circle(self.screen, pygame.Color("black"), 
                          (right_eye_center[0] + scan_offset, right_eye_center[1]), pupil_radius)
    
    def _draw_blinking_eyes(self):
        """Draw blinking eyes"""
        left_eye_center = (self.screen_width // 3, self.screen_height // 3)
        right_eye_center = (2 * self.screen_width // 3, self.screen_height // 3)
        eye_width = 80
        eye_height = 5  # Very flat for blinking
        
        # Draw thin lines for closed eyes
        pygame.draw.ellipse(self.screen, pygame.Color("white"), 
                           [left_eye_center[0] - eye_width//2, 
                            left_eye_center[1] - eye_height//2, 
                            eye_width, eye_height])
        pygame.draw.ellipse(self.screen, pygame.Color("white"), 
                           [right_eye_center[0] - eye_width//2, 
                            right_eye_center[1] - eye_height//2, 
                            eye_width, eye_height])
    
    # Mouth drawing functions
    def _draw_slight_smile(self):
        """Draw a slight smile"""
        center_x = self.screen_width // 2
        mouth_y = 2 * self.screen_height // 3
        width = 200
        height = 50
        
        # Draw a slight arc for the smile
        rect = pygame.Rect(center_x - width//2, mouth_y - height//2, width, height)
        pygame.draw.arc(self.screen, pygame.Color("white"), rect, 
                       math.pi, 2 * math.pi, 5)
    
    def _draw_big_smile(self):
        """Draw a big smile"""
        center_x = self.screen_width // 2
        mouth_y = 2 * self.screen_height // 3
        width = 250
        height = 100
        
        # Draw a wider and deeper arc for big smile
        rect = pygame.Rect(center_x - width//2, mouth_y - height//2, width, height)
        pygame.draw.arc(self.screen, pygame.Color("white"), rect, 
                       math.pi, 2 * math.pi, 8)
    
    def _draw_frown(self):
        """Draw a frown"""
        center_x = self.screen_width // 2
        mouth_y = 2 * self.screen_height // 3
        width = 200
        height = 50
        
        # Draw an upside-down arc for the frown
        rect = pygame.Rect(center_x - width//2, mouth_y, width, height)
        pygame.draw.arc(self.screen, pygame.Color("white"), rect, 
                       0, math.pi, 5)
    
    def _draw_straight_mouth(self):
        """Draw a straight mouth"""
        center_x = self.screen_width // 2
        mouth_y = 2 * self.screen_height // 3
        width = 150
        
        # Draw a straight line for the mouth
        pygame.draw.line(self.screen, pygame.Color("white"), 
                        (center_x - width//2, mouth_y),
                        (center_x + width//2, mouth_y), 5)
    
    def _draw_zigzag_mouth(self):
        """Draw a zigzag mouth for confusion"""
        center_x = self.screen_width // 2
        mouth_y = 2 * self.screen_height // 3
        width = 180
        zigzag_height = 15
        segments = 6
        
        # Draw zigzag mouth
        segment_width = width // segments
        points = []
        
        for i in range(segments + 1):
            x = center_x - width//2 + i * segment_width
            y = mouth_y + (zigzag_height if i % 2 == 0 else -zigzag_height)
            points.append((x, y))
        
        pygame.draw.lines(self.screen, pygame.Color("white"), False, points, 4)
    
    def _draw_speaking_mouth(self):
        """Draw a mouth that's speaking"""
        center_x = self.screen_width // 2
        mouth_y = 2 * self.screen_height // 3
        width = 150
        height = 80 * self.mouth_position  # Vary height based on speaking animation
        
        # Draw an ellipse that changes in height
        rect = pygame.Rect(center_x - width//2, mouth_y - height//2, width, height)
        pygame.draw.ellipse(self.screen, pygame.Color("white"), rect)