#!/usr/bin/env python3
import logging
import queue
import threading
import time
import json
import os
import numpy as np
from datetime import datetime
import pyaudio
from vosk import Model, KaldiRecognizer

logger = logging.getLogger(__name__)

class AudioManager:
    def __init__(self, command_queue, power_manager=None):
        logger.info("Initializing Audio Manager with Vosk")
        self.command_queue = command_queue
        self.power_manager = power_manager
        self.wake_words = ["assistant", "hey assistant", "computer"]
        
        # Audio recording settings
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 4000
        self.recording = False
        self.record_thread = None
        self.speech_thread = None
        
        # TTS and processing queues
        self.tts_queue = queue.Queue()
        self.audio_buffer = queue.Queue()
        
        # Load speech recognition model
        self.load_speech_models()
        
    def load_speech_models(self):
        """Load Vosk speech recognition model"""
        logger.info("Loading Vosk speech recognition model (vosk-model-small-en-us-0.15)")
        try:
            model_path = r"F:\desktop\minimo\modules\vosk-model-small-en-us-0.15"
            if not os.path.exists(model_path):
                logger.error(f"Vosk model not found at {model_path}")
                raise FileNotFoundError(f"Vosk model not found at {model_path}")
                
            self.model = Model(model_path)
            self.recognizer = KaldiRecognizer(self.model, self.RATE)
            
            # Configure recognizer for wake word detection
            self.recognizer.SetWords(True)
            logger.info("Vosk speech recognition model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Vosk model: {e}")
            raise
        
    def run(self, shutdown_event):
        """Main audio processing loop that runs continuously"""
        logger.info("Starting audio monitoring with Vosk")
        
        # Start recording microphone
        self.start_recording()
        
        # Start TTS processing thread
        self.start_tts_thread(shutdown_event)
        
        # Process audio continuously
        while not shutdown_event.is_set():
            try:
                if not self.audio_buffer.empty():
                    # Process audio data with Vosk
                    audio_data = self.audio_buffer.get()
                    self.process_audio(audio_data)
                else:
                    # Short sleep to prevent CPU hogging
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error in audio processing: {e}")
                time.sleep(1)  # Pause before retry
                
        # Clean up
        self.stop_recording()
        logger.info("Audio Manager stopped")
    
    def start_recording(self):
        """Start the microphone recording thread"""
        logger.info("Starting microphone recording")
        self.recording = True
        
        # Request microphone resource
        if self.power_manager:
            self.power_manager.request_resource('audio_input')
        
        # Start recording thread
        self.record_thread = threading.Thread(target=self._record_audio, daemon=True)
        self.record_thread.start()
    
    def stop_recording(self):
        """Stop the microphone recording"""
        logger.info("Stopping microphone recording")
        self.recording = False
        
        # Release microphone resource
        if self.power_manager:
            self.power_manager.release_resource('audio_input')
            
        # Wait for thread to end
        if self.record_thread:
            self.record_thread.join(timeout=1.0)
    
    def _record_audio(self):
        """Record audio from microphone and add to buffer"""
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=self.FORMAT,
                          channels=self.CHANNELS,
                          rate=self.RATE,
                          input=True,
                          frames_per_buffer=self.CHUNK)
                          
            logger.info("Microphone recording started")
            
            while self.recording:
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    self.audio_buffer.put(data)
                except IOError as e:
                    logger.warning(f"Audio overflow detected: {e}")
                    
            # Clean up
            stream.stop_stream()
            stream.close()
            p.terminate()
            logger.info("Microphone recording stopped")
            
        except Exception as e:
            logger.error(f"Error in recording thread: {e}")
            self.recording = False
    
    def process_audio(self, audio_data):
        """Process audio data with Vosk recognizer"""
        try:
            # Check if this chunk of audio contains speech
            if self.recognizer.AcceptWaveform(audio_data):
                result = json.loads(self.recognizer.Result())
                
                if 'text' in result and result['text']:
                    text = result['text'].lower()
                    logger.debug(f"Recognized: {text}")
                    
                    # Check for wake words
                    wake_word_detected = any(wake_word in text for wake_word in self.wake_words)
                    
                    if wake_word_detected:
                        logger.info(f"Wake word detected in: '{text}'")
                        self.command_queue.put({
                            "type": "wake_word", 
                            "text": text,
                            "time": datetime.now().isoformat()
                        })
                        
                        # Extract command (text after wake word)
                        command = None
                        for wake_word in self.wake_words:
                            if wake_word in text:
                                parts = text.split(wake_word, 1)
                                if len(parts) > 1 and parts[1].strip():
                                    command = parts[1].strip()
                                    break
                                    
                        if command:
                            logger.info(f"Command detected: '{command}'")
                            self.command_queue.put({
                                "type": "voice_command", 
                                "text": command,
                                "time": datetime.now().isoformat()
                            })
            else:
                # Handle partial results for better responsiveness
                partial = json.loads(self.recognizer.PartialResult())
                if 'partial' in partial and partial['partial']:
                    partial_text = partial['partial'].lower()
                    
                    # Check for wake words in partial results
                    for wake_word in self.wake_words:
                        if wake_word in partial_text:
                            logger.debug(f"Possible wake word in partial: '{partial_text}'")
                            # We don't send an event here, but could update UI to show listening state
                            break
                            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
    
    def speak(self, text):
        """Add text to the TTS queue"""
        logger.info(f"Adding to TTS queue: {text}")
        self.tts_queue.put(text)
        
        # Request audio output resource
        if self.power_manager:
            self.power_manager.request_resource('audio_output')
    
    def start_tts_thread(self, shutdown_event):
        """Start the TTS processing thread"""
        self.speech_thread = threading.Thread(
            target=self._process_tts_queue,
            args=(shutdown_event,),
            daemon=True
        )
        self.speech_thread.start()
    
    def _process_tts_queue(self, shutdown_event):
        """Process the TTS queue and speak items"""
        logger.info("TTS processing thread started")
        
        while not shutdown_event.is_set():
            try:
                # Get text from queue with timeout
                try:
                    text = self.tts_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                    
                # Perform text-to-speech
                self.text_to_speech(text)
                
                # Mark task as done
                self.tts_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in TTS thread: {e}")
                
        logger.info("TTS processing thread stopped")
    
    def text_to_speech(self, text):
        """Convert text to speech and play it using pyttsx3 (or similar)"""
        logger.info(f"TTS: {text}")
        
        # Request audio output resource if not already done
        if self.power_manager:
            self.power_manager.request_resource('audio_output')
            
        try:
            # In a real implementation, we would use a TTS library here
            # For example with pyttsx3:
            # import pyttsx3
            # engine = pyttsx3.init()
            # engine.say(text)
            # engine.runAndWait()
            
            # Simulate TTS processing time
            time.sleep(len(text) * 0.05)  # Rough estimate of speaking time
            
            # Release audio output resource after speaking
            if self.power_manager:
                self.power_manager.release_resource('audio_output')
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
            if self.power_manager:
                self.power_manager.release_resource('audio_output')