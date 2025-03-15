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
import pyttsx3  # Added import for TTS engine

logger = logging.getLogger(__name__)

class AudioManager:
    def __init__(self, command_queue, power_manager=None):
        logger.info("Initializing Audio Manager with Vosk")
        self.command_queue = command_queue
        self.power_manager = power_manager
        self.wake_words = ["assistant", "hey assistant", "computer"]
        
        # Audio recording settings - optimized for real-time responsiveness
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1600  # Reduced for lower latency (0.1s chunks)
        self.recording = False
        self.record_thread = None
        self.speech_thread = None
        
        # TTS and processing queues with priority
        self.tts_queue = queue.PriorityQueue()
        self.audio_buffer = queue.Queue(maxsize=100)  # Limit buffer size to prevent buildup
        
        # Wake word detection state
        self.detected_speech = ""
        self.last_speech_time = datetime.now()
        self.speech_timeout = 2.0  # seconds to wait after speech ends
        
        # Initialize TTS engine
        self.init_tts_engine()
        
        # Load speech recognition model
        self.load_speech_models()
        
        # Real-time processing
        self.process_thread = None
        self.processing = False
    
    def init_tts_engine(self):
        """Initialize the TTS engine for better real-time performance"""
        try:
            logger.info("Initializing TTS engine")
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS properties for optimal real-time performance
            self.tts_engine.setProperty('rate', 180)  # Slightly faster than default for responsiveness
            self.tts_engine.setProperty('volume', 0.9)  # 90% volume
            
            # Get available voices and select a good one if available
            voices = self.tts_engine.getProperty('voices')
            if len(voices) > 0:
                # Try to find a good quality voice
                for voice in voices:
                    if 'premium' in voice.name.lower() or 'high' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        logger.info(f"Selected premium voice: {voice.name}")
                        break
                else:
                    # Default to the first female voice if available (often better for assistants)
                    female_voices = [v for v in voices if v.gender == 'female']
                    if female_voices:
                        self.tts_engine.setProperty('voice', female_voices[0].id)
                        logger.info(f"Selected voice: {female_voices[0].name}")
                    else:
                        logger.info(f"Using default voice: {voices[0].name}")
            
            logger.info("TTS engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            # Create a fallback TTS engine with minimal settings
            self.tts_engine = pyttsx3.init()
    
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
        
        # Start real-time audio processing thread
        self.start_processing_thread(shutdown_event)
        
        # Monitor threads and restart if needed
        while not shutdown_event.is_set():
            # Check if recording thread is still alive
            if self.record_thread and not self.record_thread.is_alive():
                logger.warning("Recording thread died, restarting...")
                self.start_recording()
                
            # Check if processing thread is still alive
            if self.process_thread and not self.process_thread.is_alive():
                logger.warning("Processing thread died, restarting...")
                self.start_processing_thread(shutdown_event)
                
            # Check if TTS thread is still alive
            if self.speech_thread and not self.speech_thread.is_alive():
                logger.warning("TTS thread died, restarting...")
                self.start_tts_thread(shutdown_event)
                
            time.sleep(1)
                
        # Clean up
        self.stop_recording()
        self.processing = False
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
                    
                    # Only add to buffer if not full
                    if self.audio_buffer.full():
                        # Drop oldest chunk if buffer is full
                        try:
                            self.audio_buffer.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.audio_buffer.put_nowait(data)
                except IOError as e:
                    logger.warning(f"Audio overflow detected: {e}")
                    time.sleep(0.01)
                    
            # Clean up
            stream.stop_stream()
            stream.close()
            p.terminate()
            logger.info("Microphone recording stopped")
            
        except Exception as e:
            logger.error(f"Error in recording thread: {e}")
            self.recording = False
    
    def start_processing_thread(self, shutdown_event):
        """Start dedicated audio processing thread for real-time performance"""
        self.processing = True
        self.process_thread = threading.Thread(
            target=self._process_audio_stream,
            args=(shutdown_event,),
            daemon=True
        )
        self.process_thread.start()
        
    def _process_audio_stream(self, shutdown_event):
        """Dedicated thread for real-time audio processing"""
        logger.info("Audio processing thread started")
        
        # Keep track of streaming context
        continuous_silence_time = 0
        command_in_progress = False
        command_buffer = ""
        last_partial = ""
        
        while self.processing and not shutdown_event.is_set():
            try:
                if not self.audio_buffer.empty():
                    # Process audio data with Vosk
                    audio_data = self.audio_buffer.get(timeout=0.1)
                    
                    # Check if this chunk of audio contains speech
                    if self.recognizer.AcceptWaveform(audio_data):
                        result = json.loads(self.recognizer.Result())
                        
                        if 'text' in result and result['text']:
                            text = result['text'].lower()
                            logger.debug(f"Recognized: {text}")
                            continuous_silence_time = 0
                            
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
                                    # Start tracking for follow-up command
                                    command_in_progress = True
                                    command_buffer = ""
                            
                            # If we're awaiting a command after wake word
                            elif command_in_progress:
                                command_buffer += " " + text
                                # Check if it seems like a complete command
                                if len(text.split()) >= 3 or "?" in text:
                                    logger.info(f"Follow-up command detected: '{command_buffer.strip()}'")
                                    self.command_queue.put({
                                        "type": "voice_command", 
                                        "text": command_buffer.strip(),
                                        "time": datetime.now().isoformat()
                                    })
                                    command_in_progress = False
                            else:
                                # Non-wake-word speech detected
                                logger.debug(f"Non-wake-word speech detected: '{text}'")
                                # Always use dictionary format for consistency
                                self.command_queue.put({
                                    "type": "speech", 
                                    "text": text,
                                    "time": datetime.now().isoformat()
                                })
                        else:
                            continuous_silence_time += len(audio_data) / (self.RATE * 2)  # Approx time in seconds
                            
                            # If silence after wake word for more than 1.5 seconds, send whatever we have
                            if command_in_progress and continuous_silence_time > 1.5 and command_buffer.strip():
                                logger.info(f"Command after silence: '{command_buffer.strip()}'")
                                self.command_queue.put({
                                    "type": "voice_command", 
                                    "text": command_buffer.strip(),
                                    "time": datetime.now().isoformat()
                                })
                                command_in_progress = False
                    else:
                        # Process partial results for better responsiveness
                        partial = json.loads(self.recognizer.PartialResult())
                        if 'partial' in partial and partial['partial']:
                            partial_text = partial['partial'].lower()
                            
                            # Only log if partial changed significantly
                            if partial_text and len(partial_text) > len(last_partial) + 5:
                                logger.debug(f"Partial: '{partial_text}'")
                                last_partial = partial_text
                            
                            # Check for wake words in partial results for faster response
                            for wake_word in self.wake_words:
                                if wake_word in partial_text and not command_in_progress:
                                    logger.debug(f"Possible wake word in partial: '{partial_text}'")
                                    # We could update UI to show listening state
                                    # But we'll wait for full confirmation before sending event
                                    break
                                    
                            # Update command in progress
                            if command_in_progress:
                                command_buffer = partial_text
                                continuous_silence_time = 0
                else:
                    # Short sleep to prevent CPU hogging if buffer is empty
                    time.sleep(0.01)
                    
            except queue.Empty:
                # Normal timeout, just continue
                pass
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                time.sleep(0.1)  # Brief pause before retry
                
        logger.info("Audio processing thread stopped")
    
    def process_audio(self, audio_data):
        """Process audio data with Vosk recognizer (legacy method)"""
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
                        # Non-wake-word speech detected - use dictionary format always
                        logger.debug(f"Non-wake-word speech detected: '{text}'")
                        self.command_queue.put({
                            "type": "speech", 
                            "text": text,
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
    
    def speak(self, text, priority=1):
        """Add text to the TTS queue with priority (lower number = higher priority)"""
        logger.info(f"Adding to TTS queue (priority {priority}): {text}")
        self.tts_queue.put((priority, text))
        
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
                    priority, text = self.tts_queue.get(timeout=0.5)
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
        """Convert text to speech using pyttsx3 for real-time TTS"""
        logger.info(f"TTS: {text}")
        
        # Request audio output resource if not already done
        if self.power_manager:
            self.power_manager.request_resource('audio_output')
            
        try:
            # Split long text into sentences for more responsive playback
            sentences = text.replace('!', '.').replace('?', '?|').replace('.', '.|').split('|')
            
            # Remove empty sentences
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Speak each sentence immediately
            for sentence in sentences:
                # Use our pre-initialized TTS engine (faster than creating a new one each time)
                self.tts_engine.say(sentence)
                self.tts_engine.runAndWait()
                
                # Small pause between sentences for more natural speech
                if len(sentences) > 1:
                    time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
            # Try to recover with a new TTS engine instance
            try:
                logger.info("Attempting TTS recovery...")
                self.init_tts_engine()
                # Try one more time with the new engine
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e2:
                logger.error(f"TTS recovery failed: {e2}")
        finally:
            # Always release audio output resource
            if self.power_manager:
                self.power_manager.release_resource('audio_output')