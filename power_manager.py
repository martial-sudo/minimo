import threading
import time
import logging
import psutil
try:
    import RPi.GPIO as GPIO
except ImportError:
    # For development on non-Raspberry Pi systems
    print("Warning: RPi.GPIO not available. Using mock implementation.")
    class GPIO:
        @staticmethod
        def setmode(*args): pass
        @staticmethod
        def setup(*args, **kwargs): pass
        @staticmethod
        def output(*args): pass
        @staticmethod
        def cleanup(*args): pass
        BCM = 'BCM'
        OUT = 'OUT'
        HIGH = 1
        LOW = 0

class PowerManager:
    def __init__(self, battery_pin=18, power_led_pin=17):
        """Initialize the power management system."""
        self.logger = logging.getLogger('power_manager')
        self.active_resources = {
            'camera': False,
            'audio_input': True,  # Always on for wake word detection
            'audio_output': False,
            'display': True,  # Always on for showing the face
            'processing': True   # Always needs some processing power
        }
        self.resource_locks = {resource: threading.Lock() for resource in self.active_resources}
        self.power_mode = "normal"  # normal, eco, ultra_eco
        self.battery_level = 100
        self.battery_pin = battery_pin
        self.power_led_pin = power_led_pin
        self.thread = None
        self.running = False
        
        # Set up GPIO for battery monitoring and power LED
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.power_led_pin, GPIO.OUT)
        GPIO.output(self.power_led_pin, GPIO.HIGH)  # Power LED on
        
    def start(self):
        """Start the power management thread."""
        if self.thread is None or not self.thread.is_alive():
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            self.logger.info("Power manager started")
            
    def stop(self):
        """Stop the power management thread."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
            self.logger.info("Power manager stopped")
        
    def _run(self):
        """Main loop for power management."""
        check_interval = 30  # seconds
        last_check = 0
        
        while self.running:
            current_time = time.time()
            
            # Periodically check battery and adjust power mode
            if current_time - last_check >= check_interval:
                self._check_battery_level()
                self._update_power_mode()
                last_check = current_time
                
            # Monitor CPU and memory usage
            self._monitor_system_resources()
            
            # Sleep to reduce CPU usage
            time.sleep(5)
            
    def _check_battery_level(self):
        """Check the current battery level."""
        # In a real implementation, this would read from an ADC connected to the battery
        # For simulation, we'll decrease by 1% every check
        self.battery_level = max(0, self.battery_level - 1)
        self.logger.debug(f"Battery level: {self.battery_level}%")
        
        # Update power LED brightness based on battery level
        if self.battery_level < 20:
            # Blink LED for low battery
            GPIO.output(self.power_led_pin, GPIO.LOW)
            time.sleep(0.2)
            GPIO.output(self.power_led_pin, GPIO.HIGH)
        
    def _update_power_mode(self):
        """Update the power mode based on battery level."""
        if self.battery_level < 15:
            self.set_power_mode("ultra_eco")
        elif self.battery_level < 30:
            self.set_power_mode("eco")
        else:
            self.set_power_mode("normal")
            
    def set_power_mode(self, mode):
        """Set the power conservation mode."""
        if mode != self.power_mode:
            self.power_mode = mode
            self.logger.info(f"Power mode changed to: {mode}")
            
            # Apply power-saving measures based on mode
            if mode == "eco":
                # Reduce display brightness, CPU frequency, etc.
                pass
            elif mode == "ultra_eco":
                # Minimal power usage, essential functions only
                pass
            else:  # normal
                # Standard operation
                pass
                
    def request_resource(self, resource):
        """Request access to a power-consuming resource."""
        if resource not in self.active_resources:
            self.logger.error(f"Unknown resource requested: {resource}")
            return False
            
        with self.resource_locks[resource]:
            # Check if we can activate this resource in current power mode
            if self.power_mode == "ultra_eco" and resource not in ['audio_input', 'processing']:
                self.logger.warning(f"Resource {resource} denied due to ultra_eco mode")
                return False
                
            self.active_resources[resource] = True
            self.logger.debug(f"Resource activated: {resource}")
            return True
            
    def release_resource(self, resource):
        """Release a power-consuming resource."""
        if resource not in self.active_resources:
            self.logger.error(f"Unknown resource released: {resource}")
            return
            
        with self.resource_locks[resource]:
            self.active_resources[resource] = False
            self.logger.debug(f"Resource deactivated: {resource}")
    
    def disable_non_essential_services(self):
        """Disable all non-essential services to save power during security mode."""
        self.logger.info("Disabling non-essential services for security mode")
        
        # Keep only camera and minimal processing active
        with self.resource_locks['audio_output']:
            self.active_resources['audio_output'] = False
        
        # We can keep audio_input for wake word detection
        # We need to keep the camera on for security
        with self.resource_locks['camera']:
            self.active_resources['camera'] = True
        
        # Keep display on but maybe at lower brightness
        # In a real implementation, you might reduce display brightness
        
        # Put system into eco mode to save power
        self.set_power_mode("eco")
        
    def enable_all_services(self):
        """Re-enable all services when exiting security mode."""
        self.logger.info("Re-enabling all services after security mode")
        
        # Reset power mode to normal
        self.set_power_mode("normal")
        
        # All services can be activated as needed
        # We don't automatically turn everything on, but make them available
        # when requested through request_resource()
            
    def _monitor_system_resources(self):
        """Monitor system resources like CPU and memory usage."""
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        if cpu_percent > 80 or memory_percent > 80:
            self.logger.warning(f"High resource usage: CPU {cpu_percent}%, Memory {memory_percent}%")
            
        # Log detailed resource information at debug level
        self.logger.debug(f"System resources - CPU: {cpu_percent}%, Memory: {memory_percent}%")
        
    def get_battery_level(self):
        """Get the current battery level percentage."""
        return self.battery_level
        
    def get_power_status(self):
        """Get the current power status."""
        return {
            'battery_level': self.battery_level,
            'power_mode': self.power_mode,
            'active_resources': self.active_resources.copy()
        }
        
    def cleanup(self):
        """Clean up GPIO and other resources."""
        self.stop()
        GPIO.cleanup([self.power_led_pin, self.battery_pin])