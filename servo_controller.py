"""
Servo Controller for Panorama Camera
Controls servo to rotate 180 degrees for panorama capture on Raspberry Pi
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import RPi.GPIO (only works on Raspberry Pi)
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logger.warning("RPi.GPIO not available - running in simulation mode")


class ServoController:
    """Controls servo motor for panorama rotation"""
    
    def __init__(self, gpio_pin: int = 18, min_angle: float = 0, max_angle: float = 180):
        """
        Initialize servo controller.
        
        Args:
            gpio_pin: GPIO pin number (BCM mode). Default 18 (PWM capable)
            min_angle: Minimum angle (degrees)
            max_angle: Maximum angle (degrees)
        """
        self.gpio_pin = gpio_pin
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.current_angle = min_angle
        self.pwm: Optional[object] = None
        self.initialized = False
        
        # Servo timing (typical values for SG90/MG996R servos)
        self.min_duty = 2.5    # Duty cycle for 0 degrees
        self.max_duty = 12.5   # Duty cycle for 180 degrees
        self.frequency = 50    # 50 Hz (20ms period)
    
    def initialize(self) -> bool:
        """Initialize GPIO and PWM for servo control"""
        if not GPIO_AVAILABLE:
            logger.info("Servo simulation mode (no GPIO)")
            self.initialized = True
            return True
        
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.gpio_pin, GPIO.OUT)
            self.pwm = GPIO.PWM(self.gpio_pin, self.frequency)
            self.pwm.start(0)
            self.initialized = True
            logger.info(f"Servo initialized on GPIO {self.gpio_pin}")
            return True
        except Exception as e:
            logger.error(f"Servo init error: {e}")
            return False
    
    def _angle_to_duty(self, angle: float) -> float:
        """Convert angle (0-180) to duty cycle"""
        # Linear interpolation
        duty = self.min_duty + (angle / 180.0) * (self.max_duty - self.min_duty)
        return duty
    
    def set_angle(self, angle: float, smooth: bool = True, step_delay: float = 0.02):
        """
        Set servo to specific angle.
        
        Args:
            angle: Target angle (0-180 degrees)
            smooth: If True, move gradually. If False, move immediately.
            step_delay: Delay between steps for smooth movement (seconds)
        """
        angle = max(self.min_angle, min(self.max_angle, angle))
        
        if not GPIO_AVAILABLE:
            if smooth:
                # Simulate smooth movement time
                steps = abs(angle - self.current_angle)
                time.sleep(steps * step_delay)
            self.current_angle = angle
            logger.debug(f"[SIM] Servo at {angle}°")
            return
        
        if not self.initialized or not self.pwm:
            logger.error("Servo not initialized")
            return
        
        if smooth:
            # Move gradually
            step = 1 if angle > self.current_angle else -1
            current = int(self.current_angle)
            target = int(angle)
            
            for a in range(current, target + step, step):
                duty = self._angle_to_duty(a)
                self.pwm.ChangeDutyCycle(duty)
                time.sleep(step_delay)
        else:
            # Move immediately
            duty = self._angle_to_duty(angle)
            self.pwm.ChangeDutyCycle(duty)
            time.sleep(0.3)  # Wait for servo to reach position
        
        self.pwm.ChangeDutyCycle(0)  # Stop PWM to prevent jitter
        self.current_angle = angle
        logger.debug(f"Servo at {angle}°")
    
    def rotate_for_panorama(self, num_positions: int = 8, callback=None):
        """
        Rotate servo for panorama capture, stopping at each position.
        
        Args:
            num_positions: Number of capture positions
            callback: Function to call at each position (receives angle)
        
        Returns:
            List of angles where captures were made
        """
        angles = []
        step = (self.max_angle - self.min_angle) / (num_positions - 1)
        
        logger.info(f"Starting panorama rotation: {num_positions} positions")
        
        # Start at 0 degrees
        self.set_angle(self.min_angle, smooth=True)
        time.sleep(0.5)
        
        for i in range(num_positions):
            angle = self.min_angle + (i * step)
            self.set_angle(angle, smooth=True)
            time.sleep(0.3)  # Stabilization time
            
            angles.append(angle)
            logger.info(f"Position {i+1}/{num_positions}: {angle:.1f}°")
            
            if callback:
                callback(angle)
        
        logger.info("Panorama rotation complete")
        return angles
    
    def reset(self):
        """Reset servo to 0 degrees"""
        self.set_angle(0, smooth=True)
    
    def shutdown(self):
        """Clean up GPIO resources"""
        if GPIO_AVAILABLE and self.pwm:
            self.pwm.stop()
            GPIO.cleanup(self.gpio_pin)
        self.initialized = False
        logger.info("Servo shutdown")


def test_servo():
    """Test servo movement"""
    logging.basicConfig(level=logging.INFO)
    
    servo = ServoController(gpio_pin=18)
    
    if servo.initialize():
        print("\n🔄 Testing servo 0° → 180° → 0°")
        
        servo.set_angle(0)
        print("At 0°")
        time.sleep(1)
        
        servo.set_angle(90)
        print("At 90°")
        time.sleep(1)
        
        servo.set_angle(180)
        print("At 180°")
        time.sleep(1)
        
        servo.set_angle(0)
        print("Back to 0°")
        
        servo.shutdown()
        print("✅ Test complete!")


if __name__ == "__main__":
    test_servo()
