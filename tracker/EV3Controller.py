import logging
import time

import yaml
from ev3_usb import EV3_USB

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
ev3_config = config['ev3_controller']

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ev3_config['log_file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EV3Controller:
    def __init__(self):
        self.deadzone_x = ev3_config['deadzone']['x']
        self.deadzone_y = ev3_config['deadzone']['y']
        self.speed_factor = ev3_config['speed_factor']
        self.max_speed = ev3_config['max_speed']
        self.invert_x = ev3_config['invert_x']
        self.invert_y = ev3_config['invert_y']
        self.command_cooldown = ev3_config['command_cooldown']
        self.horizontal_motor_port = ev3_config['motors']['horizontal']
        self.vertical_motor_port = ev3_config['motors']['vertical']
        self.led_connected_color = ev3_config['led']['connected']
        self.led_disconnected_color = ev3_config['led']['disconnected']
        self.cam_width = ev3_config['camera']['width']
        self.cam_height = ev3_config['camera']['height']
        self.cam_horizontal_fov = ev3_config['camera']['horizontal_fov']
        self.cam_vertical_fov = ev3_config['camera']['vertical_fov']

        self.ev3 = None
        self.motor_a = None  # Horizontal control
        self.motor_b = None  # Vertical control
        self.connected = False
        self.last_command_time = 0
        self.connect()

    def connect(self):
        try:
            logger.info(f"Connecting to EV3 via USB...")

            # Initialize EV3 connection
            self.ev3 = EV3_USB()

            # Try to initialize motors with error handling
            logger.debug(f"Initializing Motor A (Horizontal)...")
            self.motor_a = self.ev3.Motor(self.horizontal_motor_port) 
            logger.debug(f"Initializing Motor B (Vertical)...")
            self.motor_b = self.ev3.Motor(self.vertical_motor_port)

            self.connected = True
            logger.info("EV3 connected successfully!")
            logger.debug(f"Motor A (Port {self.horizontal_motor_port}): Horizontal control")
            logger.debug(f"Motor B (Port {self.vertical_motor_port}): Vertical control")

            try:
                self.ev3.Led(self.led_connected_color, 'pulse')
                logger.info(f"LED set to {self.led_connected_color}")
            except Exception as e:
                logger.warning(f"Could not set LED: {e}")

            self.stop_motors()

            return

        except Exception as e:
            logger.error(f"Failed to connect to EV3: {e}")
            logger.error("Please check:")
            logger.error("  1. EV3 is turned on")
            logger.error("  2. EV3 is connected via USB")
            logger.error("  3. Motors are connected to ports A and B")
            self.connected = False
            self.cleanup_failed_connection()

    def cleanup_failed_connection(self):
        try:
            if self.motor_a:
                self.motor_a = None
            if self.motor_b:
                self.motor_b = None
            if self.ev3:
                self.ev3 = None
        except:
            pass

    def calculate_motor_turn_degree_and_speed(self, shift, deadzone, invert=False, axis='x'):
        if abs(shift) < deadzone:
            return None, 0
        
        speed = max(abs(shift) / 100.0 * self.speed_factor, 5)
        speed = min(speed, self.max_speed)
        if invert:
            shift = -shift
        if axis == 'x':
            # OV5647 has 128 degrees horizontal
            degree_coeff = self.cam_width / self.cam_horizontal_fov
        else:
            # OV5647 has 96 degrees vertical
            degree_coeff = self.cam_height / self.cam_vertical_fov
        degree = shift / degree_coeff
        
        return int(degree), int(speed)

    def update_motors(self, shift_x, shift_y):
        if not self.connected or self.ev3 is None:
            return

        current_time = time.time()
        if current_time - self.last_command_time < self.command_cooldown:
            return

        try:
            degree_a, speed_a = self.calculate_motor_turn_degree_and_speed(
                shift_x, self.deadzone_x, self.invert_x, axis='x'
            )

            degree_b, speed_b = self.calculate_motor_turn_degree_and_speed(
                shift_y, self.deadzone_y, self.invert_y, axis='y'
            )

            if degree_a is not None and speed_a > 0:
                try:
                    self.motor_a.run_to(degrees=degree_a, speed=speed_a)
                    logger.debug(f"Motor A: degrees={degree_a}, speed={speed_a}, shift_x={shift_x}")
                except Exception as e:
                    logger.error(f"Motor A error: {e}")
            else:
                try:
                    self.motor_a.stop()
                except Exception as e:
                    logger.debug(f"Motor A stop error: {e}")

            if degree_b is not None and speed_b > 0:
                try:
                    self.motor_b.run_to(degrees=degree_b, speed=speed_b)
                    logger.debug(f"Motor B: degrees={degree_b}, speed={speed_b}, shift_y={shift_y}")
                except Exception as e:
                    logger.error(f"Motor B error: {e}")
            else:
                try:
                    self.motor_b.stop()
                except Exception as e:
                    logger.debug(f"Motor B stop error: {e}")

            self.last_command_time = current_time

        except Exception as e:
            logger.error(f"Error updating motors: {e}")

    def stop_motors(self):
        if not self.connected:
            return

        try:
            if self.motor_a:
                try:
                    self.motor_a.stop()
                except Exception as e:
                    logger.debug(f"Error stopping motor A: {e}")
            if self.motor_b:
                try:
                    self.motor_b.stop()
                except Exception as e:
                    logger.debug(f"Error stopping motor B: {e}")
        except Exception as e:
            logger.error(f"Error stopping motors: {e}")

    def disconnect(self):
        if self.connected:
            self.stop_motors()

            try:
                if self.ev3:
                    self.ev3.Led(self.led_disconnected_color, 'static')
                    logger.info(f"LED set to {self.led_disconnected_color}")
            except Exception as e:
                logger.debug(f"Could not set LED on disconnect: {e}")

            try:
                self.motor_a = None
                self.motor_b = None
                self.ev3 = None
            except:
                pass

            self.connected = False
            logger.info("EV3 disconnected")

