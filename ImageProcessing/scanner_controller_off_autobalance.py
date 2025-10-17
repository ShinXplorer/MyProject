# scanner_controller.py
# Background worker that runs the Arduino + Camera loop without blocking the UI.

import os
import time
import cv2
import numpy as np
import serial
import serial.tools.list_ports
from PyQt5 import QtCore


class CaptureWorker(QtCore.QThread):
    # Qt signals to communicate with the UI
    progress = QtCore.pyqtSignal(str)   # emits logs or status updates
    error = QtCore.pyqtSignal(str)      # emits error messages
    finished = QtCore.pyqtSignal()      # emitted when the process ends (clean exit or error)

    def __init__(self, angle_deg: int, save_folder: str, parent=None,
                 wait_before_capture: float = 5.0,
                 post_capture_delay: float = 2.0,
                 angle_settle: float = 1.0):
        super().__init__(parent)
        self.angle_deg = int(angle_deg)
        self.save_folder = save_folder
        self._stop = False

        # timing tunables (seconds) - adjustable from the UI/controller
        # wait_before_capture: sleep after Arduino 'capture' signal and before grabbing the frame
        # post_capture_delay: small delay after saving image and before sending ack back to Arduino
        # angle_settle: time to wait after sending angle command before attempting to read confirmation
        self.wait_before_capture = float(wait_before_capture)
        self.post_capture_delay = float(post_capture_delay)
        self.angle_settle = float(angle_settle)

        # runtime handles
        self.ser = None
        self.cap = None
        self.camera_port = None
        self.buffer = ""              # Buffer for incomplete messages
        self.capture_mode = False     # Track if we're in capture mode

    def stop(self):
        self._stop = True

    def find_arduino_nano_port(self):
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if ("CH340" in port.description
                or "wchusbserial" in port.device
                or "USB-SERIAL" in port.description
                or "Arduino" in port.description):
                self.progress.emit(f"Arduino detected on {port.device}")
                return port.device
        return None

    def connect_to_arduino(self, port, baudrate=9600, timeout=1):
        try:
            ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
            time.sleep(2)  # Allow Arduino reset
            # Send HELLO to establish connection
            ser.write(b'HELLO\n')
            ser.flush()
            # Wait for response (Arduino will set pythonConnected to true)
            time.sleep(1)
            # Clear any leftover data in the buffer
            ser.reset_input_buffer()
            return ser
        except serial.SerialException as e:
            self.error.emit(f"Failed to connect to {port}: {e}")
            return None

    def find_available_camera(self, max_ports=10):
        """
        Simplified camera detection - looks for single USB camera starting from index 1
        Same as your peer's code
        """
        self.progress.emit("Checking available cameras...")
        for cam_index in range(1, max_ports):  # Start from index 1 to skip built-in camera
            cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                # Check camera resolution to confirm it's the external camera
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                self.progress.emit(f"Camera detected on port {cam_index} with resolution {width}x{height}")
                return cam_index
            cap.release()
        return None

    def open_camera(self, camera_index):
        """
        Open single USB camera with stereo settings 
        Same as your peer's code
        """
        if camera_index is None:
            return None
            
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return None
            
        # Set to stereo camera resolution (side-by-side)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1520)
        
        # Verify the settings
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.progress.emit(f"Camera opened with resolution: {actual_width}x{actual_height}")
        
        # Try to disable auto white balance for consistent colors between left/right
        try:
            self._disable_auto_white_balance(cap)
        except Exception as e:
            self.progress.emit(f"Warning: unable to disable auto white balance: {e}")
        
        return cap

    def _disable_auto_white_balance(self, cap):
        """Attempt to disable camera auto-white-balance using OpenCV properties.
        This is best-effort: not all camera drivers/backends support these controls.
        We try common properties and verify by reading them back.
        """
        tried = []
        succeeded = False

        # Common OpenCV property for auto white balance (may or may not exist)
        if hasattr(cv2, 'CAP_PROP_AUTO_WB'):
            tried.append('CAP_PROP_AUTO_WB')
            try:
                ok = cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                val = cap.get(cv2.CAP_PROP_AUTO_WB)
                if ok or (val == 0):
                    self.progress.emit('Auto white balance disabled via CAP_PROP_AUTO_WB')
                    succeeded = True
                else:
                    self.progress.emit('CAP_PROP_AUTO_WB set attempted but not confirmed')
            except Exception:
                pass

        # Some backends (especially on Windows) might expose white balance via these props
        for prop_name in ('CAP_PROP_WHITE_BALANCE_BLUE_U', 'CAP_PROP_WHITE_BALANCE_RED_V'):
            if hasattr(cv2, prop_name):
                tried.append(prop_name)
                try:
                    # read current value and re-write it to force manual mode on some drivers
                    cur = cap.get(getattr(cv2, prop_name))
                    # attempt to write the same value (some drivers switch to manual when user writes value)
                    cap.set(getattr(cv2, prop_name), cur if cur != -1 else 4000)
                    self.progress.emit(f'Tried toggling {prop_name} (value {cur})')
                    succeeded = True
                except Exception:
                    pass

        # If nothing worked, inform the user
        if not succeeded:
            self.progress.emit('Could not programmatically disable AWB; driver may not support it via OpenCV.')
            self.progress.emit(f'Tried properties: {tried}')
        return succeeded

    def capture_once(self, img_path):
        """
        Capture from single USB camera (side-by-side stereo)
        Same as your peer's code
        """
        try:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return cv2.imwrite(img_path, frame)

            # Fallback: dummy image
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, f"Test Image {os.path.basename(img_path)}",
                        (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return cv2.imwrite(img_path, img)
        except Exception:
            return False

    def run(self):
        try:
            # Folder prep
            os.makedirs(self.save_folder, exist_ok=True)
            self.progress.emit(f"Saving images to: {self.save_folder}")

            # 1) Connect Arduino
            self.progress.emit("Detecting Arduino Nano...")
            nano_port = self.find_arduino_nano_port()
            if nano_port is None:
                self.error.emit("Arduino Nano not found.")
                return
            self.ser = self.connect_to_arduino(nano_port)
            if self.ser is None:
                return

            # 2) Try to Connect Camera (single USB camera)
            self.progress.emit("Detecting stereo camera...")
            self.camera_port = self.find_available_camera()
            if self.camera_port is None:
                self.progress.emit("No camera detected. Using test mode.")
            else:
                self.cap = self.open_camera(self.camera_port)
                if self.cap is None:
                    self.progress.emit("Failed to open stereo camera. Using test mode.")
                else:
                    self.progress.emit("Camera ready.")

            # 3) Start rotation-capture cycle
            self.progress.emit("Initializing capture cycle...")

            # Clear any leftover data
            self.ser.reset_input_buffer()
            self.buffer = ""

            # Send "on" command
            self._write_line("on")
            time.sleep(self.angle_settle)  # Give Arduino time to process

            # Send the angle value
            self._write_line(str(self.angle_deg))

            # Wait a short settle period (configurable) before attempting to read the degrees confirmation
            time.sleep(self.angle_settle)
            # Wait for Arduino to send the degrees confirmation (4 bytes)
            degrees_confirm = self._read_confirm_int(4)
            if degrees_confirm is None:
                self.progress.emit("Did not receive angle confirmation. Continuing anyway...")
            else:
                self.progress.emit(f"Arduino confirmed degrees: {degrees_confirm}")

            # Now proceed with the capture loop
            img_count = 1

            while not self._stop:
                if self.ser.in_waiting > 0:
                    # Read all available data
                    data = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                    self.buffer += data

                    # Process complete lines from the buffer
                    while '\n' in self.buffer:
                        line, self.buffer = self.buffer.split('\n', 1)
                        line = line.strip()
                        if line:
                            self._process_line(line, img_count)
                            if "capture" in line.lower() and "exit" not in line.lower():
                                img_count += 1

                            if "exit" in line.lower():
                                self.progress.emit("Arduino signaled exit.")
                                return

                self.msleep(5)  # Yield to Qt event loop

        except Exception as e:
            self.error.emit(f"Unexpected error: {e}")
        finally:
            # Cleanup
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:
                pass
            try:
                if self.ser is not None and self.ser.is_open:
                    self.ser.close()
            except Exception:
                pass
            self.finished.emit()

    def _process_line(self, line, img_count):
        self.progress.emit(f"Received: {line}")

        if "capture" in line.lower() and "exit" not in line.lower():
            if not self.capture_mode:
                self.progress.emit("Starting capture sequence...")
                self.capture_mode = True

            img_name = f"image_{img_count:03d}.jpg"
            img_path = os.path.join(self.save_folder, img_name)
            # allow the platform / motor to settle before grabbing the frame
            if self.wait_before_capture > 0:
                time.sleep(self.wait_before_capture)

            if self.capture_once(img_path):
                self.progress.emit(f"Saved: {img_name}")
                # slight pause to ensure filesystem flush and avoid bus congestion
                if self.post_capture_delay > 0:
                    time.sleep(self.post_capture_delay)
                self._write_line("picture captured")
            else:
                self.error.emit("Camera capture failed.")

    def _write_line(self, s: str):
        if self.ser and self.ser.is_open:
            self.ser.write((s + "\n").encode("utf-8"))
            self.ser.flush()  # Ensure data is sent immediately

    def _read_confirm_int(self, nbytes: int):
        """Read 4-byte confirmation from Arduino (same as peer's code)"""
        start_time = time.time()
        buf = b""
        while len(buf) < nbytes and (time.time() - start_time) < 5.0:
            if self.ser.in_waiting > 0:
                buf += self.ser.read(1)
            self.msleep(10)
            
        if len(buf) == nbytes:
            try:
                value = int.from_bytes(buf, byteorder="little", signed=False)
                return value
            except:
                return None
        return None
