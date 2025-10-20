# scanner_controller.py
# Runs Arduino + camera loop in a background QThread with a binary 4-byte
# degree echo followed by a 'go' handshake to avoid stream corruption.

import os
import time
import cv2
import numpy as np
import serial
import serial.tools.list_ports
from PyQt5 import QtCore

class CaptureWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)
    error    = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, angle_deg: int, save_folder: str, parent=None,
                 wait_before_capture: float = 2.0,
                 post_capture_delay: float = 0.25,
                 angle_settle: float = 0.15):
        super().__init__(parent)
        self.angle_deg = int(angle_deg)
        self.save_folder = save_folder
        self._stop = False

        self.wait_before_capture = float(wait_before_capture)
        self.post_capture_delay  = float(post_capture_delay)
        self.angle_settle        = float(angle_settle)

        self.img_count   = 1
        self.ser         = None
        self.cap         = None
        self.camera_port = None
        self.buffer      = ""
        self.capture_mode = False

    # ---------- Serial helpers ----------
    def find_arduino_nano_port(self):
        for p in serial.tools.list_ports.comports():
            if ("CH340" in p.description
                or "wchusbserial" in p.device
                or "USB-SERIAL" in p.description
                or "Arduino" in p.description):
                self.progress.emit(f"Arduino detected on {p.device}")
                return p.device
        return None

    def connect_to_arduino(self, port, baudrate=9600, timeout=1):
        try:
            ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
            time.sleep(2)                # allow board reset
            ser.write(b'HELLO\n')        # handshake
            ser.flush()
            time.sleep(0.2)
            ser.reset_input_buffer()     # start clean
            return ser
        except serial.SerialException as e:
            self.error.emit(f"Failed to connect to {port}: {e}")
            return None

    def _write_line(self, s: str):
        if self.ser and self.ser.is_open:
            self.ser.write((s + "\n").encode("utf-8"))
            self.ser.flush()

    def _read_confirm_int(self, nbytes: int, overall_timeout_s: float = 5.0):
        """Blocking read of exactly nbytes (returns int or None)."""
        deadline = time.time() + overall_timeout_s
        buf = b""
        while len(buf) < nbytes and time.time() < deadline:
            chunk = self.ser.read(nbytes - len(buf))  # blocks up to ser.timeout
            if chunk:
                buf += chunk
            else:
                self.msleep(5)
        if len(buf) == nbytes:
            try:
                return int.from_bytes(buf, byteorder="little", signed=False)
            except Exception:
                return None
        return None

    # ---------- Camera helpers ----------
    def find_available_camera(self, max_ports=10):
        self.progress.emit("Checking available cameras...")
        for cam_index in range(1, max_ports):
            cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                self.progress.emit(f"Camera detected on port {cam_index} ({w}x{h})")
                return cam_index
            cap.release()
        return None

    def open_camera(self, camera_index):
        if camera_index is None:
            return None
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return None
        # set your preferred res; adapt as needed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.progress.emit(f"Camera opened: {w}x{h}")
        return cap

    def capture_once(self, img_path):
        try:
            if self.cap is not None and self.cap.isOpened():
                ok, frame = self.cap.read()
                if ok and frame is not None:
                    return cv2.imwrite(img_path, frame)
            # fallback dummy image
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, os.path.basename(img_path), (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            return cv2.imwrite(img_path, img)
        except Exception:
            return False

    # ---------- QThread ----------
    def stop(self):
        self._stop = True

    def run(self):
        try:
            os.makedirs(self.save_folder, exist_ok=True)
            self.img_count = 1
            self.capture_mode = False
            self.progress.emit(f"Saving images to: {self.save_folder}")

            # Arduino
            port = self.find_arduino_nano_port()
            if port is None:
                self.error.emit("Arduino Nano not found.")
                return
            self.ser = self.connect_to_arduino(port)
            if self.ser is None:
                return

            # Camera
            self.camera_port = self.find_available_camera()
            if self.camera_port is None:
                self.progress.emit("No camera found. Using test images.")
            else:
                self.cap = self.open_camera(self.camera_port)
                if self.cap is None:
                    self.progress.emit("Failed to open camera. Using test images.")
                else:
                    self.progress.emit("Camera ready.")

            # --- Start cycle ---
            self.progress.emit("Initializing capture cycle...")

            # 1) Send "on", then angle
            self._write_line("on")
            time.sleep(self.angle_settle)
            self._write_line(str(self.angle_deg))

            # 2) Read 4-byte confirm (blocking)
            deg_conf = self._read_confirm_int(4)
            if deg_conf is None:
                self.progress.emit("No angle confirmation. Continuing anyway...")
            else:
                self.progress.emit(f"Arduino confirmed degrees: {deg_conf}")

            # 3) Clear any leftover bytes (binary echo is done), then send 'go'
            self.ser.reset_input_buffer()
            self._write_line("go")

            # 4) Now only line traffic: 'capture' ... 'picture captured' ... 'exit'
            self.buffer = ""
            while not self._stop:
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                    self.buffer += data
                    while '\n' in self.buffer:
                        line, self.buffer = self.buffer.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        self._process_line(line)
                        if line.lower() == "exit":
                            # Log count including 0°
                            expected = max(1, 360 // max(1, self.angle_deg))
                            if deg_conf == self.angle_deg:
                                self.progress.emit(
                                    f"Arduino signaled exit. Saved {self.img_count-1} images (expected ~{expected})."
                                )
                            else:
                                self.progress.emit("Arduino signaled exit.")
                            return
                self.msleep(5)

        except Exception as e:
            self.error.emit(f"Unexpected error: {e}")
        finally:
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

    def _process_line(self, line: str):
        self.progress.emit(f"Received: {line}")
        if line.lower() == "capture":
            if not self.capture_mode:
                self.capture_mode = True
                self.progress.emit("Starting capture sequence...")

            img_name = f"image_{self.img_count:03d}.jpg"
            img_path = os.path.join(self.save_folder, img_name)

            if self.wait_before_capture > 0:
                time.sleep(self.wait_before_capture)

            if self.capture_once(img_path):
                self.progress.emit(f"Saved: {img_name}")
                if self.post_capture_delay > 0:
                    time.sleep(self.post_capture_delay)
                # ACK back to Arduino so it can proceed
                self._write_line("picture captured")
                self.img_count += 1
            else:
                self.error.emit("Camera capture failed.")
