#!/usr/bin/env python3
"""
7-in-1 NPK Sensor reader (Modbus RTU) via USB serial on Raspberry Pi
Author: Rohit Kadam, www.robosap.in
Notes:
- Default serial port: /dev/ttyUSB0 (change below if needed)
- Baud: 4800 
"""

import time
import serial

SERIAL_PORT = "/dev/ttyUSB0"  # e.g., /dev/ttyUSB0, /dev/ttyUSB1, /dev/ttyAMA0
BAUD = 4800
TIMEOUT = 0.3  # seconds
POLL_MS = 5000

SLAVE_ADDR = 0x01
START_REG = 0x0000
REG_COUNT = 0x0007  # 7 registers → 14 data bytes

# Scaling (adjust if your probe differs)
MOIST_SCALE = 0.1
TEMP_SCALE = 0.1
PH_SCALE = 0.1
EC_SCALE = 1.0  # some probes use 0.1 or 10.0

RESP_LEN = 19  # [addr][func][byteCount=14][14 data][CRCLo][CRCHi]


def crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def build_request(slave: int, start_reg: int, reg_count: int) -> bytes:
    frame = bytearray(8)
    frame[0] = slave
    frame[1] = 0x03  # Read Holding Registers
    frame[2] = (start_reg >> 8) & 0xFF
    frame[3] = start_reg & 0xFF
    frame[4] = (reg_count >> 8) & 0xFF
    frame[5] = reg_count & 0xFF
    crc = crc16_modbus(frame[:6])
    frame[6] = crc & 0xFF  # CRC Lo
    frame[7] = (crc >> 8) & 0xFF  # CRC Hi
    return bytes(frame)


def read_exact(ser: serial.Serial, n: int, timeout_s: float) -> bytes:
    """Read exactly n bytes or return b'' on timeout."""
    deadline = time.time() + timeout_s
    out = bytearray()
    while len(out) < n and time.time() < deadline:
        chunk = ser.read(n - len(out))
        if chunk:
            out.extend(chunk)
        else:
            time.sleep(0.001)
    return bytes(out) if len(out) == n else b""


def parse_payload(resp: bytes):
    """Parse 19-byte Modbus response into sensor values."""
    if len(resp) != RESP_LEN:
        raise ValueError("Bad length")

    if resp[2] != 14:
        raise ValueError("Bad byteCount")

    # CRC check
    rx_crc = (resp[18] << 8) | resp[17]
    calc_crc = crc16_modbus(resp[:-2])
    if rx_crc != calc_crc:
        raise ValueError("CRC mismatch")

    # Header checks
    if resp[0] != SLAVE_ADDR or resp[1] != 0x03:
        raise ValueError("Bad addr/func")

    # Extract 7 registers from data bytes 3..16
    d = resp[3:17]
    regs = []
    for i in range(0, 14, 2):
        regs.append((d[i] << 8) | d[i+1])

    return {
        "moisture_pct": regs[0] * MOIST_SCALE,
        "temperature_c": regs[1] * TEMP_SCALE,
        "ec_uScm": regs[2] * EC_SCALE,
        "ph": regs[3] * PH_SCALE,
        "nitrogen_mgkg": regs[4],
        "phosphorus_mgkg": regs[5],
        "potassium_mgkg": regs[6],
    }


def read_once(ser: serial.Serial, retries: int = 3):
    req = build_request(SLAVE_ADDR, START_REG, REG_COUNT)

    for attempt in range(1, retries + 1):
        ser.reset_input_buffer()
        ser.write(req)
        resp = read_exact(ser, RESP_LEN, TIMEOUT)
        if not resp:
            if attempt == retries:
                raise TimeoutError("No/short response from sensor")
            continue

        try:
            return parse_payload(resp)
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(0.05)
    raise RuntimeError("Unexpected read error")


def main():
    print("Soil NPK Sensor - USB Serial Reader (Modbus RTU)")
    print(f"Port={SERIAL_PORT}, {BAUD} 8N1, slave=0x{SLAVE_ADDR:02X}")
    
    while True:
        try:
            with serial.Serial(SERIAL_PORT, BAUD, timeout=0, bytesize=8, 
                             parity=serial.PARITY_NONE, stopbits=1) as ser:
                while True:
                    try:
                        reading = read_once(ser, retries=3)
                        print("\n---- Soil Sensor Readings ----")
                        print(f"Moisture:     {reading['moisture_pct']:.1f} %")
                        print(f"Temperature:  {reading['temperature_c']:.1f} °C")
                        print(f"Conductivity: {reading['ec_uScm']:.1f} uS/cm")
                        print(f"pH:           {reading['ph']:.1f}")
                        print(f"N: {reading['nitrogen_mgkg']} P: {reading['phosphorus_mgkg']} K: {reading['potassium_mgkg']} (mg/kg)")
                        print("--------------------------------")
                    except Exception as e:
                        print(f"[WARN] Read failed: {e}")
                    time.sleep(POLL_MS / 1000.0)
        except serial.SerialException as e:
            print(f"[ERROR] Serial open failed on {SERIAL_PORT}: {e}")
            print("Retrying in 3s...")
            time.sleep(3)


if __name__ == "__main__":
    main()


# =============================================================================
# PREVIOUS VERSION (COMMENTED OUT)
# =============================================================================
"""
# Previous NPK Sensor implementation by Claude
# Supports command-line arguments and more features
# Uncomment this section if you need the advanced version

'''
NPK Sensor Reader for Raspberry Pi 4
Reads Nitrogen, Phosphorus, Potassium (and more) from 7-in-1 Soil Sensor
Using Modbus RTU protocol over RS485

Hardware Setup:
- NPK Sensor (RS485) → USB-RS485 Adapter → RPi USB Port
- Or: NPK Sensor → RS485-to-TTL Module → RPi GPIO UART

Install: pip install pyserial
'''

# import serial
# import struct
# import time
# import logging
# from typing import Dict, Optional, Tuple
# from dataclasses import dataclass
# 
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# 
# 
# @dataclass
# class SoilData:
#     '''Soil sensor reading data'''
#     nitrogen: float      # mg/kg
#     phosphorus: float    # mg/kg
#     potassium: float     # mg/kg
#     moisture: float      # %
#     temperature: float   # °C
#     conductivity: float  # µS/cm
#     ph: float            # pH value
#     timestamp: float
# 
# 
# class NPKSensor:
#     '''
#     7-in-1 Soil NPK Sensor using Modbus RTU over RS485
#     
#     Typical register layout (may vary by sensor model):
#     - Register 0x00: Moisture (0.1%)
#     - Register 0x01: Temperature (0.1°C)
#     - Register 0x02: Conductivity (µS/cm)
#     - Register 0x03: pH (0.01)
#     - Register 0x04: Nitrogen (mg/kg)
#     - Register 0x05: Phosphorus (mg/kg)
#     - Register 0x06: Potassium (mg/kg)
#     '''
#     
#     DEFAULT_ADDRESS = 0x01
#     FUNCTION_READ_HOLDING = 0x03
#     START_REGISTER = 0x0000
#     NUM_REGISTERS = 7
#     
#     def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 9600,
#                  address: int = 0x01, timeout: float = 1.0):
#         self.port = port
#         self.baudrate = baudrate
#         self.address = address
#         self.timeout = timeout
#         self.serial = None
#     
#     def connect(self) -> bool:
#         try:
#             self.serial = serial.Serial(
#                 port=self.port,
#                 baudrate=self.baudrate,
#                 bytesize=serial.EIGHTBITS,
#                 parity=serial.PARITY_NONE,
#                 stopbits=serial.STOPBITS_ONE,
#                 timeout=self.timeout
#             )
#             self.serial.reset_input_buffer()
#             self.serial.reset_output_buffer()
#             logger.info(f"Connected to NPK sensor on {self.port} @ {self.baudrate} baud")
#             return True
#         except serial.SerialException as e:
#             logger.error(f"Failed to connect: {e}")
#             return False
#     
#     def disconnect(self):
#         if self.serial and self.serial.is_open:
#             self.serial.close()
#             logger.info("Disconnected from NPK sensor")
#     
#     # ... (rest of the class methods)
#     
#     def read(self):
#         '''Read all sensor values'''
#         # Implementation here
#         pass
# 
# 
# def main():
#     import argparse
#     parser = argparse.ArgumentParser(description='NPK Soil Sensor Reader')
#     parser.add_argument('--port', '-p', default='/dev/ttyUSB0')
#     parser.add_argument('--baud', '-b', type=int, default=9600)
#     parser.add_argument('--address', '-a', type=int, default=1)
#     parser.add_argument('--continuous', '-c', action='store_true')
#     parser.add_argument('--list-ports', '-l', action='store_true')
#     args = parser.parse_args()
#     
#     # Implementation here
#     pass
# 
# if __name__ == "__main__":
#     main()
"""
