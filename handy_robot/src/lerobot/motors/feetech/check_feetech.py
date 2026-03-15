# save as check_feetech.py and run:  python check_feetech.py --port /dev/ttyACM0 --baud 115200
import argparse, logging
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorCalibration

parser = argparse.ArgumentParser()
parser.add_argument("--port", required=True)
parser.add_argument("--baud", type=int, default=115200)  # try 115200 then 1000000
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
# Minimal "virtual" config: one placeholder motor entry so the bus can be created.
from lerobot.motors.motors_bus import MotorCalibration, Motor

# Add the missing field (norm_mode), usually just "default" or 0
motors = {"tmp": Motor(id=1, model="sts3215", norm_mode="default")}
calibration = {"tmp": MotorCalibration(id=1, drive_mode=0, homing_offset=0, range_min=0, range_max=0)}
bus = FeetechMotorsBus(port=args.port, motors=motors, calibration=calibration, protocol_version=0)
 # ID is placeholder; model should match your unit if known
bus = FeetechMotorsBus(port=args.port, motors=motors, calibration={"tmp": MotorCalibration(1,0,0,0,0)}, protocol_version=0)

# Try two common baudrates
for b in [args.baud, 1_000_000]:
    print(f"\n=== Trying baud {b} ===")
    bus.set_baudrate(b)
    ids_to_model = bus.broadcast_ping(raise_on_error=False)  # will also log error flags
    print("IDs found -> modelNumber:", ids_to_model)

print("\nIf you see an ID but also an 'Input voltage error', fix power first (voltage/current/ground).")
