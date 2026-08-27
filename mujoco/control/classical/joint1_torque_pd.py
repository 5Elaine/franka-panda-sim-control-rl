import os
import csv
from pathlib import Path
import numpy as np
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MENAGERIE_ROOT = Path(os.environ.get(
    "MUJOCO_MENAGERIE_PATH",
    PROJECT_ROOT / "third_party" / "mujoco_menagerie",
)).expanduser()
MODEL_PATH = str(MENAGERIE_ROOT / "franka_emika_panda" / "scene.xml")
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "classical" / "joint1_torque_pd.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Load Panda model
# ============================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

ARM_ACTUATORS = 7

# ============================================================
# 2. Convert actuator1:
#
# Original:
#   position reference
#       ↓
#   built-in servo
#       ↓
#   actuator torque
#
# New:
#   our PD controller
#       ↓
#   torque command
#       ↓
#   actuator
#
# For actuator 0:
#
# actuator force =
#     gain * ctrl + bias
#
# Set:
#     gain = 1
#     bias = 0
#
# Therefore:
#     actuator force ≈ ctrl
# ============================================================

model.actuator_gainprm[0, :] = 0.0
model.actuator_gainprm[0, 0] = 1.0

model.actuator_biasprm[0, :] = 0.0

# ctrl now represents torque, so use Panda joint1 torque limit
model.actuator_ctrlrange[0, 0] = -87.0
model.actuator_ctrlrange[0, 1] = 87.0
model.actuator_ctrllimited[0] = 1

model.actuator_forcerange[0, 0] = -87.0
model.actuator_forcerange[0, 1] = 87.0
model.actuator_forcelimited[0] = 1

# ============================================================
# 3. Controller parameters
# ============================================================

TARGET = 0.5       # rad

KP = 40.0
KD = 8.0

TORQUE_LIMIT = 87.0

SIM_TIME = 4.0
DT = model.opt.timestep
STEPS = int(SIM_TIME / DT)

# ============================================================
# 4. Initial state
# ============================================================

mujoco.mj_forward(model, data)

# Keep joints 2-7 at zero using their original position servos.
# actuator8 is the gripper and is left unchanged.
data.ctrl[1:ARM_ACTUATORS] = 0.0

records = []

print("\n========== JOINT1 TORQUE PD ==========")
print(f"Target       : {TARGET:.3f} rad")
print(f"Kp           : {KP:.3f}")
print(f"Kd           : {KD:.3f}")
print(f"Torque limit : ±{TORQUE_LIMIT:.1f} N·m")
print(f"Timestep     : {DT:.4f} s")
print(f"Simulation   : {SIM_TIME:.1f} s")

print("\nInitial joint1:")
print(f"q   = {data.qpos[0]:.6f} rad")
print(f"dq  = {data.qvel[0]:.6f} rad/s")

# ============================================================
# 5. Closed-loop simulation
# ============================================================

for step in range(STEPS):

    q = float(data.qpos[0])
    dq = float(data.qvel[0])

    error = TARGET - q

    # --------------------------------------------------------
    # THIS is our controller
    # --------------------------------------------------------

    tau = KP * error - KD * dq

    tau = float(
        np.clip(
            tau,
            -TORQUE_LIMIT,
            TORQUE_LIMIT
        )
    )

    # Our torque command goes directly to actuator1
    data.ctrl[0] = tau

    # Other arm joints still use original position control
    data.ctrl[1:ARM_ACTUATORS] = 0.0

    mujoco.mj_step(model, data)

    records.append(
        [
            data.time,
            data.qpos[0],
            data.qvel[0],
            TARGET - data.qpos[0],
            tau
        ]
    )

# ============================================================
# 6. Metrics
# ============================================================

records_np = np.asarray(records)

time = records_np[:, 0]
q = records_np[:, 1]
dq = records_np[:, 2]
error = records_np[:, 3]
tau = records_np[:, 4]

final_q = q[-1]
final_error = abs(TARGET - final_q)

peak_q = np.max(q)

overshoot = max(
    0.0,
    peak_q - TARGET
)

peak_speed = np.max(np.abs(dq))
peak_torque = np.max(np.abs(tau))

# Settling time:
# error <= 0.01 rad and never leaves afterward

threshold = 0.01
settling_time = np.nan

for i in range(len(error)):
    if np.all(np.abs(error[i:]) <= threshold):
        settling_time = time[i]
        break

# ============================================================
# 7. Save raw data
# ============================================================

with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(
        [
            "time_s",
            "joint1_position_rad",
            "joint1_velocity_rad_s",
            "position_error_rad",
            "command_torque_Nm"
        ]
    )

    writer.writerows(records)

# ============================================================
# 8. Report
# ============================================================

print("\n========== RESULT ==========")

print(f"Final q            : {final_q:.6f} rad")
print(f"Final error        : {final_error:.6f} rad")
print(f"Peak q             : {peak_q:.6f} rad")
print(f"Overshoot          : {overshoot:.6f} rad")
print(f"Peak joint speed   : {peak_speed:.6f} rad/s")
print(f"Peak command torque: {peak_torque:.6f} N·m")

if np.isnan(settling_time):
    print("Settling time      : NOT settled within simulation")
else:
    print(f"Settling time      : {settling_time:.6f} s")

print(f"\nCSV saved to:")
print(OUTPUT_PATH)

if final_error <= 0.01:
    print("\n[PASS] joint1 torque-PD tracking succeeded")
else:
    print("\n[WARN] joint1 did not reach the 0.01 rad final-error threshold")
