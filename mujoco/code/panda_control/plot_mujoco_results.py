from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mujoco_analysis"

JOINT_CSV = BASE_DIR / "panda_home_target_home.csv"
HAND_CSV = BASE_DIR / "panda_hand_pose_tracking.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    if data.size == 0:
        raise RuntimeError(f"CSV file is empty: {path}")

    return data


def plot_joint_positions(data: np.ndarray) -> None:
    time = data["time"]

    figure, axes = plt.subplots(
        7,
        1,
        figsize=(10, 13),
        sharex=True,
    )

    for index, axis in enumerate(axes, start=1):
        axis.plot(
            time,
            data[f"q{index}"],
            label=f"joint{index} actual",
        )
        axis.plot(
            time,
            data[f"target{index}"],
            linestyle="--",
            label=f"joint{index} target",
        )

        axis.set_ylabel("rad")
        axis.grid(True)
        axis.legend(loc="best")

    axes[-1].set_xlabel("Simulation time (s)")
    figure.suptitle(
        "MuJoCo Panda Joint Position Tracking",
        fontsize=14,
    )
    figure.tight_layout()

    output = OUTPUT_DIR / "mujoco_joint_position_tracking.png"
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print("Saved:", output)


def plot_joint_velocities(data: np.ndarray) -> None:
    time = data["time"]

    figure, axis = plt.subplots(figsize=(10, 6))

    for index in range(1, 8):
        axis.plot(
            time,
            data[f"dq{index}"],
            label=f"joint{index}",
        )

    axis.set_title("MuJoCo Panda Joint Velocities")
    axis.set_xlabel("Simulation time (s)")
    axis.set_ylabel("Angular velocity (rad/s)")
    axis.grid(True)
    axis.legend(ncol=2)

    output = OUTPUT_DIR / "mujoco_joint_velocities.png"
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print("Saved:", output)


def plot_hand_xyz(data: np.ndarray) -> None:
    time = data["time"]

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(time, data["hand_x"], label="x")
    axis.plot(time, data["hand_y"], label="y")
    axis.plot(time, data["hand_z"], label="z")

    axis.set_title("MuJoCo Panda Hand Position")
    axis.set_xlabel("Simulation time (s)")
    axis.set_ylabel("Position (m)")
    axis.grid(True)
    axis.legend()

    output = OUTPUT_DIR / "mujoco_hand_xyz.png"
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print("Saved:", output)


def plot_hand_path_3d(data: np.ndarray) -> None:
    figure = plt.figure(figsize=(9.5, 7.5))
    axis = figure.add_subplot(111, projection="3d")

    x = data["hand_x"]
    y = data["hand_y"]
    z = data["hand_z"]

    axis.plot(x, y, z, label="hand path")
    axis.scatter(
        [x[0]],
        [y[0]],
        [z[0]],
        marker="o",
        s=60,
        label="start",
    )

    furthest_index = int(
        np.argmax(
            np.sqrt(
                (x - x[0]) ** 2
                + (y - y[0]) ** 2
                + (z - z[0]) ** 2
            )
        )
    )

    axis.scatter(
        [x[furthest_index]],
        [y[furthest_index]],
        [z[furthest_index]],
        marker="^",
        s=70,
        label="target region",
    )

    axis.set_title("MuJoCo Panda Hand 3D Path")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)",labelpad=12)
    axis.legend(loc="upper left")

    figure.subplots_adjust(
    left=0.06,
    right=0.86,
    bottom=0.08,
    top=0.90,
    )

    output = OUTPUT_DIR / "mujoco_hand_3d_path.png"
    figure.savefig(
    output,
    dpi=200,
    bbox_inches="tight",
    pad_inches=0.25,
    )
    plt.close(figure)

    print("Saved:", output)


def print_summary(
    joint_data: np.ndarray,
    hand_data: np.ndarray,
) -> None:
    final_joint_error = []

    for index in range(1, 8):
        error = abs(
            joint_data[f"target{index}"][-1]
            - joint_data[f"q{index}"][-1]
        )
        final_joint_error.append(error)

    xyz = np.column_stack(
        (
            hand_data["hand_x"],
            hand_data["hand_y"],
            hand_data["hand_z"],
        )
    )

    displacement = np.linalg.norm(
        xyz - xyz[0],
        axis=1,
    )

    print()
    print("===== Quantitative summary =====")
    print(
        "Maximum final joint error:",
        f"{max(final_joint_error):.6f} rad",
    )
    print(
        "Maximum hand displacement:",
        f"{np.max(displacement):.6f} m",
    )
    print(
        "Final hand return distance:",
        f"{np.linalg.norm(xyz[-1] - xyz[0]):.6f} m",
    )


def main() -> None:
    joint_data = load_csv(JOINT_CSV)
    hand_data = load_csv(HAND_CSV)

    plot_joint_positions(joint_data)
    plot_joint_velocities(joint_data)
    plot_hand_xyz(hand_data)
    plot_hand_path_3d(hand_data)
    print_summary(joint_data, hand_data)

    print()
    print("[PASS] MuJoCo figures generated successfully.")


if __name__ == "__main__":
    main()
