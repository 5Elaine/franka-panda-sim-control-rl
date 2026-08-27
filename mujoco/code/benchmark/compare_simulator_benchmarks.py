#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = PROJECT_ROOT / "outputs" / "simulator_benchmark"

MUJOCO_SUMMARY = (
    RESULT_DIR
    / "mujoco_ros_benchmark_v2_summary.csv"
)

GAZEBO_SUMMARY = (
    RESULT_DIR
    / "gazebo_ros_benchmark_summary.csv"
)

COMPARISON_CSV = (
    RESULT_DIR
    / "gazebo_mujoco_comparison.csv"
)

COMPARISON_PLOT = (
    RESULT_DIR
    / "gazebo_mujoco_comparison.png"
)


NUMERIC_FIELDS = [
    "arrival_time_s",
    "settling_time_s",
    "peak_joint_speed_rad_s",
    "final_max_error_rad",
    "rms_tracking_error_rad",
]


def load_summary(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Summary CSV not found: {path}"
        )

    with path.open(
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise RuntimeError(
            f"Summary CSV is empty: {path}"
        )

    result = {}

    for row in rows:
        phase = row["phase"].strip().upper()

        result[phase] = {
            field: float(row[field])
            for field in NUMERIC_FIELDS
        }

    for required_phase in ("TARGET", "HOME"):
        if required_phase not in result:
            raise RuntimeError(
                f"{path} does not contain "
                f"{required_phase} data."
            )

    return result


def save_comparison_csv(
    mujoco_data: dict,
    gazebo_data: dict,
) -> None:
    fieldnames = [
        "phase",
        "metric",
        "mujoco",
        "gazebo",
        "gazebo_minus_mujoco",
        "gazebo_over_mujoco",
    ]

    with COMPARISON_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for phase in ("TARGET", "HOME"):
            for metric in NUMERIC_FIELDS:
                mujoco_value = mujoco_data[phase][metric]
                gazebo_value = gazebo_data[phase][metric]

                ratio = (
                    gazebo_value / mujoco_value
                    if abs(mujoco_value) > 1e-12
                    else float("nan")
                )

                writer.writerow(
                    {
                        "phase": phase,
                        "metric": metric,
                        "mujoco": f"{mujoco_value:.9f}",
                        "gazebo": f"{gazebo_value:.9f}",
                        "gazebo_minus_mujoco":
                            f"{gazebo_value - mujoco_value:.9f}",
                        "gazebo_over_mujoco":
                            f"{ratio:.9f}",
                    }
                )


def plot_comparison(
    mujoco_data: dict,
    gazebo_data: dict,
) -> None:
    phases = ["TARGET", "HOME"]
    x = np.arange(len(phases))
    width = 0.34

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
    )

    metric_settings = [
        (
            "arrival_time_s",
            "Arrival Time",
            "Time (s)",
        ),
        (
            "peak_joint_speed_rad_s",
            "Peak Joint Speed",
            "Speed (rad/s)",
        ),
        (
            "final_max_error_rad",
            "Final Maximum Error",
            "Error (rad)",
        ),
        (
            "rms_tracking_error_rad",
            "RMS Tracking Error",
            "Error (rad)",
        ),
    ]

    for axis, (
        metric,
        title,
        ylabel,
    ) in zip(axes.flat, metric_settings):
        mujoco_values = [
            mujoco_data[phase][metric]
            for phase in phases
        ]

        gazebo_values = [
            gazebo_data[phase][metric]
            for phase in phases
        ]

        axis.bar(
            x - width / 2,
            mujoco_values,
            width,
            label="MuJoCo",
        )

        axis.bar(
            x + width / 2,
            gazebo_values,
            width,
            label="Gazebo",
        )

        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.set_xticklabels(
            ["Target", "Home"]
        )
        axis.grid(
            True,
            axis="y",
        )
        axis.legend()

        maximum_value = max(
            mujoco_values + gazebo_values
        )

        offset = (
            maximum_value * 0.025
            if maximum_value > 0
            else 0.001
        )

        for index, value in enumerate(
            mujoco_values
        ):
            axis.text(
                index - width / 2,
                value + offset,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        for index, value in enumerate(
            gazebo_values
        ):
            axis.text(
                index + width / 2,
                value + offset,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    figure.suptitle(
        "Gazebo–MuJoCo Panda Joint-Control Benchmark",
        fontsize=15,
    )

    figure.tight_layout()
    figure.subplots_adjust(top=0.91)

    figure.savefig(
        COMPARISON_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def print_comparison(
    mujoco_data: dict,
    gazebo_data: dict,
) -> None:
    print("===== Gazebo–MuJoCo comparison =====")

    for phase in ("TARGET", "HOME"):
        print()
        print(f"----- {phase} -----")

        for field in NUMERIC_FIELDS:
            mujoco_value = mujoco_data[phase][field]
            gazebo_value = gazebo_data[phase][field]

            print(
                f"{field:28s} | "
                f"MuJoCo={mujoco_value:.6f} | "
                f"Gazebo={gazebo_value:.6f} | "
                f"difference="
                f"{gazebo_value - mujoco_value:+.6f}"
            )


def main() -> None:
    mujoco_data = load_summary(
        MUJOCO_SUMMARY
    )
    gazebo_data = load_summary(
        GAZEBO_SUMMARY
    )

    save_comparison_csv(
        mujoco_data,
        gazebo_data,
    )

    plot_comparison(
        mujoco_data,
        gazebo_data,
    )

    print_comparison(
        mujoco_data,
        gazebo_data,
    )

    print()
    print("Comparison CSV:", COMPARISON_CSV)
    print("Comparison plot:", COMPARISON_PLOT)
    print(
        "[PASS] Gazebo–MuJoCo comparison generated."
    )


if __name__ == "__main__":
    main()
