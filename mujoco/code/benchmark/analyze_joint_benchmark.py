#!/usr/bin/env python3

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


JOINT_COUNT = 7


@dataclass
class CommandSegment:
    start_index: int
    end_index: int
    target: np.ndarray


def load_benchmark(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Benchmark CSV was not found: {path}"
        )

    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=float,
        encoding="utf-8",
    )

    if data.size == 0:
        raise RuntimeError(
            f"Benchmark CSV contains no samples: {path}"
        )

    # 只有一行时，genfromtxt 会返回零维结构化数组。
    if data.ndim == 0:
        data = np.asarray([data], dtype=data.dtype)

    time_values = np.asarray(
        data["elapsed_time_s"],
        dtype=float,
    )

    positions = np.column_stack(
        [
            np.asarray(data[f"q{index}"], dtype=float)
            for index in range(1, JOINT_COUNT + 1)
        ]
    )

    velocities = np.column_stack(
        [
            np.asarray(data[f"dq{index}"], dtype=float)
            for index in range(1, JOINT_COUNT + 1)
        ]
    )

    targets = np.column_stack(
        [
            np.asarray(data[f"target{index}"], dtype=float)
            for index in range(1, JOINT_COUNT + 1)
        ]
    )

    return (
        time_values,
        positions,
        velocities,
        targets,
    )


def find_command_segments(
    targets: np.ndarray,
    change_tolerance: float = 1e-6,
) -> List[CommandSegment]:
    valid_rows = np.all(
        np.isfinite(targets),
        axis=1,
    )

    valid_indices = np.flatnonzero(valid_rows)

    if valid_indices.size == 0:
        raise RuntimeError(
            "No valid target commands were found."
        )

    segments: List[CommandSegment] = []

    segment_start = int(valid_indices[0])
    current_target = targets[segment_start].copy()

    for index in valid_indices[1:]:
        index = int(index)

        target_change = float(
            np.max(
                np.abs(
                    targets[index] - current_target
                )
            )
        )

        if target_change > change_tolerance:
            segments.append(
                CommandSegment(
                    start_index=segment_start,
                    end_index=index,
                    target=current_target.copy(),
                )
            )

            segment_start = index
            current_target = targets[index].copy()

    segments.append(
        CommandSegment(
            start_index=segment_start,
            end_index=int(valid_indices[-1]) + 1,
            target=current_target.copy(),
        )
    )

    return segments


def first_arrival_index(
    maximum_error: np.ndarray,
    threshold: float,
) -> int | None:
    candidates = np.flatnonzero(
        maximum_error <= threshold
    )

    if candidates.size == 0:
        return None

    return int(candidates[0])


def settling_index(
    maximum_error: np.ndarray,
    threshold: float,
) -> int | None:
    """
    返回第一次进入阈值后，直到该命令段结束都不再离开
    阈值范围的采样点。
    """
    for index in range(maximum_error.size):
        if np.all(
            maximum_error[index:] <= threshold
        ):
            return index

    return None


def seconds_or_nan(
    index: int | None,
    segment_time: np.ndarray,
) -> float:
    if index is None:
        return float("nan")

    return float(
        segment_time[index] - segment_time[0]
    )


def analyze_segment(
    label: str,
    segment: CommandSegment,
    time_values: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    error_threshold: float,
) -> dict:
    start = segment.start_index
    end = segment.end_index

    segment_time = time_values[start:end]
    segment_positions = positions[start:end]
    segment_velocities = velocities[start:end]

    if segment_time.size == 0:
        raise RuntimeError(
            f"Command segment {label} contains no samples."
        )

    joint_errors = (
        segment.target.reshape(1, -1)
        - segment_positions
    )

    absolute_errors = np.abs(joint_errors)

    maximum_error = np.max(
        absolute_errors,
        axis=1,
    )

    maximum_speed = np.max(
        np.abs(segment_velocities),
        axis=1,
    )

    arrival_sample = first_arrival_index(
        maximum_error,
        error_threshold,
    )

    settled_sample = settling_index(
        maximum_error,
        error_threshold,
    )

    rms_error = float(
        np.sqrt(
            np.mean(
                joint_errors ** 2
            )
        )
    )

    result = {
        "phase": label,
        "command_time_s": float(segment_time[0]),
        "segment_duration_s": float(
            segment_time[-1] - segment_time[0]
        ),
        "arrival_time_s": seconds_or_nan(
            arrival_sample,
            segment_time,
        ),
        "settling_time_s": seconds_or_nan(
            settled_sample,
            segment_time,
        ),
        "peak_joint_speed_rad_s": float(
            np.max(maximum_speed)
        ),
        "final_max_error_rad": float(
            maximum_error[-1]
        ),
        "rms_tracking_error_rad": rms_error,
        "samples": int(segment_time.size),
    }

    return result


def save_summary(
    path: Path,
    results: List[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "phase",
        "command_time_s",
        "segment_duration_s",
        "arrival_time_s",
        "settling_time_s",
        "peak_joint_speed_rad_s",
        "final_max_error_rad",
        "rms_tracking_error_rad",
        "samples",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def plot_benchmark(
    output_path: Path,
    time_values: np.ndarray,
    positions: np.ndarray,
    targets: np.ndarray,
    segments: List[CommandSegment],
    error_threshold: float,
) -> None:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 9),
        sharex=True,
    )

    position_axis = axes[0]
    error_axis = axes[1]

    for joint_index in range(JOINT_COUNT):
        position_axis.plot(
            time_values,
            positions[:, joint_index],
            label=f"joint{joint_index + 1}",
        )

    for segment in segments:
        start = segment.start_index
        end = segment.end_index

        for joint_index in range(JOINT_COUNT):
            position_axis.plot(
                time_values[start:end],
                np.full(
                    end - start,
                    segment.target[joint_index],
                ),
                linestyle="--",
                linewidth=1,
            )

    position_axis.set_title(
        "Panda ROS Benchmark: Joint Positions"
    )
    position_axis.set_ylabel("Position (rad)")
    position_axis.grid(True)
    position_axis.legend(
        ncol=2,
        loc="best",
    )

    maximum_error = np.full(
        time_values.shape,
        np.nan,
        dtype=float,
    )

    for segment in segments:
        start = segment.start_index
        end = segment.end_index

        maximum_error[start:end] = np.max(
            np.abs(
                segment.target.reshape(1, -1)
                - positions[start:end]
            ),
            axis=1,
        )

    error_axis.plot(
        time_values,
        maximum_error,
        label="maximum absolute joint error",
    )

    error_axis.axhline(
        error_threshold,
        linestyle="--",
        label=(
            f"threshold = "
            f"{error_threshold:.3f} rad"
        ),
    )

    for segment in segments:
        error_axis.axvline(
            time_values[segment.start_index],
            linestyle=":",
        )

    error_axis.set_title(
        "Maximum Joint Tracking Error"
    )
    error_axis.set_xlabel("Elapsed wall time (s)")
    error_axis.set_ylabel("Error (rad)")
    error_axis.grid(True)
    error_axis.legend()

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Panda joint benchmark data "
            "using identical metrics for simulators."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input benchmark CSV.",
    )

    parser.add_argument(
        "--output-summary",
        type=Path,
        required=True,
        help="Output metric summary CSV.",
    )

    parser.add_argument(
        "--output-plot",
        type=Path,
        required=True,
        help="Output benchmark plot.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help=(
            "Maximum absolute joint error threshold "
            "in radians."
        ),
    )

    arguments = parser.parse_args()

    if arguments.threshold <= 0:
        raise ValueError(
            "--threshold must be greater than zero."
        )

    (
        time_values,
        positions,
        velocities,
        targets,
    ) = load_benchmark(
        arguments.input.expanduser().resolve()
    )

    segments = find_command_segments(targets)

    print("===== Detected command segments =====")

    for index, segment in enumerate(
        segments,
        start=1,
    ):
        print(
            f"Segment {index}: "
            f"rows [{segment.start_index}, "
            f"{segment.end_index}) | "
            f"target={segment.target}"
        )

    if len(segments) < 2:
        raise RuntimeError(
            "Expected at least Target and Home commands."
        )

    labels = [
        "TARGET",
        "HOME",
    ]

    results = []

    for index, segment in enumerate(segments):
        label = (
            labels[index]
            if index < len(labels)
            else f"COMMAND_{index + 1}"
        )

        results.append(
            analyze_segment(
                label=label,
                segment=segment,
                time_values=time_values,
                positions=positions,
                velocities=velocities,
                error_threshold=arguments.threshold,
            )
        )

    save_summary(
        arguments.output_summary.expanduser().resolve(),
        results,
    )

    plot_benchmark(
        arguments.output_plot.expanduser().resolve(),
        time_values,
        positions,
        targets,
        segments,
        arguments.threshold,
    )

    duration = float(
        time_values[-1] - time_values[0]
    )

    average_rate = (
        (time_values.size - 1) / duration
        if duration > 0
        else float("nan")
    )

    print()
    print("===== Benchmark summary =====")
    print("Samples:", time_values.size)
    print(f"Duration: {duration:.3f} s")
    print(
        f"Average sample rate: "
        f"{average_rate:.3f} Hz"
    )
    print(
        f"Error threshold: "
        f"{arguments.threshold:.3f} rad"
    )

    for result in results:
        print()
        print(
            f"----- {result['phase']} -----"
        )
        print(
            "Segment duration:",
            f"{result['segment_duration_s']:.3f} s",
        )
        print(
            "Arrival time:",
            f"{result['arrival_time_s']:.3f} s",
        )
        print(
            "Settling time:",
            f"{result['settling_time_s']:.3f} s",
        )
        print(
            "Peak joint speed:",
            f"{result['peak_joint_speed_rad_s']:.6f} rad/s",
        )
        print(
            "Final maximum error:",
            f"{result['final_max_error_rad']:.6f} rad",
        )
        print(
            "RMS tracking error:",
            f"{result['rms_tracking_error_rad']:.6f} rad",
        )

    print()
    print(
        "Summary saved to:",
        arguments.output_summary.expanduser().resolve(),
    )
    print(
        "Plot saved to:",
        arguments.output_plot.expanduser().resolve(),
    )
    print()
    print(
        "[PASS] Benchmark metrics generated successfully."
    )


if __name__ == "__main__":
    main()
