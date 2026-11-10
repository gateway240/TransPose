from __future__ import annotations
from articulate.math import rotation_matrix_to_euler_angle, quaternion_to_rotation_matrix

import argparse
import logging
import pathlib
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from collections.abc import Hashable

    import scipy
    # from pandas.io.formats.style_render import ExtFormatter

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PLOT_RESULTS = True
SENSOR_MAPPING = [
    "radius_l_imu",
    "radius_r_imu",
    "tibia_l_imu",
    "tibia_r_imu",
    "head_imu",
    "pelvis_imu",
]

LINE_STYLES = [
    "-",   # radius_l_imu
    "--",  # radius_r_imu
    "-.",  # tibia_l_imu
    ":",   # tibia_r_imu
    (0, (5, 1)),         # head_imu
    (0, (3, 1, 1, 1)),   # pelvis_imu
]

YAW_OFFSETS = torch.deg2rad(
    torch.tensor([
        -30.923927739673406,  # radius_l_imu
        -44.63515206908284,   # radius_r_imu
        -48.66927350618733,   # tibia_l_imu
        -56.067362769802585,  # tibia_r_imu
        -36.847138420633556,  # head_imu
        -30.576712572777073,  # pelvis_imu
    ], dtype=torch.float32)
)

def collect_motion_files(
    root_dir: Path,
) -> dict[str, dict[str, Path]]:
    """
    Key = (participant, motion)
    """
    motions: defaultdict[str, dict[str, Path]] = defaultdict(dict)

    directory = root_dir
    for fname in directory.iterdir():
        motion = fname.name.rsplit("_", 1)[0]
        category = "NONE"
        if "acc" in fname.name:
            category= "acc"
        elif "ori" in fname.name:
            category = "ori"
        # logger.info(motion)
        motions[motion][category] = fname

    return motions


def parse_quaternion(q_str: str) -> np.ndarray:
    """Convert string 'w,x,y,z' → list of floats"""
    return np.array([float(x) for x in q_str.split(",")])


def quaternion_series_to_euler(series: pd.Series) -> scipy.Array:
    """Convert a pandas Series of quaternion strings to Euler angles"""
    quats = np.vstack(
        series.dropna().map(parse_quaternion).tolist(),
    )
    # scipy expects [x, y, z, w], so reorder if needed
    # assuming your format is [w, x, y, z]
    quats_xyzw = np.column_stack([quats[:, 1], quats[:, 2], quats[:, 3], quats[:, 0]])

    rotations = Rotation.from_quat(quats_xyzw)
    return rotations.as_euler("xyz", degrees=True)  # roll, pitch, yaw


def plot_euler(
    acc: torch.Tensor,   # [frames, sensors, 3]
    ori: torch.Tensor,   # [frames, sensors, 3] Euler angles
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        6,
        1,
        figsize=(12, 16),
        sharex=True,
    )

    (
        ax_roll,
        ax_pitch,
        ax_yaw,
        ax_acc_x,
        ax_acc_y,
        ax_acc_z,
    ) = axes

    num_frames, num_sensors = ori.shape[:2]

    for sensor_idx in range(num_sensors):
        label = (
            SENSOR_MAPPING[sensor_idx]
            if sensor_idx < len(SENSOR_MAPPING)
            else f"sensor_{sensor_idx}"
        )

        linestyle = (
            LINE_STYLES[sensor_idx]
            if sensor_idx < len(LINE_STYLES)
            else "-"
        )

        # Euler
        ax_roll.plot(
            ori[:, sensor_idx, 0].cpu(),
            label=label,
            linestyle=linestyle,
            linewidth=1.5,
        )

        ax_pitch.plot(
            ori[:, sensor_idx, 1].cpu(),
            label=label,
            linestyle=linestyle,
            linewidth=1.5,
        )

        ax_yaw.plot(
            ori[:, sensor_idx, 2].cpu(),
            label=label,
            linestyle=linestyle,
            linewidth=1.5,
        )

        # Acceleration
        ax_acc_x.plot(
            acc[:, sensor_idx, 0].cpu(),
            linestyle=linestyle,
            linewidth=1.5,
        )

        ax_acc_y.plot(
            acc[:, sensor_idx, 1].cpu(),
            linestyle=linestyle,
            linewidth=1.5,
        )

        ax_acc_z.plot(
            acc[:, sensor_idx, 2].cpu(),
            linestyle=linestyle,
            linewidth=1.5,
        )
    ax_roll.set_title("Roll (X)")
    ax_pitch.set_title("Pitch (Y)")
    ax_yaw.set_title("Yaw (Z)")

    ax_acc_x.set_title("Acceleration X")
    ax_acc_y.set_title("Acceleration Y")
    ax_acc_z.set_title("Acceleration Z")

    for ax in axes:
        ax.grid(True)
        ax.legend(fontsize=8)

    ax_roll.set_ylabel("rad")
    ax_pitch.set_ylabel("rad")
    ax_yaw.set_ylabel("rad")

    ax_acc_x.set_ylabel("m/s²")
    ax_acc_y.set_ylabel("m/s²")
    ax_acc_z.set_ylabel("m/s²")

    ax_acc_z.set_xlabel("Frame")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def _process_files(motion: str, modalities: dict[str,Path], output_dir: Path) -> dict[str | Hashable, Any]:

    # path = f
    logger.info(modalities)
    acc_file = modalities["acc"]
    ori_file = modalities["ori"]
    acc = torch.load(acc_file)
    logger.info(acc[0])
    logger.info(acc.shape)
    ori = torch.load(ori_file)
    logger.info(ori[0])
    logger.info(ori.shape)

    # logger.info(df)
    # Convert quaternions -> rotation matrices if needed
    if ori.shape[-1] == 4:
        # [frames, sensors, 4] -> [frames, sensors, 3, 3]
        ori = quaternion_to_rotation_matrix(
            ori.reshape(-1, 4)
        ).view(*ori.shape[:-1], 3, 3)

    elif ori.shape[-2:] != (3, 3):
        raise ValueError(
            f"Unsupported orientation shape {ori.shape}. "
            "Expected [frames, sensors, 4] or [frames, sensors, 3, 3]."
        )

    num_frames = ori.shape[0]

    euler = torch.stack(
        [
            rotation_matrix_to_euler_angle(ori[i])
            for i in range(num_frames)
        ],
        dim=0,
    )

    logger.info(euler.shape)
    # print("before", euler[0, :, 2])
    # euler[..., 2] += YAW_OFFSETS.to(
    #     euler.device,
    #     euler.dtype,
    # )

    # euler[..., 2] = (
    #     (euler[..., 2] + torch.pi)
    #     % (2 * torch.pi)
    # ) - torch.pi
    # print("after ", euler[0, :, 2])
    # logger.info(df_euler)
    # means, stds = df_euler.mean().to_numpy(), df_euler.std().to_numpy()

    # # interleave
    # summary = pd.DataFrame(
    #     np.column_stack([means, stds]).reshape(1, -1),
    #     columns=pd.Index([f"{c}_{stat}" for c in df_euler.columns for stat in ["mean", "std"]]),
    # )
    # logger.info(summary)
    # summary_dict = summary.iloc[0].to_dict()

    if PLOT_RESULTS:
        output_file = output_dir / f"{motion}.pdf"
        plot_euler(acc, euler, output_file)

    return {
        "trial": motion,
        "acc": acc,
        "ori": euler,
    }


def aggregate_and_plot(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    One figure per sensor.
    Each figure shows roll/pitch/yaw.
    Each curve = one participant.
    """

    # group by motion first (optional but usually useful)
    for motion, motion_df in summary_df.groupby("motion"):
        # collect per sensor: sensor -> participant -> euler
        sensor_data = {}

        for _, row in motion_df.iterrows():
            participant = row["participant"]
            df = row["df"]

            for sensor in df.columns:
                euler = quaternion_series_to_euler(df[sensor])

                if sensor not in sensor_data:
                    sensor_data[sensor] = {}

                sensor_data[sensor][participant] = euler

        # plot per sensor
        for sensor, participants in sensor_data.items():
            fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
            ax_roll, ax_pitch, ax_yaw = axes

            sorted_participants = sorted(
                participants.items(),
                key=lambda x: (
                    int("".join(filter(str.isdigit, str(x[0])))) if any(c.isdigit() for c in str(x[0])) else str(x[0])
                ),
            )

            for participant, euler in sorted_participants:
                ax_roll.plot(euler[:, 0], label=participant)
                ax_pitch.plot(euler[:, 1], label=participant)
                ax_yaw.plot(euler[:, 2], label=participant)

            ax_roll.set_title(f"{motion} - {sensor} Roll")
            ax_pitch.set_title(f"{motion} - {sensor} Pitch")
            ax_yaw.set_title(f"{motion} - {sensor} Yaw")

            for ax in axes:
                ax.set_ylabel("Degrees")
                ax.grid(True)  # ruff: ignore[boolean-positional-value-in-call]
                ax.legend(fontsize=7, loc="upper right")

            ax_yaw.set_xlabel("Frame")

            fig.tight_layout()

            out_file = output_dir / f"{motion}_{sensor}.png"
            fig.savefig(out_file)
            plt.close(fig)


def process_motion_files(
    motions:  dict[str, dict[str, Path]],
    output_dir: Path,
) -> Any:
    tasks_stage1 = [
        (trial, files, output_dir) for (trial), files in motions.items()
    ]
    logger.info(tasks_stage1)
    with Pool() as executor:
        try:
            results = executor.starmap(
                _process_files,
                tasks_stage1,
            )
        except KeyboardInterrupt:
            logger.info("Interrupted")
            executor.terminate()
            executor.join()
            raise
    return results
    # summary_df = pd.DataFrame(results)
    # if PLOT_RESULTS:
    #     aggregate_and_plot(summary_df, output_dir)
    # return summary_df


def summary_table(df_subset: pd.DataFrame, index_list: list[str]) -> pd.DataFrame:
    mean_row = df_subset.mean().round(2)
    sd_row = df_subset.std().round(2)
    range_row = df_subset.apply(lambda x: f"{x.min():.1f}–{x.max():.1f}")  # ruff: ignore[ambiguous-unicode-character-string]
    return pd.DataFrame([mean_row, sd_row, range_row], index=pd.Index(index_list))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check files")
    parser.add_argument(
        "--source_dir",
        default=Path("data/example"),
        type=Path,
        help="Root directory containing subject folders",
    )
    parser.add_argument(
        "--output_dir",
        default="out",
        type=Path,
        help="Directory to save output CSV (default: current directory)",
    )

    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    motions = collect_motion_files(args.source_dir)
    logger.info(motions)
    # logger.info(motions)
    summary_df = process_motion_files(motions, output_dir)
    logger.info(summary_df)
    # logger.info(summary_df.head())
    # summary_df = summary_df.drop(["file", "trial", "df"], axis=1)
    # summary_df = summary_df.sort_values(["participant"])
    # logger.info(summary_df)

    # output_file = output_dir / "imu-table-test.csv"
    # summary_df.to_csv(output_file, index=False)

    # # Calculate across all sensors
    # roll_cols = [c for c in summary_df.columns if "_roll_mean" in c]
    # pitch_cols = [c for c in summary_df.columns if "_pitch_mean" in c]
    # yaw_cols = [c for c in summary_df.columns if "_yaw_mean" in c]
    # summary_stats = pd.DataFrame(
    #     {
    #         "participant": summary_df["participant"],
    #         "roll_mean": summary_df[roll_cols].mean(axis=1),
    #         "roll_std": summary_df[roll_cols].std(axis=1),
    #         "roll_delta": summary_df[roll_cols].apply(
    #             lambda x: f"{x.min():.1f} – {x.max():.1f}",  # ruff: ignore[ambiguous-unicode-character-string]
    #             axis=1,
    #         ),
    #         "pitch_mean": summary_df[pitch_cols].mean(axis=1),
    #         "pitch_std": summary_df[pitch_cols].std(axis=1),
    #         "pitch_delta": summary_df[pitch_cols].apply(
    #             lambda x: f"{x.min():.1f} – {x.max():.1f}",  # ruff: ignore[ambiguous-unicode-character-string]
    #             axis=1,
    #         ),
    #         "yaw_mean": summary_df[yaw_cols].mean(axis=1),
    #         "yaw_std": summary_df[yaw_cols].std(axis=1),
    #         "yaw_delta": summary_df[yaw_cols].apply(
    #             lambda x: f"{x.min():.1f} – {x.max():.1f}",  # ruff: ignore[ambiguous-unicode-character-string]
    #             axis=1,
    #         ),
    #         # "yaw_delta":  summary_df[yaw_cols].max(axis=1)
    #         # - summary_df[yaw_cols].min(axis=1),
    #     },
    # )
    # logger.info(summary_stats)

    # output_dir_latex = pathlib.Path("out")
    # output_file = output_dir_latex / "imu-table-test-per-participant.csv"
    # summary_stats.to_csv(output_file, index=False)

    # rename_map = {
    #     "participant": r"\#",
    #     "roll_mean": r"X $\mu$",
    #     "roll_std": r"X $\sigma$",
    #     "roll_delta": r"X $\Delta$",
    #     "pitch_mean": r"Y $\mu$",
    #     "pitch_std": r"Y $\sigma$",
    #     "pitch_delta": r"Y $\Delta$",
    #     "yaw_mean": r"Z $\mu$",
    #     "yaw_std": r"Z $\sigma$",
    #     "yaw_delta": r"Z $\Delta$",
    # }

    # summary_stats = summary_stats[list(rename_map.keys())].rename(columns=rename_map)

    # fmt: ExtFormatter = {
    #     col: ("{:.2f}" if pd.api.types.is_numeric_dtype(summary_stats[col]) else "{:s}")
    #     for col in summary_stats.columns[1:]
    # }

    # latex = (
    #     summary_stats.style
    #     .format(fmt)
    #     .hide(axis="index")
    #     .to_latex(
    #         caption=(
    #             r"IMU table test results (mean $\mu$, standard deviation $\sigma$, and range $\Delta$) for each participant (\#). "  # ruff: ignore[line-too-long]
    #             "The X, Y, and Z values represent roll, pitch, and yaw in degrees (°) respectively. "
    #             "Sensors were placed on a flat, non-metallic table with a uniform orientation "
    #             "after being removed from each participant at the end of each data collection session."
    #         ),
    #         label="tab:imu_table_test_per_participant",
    #         hrules=True,
    #         position_float="centering",
    #     )
    # )

    # logger.info("\n%s", latex)
    # output_file_latex = output_dir_latex / "imu-table-test-per-participant.txt"
    # output_file_latex.write_text(latex, encoding="utf-8", newline="")

    logger.info("Done. Processed: %d trials!", len(summary_df))


if __name__ == "__main__":
    main()
