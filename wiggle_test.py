r"""
Test the system with an example IMU measurement sequence.
"""

import logging
import os
import time

import torch
from numpy.linalg import inv
from scipy.spatial.transform import Rotation

import articulate as art
from articulate import SMPLHJoint
from config import paths
from net import TransPoseNet
from utils import normalize_and_concat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sensor_wiggle(
    num_frames=200,
    num_sensors=6,
    sensor_idx=0,
    max_angle=45.0,
    axis="z",
):
    a_norm = torch.zeros(num_frames, num_sensors, 3)

    R_norm = torch.eye(3).repeat(
        num_frames,
        num_sensors,
        1,
        1,
    )

    t = torch.linspace(
        0,
        2 * torch.pi,
        num_frames,
    )

    for frame in range(num_frames):
        angle = max_angle * torch.sin(t[frame])

        rot = torch.tensor(
            Rotation.from_euler(
                axis,
                angle.item(),
                degrees=True,
            ).as_matrix(),
            dtype=torch.float32,
        )

        R_norm[frame, sensor_idx] = rot

    return a_norm, R_norm

def main() -> None:
    logger.info("Starting calibration!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = TransPoseNet().to(device)

    tran = torch.tensor(
        [
            [0.0],  # x
            [0.0],  # y
            [0.0],  # z
        ]
    ).T

    for sensor in range(6):
        for axis in ["x", "y", "z"]:
            for angle in [60, -60]:

                logger.info(
                    "sensor=%d axis=%s angle=%d",
                    sensor,
                    axis,
                    angle,
                )

                a_norm = torch.zeros(200, 6, 3)

                R_norm = torch.eye(3).repeat(
                    200,
                    6,
                    1,
                    1,
                )

                rot = torch.tensor(
                    Rotation.from_euler(
                        axis,
                        angle,
                        degrees=True,
                    ).as_matrix(),
                    dtype=torch.float32,
                )

                R_norm[:, sensor] = rot

                x = normalize_and_concat(
                    a_norm,
                    R_norm,
                ).to(device)

                pose, tran = net.forward_offline(x)

                print(
                    f"sensor={sensor} axis={axis} "
                    f"angle={angle}"
                )
                R_fix = torch.tensor(
                    Rotation.from_euler(
                        "xy",
                        [180, 180],
                        degrees=True,
                    ).as_matrix(),
                    dtype=torch.float32,
                )

                pose[:, 0] = R_fix @ pose[:, 0]
                art.ParametricModel(
                    paths.smpl_file
                ).view_motion(
                    [pose],
                    [tran],
                )


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    duration = time.perf_counter() - start
    logger.info("Finished in %.2f seconds", duration)
