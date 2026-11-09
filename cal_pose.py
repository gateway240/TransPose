r"""
Test the system with an example IMU measurement sequence.
"""
import itertools

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

tracked_joints = [
    SMPLHJoint.LELBOW,
    SMPLHJoint.RELBOW,
    SMPLHJoint.LKNEE,
    SMPLHJoint.RKNEE,
    SMPLHJoint.HEAD,
    SMPLHJoint.PELVIS,
]
# tracked_joints = [
#     SMPLHJoint.PELVIS,
#     SMPLHJoint.LKNEE,
#     SMPLHJoint.RKNEE,
#     SMPLHJoint.HEAD,
#     SMPLHJoint.LELBOW,
#     SMPLHJoint.RELBOW,
# ]

PELVIS_IDX = 5

def generate_coordinate_frame_candidates():
    """
    Generate every proper orthogonal axis mapping.

    Returns:
        [(name, R), ...]
    """

    candidates = []

    axes = [
        torch.tensor([1, 0, 0], dtype=torch.float32),
        torch.tensor([0, 1, 0], dtype=torch.float32),
        torch.tensor([0, 0, 1], dtype=torch.float32),
    ]

    for perm in itertools.permutations(range(3)):

        P = torch.stack(
            [axes[perm[0]], axes[perm[1]], axes[perm[2]]],
            dim=1,
        )

        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):

                    S = torch.diag(
                        torch.tensor(
                            [sx, sy, sz],
                            dtype=torch.float32,
                        )
                    )

                    R = P @ S

                    det = torch.det(R)

                    #
                    # Only keep proper rotations
                    #
                    if torch.isclose(
                        det,
                        torch.tensor(1.0),
                        atol=1e-5,
                    ):

                        name = (
                            f"P{perm}"
                            f"_sx{sx:+d}"
                            f"_sy{sy:+d}"
                            f"_sz{sz:+d}"
                        )

                        candidates.append(
                            (name, R)
                        )

    print(
        f"Generated {len(candidates)} candidates"
    )

    return candidates

def make_calibration_pose(num_frames: int = 1) -> torch.Tensor:
    pose = torch.eye(3).repeat(num_frames, 24, 1, 1)

    left_arm_down = torch.tensor(
        Rotation.from_euler("z", -85, degrees=True).as_matrix(),
        dtype=torch.float32,
    )

    right_arm_down = torch.tensor(
        Rotation.from_euler("z", 85, degrees=True).as_matrix(),
        dtype=torch.float32,
    )

    left_elbow_bend = torch.tensor(
        Rotation.from_euler("y", -10, degrees=True).as_matrix(),
        dtype=torch.float32,
    )

    right_elbow_bend = torch.tensor(
        Rotation.from_euler("y", 10, degrees=True).as_matrix(),
        dtype=torch.float32,
    )

    pose[:, SMPLHJoint.LSHOULDER] = left_arm_down
    pose[:, SMPLHJoint.RSHOULDER] = right_arm_down

    pose[:, SMPLHJoint.LELBOW] = left_elbow_bend
    pose[:, SMPLHJoint.RELBOW] = right_elbow_bend

    return pose


def calibrate_rotations(
    ori,  # [frames,6,3,3]
    acc,
    global_rotations,
    P_IM,
):

    # calibration pose bone orientations
    g_rot = torch.tensor(global_rotations, dtype=torch.float32)
    for j in tracked_joints:
        print(j)
        print(
            Rotation.from_matrix(
                g_rot[0, j].numpy()
            ).as_euler("xyz", degrees=True)
        )
    for j in [
        SMPLHJoint.LSHOULDER,
        SMPLHJoint.LELBOW,
        SMPLHJoint.RSHOULDER,
        SMPLHJoint.RELBOW,
    ]:
        e = Rotation.from_matrix(
            g_rot[0, j].numpy()
        ).as_euler("xyz", degrees=True)

        print(j, e)
        R_M_bone = g_rot[0, tracked_joints]

    # average calibration orientation
    R_I_sensor = ori[0, :, :, :]

    # Eq. 15
    # Pelvis sensor rotation
    # P_IM = ori[0, PELVIS_IDX, :, :]

    # Eq. 18
    R_I_offset = R_I_sensor.inverse() @ P_IM @ R_M_bone

    a_s_sensor = acc[0, :, :]
    # print(a_s_sensor.shape)
    a_m = (P_IM.inverse().unsqueeze(0) @ R_I_sensor @ a_s_sensor.unsqueeze(-1)).squeeze(
        -1
    )
    # print(a_m.shape)
    # R_check = (
    #     P_IM.inverse().unsqueeze(0)
    #     @ R_I_sensor
    #     @ R_I_offset
    # )

    # for i in range(6):
    #     err = (
    #         R_check[i].T
    #         @ R_M_bone[i]
    #     )
    #     print(i)
    #     print(err)
    return (R_I_offset, a_m)


ACC_SCALE = 30.0  # DIP uses roughly this scale


def normalize_imu_data(
    imu_rotations,  # (F,N,3,3)
    imu_accelerations,  # (F,N,3)
    R_I_offset,  # (N,3,3)
    P_IM,
    root_sensor,
    acc_scale=30.0,
):
    #
    # Eq. 16
    #
    R_bone_M = P_IM.inverse().unsqueeze(0).unsqueeze(0) @ (
        imu_rotations @ R_I_offset.unsqueeze(0)
    )
    frame = 100

    for i in range(6):
        e = Rotation.from_matrix(R_bone_M[frame, i].cpu().numpy()).as_euler(
            "xyz", degrees=True
        )

        print(i, tracked_joints[i], e)

    #
    # Transform measured accelerations into model frame
    #
    a_M = (
        P_IM.inverse().unsqueeze(0).unsqueeze(0)
        @ (imu_rotations @ imu_accelerations.unsqueeze(-1))
    ).squeeze(-1)

    #
    # Eq. 22-25
    #
    R_root = R_bone_M[:, root_sensor]
    a_root = a_M[:, root_sensor]

    R_root_inv = R_root.transpose(-2, -1)

    a_norm = (
        R_root_inv.unsqueeze(1) @ (a_M - a_root.unsqueeze(1)).unsqueeze(-1)
    ).squeeze(-1)

    R_norm = R_root_inv.unsqueeze(1) @ R_bone_M

    #
    # Eq. 24
    #
    a_norm[:, root_sensor] = (R_root_inv @ a_root.unsqueeze(-1)).squeeze(-1)

    #
    # Eq. 25
    #
    R_norm[:, root_sensor] = R_root

    #
    # network scaling
    #
    # a_norm /= acc_scale

    return a_norm, R_norm


motion = "squats_deep"


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

    acc = torch.load(os.path.join(paths.example_dir, f"{motion}_accelerations.pt"))
    # acc = torch.load(os.path.join(paths.example_dir, f'acc.pt'))
    print(acc[0])
    print(acc.shape)
    ori = torch.load(os.path.join(paths.example_dir, f"{motion}_orientations.pt"))

    print(ori[0])
    print(ori.shape)
    parent = [
        -1,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        12,
        12,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
        20,
        21,
    ]
    pose = make_calibration_pose(1)
    tran = torch.tensor(
        [
            [0.0],  # x
            [0.0],  # y
            [0.0],  # z
        ]
    ).T
    print(pose.shape, tran.shape)
    global_rotations = art.math.spatial.forward_kinematics_R(pose, parent)
    print(global_rotations)
    candidates = generate_coordinate_frame_candidates()

    n_frames = 500

    R_vis = torch.tensor(
        Rotation.from_euler(
            "xy",
            [180, 180],
            degrees=True,
        ).as_matrix(),
        dtype=torch.float32,
    )
    for name, R_fix in candidates:

        print()
        print("=" * 80)
        print(name)
        print("=" * 80)

        #
        # Apply candidate transform to experimental IMU data
        #
        ori_test = ori @ R_fix

        #
        # Recompute calibration in transformed frame
        #
        P_IM_test = ori_test[0, PELVIS_IDX]

        R_I_offset, a_m = calibrate_rotations(
            ori_test,
            acc,
            global_rotations,
            P_IM_test,
        )

        #
        # Apply normal TransPose preprocessing
        #
        a_norm, R_norm = normalize_imu_data(
            ori_test,
            acc,
            R_I_offset,
            P_IM_test,
            PELVIS_IDX,
        )
        a_test = torch.zeros_like(a_norm)
        eulers = torch.tensor([
            Rotation.from_matrix(
                R_norm[f, PELVIS_IDX].cpu().numpy()
            ).as_euler("xyz", degrees=True)
            for f in range(len(R_norm))
        ])

        peak = eulers.abs().max(0).values

        print(
            name,
            "peakX", peak[0].item(),
            "peakY", peak[1].item(),
            "peakZ", peak[2].item(),
        )
        #
        # TransPose inference
        #
        x = normalize_and_concat(
            a_test[:n_frames],
            R_norm[:n_frames],
        ).to(device)

        pose_pred, tran_pred = net.forward_offline(x)

        #
        # Visualization correction only
        #
        pose_vis = pose_pred.clone()

        if tran_pred is not None:
            tran_vis = tran_pred.clone()

            tran_vis = (
                R_vis
                @ tran_vis.unsqueeze(-1)
            ).squeeze(-1)
        else:
            tran_vis = None

        pose_vis[:, 0] = (
            R_vis
            @ pose_vis[:, 0]
        )

        print(
            "root:",
            Rotation.from_matrix(
                R_norm[0, PELVIS_IDX].cpu().numpy()
            ).as_euler("xyz", degrees=True)
        )

        art.ParametricModel(
            paths.smpl_file
        ).view_motion(
            [pose_vis],
            [tran_vis],
        )


    with open("global_rotations.py", "w", encoding="utf-8") as f:
        f.write("global_rotations = ")
        f.write(repr(global_rotations.tolist()))
        f.write("\n")
    # r = Rotation.from_euler('z', 90, degrees=True)
    # rot_matrix_np = r.as_matrix()
    # P_IM = torch.tensor(rot_matrix_np, dtype=torch.float32)
    P_IM = torch.tensor(
        [
            [0, -1, 0],  # x
            [0, 0, 1],  # y
            [-1, 0, 0],  # z
        ],
        dtype=torch.float32,
    )
    # P_IM = torch.tensor([
    #     [ 0,  0,  1],
    #     [ 0,  1,  0],
    #     [-1,  0,  0],
    # ], dtype=torch.float32)
    P_IM = ori[0, PELVIS_IDX]
    print("P_IM", P_IM)
    (R_I_offset, a_m) = calibrate_rotations(ori, acc, pose, global_rotations, P_IM)
    print("R_I_offset", R_I_offset)
    print("a_m", a_m)
    a_norm, R_norm = normalize_imu_data(ori, acc, R_I_offset, P_IM, PELVIS_IDX)
    for frame in range(0, len(R_norm), 100):
        e = Rotation.from_matrix(
            R_norm[frame, PELVIS_IDX].cpu().numpy()
        ).as_euler("xyz", degrees=True)

        print(frame, e)
    print(
        Rotation.from_matrix(R_norm[0, PELVIS_IDX].cpu().numpy()).as_euler(
            "xyz", degrees=True
        )
    )
    for i in range(6):
        e = Rotation.from_matrix(R_norm[100, i].cpu().numpy()).as_euler(
            "xyz", degrees=True
        )

        print(i, e)
    a_test = torch.zeros_like(a_norm)
    # R_fix = torch.tensor(
    #     [
    #         [0,-1,0],
    #         [1, 0,0],
    #         [0, 0,1],
    #     ],
    #     dtype=torch.float32,
    # )
    # R_fix = torch.tensor(
    #     Rotation.from_euler(
    #         "xy",
    #         [180, 180],
    #         degrees=True,
    #     ).as_matrix(),
    #     dtype=torch.float32,
    # )
    # ori_test = ori @ R_fix.T
    x = normalize_and_concat(acc, ori).to(device)
    pose, tran = net.forward_offline(x)
    art.ParametricModel(paths.smpl_file).view_motion([pose], [tran])
    # Visualize calibration pose
    pose = make_calibration_pose(1000)
    art.ParametricModel(paths.smpl_file).view_motion([pose], [tran])


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    duration = time.perf_counter() - start
    logger.info("Finished in %.2f seconds", duration)
