r"""
Test the system with an example IMU measurement sequence.
"""

from torch._tensor import Tensor

import itertools

import logging
import os
import time

import torch
from numpy.linalg import inv
from scipy.spatial.transform import Rotation

import articulate as art
from articulate import SMPLJoint
from config import paths
from net import TransPoseNet
from utils import normalize_and_concat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tracked_joints = [
    SMPLJoint.LELBOW,
    SMPLJoint.RELBOW,
    SMPLJoint.LKNEE,
    SMPLJoint.RKNEE,
    SMPLJoint.HEAD,
    SMPLJoint.PELVIS,
]
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
                        name = f"P{perm}_sx{sx:+d}_sy{sy:+d}_sz{sz:+d}"

                        candidates.append((name, R))

    print(f"Generated {len(candidates)} candidates")

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

    pose[:, SMPLJoint.LSHOULDER] = left_arm_down
    pose[:, SMPLJoint.RSHOULDER] = right_arm_down

    pose[:, SMPLJoint.LELBOW] = left_elbow_bend
    pose[:, SMPLJoint.RELBOW] = right_elbow_bend

    # left_wrist_bend = torch.tensor(
    #     Rotation.from_euler("x", 30, degrees=True).as_matrix(),
    #     dtype=torch.float32,
    # )

    # right_wrist_bend = torch.tensor(
    #     Rotation.from_euler("x", -30, degrees=True).as_matrix(),
    #     dtype=torch.float32,
    # )

    # pose[:, SMPLJoint.LWRIST] = left_wrist_bend
    # pose[:, SMPLJoint.RWRIST] = right_wrist_bend

    return pose


# def calibrate_rotations(
#     ori,  # [frames,6,3,3]
#     acc,
#     global_rotations,
#     P_IM,
# ):

#     # calibration pose bone orientations
#     # g_rot = global_rotations
#     # R_M_bone = g_rot[0, tracked_joints]

#     # pose = torch.eye(3).repeat(1, 24, 1, 1)
#     pose = torch.eye(3).repeat(1, 24, 1, 1)

#     # global_rotations = art.math.spatial.forward_kinematics_R(
#     #     pose, parent
#     # )


#     global_rotations = art.math.spatial.forward_kinematics_R(
#         pose, parent
#     )
#     for i in range(24):
#         print(
#             i,
#             "parent:", parent[i],
#             "global rotation:",
#             global_rotations[0, i],
#         )
#     for i, joint in enumerate(tracked_joints):
#         idx = int(joint)

#         print(
#             f"IMU {i}: {joint}",
#             f"index={idx}",
#             f"parent={parent[idx]}",
#         )
#     R_M_bone = global_rotations[0, tracked_joints]
#     # average calibration orientation
#     R_I_sensor = ori[0, :, :, :]

#     # Eq. 15
#     # Pelvis sensor rotation
#     # P_IM = ori[0, PELVIS_IDX, :, :]

#     # Eq. 18
#     R_I_offset = R_I_sensor.inverse() @ P_IM @ R_M_bone

#     a_s_sensor = acc[0, :, :]
#     # print(a_s_sensor.shape)
#     a_m = (P_IM.inverse().unsqueeze(0) @ R_I_sensor @ a_s_sensor.unsqueeze(-1)).squeeze(
#         -1
#     )
#     # print(a_m.shape)
#     # R_check = (
#     #     P_IM.inverse().unsqueeze(0)
#     #     @ R_I_sensor
#     #     @ R_I_offset
#     # )


#     # for i in range(6):
#     #     err = (
#     #         R_check[i].T
#     #         @ R_M_bone[i]
#     #     )
#     #     print(i)
#     #     print(err)
#     return (R_I_offset, a_m)
def calibrate_rotations(
    ori,
    acc,
    global_rotations,
    P_IM,
):
    R_M_bone = global_rotations[0, tracked_joints]
    R_I_sensor = ori[0]

    R_I_offset = R_I_sensor.transpose(-1, -2) @ P_IM @ R_M_bone
    R_expected = global_rotations[0, tracked_joints]

    R_reconstructed = P_IM.T @ ori[0] @ R_I_offset

    print((R_reconstructed - R_expected).abs().amax(dim=(-2, -1)))

    a_m = (P_IM.transpose(-1, -2) @ R_I_sensor @ acc[0].unsqueeze(-1)).squeeze(-1)

    return R_I_offset, a_m


ACC_SCALE = 30.0  # DIP uses roughly this scale


def normalize_imu_data(
    ori,  # [frames,6,3,3]
    acc,
    R_I_offset,  # (N,3,3)
    a_m,
    P_IM,
    root_sensor,
):

    # Eq 22
    # R_root = ori[:, root_sensor, :, :]
    # a_root = acc[:, root_sensor, :]

    # R_root_inv = R_root.inverse()

    # R_bone_M = (
    #     P_IM.inverse().unsqueeze(0).unsqueeze(0)
    #     @ (
    #         ori @ R_I_offset.unsqueeze(0)
    #     )
    # )
    # a_m_bone = P_IM.inverse().unsqueeze(0).unsqueeze(0) @ (
    #     acc @ a_M.unsqueeze(0)
    # )
    # a_norm = (
    #     R_root_inv.unsqueeze(1) @ (a_m_bone - a_root.unsqueeze(1)).unsqueeze(-1)
    # ).squeeze(-1)

    # R_norm = R_root_inv.unsqueeze(1) @ R_bone_M

    # #
    # # Eq. 24
    # #
    # a_norm[:, root_sensor] = (R_root_inv @ a_root.unsqueeze(-1)).squeeze(-1)

    # #
    # # Eq. 25
    # #
    # R_norm[:, root_sensor] = R_root
    R_bone_M = P_IM.inverse().unsqueeze(0).unsqueeze(0) @ (
        ori @ R_I_offset.unsqueeze(0)
    )

    a_M = (
        P_IM.inverse().unsqueeze(0).unsqueeze(0) @ (ori @ acc.unsqueeze(-1))
    ).squeeze(-1)

    a_M = a_M - a_m.unsqueeze(0)

    R_root = R_bone_M[:, root_sensor]
    R_root_inv = R_root.transpose(-2, -1)

    a_root = a_M[:, root_sensor]

    a_norm = (
        R_root_inv.unsqueeze(1) @ (a_M - a_root.unsqueeze(1)).unsqueeze(-1)
    ).squeeze(-1)

    # Equation 24: root acceleration
    a_norm[:, root_sensor] = (R_root_inv @ a_root.unsqueeze(-1)).squeeze(-1)

    R_norm = R_root_inv.unsqueeze(1) @ R_bone_M

    # Equation 25: root orientation
    R_norm[:, root_sensor] = R_root
    #
    # network scaling
    #
    # a_norm /= acc_scale

    return a_norm, R_norm


# motion = "squats_deep"
# frame_start = 0
# frame_end = 400

motion = "walking"
frame_start = 500
frame_end = 800

# motion = "jogging"
# frame_start = 100
# frame_end = 1000

# motion = "half_jacks"
# frame_start = 0
# frame_end = 400

# motion = "curls"
# frame_start = 0
# frame_end = 1000

import torch


def compute_heading_correction(
    ori,  # [frames, sensors, 3, 3]
    base_sensor=5,  # Pelvis
    base_heading_axis=(1.0, 0.0, 0.0),
    desired_heading=(1.0, 0.0, 0.0),
    up_axis=(0.0, 0.0, 1.0),
):
    """
    Compute heading correction around Xsens global +Z.

    Assumptions:
      - ori maps sensor coordinates to Xsens global coordinates.
      - Xsens frame is North-West-Up.
      - Pelvis +X points forward.
      - Desired heading is magnetic North (+X).
    """

    device = ori.device
    dtype = ori.dtype

    up = torch.tensor(up_axis, device=device, dtype=dtype)
    up = up / torch.linalg.norm(up)

    sensor_forward = torch.tensor(base_heading_axis, device=device, dtype=dtype)

    target_forward = torch.tensor(desired_heading, device=device, dtype=dtype)

    # First pelvis orientation.
    R_base = ori[0, base_sensor]

    # Pelvis forward direction expressed in global coordinates.
    heading = R_base @ sensor_forward

    # Project both directions onto the horizontal plane.
    heading = heading - torch.dot(heading, up) * up
    target_forward = target_forward - torch.dot(target_forward, up) * up

    heading = heading / torch.linalg.norm(heading)
    target_forward = target_forward / torch.linalg.norm(target_forward)

    # Signed angle from measured heading to desired heading.
    cross = torch.linalg.cross(heading, target_forward)

    sin_angle = torch.dot(up, cross)
    cos_angle = torch.dot(heading, target_forward)

    angle = torch.atan2(sin_angle, cos_angle)

    # Rotation around global vertical axis.
    K = torch.tensor(
        [
            [0.0, -up[2], up[1]],
            [up[2], 0.0, -up[0]],
            [-up[1], up[0], 0.0],
        ],
        device=device,
        dtype=dtype,
    )

    I = torch.eye(3, device=device, dtype=dtype)

    C = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

    return angle, C


def debug_calibration(
    ori,
    global_rotations,
    tracked_joints,
    R_I_offset,
    P_IM,
):
    R_expected = global_rotations[0, tracked_joints]

    # Same equation used in calibration:
    R_offset_check = ori[0].inverse() @ P_IM @ R_expected

    print("Offset calculation consistency:")
    print(
        (R_offset_check.transpose(-1, -2) @ R_I_offset)
        .diagonal(dim1=-2, dim2=-1)
        .sum(-1)
    )

    # Runtime reconstruction:
    R_reconstructed = P_IM.inverse() @ ori[0] @ R_I_offset

    # Relative rotation:
    R_error = R_expected.transpose(-1, -2) @ R_reconstructed

    cos_angle = ((R_error.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)

    angle = torch.rad2deg(torch.acos(cos_angle))

    print("Reconstruction errors (degrees):")
    for i, angle_i in enumerate(angle):
        print(f"IMU {i}: {angle_i.item():.8f}")


def main() -> None:
    logger.info("Starting calibration!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = TransPoseNet().to(device)

    # acc = torch.load(os.path.join(paths.example_dir, f"{motion}_accelerations.pt"))
    # acc = torch.load(os.path.join(paths.example_dir, f'acc.pt'))
    # print(acc[0])
    # print(acc.shape)
    # ori = torch.load(os.path.join(paths.example_dir, f"{motion}_orientations.pt"))
    # angle, C_heading = compute_heading_correction(
    #     ori,
    #     base_sensor=PELVIS_IDX,
    # )

    # print("Heading correction:", torch.rad2deg(angle).item())
    # print("Correction matrix:", C_heading)

    # # Apply global heading correction to all IMU orientations.
    # ori = C_heading @ ori
    # print(ori[0])
    # print(ori.shape)
    # parent = [
    #     -1,
    #     0,
    #     0,
    #     0,
    #     1,
    #     2,
    #     3,
    #     4,
    #     5,
    #     6,
    #     7,
    #     8,
    #     9,
    #     12,
    #     12,
    #     12,
    #     13,
    #     14,
    #     16,
    #     17,
    #     18,
    #     19,
    #     20,
    #     21,
    # ]
    # parent = [-1 for i in range(len(parent))]
    pose = make_calibration_pose(1)
    tran_0 = torch.tensor(
        [
            [0.0],  # x
            [0.0],  # y
            [0.0],  # z
        ]
    ).T
    print(pose.shape, tran_0.shape)
    global_rotations = art.math.spatial.forward_kinematics_R(pose, parent)
    print(global_rotations)

    candidates = generate_coordinate_frame_candidates()
    R_vis = torch.tensor(
        Rotation.from_euler(
            "xy",
            [180, 180],
            degrees=True,
        ).as_matrix(),
        dtype=torch.float32,
    )
    # print("Candidates", candidates)
    # for name, R_fix in candidates:
    #     print()
    #     print("=" * 80)
    #     print(name)
    #     print("=" * 80)

    #     #
    #     # Apply candidate transform to experimental IMU data
    #     #
    #     ori_test = ori @ R_fix
    #     acc_test = acc @ R_fix
    #     # ori_test = ori
    #     # acc_test = acc
    #     #
    #     # Recompute calibration in transformed frame
    #     #
    #     P_IM_test = ori_test[0, PELVIS_IDX]
    #     # P_IM_test = R_fix
    #     R_I_offset, a_m = calibrate_rotations(
    #         ori_test,
    #         acc_test,
    #         global_rotations,
    #         P_IM_test,
    #     )

    #     #
    #     # Apply normal TransPose preprocessing
    #     #
    #     a_norm, R_norm = normalize_imu_data(
    #         ori_test,
    #         acc_test,
    #         R_I_offset,
    #         a_m,
    #         P_IM_test,
    #         PELVIS_IDX,
    #     )
    #     a_test = torch.zeros_like(a_norm)
    #     eulers = torch.tensor(
    #         [
    #             Rotation.from_matrix(R_norm[f, PELVIS_IDX].cpu().numpy()).as_euler(
    #                 "xyz", degrees=True
    #             )
    #             for f in range(len(R_norm))
    #         ]
    #     )

    #     peak: Tensor = eulers.abs().max(0).values

    #     print(
    #         name,
    #         "peakX",
    #         peak[0].item(),
    #         "peakY",
    #         peak[1].item(),
    #         "peakZ",
    #         peak[2].item(),
    #     )
    #     #
    #     # TransPose inference
    #     #
    #     x = normalize_and_concat(
    #         a_norm[frame_start:frame_end],
    #         R_norm[frame_start:frame_end],
    #     ).to(device)

    #     pose_pred, tran_pred = net.forward_offline(x)

    #     #
    #     # Visualization correction only
    #     #
    #     pose_vis = pose_pred.clone()

    #     if tran_pred is not None:
    #         tran_vis = tran_pred.clone()

    #         tran_vis = (R_vis @ tran_vis.unsqueeze(-1)).squeeze(-1)
    #     else:
    #         tran_vis = None

    #     pose_vis[:, 0] = R_vis @ pose_vis[:, 0]

    #     print(
    #         "root:",
    #         Rotation.from_matrix(R_norm[0, PELVIS_IDX].cpu().numpy()).as_euler(
    #             "xyz", degrees=True
    #         ),
    #     )

    #     art.ParametricModel(paths.smpl_file).view_motion(
    #         [pose_vis],
    #         [tran_vis],
    #     )

    with open("global_rotations.py", "w", encoding="utf-8") as f:
        f.write("global_rotations = ")
        f.write(repr(global_rotations.tolist()))
        f.write("\n")

    # R_fix = torch.tensor(
    #     [
    #         [-1, 0, 0],
    #         [0, -1, 0],
    #         [0, 0, 1],
    #     ],
    #     dtype=torch.float32,
    # )
    # R_fix = torch.tensor(
    #     [
    #         [1, 0, 0],
    #         [0, 0, -1],
    #         [0, 1, 0],
    #     ],
    #     dtype=torch.float32,
    # )
    R_fix = torch.tensor(
        [
            [-1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ],
        dtype=torch.float32,
    )
    # R_fix = torch.tensor(
    #     [
    #         [0, 1, 0],
    #         [0, 0, -1],
    #         [-1, 0, 0],
    #     ],
    #     dtype=torch.float32,
    # )
    #     R_fix = torch.tensor(
    #     [
    #         [1, 0,  0],
    #         [0, 0, -1],
    #         [0, 1,  0],
    #     ],
    #     dtype=torch.float32,
    # )
    # ori = ori @ R_fix
    # acc = acc @ R_fix

    # P_IM = ori[0, PELVIS_IDX]
    #     P_IM = torch.tensor(
    #     [
    #         [0, 0, 1],
    #         [1, 0, 0],
    #         [0, 1, 0],
    #     ],
    #     dtype=torch.float32,
    # )
    # P_IM = torch.tensor([
    #     [1,  0,  0],
    #     [0, -1,  0],
    #     [0,  0, -1],
    # ], dtype=torch.float32)
    #     P_IM = torch.tensor([
    #     [ 0,  0, -1],
    #     [ 1,  0,  0],
    #     [ 0, -1,  0],
    # ], dtype=torch.float32)
    # P_IM = torch.tensor([
    #     [ 0,  0, -1],
    #     [ 0, -1,  0],
    #     [-1,  0,  0],
    # ], dtype=torch.float32)
    # print("P_IM:")
    # print(P_IM)

    # print("Determinant:", torch.det(P_IM).item())
    # print("Orthogonality error:", torch.norm(P_IM.T @ P_IM - torch.eye(3)).item())

    # R_I_offset, a_m = calibrate_rotations(ori, acc, global_rotations, P_IM)

    # debug_calibration(
    #     ori,
    #     global_rotations,
    #     tracked_joints,
    #     R_I_offset,
    #     P_IM,
    # )
    # print("a_m", a_m)
    # a_norm, R_norm = normalize_imu_data(ori, acc, R_I_offset, a_m, P_IM, PELVIS_IDX)
    # a_test = torch.zeros_like(a_norm)
    # x = normalize_and_concat(
    #     a_norm[frame_start:frame_end], R_norm[frame_start:frame_end]
    # ).to(device)
    # pose, tran = net.forward_offline(x)
    # pose_vis = pose.clone()
    # tran_vis = tran.clone()
    # pose_vis[:, 0] = R_vis @ pose_vis[:, 0]
    # tran_vis = (R_vis @ tran_vis.unsqueeze(-1)).squeeze(-1)
    # art.ParametricModel(paths.smpl_file).view_motion([pose_vis], [tran_vis])
    # Visualize calibration pose
    pose = make_calibration_pose(1000)
    pose_vis = pose.clone()
    pose_vis[:, 0] = R_vis @ pose_vis[:, 0]
    art.ParametricModel(paths.smpl_file).view_motion([pose_vis], [tran_0])


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    duration = time.perf_counter() - start
    logger.info("Finished in %.2f seconds", duration)
