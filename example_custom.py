r"""
Test the system with an example IMU measurement sequence.
"""

import itertools
from articulate.math import quaternion_to_rotation_matrix
from scipy.spatial.transform import Rotation
from articulate import SMPLJoint


import torch
from net import TransPoseNet
from config import paths
from utils import normalize_and_concat
import os
import articulate as art

# motion = 'squats_deep'
# frame_start = 0
# frame_end = 400

motion = "walking"
frame_start = 400
frame_end = 1000

# motion = "half_jacks"
# frame_start = 0
# frame_end = 400

PELVIS_IDX = 5
CAL_FRAMES = 60  # 1 sec at 60 FPS

tracked_joints = [
    SMPLJoint.LELBOW,
    SMPLJoint.RELBOW,
    SMPLJoint.LKNEE,
    SMPLJoint.RKNEE,
    SMPLJoint.HEAD,
    SMPLJoint.PELVIS,
]


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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = TransPoseNet().to(device)
acc_raw = torch.load(os.path.join(paths.example_dir, f"{motion}_accelerations.pt"))
print(acc_raw[0])
print(acc_raw.shape)
ori_raw = torch.load(os.path.join(paths.example_dir, f"{motion}_orientations.pt"))
print(ori_raw[0])
print(ori_raw.shape)
acc_raw = acc_raw[frame_start:frame_end]
ori_raw = ori_raw[frame_start:frame_end]

num_frames = ori_raw.shape[0]


pose_cal = make_calibration_pose(1)[0]  # (24,3,3)
R_bone_cal = pose_cal[tracked_joints]  # (6,3,3)
candidates = generate_coordinate_frame_candidates()
# for name, R_fix in candidates:
#     print()
#     print("=" * 80)
#     print(name)
#     print("=" * 80)

#
# Apply candidate transform to experimental IMU data
#
# ori_test = ori @ R_fix
# acc_test = acc @ R_fix
# R_fix = torch.tensor(
#     [
#         [-1, 0, 0],
#         [0, 0, 1],
#         [0, 1, 0],
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
# All data corrected to new basis

# R_fix = torch.tensor(
#     [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
#     dtype=torch.float32,
# )
# R_fix = torch.tensor(
#     [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
#     dtype=torch.float32,
# )
# R_fix = torch.tensor(
#     [[-1, 0, 0], [0, 0, -1], [0, -1, 0]],
#     dtype=torch.float32,
# )
# Good for walking
R_fix = torch.tensor(
    [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
    dtype=torch.float32,
)

# R_fix = torch.tensor(
#     [[-1,  0,  0],
#  [ 0,  0, -1],
#  [ 0, -1,  0]],
#     dtype=torch.float32,
# )
ori = torch.stack(
    [
        quaternion_to_rotation_matrix(ori_raw[i])  # (6, 4) -> (6, 3, 3)
        for i in range(num_frames)
    ],
    dim=0,
)
ori = ori @ R_fix
acc_raw = acc_raw @ R_fix


# Calibration frame
ori_cal = ori_raw[:CAL_FRAMES].mean(dim=0)
acc_cal = acc_raw[:CAL_FRAMES].mean(dim=0)
R_ori_cal = quaternion_to_rotation_matrix(ori_cal.reshape(-1, 4)).view(6, 3, 3)
R_ori_cal = R_ori_cal @ R_fix
smpl2imu = R_ori_cal[PELVIS_IDX].T
print(smpl2imu)

device2bone = smpl2imu.matmul(R_ori_cal).transpose(1, 2).matmul(R_bone_cal)
acc_offsets = smpl2imu.matmul(
    acc_cal.unsqueeze(-1)
)  # [num_imus, 3, 1], already in global inertial frame

print(device2bone.shape)
print(acc_offsets.shape)

acc_in = (smpl2imu.matmul(acc_raw.unsqueeze(-1)) - acc_offsets.unsqueeze(0)).squeeze(-1)
ori_in = smpl2imu.matmul(ori).matmul(device2bone)
# acc_in = torch.zeros_like(acc_raw)
x = normalize_and_concat(acc_in, ori_in).to(device)
pose, tran = net.forward_offline(x)  # offline
R_vis = torch.tensor(
    Rotation.from_euler(
        "xy",
        [180, 0],
        degrees=True,
    ).as_matrix(),
    dtype=torch.float32,
)
pose_vis = pose.clone()
tran_vis = tran.clone()
pose_vis[:, 0] = R_vis @ pose_vis[:, 0]
tran_vis = (R_vis @ tran_vis.unsqueeze(-1)).squeeze(-1)
# print(pose.shape, tran.shape)
# print(pose[0])
# pose, tran = [torch.stack(_) for _ in zip(*[net.forward_online(f) for f in x])]   # online
art.ParametricModel(paths.smpl_file).view_motion([pose_vis], [tran_vis])
