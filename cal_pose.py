r"""
    Test the system with an example IMU measurement sequence.
"""
from articulate import SMPLHJoint


import torch
from net import TransPoseNet
from config import paths
from utils import normalize_and_concat
import os
import articulate as art
from scipy.spatial.transform import Rotation

tracked_joints = [
    SMPLHJoint.PELVIS,
    SMPLHJoint.LKNEE,
    SMPLHJoint.RKNEE,
    SMPLHJoint.HEAD,
    SMPLHJoint.LELBOW,
    SMPLHJoint.RELBOW,
]

def make_calibration_pose(num_frames: int=1) -> torch.Tensor:
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

pose = make_calibration_pose(1)
tran = torch.tensor([
    [0.0],  # x
    [0.0],  # y
    [0.0],  # z
]).T
print(pose.shape,tran.shape)
parent = [
    -1,
     0, 0, 0,
     1, 2, 3,
     4, 5, 6,
     7, 8, 9,
    12,12,12,
    13,14,
    16,17,
    18,19,
    20,21,
]
global_rotations = art.math.spatial.forward_kinematics_R(pose, parent)
print(global_rotations)
with open("global_rotations.py", "w", encoding="utf-8") as f:
    f.write("global_rotations = ")
    f.write(repr(global_rotations.tolist()))
    f.write("\n")
pose = make_calibration_pose(1000)
art.ParametricModel(paths.smpl_file).view_motion([pose], [tran])