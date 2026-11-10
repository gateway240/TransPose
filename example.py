r"""
    Test the system with an example IMU measurement sequence.
"""
from scipy.spatial.transform import Rotation
from articulate import SMPLHJoint


import torch
from net import TransPoseNet
from config import paths
from utils import normalize_and_concat
import os
import articulate as art

motion = 'squats_deep'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = TransPoseNet().to(device)
# acc = torch.load(os.path.join(paths.example_dir, f'{motion}_accelerations.pt'))
acc = torch.load(os.path.join(paths.example_dir, f'acc.pt'))
print(acc[0])
print(acc.shape)
# ori = torch.load(os.path.join(paths.example_dir, f'{motion}_orientations.pt'))
ori = torch.load(os.path.join(paths.example_dir, f'ori.pt'))
print(ori[0])
print(ori.shape)
# acc = torch.zeros_like(acc)
x = normalize_and_concat(acc, ori).to(device)
pose, tran = net.forward_offline(x)     # offline
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
