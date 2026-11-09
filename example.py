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
x = normalize_and_concat(acc, ori).to(device)
pose, tran = net.forward_offline(x)     # offline
# print(pose.shape, tran.shape)
# print(pose[0])
# pose, tran = [torch.stack(_) for _ in zip(*[net.forward_online(f) for f in x])]   # online
art.ParametricModel(paths.smpl_file).view_motion([pose], [tran])
