import torch
import torch.nn.functional as F
import time
import os
import math
import shutil
import os.path as osp
import matplotlib.pyplot as plt
import torchvision
from collections import OrderedDict
import pandas as pd
from fvcore.nn import FlopCountAnalysis
from torchvision import transforms
from PIL import Image

from models.egformer import EGDepthModel

import matplotlib as mpl
import matplotlib.cm as cm
import argparse
import importlib
import numpy as np

# parser = argparse.ArgumentParser()
# parser.add_argument("--method",
#                     type=str,
#                     help="Method to be evaluated",
#                     default="EGformer")
#
# parser.add_argument("--eval_data",
#                     type=str,
#                     help="data category to be evaluated",
#                     default="Inference")
#
# parser.add_argument('--num_workers', type=int, default=1)
#
# parser.add_argument('--checkpoint_path', type=str, default='../pretrained_models/EGformer_pretrained.pkl')
#
# ############ Distributed Data Parallel (DDP) ############
# parser.add_argument('--world_size', type=int, default=1)
# parser.add_argument('--rank', type=int, default=0)
# parser.add_argument('--multiprocessing_distributed', default=True)
# parser.add_argument('--dist-backend', type=str, default="nccl")
# parser.add_argument('--dist-url', type=str, default="tcp://127.0.0.1:7777")
#
# config = parser.parse_args()
#
# torch.distributed.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
#                                      world_size=config.world_size, rank=config.rank)
#
#
# device = 'cuda:0'
#
# net = EGDepthModel(hybrid=False)
# net = net.to(device)
# net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[0], find_unused_parameters=True)
# net.load_state_dict(torch.load(config.checkpoint_path), strict=False)
# net.eval()
# #
# #
transform = transforms.Compose([transforms.ToTensor()])
# img_path = 'rgb_0.png'
# img = np.array(Image.open(img_path).convert('RGB'))
# print(img.shape)
#img = transform(img)
#img = torch.unsqueeze(img, 0)  # 在第一个维度上增加一个维度

#img = np.expand_dims(img, axis=0)  # 假设你想在第一个维度上添加一个维度
def depth_est(img, net):
    with torch.no_grad():
        #trans转换，模型预测，得到输出，转换为cpu
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        inputs = img.float().cuda()
        features = net(inputs)
        output = features

        disp_pp = output
        disp_pp = disp_pp.cpu().detach().numpy()
        disp_pp = disp_pp.squeeze()

        # vmax = np.percentile(disp_pp, 95)
        # normalizer = mpl.colors.Normalize(vmin=disp_pp.min(), vmax=vmax)
        # mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
        # disp_pp = (mapper.to_rgba(disp_pp)[:, :, :3] * 255).astype(np.uint8)
        #
        # plt.imsave('out.png', disp_pp, cmap='viridis')

        # print(disp_pp.max(), disp_pp.min())
        #
        ## 将像素值归一化到 0-255 范围
        # disp_pp_normalized = ((disp_pp - disp_pp.min()) / (disp_pp.max() - disp_pp.min()) * 255).astype('uint8')
        # img = Image.fromarray(disp_pp_color)

        return disp_pp

#depth_est(img, net)