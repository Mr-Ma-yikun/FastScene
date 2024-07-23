import cv2
import os
import numpy as np
from PIL import Image
import torch
import imageio as io
import json
from scipy.interpolate import griddata
from utils.option import args
from models.egformer import EGDepthModel

import argparse
import importlib
from inpaint import inpaint
from evaluate.depest import depth_est

#--------sr
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

device = 'cuda:0'
rgb = 'inpaint_data/demo.png'

mov = 0.02
step = 12
min = 6

#----------------------------------------depth estimation-EGformer
parser = argparse.ArgumentParser()
parser.add_argument("--method",
                    type=str,
                    help="Method to be evaluated",
                    default="EGformer")

parser.add_argument("--eval_data",
                    type=str,
                    help="data category to be evaluated",
                    default="Inference")

parser.add_argument('--num_workers', type=int, default=1)

parser.add_argument('--checkpoint_path', type=str, default='pretrained_models/EGformer_pretrained.pkl')

parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--multiprocessing_distributed', default=True)
parser.add_argument('--dist-backend', type=str, default="nccl")
parser.add_argument('--dist-url', type=str, default="tcp://127.0.0.1:7777")

config = parser.parse_args()

torch.distributed.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                     world_size=config.world_size, rank=config.rank)

net = EGDepthModel(hybrid=False)
net = net.to(device)
net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[0], find_unused_parameters=True)
net.load_state_dict(torch.load(config.checkpoint_path), strict=False)
net.eval()

#-------------------Inpainting AOT-GAN

net_inpa = importlib.import_module('model.' + args.model)
in_model = net_inpa.InpaintGenerator(args).to(device)
in_model.load_state_dict(torch.load(args.pre_train, map_location=device))
in_model.eval()


#-------------------SR real-GAN
model_path = r'models/RealESRGAN_x2plus.pth'
dni_weight = None
sr_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2)

upsampler = RealESRGANer(
            scale=2,
            model_path=model_path,
            dni_weight=dni_weight,
            model=sr_model,
            tile=384,
            tile_pad=20,
            pre_pad=20,
            half=False,
            device=device,
        )

def inpaint_image(mask_path, rgb_path):

    mask = mask_path
    rgb = rgb_path
    inpainted = inpaint(mask, rgb, in_model, upsampler)

    return inpainted

def depth_completion(input_rgb):

    est_depth = depth_est(input_rgb, net)

    return est_depth


def translate(crd, rgb, d, cam=[]):
    H, W = rgb.shape[0], rgb.shape[1]
    d = np.where(d == 0, -1, d)

    tmp_coord = crd - cam
    new_d = np.sqrt(np.sum(np.square(tmp_coord), axis=2))
    new_coord = tmp_coord / new_d.reshape(H, W, 1)

    new_depth = np.zeros(new_d.shape)

    [x, y, z] = new_coord[..., 0], new_coord[..., 1], new_coord[..., 2]

    idx = np.where(new_d > 0)

    theta = np.zeros(y.shape)
    phi = np.zeros(y.shape)
    x1 = np.zeros(z.shape)
    y1 = np.zeros(z.shape)

    theta[idx] = np.arctan2(y[idx], np.sqrt(np.square(x[idx]) + np.square(z[idx])))
    phi[idx] = np.arctan2(-z[idx], x[idx])

    x1[idx] = (0.5 - theta[idx] / np.pi) * H  # - 0.5  # (1 - np.sin(theta[idx]))*H/2 - 0.5
    y1[idx] = (0.5 - phi[idx] / (2 * np.pi)) * W  # - 0.5

    x, y = np.floor(x1).astype('int'), np.floor(y1).astype('int')

    img = np.zeros(rgb.shape)

    mask = (new_d > 0) & (H > x) & (x > 0) & (W > y) & (y > 0)

    # (522270,)
    x = x[mask]
    y = y[mask]
    new_d = new_d[mask]
    rgb = rgb[mask]

    reorder = np.argsort(-new_d)
    x = x[reorder]
    y = y[reorder]
    new_d = new_d[reorder]
    rgb = rgb[reorder]

    # Assign
    new_depth[x, y] = new_d
    img[x, y] = rgb

    mask = (new_depth != 0).astype(int)

    mask_index = np.argwhere(mask == 0)
    return img, new_depth.reshape(H, W, 1), tmp_coord, cam.reshape(1, 1, 3), mask, mask_index



def generate(input_rgb, input_depth, flag, dir, first):

    H, W = 512, 1024

    if first == True:
        rgb = input_rgb
        d = input_depth
    elif first == False: 
        rgb = input_rgb
        d = input_depth

    d_max = np.max(d)
    d = d / d_max
    d = d.reshape(rgb.shape[0], rgb.shape[1], 1) #/ np.max(d)
    d = np.where(d == 0, 1, d)

    _y = np.repeat(np.array(range(W)).reshape(1, W), H, axis=0)
    _x = np.repeat(np.array(range(H)).reshape(1, H), W, axis=0).T

    _theta = (1 - 2 * (_x) / H) * np.pi / 2  # latitude
    _phi = 2 * np.pi * (0.5 - (_y) / W)  # longtitude

    axis0 = (np.cos(_theta) * np.cos(_phi)).reshape(H, W, 1)
    axis1 = np.sin(_theta).reshape(H, W, 1)
    axis2 = (-np.cos(_theta) * np.sin(_phi)).reshape(H, W, 1)
    coord = np.concatenate((axis0, axis1, axis2), axis=2) * d

    cam_pos = []

    if dir == 'x': 
        pos_x = np.array([mov * flag, 0, 0])
        cam_pos = pos_x

    elif dir == 'z': 
        pos_z = np.array([0, 0, mov * flag])
        cam_pos = pos_z

    elif dir == 'xz':
        pos_xz = np.array([mov * flag, 0, mov * flag])
        cam_pos = pos_xz

    elif dir == '-xz': 
        pos__xz = np.array([mov * flag, 0, -mov * flag])
        cam_pos = pos__xz

    img1, d1, _, _, mask1, mask_index = translate(coord, rgb, d, cam_pos) 
    d1 = np.squeeze(d1, axis=-1)  
    d1 = np.stack((d1, d1, d1), axis=-1) 


    mask = np.uint8(mask1 * 255)
    img = np.uint8(img1)

    img[mask == 0] = 255
    mask = cv2.bitwise_not(mask)
    depth = d1

    return mask, img, depth[:, :, 0], mask_index


def progressive_inpaint(ori_rgb, ori_depth):
    num_inpaint = 0

    input_rgb = ori_rgb
    depth = ori_depth
    flag = 1
    for i in range(0, step):
        if i == min: 
            flag = -1
            input_rgb = ori_rgb
            depth = ori_depth
        if i == 0 or i == min:
            mask, img, depth, mask_index = generate(input_rgb, depth, flag, dir='x', first=True)
        else:
            mask, img, depth, mask_index = generate(input_rgb, depth, flag, dir='x', first=False)

        input_rgb = inpaint_image(mask, img)
        depth = depth_completion(input_rgb)

        Image.fromarray(input_rgb).save('Pano_inpaint/rgb_' + str(num_inpaint) + '.png')

        num_inpaint = num_inpaint + 1
        print('num_image', num_inpaint)

    flag = 1
    input_rgb = ori_rgb
    depth = ori_depth
    for i in range(0, step):
        if i == min:
            input_rgb = ori_rgb
            depth = ori_depth
            flag = -1
        if i == 0 or i == min:
            mask, img, depth, mask_index = generate(input_rgb, depth, flag, dir='z', first=True)
        else:
            mask, img, depth, mask_index = generate(input_rgb, depth, flag, dir='z', first=False)
        input_rgb = inpaint_image(mask, img)
        depth = depth_completion(input_rgb)

        Image.fromarray(input_rgb).save('Pano_inpaint/rgb_' + str(num_inpaint) + '.png')

        num_inpaint = num_inpaint + 1
        print('num_image', num_inpaint)

    flag = 1
    input_rgb = ori_rgb
    depth = ori_depth
    for i in range(0, step):
        if i == min:
            input_rgb = ori_rgb
            depth = ori_depth
            flag = -1
        if i == 0 or i == min:
            mask, img, depth, mask_index  = generate(input_rgb, depth, flag, dir='xz', first=True)
        else:
            mask, img, depth, mask_index  = generate(input_rgb, depth, flag, dir='xz', first=False)
        input_rgb = inpaint_image(mask, img)
        depth = depth_completion(input_rgb)

        Image.fromarray(input_rgb).save('Pano_inpaint/rgb_' + str(num_inpaint) + '.png')

        num_inpaint = num_inpaint + 1
        print('num_image', num_inpaint)

    flag = 1
    input_rgb = ori_rgb
    depth = ori_depth
    for i in range(0, step):
        if i == min:
            input_rgb = ori_rgb
            depth = ori_depth
            flag = -1
        if i == 0 or i == min:
            mask, img, depth, mask_index  = generate(input_rgb, depth, flag, dir='-xz', first=True)
        else:
            mask, img, depth, mask_index  = generate(input_rgb, depth, flag, dir='-xz', first=False)
        input_rgb = inpaint_image(mask, img)
        depth = depth_completion(input_rgb)

        Image.fromarray(input_rgb).save('Pano_inpaint/rgb_' + str(num_inpaint) + '.png')

        num_inpaint = num_inpaint + 1
        print('num_image', num_inpaint)

# load image and depth
rgb = np.array(Image.open(rgb).convert('RGB'))
depth = depth_est(rgb, net)
progressive_inpaint(ori_rgb=rgb, ori_depth=depth)

