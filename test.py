import os
import argparse
import numpy as np
from PIL import Image
from glob import glob

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import torchvision.transforms as tf

device = 'cuda:0'

def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image

def main_worker(mask, rgb, model, use_gpu=True):

    mask = mask
    image = rgb

    img_transform = tf.Compose([tf.ToTensor()])
    mask_transform = tf.Compose([tf.ToTensor()])

    image = img_transform(image)

    image = (image * 2.0 - 1.0).unsqueeze(0)
    mask = mask_transform(mask)
    mask = mask.unsqueeze(0)

    image, mask = image.to(device), mask.to(device)
    #x(1-0)+0=x
    #x(1-1)+1=1
    image_masked = image * (1 - mask.float()) + mask

    with torch.no_grad():
        pred_img = model(image_masked, mask)


    inpainted = postprocess(pred_img[0])
    return inpainted


