import cv2
import torch
import numpy as np
from imageio import imread, imwrite
import os
from test import main_worker
from PIL import Image
from Projection import Equirec2Cube
from Projection import Cube2Equirec
import subprocess
USE_GPU = True 

device = 'cuda:0'
size = 512

def inpaint(mask, rgb, in_model, upsampler):

    print
    e2c = Equirec2Cube(512, 512).to(device)
    c2e = Cube2Equirec(1024, 512).to(device)

    #最原始的全景mask
    ori_mask = (mask == 255.0)
    ori_rgb = rgb.copy()

    mask = mask.astype(np.float32) / 255.0
    batch_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    batch_img = ori_rgb.astype(np.float32) / 255.0

    batch_img = torch.FloatTensor(batch_img).permute(2, 0, 1)[None, ...].to(device)
    batch_mask = torch.FloatTensor(batch_mask).permute(2, 0, 1)[None, ...].to(device)

    cubemap_img = e2c(batch_img)
    cubemap_mask = e2c(batch_mask)
    cubemap_img = cubemap_img.permute(0, 2, 3, 1).cpu().numpy()
    cubemap_mask = cubemap_mask.permute(0, 2, 3, 1).cpu().numpy()

    inpaint_images = torch.zeros(6, 3, size, size, dtype=torch.float32).to(device)

    for i in range(6):

        cubemap_mask[i] = (cubemap_mask[i] > 0).astype(int)    #mask = 1

        face_mask = (cubemap_mask[i] * 255.0).astype(np.uint8)
        true_mask = (face_mask == 255) 

        cubemap_img[i][true_mask] = 1.0
        face_img = (cubemap_img[i] * 255.0).astype(np.uint8)

        #Image.fromarray(face_mask).save(os.path.join('input', f"image{i + 1}_mask001.png"))
        #Image.fromarray(face_img).save(os.path.join('input', f"image{i + 1}.png"))
        
        #AOT-GAN
        inpainted = main_worker(face_mask[:,:,0], face_img, in_model)

        # You can optionally use a super-resolution algorithm to enhance the visual quality.
        #inpainted, _ = upsampler.enhance(inpainted, outscale=2)

        new_inpaint = torch.FloatTensor(inpainted.astype(np.float32)/255.0).permute(2, 0, 1)[None, ...].cuda()
        inpaint_images[i] = new_inpaint

        #Image.fromarray(inpainted).save(os.path.join('output', f"inpaint_{i + 1}.png"))


    equirec_img = c2e(inpaint_images)
    equirec_img = equirec_img.permute(0, 2, 3, 1).cpu().numpy()
    equirec_img = (equirec_img * 255).astype(np.uint8)
    equirec_img = equirec_img.squeeze(0)
    #equirec_img, _ = upsampler.enhance(equirec_img, outscale=2)

    ori_rgb[ori_mask] = equirec_img[ori_mask]

    return ori_rgb
