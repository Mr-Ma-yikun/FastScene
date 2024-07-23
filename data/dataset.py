import os
import math
import numpy as np
from glob import glob

from random import shuffle
from PIL import Image, ImageFilter

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class InpaintingData(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.w = self.h = args.image_size
        self.mask_type = args.mask_type
        
        # image and mask 
        self.image_path = []
        for ext in ['*.jpg', '*.png']: 
            self.image_path.extend(glob(os.path.join(args.dir_image, ext)))
        self.mask_path = glob(os.path.join(args.dir_mask, '*.png'))
        self.image_path = sorted(self.image_path)[:8000]
        self.mask_path = sorted(self.mask_path)[:128000]
        # augmentation 
        self.img_trans = transforms.Compose([
            #transforms.RandomCrop(size=(512, 512)),
            #transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()])
        # self.mask_trans = transforms.Compose([
        #     transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.NEAREST),
        # ])

        
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        # load image
        image = Image.open(self.image_path[index]).convert('RGB')
        filename = os.path.basename(self.image_path[index])

        # if self.mask_type == 'pconv':
        #     index = np.random.randint(0, len(self.mask_path))
        #     mask = Image.open(self.mask_path[index])
        #     mask = mask.convert('L')
        # else:
        #     mask = np.zeros((self.h, self.w)).astype(np.uint8)
        #     mask[:self.h, self.w//2:] = 1
        #     mask = Image.fromarray(mask).convert('L')
        
        # augment
        image = self.img_trans(image) * 2. - 1.

        mask_index = np.random.randint(0, 16)
        mask = Image.open(self.mask_path[index*16+mask_index])
        mask = mask.convert('L')
        #mask_score = torch.from_numpy(np.random.uniform(low=0,high=1,size=(512, 512)))
        #mask = mask_score>=0.8
        #mask = mask.float().unsqueeze(0)
        mask = F.to_tensor(mask)
        mask = 1-mask

        return image, mask, filename



if __name__ == '__main__': 

    from attrdict import AttrDict
    args = {
        'dir_image': '/home/zhandandan/data/celeba/train',
        'data_train': 'celeba',
        'dir_mask': '/home/zhandandan/data/mask/train',
        'mask_type': 'pconv',
        'image_size': 256
    }
    args = AttrDict(args)

    data = InpaintingData(args)
    print(len(data), len(data.mask_path))
    img, mask, filename = data[0]
    print(img.size(), mask.size(), filename)