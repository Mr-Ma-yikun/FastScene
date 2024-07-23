import os
from torch.utils import data
from torchvision import transforms
from PIL import ImageEnhance
import random
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.utils.data import RandomSampler
import torchvision.transforms.functional as F
from imageio import imread
import numpy as np
from skimage import io
#import OpenEXR, Imath, array
import math
import os.path as osp
import torch.utils.data
from skimage.transform import rescale,resize

class S3D_loader(data.Dataset):
    def __init__(self,root,transform = None,transform_t = None):
            "makes directory list which lies in the root directory"
            if True:
                dir_path = list(map(lambda x:os.path.join(root,x),os.listdir(root)))
                dir_path_deep=[]
                left_path=[]
                right_path=[]
                self.image_paths = []
                self.depth_paths = []
                dir_sub_dir=[]
 

                for dir_sub in dir_path:
                    
                    sub_path = os.path.join(dir_sub,'2D_rendering')
                    sub_path_list = os.listdir(sub_path)
                    

                    for path in sub_path_list:
                        dir_sub_dir.append(os.path.join(sub_path,path,'panorama/full'))
                

                for final_path in dir_sub_dir:
                    self.image_paths.append(os.path.join(final_path,'rgb_rawlight.png'))                    
                    self.depth_paths.append(os.path.join(final_path,'depth.png'))                    



                self.transform = transform
                self.transform_t = transform_t

    def __getitem__(self,index):
           
        if True:
            
            image_path = self.image_paths[index]
            depth_path = self.depth_paths[index]
                
            image = Image.open(image_path).convert('RGB')
            depth = imread(depth_path,as_gray=True).astype(np.float)

            data=[]

        if self.transform is not None:
            data.append(self.transform(image))
            data.append(self.transform_t(depth))
        return data

    def __len__(self):
        
        return len(self.image_paths)


class Sample_loader(data.Dataset):
    def __init__(self,root,transform = None,transform_t = None):
            "makes directory list which lies in the root directory"
            if True:
                dir_path = list(map(lambda x:os.path.join(root,x),os.listdir(root)))
                self.image_paths = []
                self.depth_paths = []
                dir_sub_dir=[]


                index = 0 
                for path in dir_path:
                    file_list = os.listdir(path)
                    for file_name in file_list:
                        self.image_paths.append(os.path.join(path,file_name))

                        index = index + 1

                self.image_paths.sort()
                self.transform = transform
                self.transform_t = transform_t

    def __getitem__(self,index):
           
        if True:
            
            image_path = self.image_paths[index]
                
            image = Image.open(image_path).convert('RGB')

            data=[]

        if self.transform is not None:
            data.append(self.transform(image))
        return data

    def __len__(self):
        
        return len(self.image_paths)


