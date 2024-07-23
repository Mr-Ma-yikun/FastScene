import torch
import torch.nn as nn
import argparse
from evaluate import Evaluation
from data_loader import S3D_loader,Sample_loader
import os.path as osp
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision import transforms

from pano_loader.pano_loader import Pano3D
import os
from torch.backends import cudnn


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    cudnn.benchmark = True

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if config.multiprocessing_distributed:
        config.world_size = ngpus_per_node * config.world_size
        torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        main_worker(config.gpu, ngpus_per_node, config)


def main_worker(gpu, ngpus_per_node, config):
    
    transform = transforms.Compose([
            transforms.ToTensor()])
    
    input_dir = ""
    device_ids = [int(config.gpu)]

    if config.gpu is not None:
        print(f'Use GPU: {gpu} for training')

    if config.distributed:
        if config.dist_url == "envs://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            config.rank = config.rank * ngpus_per_node + gpu
        torch.distributed.init_process_group(backend=config.dist_backend, init_method=config.dist_url, world_size=config.world_size, rank=config.rank)



    val_loader = Sample_loader(config.data_path,transform = transform,transform_t = transform)

# -------------------------------------------------------
    device = torch.device('cuda', device_ids[0])

    val_dataloader = torch.utils.data.DataLoader(
    	val_loader,
        batch_size=1,
	    shuffle=False,
    	num_workers=config.num_workers,
	    drop_last=False)

    evaluation = Evaluation(
    	config,
        val_dataloader, 
	    device)
    
    evaluation.evaluate_egformer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",
                                 type=str,
                                 help="Method to be evaluated",
                                 choices=["EGformer","Panoformer"],
                                 default="EGformer")
    
    parser.add_argument("--eval_data",
                                 type=str,
                                 help="data category to be evaluated",
                                 choices=["Structure3D", "Pano3D","Inference"],
                                 default="Structure3D")
  
    parser.add_argument("--align_type",
                                 type=str,
                                 help="align_type for evaluation metric",
                                 choices=["Column", "Image"],
                                 default="Image")


    parser.add_argument('--S3D_path', help = 'Structure3D Data_path' , type=str, default='')
    parser.add_argument('--data_path', help = 'Data_path for inference' , type=str, default='')

    parser.add_argument('--num_workers' , type=int, default=1)

    parser.add_argument('--checkpoint_path', type=str, default='')


    parser.add_argument('--save_sample', help= 'if true, save depth results in the designated folder', action='store_true')
    parser.add_argument('--output_path', help = 'folder where depth results will be saved' , type=str, default='output')
    

    parser.add_argument("--pano3d_root", type=str, help="Path to the root folder containing the Pano3D extracted data.",default='/YOUR_PATH/Pano3D_folder')
    parser.add_argument("--pano3d_part", type=str, help="The Pano3D subset to load.",default='M3D_high')
    parser.add_argument("--pano3d_split", type=str, help="The Pano3D split corresponding to the selected subset that will be loaded.",default='./pano_loader/Pano3D/splits/M3D_v1_test.yaml')
    parser.add_argument('--pano3d_types', default=['color','depth'], nargs='+',
            choices=[
                'color', 'depth', 'normal', 'semantic', 'structure', 'layout',
                'color_up', 'depth_up', 'normal_up', 'semantic_up', 'structure_up', 'layout_up'
                'color_down', 'depth_down', 'normal_down', 'semantic_down', 'structure_down', 'layout_down'
                'color_left', 'depth_left', 'normal_left', 'semantic_left', 'structure_left', 'layout_left'
                'color_right', 'depth_right', 'normal_right', 'semantic_right', 'structure_right', 'layout_right'
            ],
            help='The Pano3D data types that will be loaded, one of [color, depth, normal, semantic, structure, layout], potentially suffixed with a stereo placement from [up, down, left, right].'
        )
    ############ Distributed Data Parallel (DDP) ############
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=-1)
    parser.add_argument('--gpu', type=str, default="0,1,2,3")
    parser.add_argument('--dist-url', type=str, default="tcp://127.0.0.1:7777")
    parser.add_argument('--dist-backend', type=str, default="nccl")
    parser.add_argument('--multiprocessing_distributed', default=True)
    

    config = parser.parse_args()
    main(config)


