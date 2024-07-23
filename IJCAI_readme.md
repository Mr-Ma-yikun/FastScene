# FastScene: Text-Driven Fast 3D Indoor Scene Generation via Panoramic Gaussian Splatting (IJCAI-2024)

# Introduction

This repository contains the official PyTorch implementation for the IJCAI 2024 paper titled "FastScene: Text-Driven Fast 3D Indoor Scene Generation via Panoramic Gaussian Splatting" by Yikun Ma, Dandan Zhan, and Zhi Jin.

![TWQY6AJ~~KHD2Q3YJ)SK}AD](https://private-user-images.githubusercontent.com/72637909/330794808-90bd3184-f91f-4401-b4f7-4b3421f67359.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjE3NDI2OTQsIm5iZiI6MTcyMTc0MjM5NCwicGF0aCI6Ii83MjYzNzkwOS8zMzA3OTQ4MDgtOTBiZDMxODQtZjkxZi00NDAxLWI0ZjctNGIzNDIxZjY3MzU5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA3MjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNzIzVDEzNDYzNFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTMwZmYxYzBmZTgyOTcxYzBhYjkwZTA3OWY4N2Q3MzlkZDU3ZjY0ZTdiOTgwMjgzYTEwNDA4Zjg5ODk3MWUyZjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.2wVgJ8dbYjIWVMt3lsvwbK4utjgv4Y3D43CDCDGRiz4)

# Abstract

Text-driven 3D indoor scene generation holds broad applications, ranging from gaming and smart home technologies to augmented and virtual reality (AR/VR) applications. Fast and high-fidelity scene generation is paramount for ensuring user-friendly experiences. However, existing methods are characterized by lengthy generation processes or necessitate the intricate manual specification of motion parameters, which introduces inconvenience for users. Furthermore, these methods often rely on narrow-field viewpoint iterative generations, compromising global consistency and overall scene quality. To address these issues, we propose FastScene, a framework for fast and high-quality 3D scene generation, while maintaining the scene consistency. Specifically, given a text prompt, we generate a panorama and estimate its depth, since panorama encompasses information about the entire scene and exhibits explicit geometric constraints. To obtain high-quality novel views, we introduce the Coarse View Synthesis (CVS) and Progressive Novel View Inpainting (PNVI) strategies, ensuring both scene consistency and view quality. Subsequently, we utilize Multi-View Projection (MVP) to form perspective views, and apply 3D Gaussian Splatting (3DGS) for fast scene generation. Comprehensive experiments demonstrate FastScene surpasses other methods in both generation speed and quality with better scene consistency. Notably, guided only by a text prompt, FastScene can generate a complete 3D scene within a mere 15 minutes, which is at least one hour faster than state-of-the-art methods, making it a paradigm for user-friendly scene generation.

# Installation

1. You can use https://github.com/ArcherFMY/SD-T2I-360PanoImage to generate a panorama based on a text.

```
git clone https://github.com/ArcherFMY/SD-T2I-360PanoImage.git
cd SD-T2I-360PanoImage
pip install -r requirements.txt
```

2. You can follow the environment setup guide provided by 3DGS  https://github.com/graphdeco-inria/gaussian-splattingto and install the corresponding environment.

# Getting Started

## 1. Download Pre-trained Models:

We have provided corresponding pre-trained weights, which you can download:

Link：https://pan.baidu.com/s/1Cm4ChVSHw-2NbDbPnvixZg 
Code：6666

Then, follow the settings in the `pro_inpaint.py` to place these weights in the corresponding locations.

## 2. Running

First, generate a panorama based on a text:

```
cd /path of diffusion360
python demo.py
```

Then, obtain the novel multi-view panoramas by PNVI and CVS:

```
python pro_inpaint.py
```

Then, we recommend using OpenMVG to perform MVP operations, as it integrates convenient toolkits. 

```
openMVG_sample_pano_converter -i input -o out -n n -r r
```

Then you can use the obtained perspective views to train with 3DGS!

```
cd /path of 3DGS
python convert.py -s data
python train.py -s data -m data/output -w
python render.py -m data/output
```

Specifically, you can freely control the number of viewpoints and the training termination steps, allowing you to balance between the generation speed and quality.

# Acknowledgement

This work was supported by [Frontier Vision Lab](https://fvl2020.github.io/fvl.github.com/), SUN YAT-SEN University.

# Citation

If you find this work helpful, please consider citing:

```
@article{ma2024fastscene,
  title={FastScene: Text-Driven Fast 3D Indoor Scene Generation via Panoramic Gaussian Splatting},
  author={Ma, Yikun and Zhan, Dandan and Jin, Zhi},
  journal={arXiv preprint arXiv:2405.05768},
  year={2024}
}
```

Feel free to reach out for any questions or issues related to the project. Thank you for your interest!