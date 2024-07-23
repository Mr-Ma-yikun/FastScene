
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
import numpy as np
model_path = r'models/RealESRGAN_x2plus.pth'
dni_weight = None
sr_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2)
device = 'cuda:0'

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

img = np.array(Image.open('test.png').convert('RGB'))

output_img, _ = upsampler.enhance(img, outscale=2)

Image.fromarray(output_img).save('sr.png')
