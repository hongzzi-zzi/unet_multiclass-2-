#%%
import torch
from PIL import Image

import itertools
from torchvision import datasets, transforms
#%%
tensor2PIL=transforms.ToPILImage()
PIL2tensor=transforms.ToTensor()
mapping={(0, 0, 0):0,
         (255, 255, 255):1}#(0, 0, 0): background, (255, 255, 255): teeth
#%%

## 이것도 미리돌리기

# # 두장인풋으로너어서
# def mask2label(mask_ori):## RGB tensor
#     mask=tensor2PIL(mask_ori)
#     pix=mask.load()
#     label = torch.zeros(2, 512, 512, dtype=torch.float)
#     for k in mapping:
#         v=mapping.get(k)
#         # w:가로 h: 세로
#         for w, h in itertools.product(range(mask.size[0]),range(mask.size[1])):
#             label[v][h][w]=(pix[w, h]==k)
#     return label ## tensor

# 두장인풋으로너어서
def mask2label(mask_ori):## RGB tensor
    
    mask_bg = mask_ori.mul(-1.0).add(1.0)
    label = torch.concat([mask_bg, mask_ori], dim=0)
            
    return label ## tensor