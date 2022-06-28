#%%
import os
import random
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from mapping import *


#%%
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, transform_m=None):
        self.img_dir = os.path.join(data_dir, 'm_label')
        self.mask_dir=os.path.join(data_dir, 'rgb_label')
        self.transform = transform
        self.transform_m = transform_m

    def __len__(self):
        return len(os.listdir(self.img_dir))
    
    def __getitem__(self, idx):
        ## pillow 사용
        if self.mask_dir!=None:
            img_path=os.path.join(self.img_dir,sorted(os.listdir(self.img_dir))[idx])
            mask_path=os.path.join(self.mask_dir,sorted(os.listdir(self.mask_dir))[idx])

            image = Image.open(img_path).convert('RGB')
            ##MASK 2장가져오는걸로 ㅇㅇ
            # CONVERT('l')
            mask = Image.open(mask_path).convert('L')
        
            seed=random.randint(1, 10)
            # seed 고정해주기!!!!!!!!!!!!!!
            if self.transform:
                torch.manual_seed(seed)
                image = self.transform(image)
            if self.transform_m:
                torch.manual_seed(seed)
                mask = self.transform_m(mask)
            ##label변환해주고 리턴하기
            
            ###mask2label을 2장가져와서 결과똑같이나오게
            
            label=mask2label(mask)
            return image, label, img_path
        else:
            img_path=os.path.join(self.img_dir,sorted(os.listdir(self.img_dir))[idx])
            image = Image.open(img_path).convert('RGB')
            # image=ImageEnhance.Color(image).enhance(2)
            if self.transform:
                image = self.transform(image)
            return image, img_path
        
#%%
def CustomDataLoader(dataset, val_split, batch_size, shuffle_dataset=True, random_seed=42):## training
    dataset_size=len(dataset)
    indices=list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))

    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    training_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=valid_sampler, drop_last=True)
    return training_loader,validation_loader

# %%
