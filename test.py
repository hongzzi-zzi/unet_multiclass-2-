#%%
import argparse
import gc
import os
from datetime import datetime
from tabnanny import verbose

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pyparsing import col
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from dataset import *
from model import UNet
from util import *
torch.autograd.set_detect_anomaly(True)
#%%

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4
LEARNING_RATE = 1e-3

DATA_DIR='/home/h/Desktop/data/random/test'
CKPT_DIR = 'ckpt'
RESULT_DIR = 'result'

print("device: %s" % DEVICE)
print("learning rate: %.4e" % LEARNING_RATE)
print("batch size: %d" % BATCH_SIZE)
print("data directory : %s" % DATA_DIR)
print("ckpt directory: %s" % CKPT_DIR)
print("log directory: %s" % RESULT_DIR)

# make folder if doesn't exist
if not os.path.exists(RESULT_DIR) : os.makedirs(RESULT_DIR)
#%%
# network train
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomAutocontrast(p = 1),
    transforms.ToTensor(),
    transforms.Normalize(mean = 0.5, std = 0.5)
])
transform_label = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
test_dataset = CustomDataset(
    data_dir=DATA_DIR,
    transform = transform,
    transform_m = transform_label
)
test_loader = DataLoader(
    test_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False
)
#%% network generate
model = UNet().to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# variables
num_data_test=len(test_dataset)

num_batch_test=np.ceil(num_data_test/BATCH_SIZE)

#%%
# test network
net, optim, st_epoch = load(ckpt_dir=CKPT_DIR, net=model, optim=optimizer)

with torch.no_grad(): # no backward pass 
    net.eval()
    loss_arr=[]

    for batch, data in enumerate(test_loader, 1):
        # forward pass
        input=data[0].to(DEVICE)
        label=data[1].to(DEVICE)
        output=net(input)
        
        for i in range(input.shape[0]):
            # print(output[i].shape)
            
            
            ##argmax
            
            outputimg=tensor2PIL(fn_class(output[i][1])).convert('RGBA')
            bg= Image.open('transparence.png').resize((512, 512)) 
            name=data[2][i].split('/')[-1].replace('m_label', 'eval').replace('jpg','png')

            outputimg.save(os.path.join(os.path.join(RESULT_DIR), name))
# %%
