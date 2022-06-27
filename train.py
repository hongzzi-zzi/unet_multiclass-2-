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

#%% training parameter
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
EPOCH=100
BATCH_SIZE = 4
LEARNING_RATE = 1e-3

TRAIN_CONTINUE = 'off'
DATA_DIR='/home/h/Desktop/data/random/train'
CKPT_DIR = 'ckpt'
LOG_DIR = 'log'

print("device: %s" % DEVICE)
print("learning rate: %.4e" % LEARNING_RATE)
print("batch size: %d" % BATCH_SIZE)
print("number of epoch: %d" % EPOCH)
print("data directory : %s" % DATA_DIR)
print("train continue: %s" % TRAIN_CONTINUE)
print("ckpt directory: %s" % CKPT_DIR)
print("log directory: %s" % LOG_DIR)

#%% data aug & custom dataset
transform=transforms.Compose([transforms.Resize((512, 512)),
                              transforms.RandomHorizontalFlip(), 
                              transforms.RandomVerticalFlip(), 
                              transforms.RandomAffine([-60, 60]),
                              transforms.ColorJitter(brightness=0.3, contrast=0.3),
                              transforms.RandomAutocontrast(p=1),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=0.5, std=0.5)
                              ])
transform_label=transforms.Compose([transforms.Resize((512, 512)),
                              transforms.RandomHorizontalFlip(), 
                              transforms.RandomVerticalFlip(), 
                              transforms.RandomAffine([-60, 60]),
                              transforms.ToTensor(),
                              ])
dataset=CustomDataset(DATA_DIR, transform=transform,transform_m= transform_label)
training_loader, validation_loader=CustomDataLoader(dataset, val_split=0.1, batch_size=BATCH_SIZE)
#%%  network generate
model = UNet().to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#%%
# variables
num_data_train = len(training_loader)
num_data_val = len(validation_loader)
num_batch_train = np.ceil(num_data_train / BATCH_SIZE)
num_batch_val = np.ceil(num_data_val / BATCH_SIZE)

# set summarywriter to use tensorboard
writer_train = SummaryWriter(log_dir=os.path.join(LOG_DIR, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(LOG_DIR, 'val'))

#%%
st_epoch=0

if TRAIN_CONTINUE == "on":
    model, optimizer, st_epoch = load(ckpt_dir=CKPT_DIR, net=model, optim=optimizer)

for epoch in range(st_epoch + 1, EPOCH + 1):
    model.train()
    loss_arr = []

    for batch, data in enumerate(training_loader, 1):
        # forward pass
        input=data[0].to(DEVICE)
        label=data[1].to(DEVICE)
        output = model(input)
        # backward pass
        optimizer.zero_grad()
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        
        loss_arr += [loss.item()]
        print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %(epoch, EPOCH, batch, num_data_train, np.mean(loss_arr)))

        # save to tensorboard
        input = fn_denorm(input, mean=0.5, std=0.5)
        label_t=torch.chunk(label,2, dim=1 )[-1]
        output_t=torch.chunk(label,2, dim=1 )[-1]
        
        writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NCHW')   
        writer_train.add_image('label', label_t, num_batch_train * (epoch - 1) + batch, dataformats='NCHW')
        writer_train.add_image('output', output_t, num_batch_train * (epoch - 1) + batch, dataformats='NCHW')

    writer_train.add_scalar('loss', np.mean(loss_arr), epoch)
        
    with torch.no_grad():
        model.eval()
        loss_arr = []

        for batch, data in enumerate(validation_loader, 1):
            # forward pass
            input=data[0].to(DEVICE)
            label=data[1].to(DEVICE)
            output = model(input)
            
            loss = loss_fn(output, label)
            loss_arr += [loss.item()]

            print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %(epoch, EPOCH, batch, num_batch_val, np.mean(loss_arr)))

            # save to tensorboard
            input = fn_denorm(input, mean=0.5, std=0.5)
            label_t=torch.chunk(label,2, dim=1 )[-1]
            output_t=torch.chunk(label,2, dim=1 )[-1]
            
            writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NCHW')   
            writer_val.add_image('label', label_t, num_batch_val * (epoch - 1) + batch, dataformats='NCHW')
            writer_val.add_image('output', output_t, num_batch_val * (epoch - 1) + batch, dataformats='NCHW')
    writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
    
    if epoch % 20 == 0:
        save(ckpt_dir=CKPT_DIR, net=model, optim=optimizer, epoch=epoch)

writer_train.close()
writer_val.close()
#%%
