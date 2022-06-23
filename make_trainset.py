#%%
# package
import itertools
import os
import random
import re
import shutil
from glob import glob

from PIL import Image
from torchvision import transforms

#%%
########################################################################################
INPUT_DIR=glob('/home/h/Desktop/data/*/m_label/*')
LABEL_DIR=glob('/home/h/Desktop/data/*/t_label/*')
TRAIN_PATH="/home/h/Desktop/data/random/train"
TEST_PATH="/home/h/Desktop/data/random/test"
########################################################################################
#%%
tensor2PIL=transforms.ToPILImage()
PIL2tensor=transforms.ToTensor()
def mask2RGBmask(list, path):## 굳이 안써도 되지만,,, 2가지 이상의 채널일 경우도 있으니까 ㅇㅅㅇ
    for i in list:
        mask_ori=Image.open(i).resize((512, 512)).convert('RGBA').split()[-1]
        pix=mask_ori.load()
        rgb_mask=Image.new(mode="RGB", size=(512, 512),color=(0, 0, 0))
        for x, y in itertools.product(range(512),range(512)):
            if pix[x, y]!=0:
                rgb_mask.load()[x, y]=(255, 255, 255)
        rgb_mask.save(path)
#%%
if os.path.exists(TEST_PATH):
    shutil.rmtree(TEST_PATH)
if os.path.exists(TRAIN_PATH):
    shutil.rmtree(TRAIN_PATH)

os.makedirs(os.path.join(TEST_PATH, 'm_label'))
os.makedirs(os.path.join(TEST_PATH, 'rgb_label'))
os.makedirs(os.path.join(TRAIN_PATH, 'm_label'))
os.makedirs(os.path.join(TRAIN_PATH, 'rgb_label'))

input_lst=sorted(INPUT_DIR)
label_lst=sorted(LABEL_DIR)
allfile_lst=[[i, l]for i, l in zip(input_lst, label_lst)]
img_cnt=len(allfile_lst)

test_cnt=int(img_cnt*0.1)
train_cnt=img_cnt-test_cnt

random.shuffle(allfile_lst)
test_lst=allfile_lst[:test_cnt]
train_lst=allfile_lst[test_cnt:]

for i in test_lst:
    shutil.copyfile(i[0], os.path.join(TEST_PATH, i[0].split('/')[-2], i[0].split('/')[-1]))## input
    mask2RGBmask(i, os.path.join(TEST_PATH,i[1].split('/')[-2], i[1].split('/')[-1]).replace('t_label','rgb_label'))##rgb_label
    
for i in train_lst:
    shutil.copyfile(i[0], os.path.join(TRAIN_PATH, i[0].split('/')[-2], i[0].split('/')[-1]))##input
    mask2RGBmask(i, os.path.join(TRAIN_PATH,i[1].split('/')[-2], i[1].split('/')[-1]).replace('t_label','rgb_label'))##rgb_label
# %%
