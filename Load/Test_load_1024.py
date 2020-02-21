
# coding: utf-8

# In[1]:


import torch
import torchvision
from torch import nn
import logging
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np

import time
import os
import copy
import logging
import pylab as plt
from PIL import Image
import sys
sys.path.append('../')
from Model.Unet_universal import UNet
from Data.get_segmentation_data import get_dataloader

from Options.USeg_options import * 

config = unet_seg_v1
# config = unet_v1


# In[2]

print("----Config name: %s-----------batch_size: %d----------"%(config.name,config.batch_size))

model = UNet(in_channels=config.input_nc, out_channels=config.n_classes,           num_hidden_features=config.num_channels,n_blocks=config.num_blocks,          num_dilated_convs=0, dropout_min=0, dropout_max=0,          block_type=config.block_type, padding=1, kernel_size=3,group_norm=0)

Test_loader= get_dataloader(batch_size=config.batch_size,                                      root_dir = config.train_path_sample,                                      mask_dir=config.train_path_mask,num_workers=config.num_workers),


# In[3]:


state_dict=torch.load(os.path.expanduser('~')+'/DATA_CRLM/Patches/Checkpoints/U_Seg/unet_seg_v1/unet_006.pth')
new_state_dict = {}
for key in model.state_dict():
    new_state_dict[key] = state_dict['module.'+key].double()
    
model.load_state_dict(new_state_dict)


# In[4]:


# dataloaders = {'train':get_dataloader(batch_size=1,                                      root_dir = config.train_path_sample,                                      mask_dir=config.train_path_mask,num_workers=config.num_workers),
#                'val':get_dataloader(batch_size= 1,\
#                                     root_dir = config.test_path_sample,\
#                                     mask_dir=config.test_path_mask,num_workers=config.num_workers)}

# Train_loader =dataloaders['train']
Test_loader =get_dataloader(batch_size= 1,\
                                    root_dir = config.test_path_sample,\
                                    mask_dir=config.test_path_mask,num_workers=config.num_workers)


# In[7]:

i = 0
model.cuda()


def process_patch(pindex):
    a,b = Test_loader.dataset.__getitem__(pindex)

    c=model(a.cuda().float().unsqueeze(0))
    result = c[0].detach().cpu().numpy().argmax(0)
    #rim = Image.fromarray(np.array(result,dtype=np.uint8))
    #rim.save('/home/zyx31/DATA_CRLM/Downloads/')
    plt.figure(figsize=(27,9))
    plt.subplot(1,3,1)
    plt.imshow(a.permute((1,2,0))/2+0.5)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(b)
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(result)
    plt.axis('off')
    plt.savefig(os.path.expanduser('~')+'/Downloads/Test_1024_%d.png'%pindex)

    torch.cuda.empty_cache()
    
for i in range(0,3000,300):
    process_patch(i)