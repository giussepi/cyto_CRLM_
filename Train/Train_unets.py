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

import sys
sys.path.append('../')
from Model.Unet_universal import UNet
from Data.get_super_synth_loader import get_dataloader

from Model.loss import CrossEntropyLoss2d_unet as CrossEntropyLoss2d

from Train_unet_module import train_model

from Options.Unet_options import * 
from Options.Ures_options import * 


#config = ures_v1
config = unet_v1



print("----Config name: %s-----------batch_size: %d----------"%(config.name,config.batch_size))

model = UNet(in_channels=3, out_channels=config.n_classes, \
          num_hidden_features=config.num_features,n_blocks=config.n_blocks,\
          num_dilated_convs=0, dropout_min=0, dropout_max=config.dropout,\
          block_type=config.block_type, padding=1, kernel_size=3,group_norm=0)


device_ids=config.device_ids
device = torch.device('cuda:{}'.format(','.join([str(i) for i in device_ids])) \
                      if torch.cuda.device_count()>0 else torch.device('cpu'))

model_ft = nn.DataParallel(model, device_ids, dim=0)
model_ft.to(device)

criterion = CrossEntropyLoss2d()

# Observe that all parameters are being optimized
if config.optim == 'sgd':
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
else:
    optimizer_ft = optim.RMSprop(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=config.milestones, gamma=0.1, )

dataloaders = {'train':get_dataloader(batch_size=config.batch_size,\
                        root_dir = config.train_path,mask_dir=config.mask_dir,\
                        num_workers=config.num_workers),\
               'val':get_dataloader(batch_size= config.batch_size,\
                      root_dir = config.test_path,mask_dir=config.mask_dir,\
                                   num_workers=config.num_workers)}

model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,\
                       num_epochs=config.num_epochs,save_epoch=config.save_epoch,\
                       display_size=config.display_size,save_path= config.save_path)
