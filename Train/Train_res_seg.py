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
from Model.Res_Seg import Res_Seg,get_norm_layer,init_weights
from Data.get_segmentation_data import get_dataloader

from Model.loss import CrossEntropyLoss2d_unet as CrossEntropyLoss2d

from Train_unet_module import train_model

from Options.RSeg_options import * 




#config = res_seg_v1
config = eval(sys.argv[1])


print("----Config name: %s-----------batch_size: %d----------"%(config.name,config.batch_size))

gpu_ids = config.device_ids
use_gpu = len(gpu_ids) > 0
norm_layer = get_norm_layer(norm_type=config.norm)
model = Res_Seg(input_nc=config.input_nc, output_nc=config.output_nc, n_blocks=config.n_blocks,\
                 gpu_ids=config.device_ids,use_dropout=config.use_dropout,n_downsampling=config.n_downsampling)

init_weights(model, init_type=config.init_type)


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
                                      root_dir = config.train_path_sample,\
                                      mask_dir=config.train_path_mask,num_workers=config.num_workers),
               'val':get_dataloader(batch_size= config.batch_size,\
                                    root_dir = config.test_path_sample,\
                                    mask_dir=config.test_path_mask,num_workers=config.num_workers)}


model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,\
                       num_epochs=config.num_epochs,save_epoch=config.save_epoch,\
                       display_size=config.display_size,save_path= config.save_path)
