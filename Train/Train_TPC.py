import os

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler

from Train_TPC_module import train_model
import sys
sys.path.append('../')
from Model.TPC import TPC
from Data.get_dataloader import get_dataloader
from Options.TPC_options import *    


if len(sys.argv)==1 or len(sys.argv)>2:
    raise Exception("no configuration files")
else:
    config = eval(sys.argv[1])

    
if not os.path.exists(config.save_path):
    os.makedirs(config.save_path, exist_ok=True)

print("----Config name: %s-----------batch_size: %d----------"%(config.model_name,config.batch_size))
print("----Num of workers: %s-----------"%(config.num_workers))

model =TPC(base_name=config.base_name,patch_size=config.patch_size,dropout_rate=config.dropout_rate)

device_ids=config.device_ids

device = torch.device('cuda:{}'.format(','.join([str(i) for i in device_ids])) \
                      if torch.cuda.device_count()>0 else torch.device('cpu'))

model_ft = nn.DataParallel(model,device_ids,dim=0)
model_ft.to(device)

criterion = nn.CrossEntropyLoss()
criterion.to(device)

# Observe that all parameters are being optimized
if config.optim == 'sgd':
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=config.momentum,)
else:
    optimizer_ft = optim.RMSprop(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


milestone_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=config.milestones, gamma=config.gamma, )

dataloaders = {'train':get_dataloader(batch_size=config.batch_size,patch_size=config.patch_size,\
                        root_dir = config.train_path,\
                        num_workers=config.num_workers,\
                        mean=config.mean,std=config.std,color_aug_param=config.color_aug_param_train),\
               'val':get_dataloader(batch_size= config.batch_size,patch_size=config.patch_size,\
                      root_dir = config.test_path,\
                      num_workers=config.num_workers,\
                      mean=config.mean,std=config.std,color_aug_param=config.color_aug_param_test)}

train_model(model_ft,dataloaders,criterion,optimizer_ft,\
            scheduler=milestone_lr_scheduler,\
           config=config)

# train_model(model,dataloaders,criterion,optimizer_ft,\
#             scheduler=milestone_lr_scheduler,\
#             num_epochs=config.num_epochs,\
#             display_size=config.display_size,save_epoch=config.save_epoch,\
#             save_path=config.save_path)