#########
#  Change  the 
#
#
import torch
import torch.nn as nn

from torch.autograd import Variable
from train_module import *
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
#import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image

import sys
sys.path.append('../')
from Data.get_dataloader import get_dataloader
from Model.PatchCNNv3 import PatchCNN
from Model.loss import CrossEntropyLoss2d


batch_size = 32
patch_size = 448
num_workers = 32
num_epochs = 100

num_layers = [3,4,6,3]  # res34
#num_layers = [2,2,2,2] # res18
dropout_rate = 0.5

train_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_448/All/'
test_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_448/Test/'
save_path = os.path.expanduser('~')+'/DATA_CRLM/Checkpoint/PatchCNN/PatchCNN_448_res34_base/'

if not os.path.exists(save_path):
    os.mkdir(save_path)

device_ids=[0]

logger = logging.getLogger()
logging.basicConfig(filename=save_path+'training.log',level=logging.DEBUG)
logger.setLevel(logging.DEBUG)

logging.info("Start Training")

device = torch.device('cuda:{}'.format(','.join([str(i) for i in device_ids])) if torch.cuda.device_count()>0 else torch.device('cpu'))

model_ft = PatchCNN(layers=num_layers,dropout_rate = dropout_rate)

model_ft = nn.DataParallel(model_ft, device_ids, dim=0)
model_ft.to(device)


criterion = CrossEntropyLoss2d()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)


dataloaders = {'train':get_dataloader(batch_size=batch_size,patch_size=patch_size,
                        root_dir = train_path,num_workers=num_workers),\
               'val':get_dataloader(batch_size= batch_size,patch_size=patch_size,
                      root_dir=test_path,num_workers=num_workers)}


model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs,save_epoch=10,display_size=100,save_path = save_path)

torch.save(model_ft.cpu().state_dict(),save_path+'/PatchCNN_best.pth')