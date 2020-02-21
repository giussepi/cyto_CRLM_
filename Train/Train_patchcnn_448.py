# -*- coding: utf-8 -*-
""" Train/Train_patchcnn_448 """

import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from Data.get_dataloader import get_dataloader
from Model.loss import CrossEntropyLoss2d
from Model.PatchCNN import PatchCNN
from settings import TRAIN_PATH, TRAIN_TEST_PATH, TRAIN_SAVE_PATH
from .train_module import train_model


def run():
    batch_size = 16  # 64
    patch_size = 448
    num_workers = 32
    num_layers = [3, 4, 6, 3]  # res34
    # num_layers = [2,2,2,2] # res18
    dropout_rate = 0.5

    if not os.path.exists(TRAIN_SAVE_PATH):
        os.mkdir(TRAIN_SAVE_PATH)

    device_ids = [0]
    logger = logging.getLogger()
    logging.basicConfig(filename=TRAIN_SAVE_PATH+'training.log', level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logging.info("Start Training")
    device = torch.device('cuda:{}'.format(','.join([str(i) for i in device_ids]))
                          if torch.cuda.device_count() > 0 else torch.device('cpu'))
    model_ft = PatchCNN(layers=num_layers, dropout_rate=dropout_rate)
    model_ft = nn.DataParallel(model_ft, device_ids, dim=0)
    model_ft.to(device)
    criterion = CrossEntropyLoss2d()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
    dataloaders = {'train': get_dataloader(batch_size=batch_size, patch_size=patch_size,
                                           root_dir=TRAIN_PATH, num_workers=num_workers),
                   'val': get_dataloader(batch_size=batch_size, patch_size=patch_size,
                                         root_dir=TRAIN_TEST_PATH, num_workers=num_workers)}
    model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=100, save_epoch=10, display_size=100,
                           save_path=TRAIN_SAVE_PATH)
    torch.save(model_ft.cpu().state_dict(), TRAIN_SAVE_PATH+'/PatchCNN_best.pth')
