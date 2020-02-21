from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pylab as plt
import cv2
from .UnetAugmentation import *


class SynthDataset(Dataset):
    """Synthetic Data from superpixel"""
    def __init__(self, root_dir, mask_dir, patch_size=1024,transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            mask_dir (string): Directory with artifical Masks
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir  # image root
        self.mask_dir = mask_dir   # Mask Directory
        self.transform = transform 
        self.filelist = self.get_filelist()
        #self.masklist = self.get_masklist()
        self.patch_size= patch_size
        
    def get_filelist(self):
        flist = os.listdir(self.root_dir)
        flist.sort()
        return flist
    
    def get_masklist(self):
        flist = os.listdir(self.mask_dir)
        flist.sort()
        return flist

    def __len__(self):
        return len(self.filelist)

    
    def __getitem__(self, idx):
        sample_name = self.filelist[idx]
        sythetic_image = np.array(Image.open(self.root_dir + sample_name))[:,:,:3]
        sythetic_mask  = np.array(Image.open(self.mask_dir + sample_name))
        #sythetic_mask  = np.array(plt.imread(self.mask_dir + sample_name))

        if self.transform is not None:
            return self.transform(sythetic_image,sythetic_mask)
        else:
            if sythetic_image.max()>10:
                return torch.from_numpy(sythetic_image/255.0).permute(2,0,1), torch.from_numpy(sythetic_mask)
            else:
                return torch.from_numpy(sythetic_image).permute(2,0,1), torch.from_numpy(sythetic_mask)



def get_dataloader(batch_size = 1,\
                   root_dir=os.path.expanduser('~')+\
                   '/DATA_CRLM/Patches/Patches_Segment/Synth_Patch_1024_Paired/Training/Sample/',\
                   mask_dir=os.path.expanduser('~')+\
                   '/DATA_CRLM/Patches/Patches_Segment/Synth_Patch_1024_Paired/Training/Mask/',\
                   num_workers=4):
    
    class UnetAugmentation(object):
        def __init__(self, size=1024, mean=(0.485,0.456,0.406),std=(0.5,0.5,0.5),scale=(0.64, 1)):
            self.augment = Compose([
                                    ConvertFromInts(),
                                    PhotometricDistort(delta=18.0,con_lower=0.8, \
                                                       con_upper=1.2,sat_lower=0.8, sat_upper=1.2),
                                    #RandomResizedCrop(size=size,scale=scale),
                                    RandomMirror(),
                                    RandomFlip(),
                                    #Resize(size),
                                    ToTensor(),
                                    Normalize(mean,std),
                                ])

        def __call__(self, img, masks):
            return self.augment(img, masks)
    
    data_transforms = UnetAugmentation()

    train_dataset = SynthDataset(root_dir,mask_dir,transform=data_transforms)
    dataset_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,\
                                                 shuffle=True,num_workers=num_workers)
    return dataset_loader

