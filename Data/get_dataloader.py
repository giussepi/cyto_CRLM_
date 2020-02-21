from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
 
 
 
class TestDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.filelist = self.get_filelist()
        self.label_name_list =['H', 'N', 'F', 'T', 'I', 'M', 'B', 'D' ,'C', 'G','Y']
         
    def get_filelist(self):
        return os.listdir(self.root_dir)
 
    def __len__(self):
        return len(self.filelist)
 
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.filelist[idx])
        image = Image.open(img_name)
        sample = Image.fromarray(np.array(image)[:,:,:3])
 
        if self.transform:
            sample = self.transform(sample)
        label = self.label_name_list.index(self.filelist[idx][0])
        return sample,label
 
 
def get_dataloader(batch_size = 5,patch_size=224,\
                   root_dir='/mnt/DATA_CRLM/Patches/Patches_Level0/Patches_224/All/',\
                   num_workers = 64,\
                   mean = [0.485, 0.456, 0.406],\
                   std = [0.5, 0.5, 0.5],
                   color_aug_param = [0.2,0.2,0.1,0.03]): 
    #std = [0.229, 0.224, 0.225],
    data_transforms = transforms.Compose([
        #transforms.RandomResizedCrop(patch_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=color_aug_param[0],contrast=color_aug_param[1],\
                               hue=color_aug_param[2],saturation=color_aug_param[3]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])
 
    train_dataset = TestDataset(root_dir,transform=data_transforms)
    dataset_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True,num_workers=num_workers)
    return dataset_loader