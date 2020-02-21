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
    def __init__(self, root_dir, mask_dir, patch_size=448,transform=None):
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
        self.masklist = self.get_masklist()
        self.label_name_list =['H', 'N', 'F', 'T', 'I', 'M', 'B', 'D' ,'C', 'G','Y']
        label_point = [(35981,81083),(93094,105493),(3888,31980),(105494,128220),(81084,82911),\
                       (82912,93093),(0,1030),(3617,3887),(1031,3616),(31981,35980),(128221,128601)]
        self.label_dict = dict(zip(self.label_name_list,label_point))  
        self.patch_size= patch_size
        
    def get_filelist(self):
        flist = os.listdir(self.root_dir)
        flist.sort()
        return flist
    
    def get_rand_patch(self,num_seg):
        flist_sep = [[i for i in self.filelist if i[0]==tls] for tls in self.label_name_list]
        rand_patch = np.random.randint(0,11,size=num_seg)
        return [cv2.imread(self.root_dir+'/'+flist_sep[ti][np.random.randint(len(flist_sep[ti]))]) for ti in rand_patch],rand_patch

    
    def get_masklist(self):
        flist = os.listdir(self.mask_dir)
        flist.sort()
        return flist

    def __len__(self):
        return len(self.filelist)

    
    def __getitem__(self, idx):
        sythetic_image = np.zeros((self.patch_size,self.patch_size,3))
        sythetic_mask = np.zeros((self.patch_size,self.patch_size),dtype=np.int)
        mask = cv2.imread(self.mask_dir+self.masklist[np.random.randint(len(self.masklist))],0)
        mask = mask-mask.min()
        num_seg = mask.max()+1
        test_images,test_labels = self.get_rand_patch(num_seg)
        
        for i in range(0,num_seg):
            ttimage = np.array(test_images[i]*(np.random.random_sample()*0.3+0.8),dtype=np.uint8)
            sythetic_image[mask==i] = ttimage[mask==i]
            #sythetic_image[mask==i] = test_images[i][mask==i]
            sythetic_mask[mask==i] = test_labels[i]

        #return torch.from_numpy(sythetic_image/255.0).permute(2,0,1), torch.from_numpy(mask)
        return self.transform(sythetic_image,mask)

# train_dataset = SynthDataset(root_dir=os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_448/All',                            mask_dir=os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Segment/Patches_Syth_mask_448/',                            transform=None)
# syn_image, syn_mask = train_dataset.__getitem__(0)




def get_dataloader(batch_size = 1,\
                   root_dir=os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_448/All/',\
                   mask_dir=os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Segment/Patches_Syth_mask_448/',\
                   num_workers=4):
    
    class UnetAugmentation(object):
        def __init__(self, size=448, mean=(0.485,0.456,0.406),std=(0.5,0.5,0.5),scale=(0.64, 1)):
            self.augment = Compose([
                                    ConvertFromInts(),
                                    PhotometricDistort(),
                                    RandomResizedCrop(size=size,scale=scale),
                                    RandomMirror(),
                                    RandomFlip(),
                                    Resize(size),
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


