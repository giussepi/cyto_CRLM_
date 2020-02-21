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
    def __init__(self, root_dir, patch_size=448,transform=None):
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
        label_point = [(35981,81083),(93094,105493),(3888,31980),(105494,128220),(81084,82911),(82912,93093),(0,1030),(3617,3887),(1031,3616),(31981,35980),(128221,128601)]
        self.label_dict = dict(zip(self.label_name_list,label_point))  
            
        self.patch_size= patch_size
        
    def get_filelist(self):
        flist = os.listdir(self.root_dir)
        flist.sort()
        return flist

    def __len__(self):
        return len(self.filelist)
    
    """
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.filelist[idx])
        image = Image.open(img_name)
        sample = image

        if self.transform:
            sample = self.transform(sample)
        label = self.label_name_list.index(self.filelist[idx][0])
        return sample,label
    """
    def get_rand_patch(self):
        rand_label = np.random.randint(0,len(self.label_name_list))
        r_h = np.random.randint(self.label_dict[self.label_name_list[rand_label]][0],self.label_dict[self.label_name_list[rand_label]][1])
        return r_h,rand_label
        
    def get_single(self):
        r_h1,r_b1 = self.get_rand_patch()
        img = test_h = np.array(Image.open(self.root_dir+self.filelist[r_h]))
        
        labels = np.ones((4,4))*r_b1
        return img,labels
        
    
    def get_image_uniform(self):
        r_h,r_b = self.get_rand_patch()
        test_h = np.array(Image.open(self.root_dir+self.filelist[r_h]))
        new_img = np.zeros(shape=(self.patch_size,self.patch_size,3), dtype = np.uint8)
        
        tseg = int(self.patch_size/2)
        new_img[:tseg,:tseg,:] = np.array(test_h)[112:336,112:336,:3]
        new_img[:tseg,tseg:,:] = np.array(test_h)[112:336,112:336,:3]
        new_img[tseg:,:tseg,:] = np.array(test_h)[112:336,112:336,:3]
        new_img[tseg:,tseg:,:] = np.array(test_h)[112:336,112:336,:3]
        labels = np.ones((4,4))*r_b
        return new_img,r_h,labels

    def get_image_mixed(self):
        r_h1,r_b1 = self.get_rand_patch()
        r_h2,r_b2 = self.get_rand_patch()
        r_h3,r_b3 = self.get_rand_patch()
        r_h4,r_b4 = self.get_rand_patch()

        test_h1 = Image.open(self.root_dir+self.filelist[r_h1])
        test_h2 = Image.open(self.root_dir+self.filelist[r_h2])
        test_h3 = Image.open(self.root_dir+self.filelist[r_h3])
        test_h4 = Image.open(self.root_dir+self.filelist[r_h4])

        new_img = np.zeros(shape=(self.patch_size,self.patch_size,3),dtype = np.uint8)
        
        tseg = int(self.patch_size/2)
        new_img[:tseg,:tseg,:] = np.array(test_h1)[112:336,112:336,:3]
        new_img[:tseg,tseg:,:] = np.array(test_h2)[112:336,112:336,:3]
        new_img[tseg:,:tseg,:] = np.array(test_h3)[112:336,112:336,:3]
        new_img[tseg:,tseg:,:] = np.array(test_h4)[112:336,112:336,:3]
        
        tlabels = np.zeros((int(self.patch_size/112),int(self.patch_size/112)))
        ##TODO
        tlabels[:2,:2] = r_b1 
        tlabels[:2,2:] =  r_b2
        tlabels[2:,:2] =  r_b3
        tlabels[2:,2:] =  r_b4
        
        #return new_img,(r_h1,r_h2,r_h3,r_h4),np.array([[r_b1,r_b2],[r_b3,r_b4]])
        return new_img,(r_h1,r_h2,r_h3,r_h4),tlabels
    def __getitem__(self, idx):
        if idx%3 ==0:
            sample,_,labels = self.get_image_mixed()
        elif idx%2 == 0:
            sample,_,labels = self.get_image_uniform()
        else:
            sample,_,labels = self.get_single()
        sample = Image.fromarray(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample, labels
        
        
def get_dataloader(batch_size = 5,rootdir='/mnt/DATA_CRLM/Patches/Patches_Level0/Patches_448/All/'):
    data_transforms = transforms.Compose([
        #transforms.RandomResizedCrop(patch_size),
        #transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3,contrast=0.3,hue=0.1,saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = TestDataset(rootdir,transform=data_transforms)
    dataset_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True,num_workers=4)
    return dataset_loader
