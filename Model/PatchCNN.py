
import torch
from torch import nn
from .initialize import *
from .Resnet_base import *

class PatchCNN(nn.Module):
    def __init__(self,layers= [2,2,2,2],num_classes=11,base_num = 64, dropout_rate=0, **kwargs):
        super(PatchCNN,self).__init__()
        self.base = ResNet(BasicBlock, layers, **kwargs)
        if dropout_rate !=0:
            self.down = nn.Sequential(nn.Conv2d(self.base.outdim,base_num*4,kernel_size=5,stride=2,padding=2),\
                                                  nn.InstanceNorm2d(base_num*4),
                                                  nn.LeakyReLU(0.2, True),
                                                  nn.Dropout(p=dropout_rate),

                                                  nn.Conv2d(base_num*4,base_num,kernel_size=5,stride=2,padding=2),\
                                                  nn.InstanceNorm2d(base_num),
                                                  nn.LeakyReLU(0.2, True),
                                                  nn.Dropout(p=dropout_rate),
                                                 )
        else:
            self.down = nn.Sequential(nn.Conv2d(self.base.outdim,base_num*4,kernel_size=5,stride=2,padding=2),\
                                          nn.InstanceNorm2d(base_num*4),
                                          nn.LeakyReLU(0.2, True),
                                          
                                          nn.Conv2d(base_num*4,base_num,kernel_size=5,stride=2,padding=2),\
                                          nn.InstanceNorm2d(base_num),
                                          nn.LeakyReLU(0.2, True),
                                         )    
        self.patch_classifier = nn.Conv2d(base_num,num_classes, kernel_size=3,padding=1,stride=1)
    
    def initialize(self,init_type='kaiming'):
        init_weights(self.base,init_type)
        init_weights(self.down,init_type)
        init_weights(self.patch_classifier,init_type)
    
    def forward(self,x):
        x = self.base(x)
        #print x.shape
        x = self.down(x)
        #print x.shape
        x = self.patch_classifier(x)
        return x
    
    
"""
class PatchCNN(nn.Module):
    def __init__(self,num_classes=11,**kwargs):
        super(PatchCNN,self).__init__()
        self.base = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
        self.patch_classifier = nn.Conv2d(self.base.outdim,num_classes,kernel_size=5,stride=1,padding=2)
    
    def initialize(self,init_type='kaiming'):
        init_weights(self.base,init_type)
        init_weights(self.patch_classifier,init_type)
    
    def forward(self,x):
        x = self.base(x)
        x = self.patch_classifier(x)
        return x
"""