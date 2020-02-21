import torch
from torch import nn
from .Alexnet_base import AlexNet
from .VGG_base import VGG
from .Resnet_base import ResNet,BasicBlock,Bottleneck
from .Dense_base import DenseNet

class TPC(nn.Module):
    """
    base_name : 'alexnet','vgg16','res18','res34','res50','dense'
    """
    def __init__(self,base_name,patch_size=224,num_classes=11,dropout_rate=0.5):
        super(TPC,self).__init__()
        self.patch_size = patch_size
        self.base_name = base_name
        if base_name =='alexnet':
            self.features = AlexNet()
            features_in=256
        elif base_name =='vgg16':
            self.features = VGG(cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'])
            features_in= 512 
        elif base_name =='res18':
            self.features = ResNet(BasicBlock, [2, 2, 2, 2])
            features_in= 512 if patch_size==224 else 2048
        elif base_name =='res34':
            self.features = ResNet(BasicBlock, [3, 4, 6, 3])
            features_in= 512 if patch_size==224 else 2048
        elif base_name =='res50':
            self.features = ResNet(Bottleneck, [3, 4, 6, 3])
            features_in= 2048 if patch_size==224 else 2048*4
        elif base_name =='dense':
            self.features = DenseNet(num_init_features=64, growth_rate=48, block_config=(3,4,6,3))
            #self.features = DenseNet(num_init_features=64, growth_rate=48, block_config=(6, 12, 24, 16))
            features_in= 362 if patch_size==224 else 362*4
        else:
            raise Exception("No model Found")
        
        if self.base_name in ['res18','res34','res50','dense']:
            stride = 1 if patch_size==224 else 7

            self.pool = nn.AvgPool2d(7, stride=stride)

            self.classifier1 = nn.Linear(features_in, num_classes)
        else:
            if patch_size==224:
                self.classifier2 = nn.Sequential(
                    nn.Linear(features_in * 7 * 7, 2048),
                    nn.ReLU(True),
                    nn.Dropout(dropout_rate),
                    nn.Linear(2048, 1024),
                    nn.ReLU(True),
                    nn.Dropout(dropout_rate),
                    nn.Linear(1024, num_classes),
                )
            else:
                self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
                self.classifier2 = nn.Sequential(
                    nn.Linear(features_in * 7 * 7, 2048),
                    nn.ReLU(True),
                    nn.Dropout(dropout_rate),
                    nn.Linear(2048, 1024),
                    nn.ReLU(True),
                    nn.Dropout(dropout_rate),
                    nn.Linear(1024, num_classes),
                )
        
    def forward(self,x):
        x = self.features(x)
        if self.base_name in ['res18','res34','res50']:
            x = self.pool(x)
            x= x.view(x.size(0), -1)
            #print(x.shape)
            x = self.classifier1(x)
            return(x)
        elif self.base_name in ['dense']:
            x = self.pool(x)
            x= x.view(x.size(0), -1)
            x = self.classifier1(x)
            return(x)
        else:
            if self.patch_size ==448:
                x = self.pool2(x)
            x= x.view(x.size(0), -1)
            #print(x.shape)
            x = self.classifier2(x)
            return(x)

if __name__=='__main__':

    patch_size = 224
    x = torch.randn(1,3,patch_size,patch_size)

    tpc_alx =TPC(base_name='alexnet',patch_size=patch_size)
    tpc_vgg =TPC(base_name='vgg16',patch_size=patch_size)
    tpc_res18 =TPC(base_name='res18',patch_size=patch_size)
    tpc_res34 =TPC(base_name='res34',patch_size=patch_size)
    tpc_res50 =TPC(base_name='res50',patch_size=patch_size)
    tpc_des =TPC(base_name='dense',patch_size=patch_size)

    print(tpc_alx(x).shape)
    print(tpc_vgg(x).shape)
    print(tpc_res18(x).shape)
    print(tpc_res34(x).shape)
    print(tpc_res50(x).shape)
    print(tpc_des(x).shape)