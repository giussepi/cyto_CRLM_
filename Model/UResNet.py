import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.downsample(x)
        out = F.relu(out)
        return out


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=0, bias=True)

    def forward(self, x, skip):
        out = self.convTrans(x)
        #diffX = out.size(2)-skip.size(2)
        #diffY = out.size(3)-skip.size(3)
        #out_skip = torch.nn.functional.pad(skip, (diffX // 2, int(diffX / 2),
                        #diffY // 2, int(diffY / 2)))        
        out = center_crop(out, skip.size(2), skip.size(3))
        #out = torch.cat([out, out_skip], 1)
        out = torch.cat([out, skip], 1)
        return out

def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]





class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class ResBlock(nn.Module):
    def __init__(self, planes, num_blocks, stride):
        super().__init__()
        strides = [stride] + [1]*(num_blocks-1)
        self.layers = nn.ModuleList()
        for stride in strides:
            layers.append(Bottleneck(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
    
    def forward(self, x):
        for layer in self.layers:
            x= layer(x)
        return x
        
class ResU(nn.Module):
    def __init__(self, num_classes=2, num_blocks=[2,2,2,2,2],\
                 num_channels=[64,64,128,128,256],\
                 strides =[2,2,2,2,2],block=Bottleneck):
        super(ResU, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        
        self.down_layers = nn.ModuleList()
        for i in range(len(num_blocks)):
            self.down_layers.append(self._make_layer(block, num_channels[i],num_blocks[i], stride=strides[i]))
        
        self.up_layers = nn.ModuleList()
        
        for j in reversed(range(len(num_blocks))):
            if j==0:
                self.in_planes = num_channels[1]*block.expansion 
                self.up_layers.append(nn.ConvTranspose2d(in_channels=self.in_planes, \
                                                         out_channels= num_channels[0],\
                                                         kernel_size=3, stride=2, padding=0, bias=True))
                self.in_planes = num_channels[0]
                self.up_layers.append(self._make_layer(block, num_channels[0], num_blocks[0], stride=1))
                
            else:
                self.up_layers.append(TransitionUp(num_channels[j]*block.expansion,num_channels[j-1]*block.expansion))
                self.in_planes = (num_channels[j-1]+num_channels[j])*block.expansion
                self.up_layers.append(self._make_layer_up(block,num_channels[j-1], num_blocks[j], stride=1))

        
        self.convf = nn.Conv2d(num_channels[0]*block.expansion, num_classes, \
                               kernel_size=3, stride=1, padding=1, bias=False)
        
        #self.softmax = nn.LogSoftmax(dim=1)
        self.fact = nn.Sigmoid()
        
        
    def forward(self,x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x = F.relu(x0)
        #x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        down_list= []
        for ti in range(len(self.down_layers)):
            x= self.down_layers[ti](x)
            print(ti,x.shape)
            down_list.append(x)
        reversed_list = list(reversed(down_list))
        
        for ti in range(0,len(self.up_layers),2):
            x= self.up_layers[ti](x,reversed_list[ti])
            print(ti,x.shape)
            x = self.up_layers[ti+1](x)
            print(ti,x.shape)
            
        out = self.convf(btn)
        #out = self.softmax(out)
        out = self.fact(out)
        return out

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _make_layer_up(self, block, planes, num_blocks, stride=1):
        """need to optimize later"""
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

