## from https://github.com/vlievin/Unet/blob/master/unet.py
import numpy as np
import torch
import torch.nn as nn

class gated_resnet(nn.Module):
    """
    Gated Residual Block
    """
    def __init__(self, num_filters, kernel_size, padding, nonlinearity=nn.ReLU, dropout=0.2, dilation=1,batchNormObject=nn.BatchNorm2d):
        super(gated_resnet, self).__init__()
        self.gated = True
        num_hidden_filters =2 * num_filters if gated else num_filters
        self.conv_input = nn.Conv2d(num_filters, num_hidden_filters, kernel_size=kernel_size,stride=1,padding=padding,dilation=dilation )
        self.dropout = nn.Dropout2d(dropout)
        self.nonlinearity = nonlinearity()
        self.batch_norm1 = batchNormObject(num_hidden_filters)
        self.conv_out = nn.Conv2d(num_hidden_filters, num_hidden_filters, kernel_size=kernel_size,stride=1,padding=padding,dilation=dilation )
        self.batch_norm2 = batchNormObject(num_filters)

    def forward(self, og_x):
        x = self.conv_input(og_x)
        x = self.batch_norm1(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        if self.gated:
            a, b = torch.chunk(x, 2, dim=1)
            c3 = a * F.sigmoid(b)
        else:
            c3 = x
        out = og_x + c3
        out = self.batch_norm2(out)
        return out
    
class ResidualBlock(nn.Module):
    """
    Residual Block
    """
    def __init__(self, num_filters, kernel_size, padding, nonlinearity=nn.ReLU, dropout=0.2, dilation=1,batchNormObject=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        num_hidden_filters = num_filters
        self.conv1 = nn.Conv2d(num_filters, num_hidden_filters, kernel_size=kernel_size,stride=1,padding=padding,dilation=dilation )
        self.dropout = nn.Dropout2d(dropout)
        self.nonlinearity = nonlinearity(inplace=False)
        self.batch_norm1 = batchNormObject(num_hidden_filters)
        self.conv2 = nn.Conv2d(num_hidden_filters, num_hidden_filters, kernel_size=kernel_size,stride=1,padding=padding,dilation=dilation )
        self.batch_norm2 = batchNormObject(num_filters)

    def forward(self, og_x):
        x = og_x
        x = self.dropout(x)
        x = self.conv1(og_x)
        x = self.batch_norm1(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        out = og_x + x
        out = self.batch_norm2(out)
        out = self.nonlinearity(out)
        return out

class ConvBlock(nn.Module):
    """
    Conv Block
    """
    def __init__(self, num_filters, kernel_size, padding, nonlinearity=nn.ReLU, dropout=0.2, dilation=1,batchNormObject=nn.BatchNorm2d):
        super(ConvBlock, self).__init__()
        num_hidden_filters = num_filters
        self.conv1 = nn.Conv2d(num_filters, num_hidden_filters, kernel_size=kernel_size,stride=1,padding=padding,dilation=dilation )
        self.dropout = nn.Dropout2d(dropout)
        self.nonlinearity = nonlinearity(inplace=False)
        self.batch_norm1 = batchNormObject(num_hidden_filters)

    def forward(self,x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.nonlinearity(x)
        return x
    
    


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate,kernel_size=3,\
                 padding=1,nonlinearity=nn.ReLU, dropout=0.2, \
                 dilation=1,batchNormObject=nn.BatchNorm2d):
        super().__init__()
        self.add_module('norm', batchNormObject(in_channels))
        self.add_module('relu', nonlinearity(True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size,
                                          stride=1, padding=padding, bias=True))
        self.add_module('drop', nn.Dropout2d(dropout))

    def forward(self, x):
        return super().forward(x)

#------------No Working---------------
#-------------To do---------------
class DenseBlock(nn.Module):
    def __init__(self, num_filters, kernel_size, padding, nonlinearity=nn.ReLU, dropout=0.2, dilation=1,batchNormObject=nn.BatchNorm2d,upsample=False):
        
    #def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(in_channels + i*growth_rate, growth_rate,\
                                                kernel_size=kernel_size,padding=padding,\
                                                nlinearity=nn.ReLU, dropout=0.2, dilation=1,\
                                                batchNormObject=nn.BatchNorm2d) \
                                     for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) # 1 = channel axis
            return x

    
class ConvolutionalEncoder(nn.Module):
    """
    Convolutional Encoder providing skip connections
    """
    def __init__(self,n_features_input,num_hidden_features,kernel_size,padding,n_resblocks,dropout_min=0,dropout_max=0.2, blockObject=ResidualBlock,batchNormObject=nn.BatchNorm2d):
        """
        n_features_input (int): number of intput features
        num_hidden_features (list(int)): number of features for each stage
        kernel_size (int): convolution kernel size
        padding (int): convolution padding
        n_resblocks (int): number of residual blocks at each stage
        dropout (float): dropout probability
        blockObject (nn.Module): Residual block to use. Default is ResidualBlock
        batchNormObject (nn.Module): normalization layer. Default is nn.BatchNorm2d
        """
        super(ConvolutionalEncoder,self).__init__()
        self.n_features_input = n_features_input
        self.num_hidden_features = num_hidden_features
        self.stages = nn.ModuleList()
        dropout_list = [(1-t)*dropout_min + t*dropout_max   for t in np.linspace(0,1,(len(num_hidden_features)))]
        dropout = iter(dropout_list)
        # input convolution block
        block = [nn.Conv2d(n_features_input, num_hidden_features[0], kernel_size=kernel_size,stride=1, padding=padding),\
                batchNormObject(num_hidden_features[0]),
                nn.ReLU()]
        for _ in range(n_resblocks):
            p = next(dropout)
            block += [blockObject(num_hidden_features[0], kernel_size, padding, dropout=p,batchNormObject=batchNormObject)]
        self.stages.append(nn.Sequential(*block))
        # layers
        dropout = iter(dropout_list)
        for features_in,features_out in [num_hidden_features[i:i+2] for i in range(0,len(num_hidden_features), 1)][:-1]:
            # downsampling
            block = [nn.MaxPool2d(2),nn.Conv2d(features_in, features_out, kernel_size=1,padding=0 ),batchNormObject(features_out),nn.ReLU()]
            #block = [nn.Conv2d(features_in, features_out, kernel_size=kernel_size,stride=2,padding=padding ),nn.BatchNorm2d(features_out),nn.ReLU()]
            # residual blocks
            p = next(dropout)
            for _ in range(n_resblocks):
                block += [blockObject(features_out, kernel_size, padding, dropout=p,batchNormObject=batchNormObject)]
            self.stages.append(nn.Sequential(*block)) 
            
    def forward(self,x):
        skips = []
        for stage in self.stages:
            x = stage(x)
            #print(x.shape)
            skips.append(x)
        return x,skips
    def getInputShape(self):
        return (-1,self.n_features_input,-1,-1)
    def getOutputShape(self):
        return (-1,self.num_hidden_features[-1], -1,-1)
    
            
class ConvolutionalDecoder(nn.Module):
    """
    Convolutional Decoder taking skip connections
    """
    def __init__(self,n_features_output,num_hidden_features,\
                     kernel_size,padding,n_resblocks,dropout_min=0,\
                     dropout_max=0.2,\
                     blockObject=ResidualBlock,batchNormObject=nn.BatchNorm2d):
        """
        n_features_output (int): number of output features
        num_hidden_features (list(int)): number of features for each stage
        kernel_size (int): convolution kernel size
        padding (int): convolution padding
        n_resblocks (int): number of residual blocks at each stage
        dropout (float): dropout probability
        blockObject (nn.Module): Residual block to use. Default is ResidualBlock
        batchNormObject (nn.Module): normalization layer. Default is nn.BatchNorm2d
        """
        super(ConvolutionalDecoder,self).__init__()
        self.n_features_output = n_features_output
        self.num_hidden_features = num_hidden_features
        self.upConvolutions = nn.ModuleList()
        self.skipMergers = nn.ModuleList()
        self.residualBlocks = nn.ModuleList()
        dropout = iter([(1-t)*dropout_min + t*dropout_max   for t in np.linspace(0,1,(len(num_hidden_features)))][::-1])
        # input convolution block
        # layers
        for features_in,features_out in [num_hidden_features[i:i+2] for i in range(0,len(num_hidden_features), 1)][:-1]:
            # downsampling
            self.upConvolutions.append(nn.Sequential(nn.ConvTranspose2d(features_in, features_out, kernel_size=3, stride=2,padding=1,output_padding=1),batchNormObject(features_out),nn.ReLU()))
            self.skipMergers.append(nn.Conv2d(2*features_out, features_out, kernel_size=kernel_size,stride=1, padding=padding))
            # residual blocks
            block = []
            p = next(iter(dropout))
            for _ in range(n_resblocks):
                block += [blockObject(features_out, kernel_size, padding, dropout=p,batchNormObject=batchNormObject)]
            self.residualBlocks.append(nn.Sequential(*block))   
        # output convolution block
        block = [nn.Conv2d(num_hidden_features[-1],n_features_output, kernel_size=kernel_size,stride=1, padding=padding)]
        self.output_convolution = nn.Sequential(*block)

    def forward(self,x, skips):
        for up,merge,conv,skip in zip(self.upConvolutions,self.skipMergers, self.residualBlocks,skips):
            x = up(x)
            cat = torch.cat([x,skip],1)
            x = merge(cat)
            x = conv(x)
        return self.output_convolution(x)
    
    def getInputShape(self):
        return (-1,self.num_hidden_features[0],-1,-1)
    def getOutputShape(self):
        return (-1,self.n_features_output, -1,-1)
    
    
class DilatedConvolutions(nn.Module):
    """
    Sequential Dialted convolutions
    """
    def __init__(self, n_channels, n_convolutions, dropout):
        super(DilatedConvolutions, self).__init__()
        kernel_size = 3
        padding = 1
        self.dropout = nn.Dropout2d(dropout)
        self.non_linearity = nn.ReLU(inplace=True)
        self.strides = [2**(k+1) for k in range(n_convolutions)]
        convs = [nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size,dilation=s, padding=s) for s in self.strides ]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for c in convs:
            self.convs.append(c)
            self.bns.append(nn.BatchNorm2d(n_channels))
    def forward(self,x):
        skips = []
        for (c,bn,s) in zip(self.convs,self.bns,self.strides):
            x_in = x
            x = c(x)
            x = bn(x)
            x = self.non_linearity(x)
            x = self.dropout(x)
            x = x_in + x
            skips.append(x)
        return x,skips
    
class DilatedConvolutions2(nn.Module):
    """
    Sequential Dialted convolutions
    """
    def __init__(self, n_channels, n_convolutions,dropout,kernel_size,blockObject=ResidualBlock,batchNormObject=nn.BatchNorm2d):
        super(DilatedConvolutions2, self).__init__()
        self.dilatations = [2**(k+1) for k in range(n_convolutions)]
        self.blocks = nn.ModuleList([blockObject(n_channels, kernel_size, d, dropout=dropout, dilation=d,batchNormObject=batchNormObject) for d in self.dilatations ])
    def forward(self,x):
        skips = []
        for b in self.blocks:
            x = b(x)
            skips.append(x)
        return x, skips
    
class UNet(nn.Module):
    """
    U-Net model with dynamic number of layers, Residual Blocks, Dilated Convolutions, Dropout and Group Normalization
    """
    def __init__(self, in_channels, out_channels, num_hidden_features,\
                 n_blocks,num_dilated_convs, dropout_min=0, dropout_max=0,\
                 block_type='res', padding=1, kernel_size=3,group_norm=32):
        
        """
        initialize the model
        Args:
            in_channels (int): number of input channels (image=3)
            out_channels (int): number of output channels (n_classes)
            num_hidden_features (list(int)): number of hidden features for each layer (the number of layer is the lenght of this list)
            n_resblocks (int): number of residual blocks at each layer 
            num_dilated_convs (int): number of dilated convolutions at the last layer
            dropout (float): float in [0,1]: dropout probability
            gated (bool): use gated Convolutions, default is False
            padding (int): padding for the convolutions
            kernel_size (int): kernel size for the convolutions
            group_norm (bool): number of groups to use for Group Normalization, default is 32, if zero: use nn.BatchNorm2d
            
            n_resblocks --> n_blocks
            gated --> block_type    ['res','gated','conv','dense']
            
        """
        super(UNet, self).__init__()
        if group_norm > 0:
            for h in num_hidden_features:
                assert h%group_norm==0, "Number of features at each layer must be divisible by 'group_norm'"
        
        #blockObject = gated_resnet if gated else ResidualBlock
        if block_type == 'gated':
            blockObject = gated_resnet
        elif block_type == 'conv':
            blockObject = ConvBlock
        elif block_type == 'res':
            blockObject = ResidualBlock
        elif block_type == 'dense':
            blockObject = DenseBlock
        else:
            blockObject = ResidualBlock
        
        n_resblocks = n_blocks
        batchNormObject = lambda n_features : nn.GroupNorm(group_norm,n_features) \
                        if group_norm > 0 else  nn.BatchNorm2d(n_features)
        self.encoder = ConvolutionalEncoder(in_channels,num_hidden_features,\
                                            kernel_size,padding,n_resblocks,\
                                            dropout_min=dropout_min,dropout_max=dropout_max,\
                                            blockObject=blockObject,batchNormObject=batchNormObject)
        if num_dilated_convs > 0:
            #self.dilatedConvs = DilatedConvolutions2(num_hidden_features[-1], num_dilated_convs,dropout_max,kernel_size,blockObject=blockObject,batchNormObject=batchNormObject)
            self.dilatedConvs = DilatedConvolutions(num_hidden_features[-1],num_dilated_convs,dropout_max) # <v11 uses dilatedConvs2
        else:
            self.dilatedConvs = None
        self.decoder = ConvolutionalDecoder(out_channels,num_hidden_features[::-1],\
                                            kernel_size,padding,n_resblocks,\
                                            dropout_min=dropout_min,dropout_max=dropout_max,\
                                            blockObject=blockObject,batchNormObject=batchNormObject)
        
    def forward(self, x):
        x,skips = self.encoder(x)
        if self.dilatedConvs is not None:
            x,dilated_skips = self.dilatedConvs(x)
            for d in dilated_skips:
                x += d
            x += skips[-1]
        x = self.decoder(x,skips[:-1][::-1])
        return x