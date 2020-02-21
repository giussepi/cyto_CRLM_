
import torch
import torch.nn as nn

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        #tmp_targets = targets.view(-1,1,1)
        if targets.is_cuda:
            tmp_device = targets.get_device()
            if  len(targets.size())==2 or  len(targets.size())==1:
                tmp_targets = torch.ones(size=(inputs.shape[0],inputs.shape[2],inputs.shape[3]),device=torch.device(tmp_device))*(targets.view(-1,1,1).float())
            elif len(targets.size())==3:
                tmp_targets = targets.to(tmp_device)
            else:
                raise RuntimeError('dimension Error')
        else:
            if  len(targets.size())==2:
                tmp_targets = torch.ones(size=(inputs.shape[0],inputs.shape[2],inputs.shape[3]))*(targets.view(-1,1,1))
            elif len(targets.size())==3:
                pass
            else:
                raise RuntimeError('dimension Error')
        tmp_targets = tmp_targets.long()
        return self.nll_loss(torch.nn.functional.log_softmax(inputs), tmp_targets)
    

class CrossEntropyLoss2d_unet(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-1):
        super(CrossEntropyLoss2d_unet, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(torch.nn.functional.log_softmax(inputs, dim=1), targets)
