import torch
import sys
import numpy as np
import pylab as plt
import os

sys.path.append('../../')
from Model.PatchCNN import PatchCNN
from Load_trained.process_large_patches import process_large_image
from Data.get_dataloader import get_dataloader

device = torch.device("cuda:0")
num_layers = [3,4,6,3]  # res34
dropout_rate = 0
model = PatchCNN(layers=num_layers,dropout_rate=dropout_rate)
state_dict = torch.load('/mnt/DATA_CRLM/Patches/Checkpoints/PatchCNN/Legacy/PatchCNN_448_res34/PatchCNN_best.pth')
new_state_dict = {}
for key in model.state_dict():
    new_state_dict[key] = state_dict['module.'+key].double()
model.load_state_dict(new_state_dict)
model.eval()
model.to(device)


import shutil
label_name_list =['H', 'N', 'F', 'T', 'I', 'M', 'B', 'D' ,'C', 'G','Y']
proot ='/home/zyx31/DATA_CRLM/Patches/Patches_Level0/Patches_1024/All/'
proot_mix = '/home/zyx31/DATA_CRLM/Patches/Patches_Level0/Patches_1024/Mixed/'
plist = os.listdir(proot)
pindex = 0



for pindex in range(len(plist)):
    try:
        print('Processing image %s'%plist[pindex])
        ppath = proot + plist[pindex]
        label = label_name_list.index(plist[pindex][0])
        patch = np.array(plt.imread(ppath))[:,:,(2,1,0)]

        pr = process_large_image(model,patch,step=224)
        plprob ,plabel = torch.max(torch.tensor(pr),dim=2)
        a,b,c,d = (label_name_list[label], torch.sum(plabel==label),\
                   torch.mean(plprob[plabel==label]),\
                   torch.std((plprob[plabel==label])))
    except Exception as e:
        print(e)
        continue

    if b < 140 or c<0.98 or d > 0.05:
        shutil.move(ppath,proot_mix+plist[pindex])
        print(plist[pindex])
        
    if pindex %200 ==0:
        print('--------------------------->>>',pindex)
            
print(">>>>----------<<<<<<-DONE->>>>>>>>-----------<<<<")