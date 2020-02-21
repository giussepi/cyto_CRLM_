import sys
import numpy as np
import pylab as plt
import os
from PIL import Image

label_name_list =['H', 'N', 'F', 'T', 'I', 'M', 'B', 'D' ,'C', 'G','Y']
def get_image_dict(fpath):
    fname_dict = dict(zip(label_name_list,[[],[],[],[],[],[],[],[],[],[],[]]))
    flist = os.listdir(fpath)
    #print(fname_dict.keys())
    for f in flist:
        fname_dict[f[0]].append(f)
    return fname_dict 

def get_rand_image_pair(fname_dict):
    rand_list = [['F','H'],['T','F'],['T','N'],['T','H'],\
                 ['F','H'],['T','F'],['T','N'],['T','H'],\
                 ['F','H'],['T','F'],['T','N'],['T','H'],\
                 ['T','I'],['I','H'],['F','I']]    # Manually adjust the weight
    #rand_list = [['T','H'],['F','H'],['T','F']]
    l1,l2 = rand_list[np.random.randint(len(rand_list))]
    #print(l1,l2)
    return fname_dict[l1][np.random.randint(len(fname_dict[l1]))],fname_dict[l2][np.random.randint(len(fname_dict[l2]))]

froot='/home/zyx31/DATA_CRLM/Patches/Patches_Level0/Patches_1024/Norm/'
fname_dict = get_image_dict(froot)


mroot = '/home/zyx31/DATA_CRLM/Patches/Patches_Segment/Patches_Syth_mask_1024/'
mlist= os.listdir(mroot)

ssroot = '/home/zyx31/DATA_CRLM/Patches/Patches_Segment/Synth_Patch_1024_Paired/Training/Sample/'
smroot = '/home/zyx31/DATA_CRLM/Patches/Patches_Segment/Synth_Patch_1024_Paired/Training/Mask/'

# ssroot = '/home/zyx31/DATA_CRLM/Patches/Patches_Segment/Synth_Patch_1024_Paired/Testing/Sample/'
# smroot = '/home/zyx31/DATA_CRLM/Patches/Patches_Segment/Synth_Patch_1024_Paired/Testing/Mask/'


sindex= 8000
nrepeat = 3
#mindex = 1

for mindex in range(len(mlist)):
    if mindex % 100==0:
        print(mindex,sindex)
    mpath = mroot+mlist[mindex]


    for _ in range(nrepeat):
        tmask = np.array(plt.imread(mpath), dtype = np.int)
        tmask[tmask == tmask.max()] =200
        tmask[tmask == tmask.min()] =100
        fname1,fname2 = get_rand_image_pair(fname_dict)
        tlabel1,tlabel2 = label_name_list.index(fname1[0]),label_name_list.index(fname2[0])
        tim1 = np.array(plt.imread(froot+fname1))[:,:,:3]
        tim2 = np.array(plt.imread(froot+fname2))[:,:,:3]

        patch_size = 1024
        synim = np.zeros((patch_size,patch_size,3))

        synim[tmask==100]=tim1[tmask==100]
        synim[tmask==200]=tim2[tmask==200]
        
        tmask[tmask==100] = tlabel1
        tmask[tmask==200] = tlabel2

        plt.imsave(ssroot+'synth_%04d.png'%sindex,synim)
        #plt.imsave(smroot+'synth_%04d.png'%sindex,np.array(tmask,dtype=np.int8))
        tmask_out = Image.fromarray(np.array(tmask,dtype=np.int8))
        tmask_out.save(smroot+'synth_%04d.png'%sindex)

        sindex +=1

print('<<<<------------Done------------->>>>')