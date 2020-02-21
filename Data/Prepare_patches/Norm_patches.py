import sys
import numpy as np
import pylab as plt
import os
sys.path.append('../../')

from Utils.StainNormalization import StainNormalizationWithVector,StainNormalizationWithVector2

h_ref = np.array([[0.24502717, 0.80708244, 0.34289746],[0.63904356, 0.67133489, 0.30034316]])

proot ='/home/zyx31/DATA_CRLM/Patches/Patches_Level0/Patches_1024/All/'
proot_norm = '/home/zyx31/DATA_CRLM/Patches/Patches_Level0/Patches_1024/Norm/'

pindex = 0
plist = os.listdir(proot)

for pindex in range(len(plist)):
    ppath = proot + plist[pindex]
    #ppath = proot + fname_dict['T'][0]
    print("processing %s"%plist[pindex])
    try:
        patch = np.array(plt.imread(ppath))[:,:,:3]
        normp  = StainNormalizationWithVector(patch,h_ref=h_ref,\
                                          illuminant_ref=[255,255,255],no_channel=2,Df=5, init=[260,320])

        #plt.imshow(normp)
        plt.imsave(proot_norm+plist[pindex],normp)
    
    except Exception as e:
        print(e)
    
    if pindex %200 ==0:
        print("----------------->",pindex)
    