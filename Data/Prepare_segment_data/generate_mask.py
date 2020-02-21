import os
import numpy as np

from skimage.measure import regionprops
import openslide as ops
import pylab as plt
import copy

import cv2
from skimage.measure import label

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries



PATH = '/mnt/DATA_CRLM/Patches/CRLM_immuno/Original_Patches/Met_immuno/'

flist = os.listdir(PATH)


def get_masks(segments,patch_size= 448,min_segs = 2,max_segs=3):
    rx = np.random.randint(0,segments.shape[0]-448)
    ry = np.random.randint(0,segments.shape[1]-448)
    
    tregion = segments[rx:rx+448,ry:ry+448]
    sregions = regionprops(tregion)
    
    def check_mask(sregions):
        if len(sregions)<min_segs or len(sregions)>max_segs:
            return False
        for i in sregions:
            if i.area/(patch_size*patch_size*1.0)>0.7:
                return False
        return True

    def refine_mask(tmask,min_mask_thresh=0.2):
        nregions = label(tmask)
        fregions = copy.copy(nregions)
        nsregions = regionprops(nregions)
        for ti,treg in enumerate(nsregions):
            #print treg.label,treg.area/(patch_size*patch_size*1.0)
            if treg.area/(patch_size*patch_size*1.0)< min_mask_thresh:
                tregions = copy.copy(nregions)

                for tti in range(len(nsregions)):
                    tregions = copy.deepcopy(nregions)
                    tregions[nregions==treg.label] = tti+1
                    if label(tregions).max()<nregions.max():
                        fregions[nregions==treg.label] = tti+1
                        #print tnum,len(regionprops(label(fregions)))
                        break   
                        
        return label(fregions)
    
    tmp_count = 0
    while not (check_mask(sregions)):
        rx = np.random.randint(0,segments.shape[0]-448)
        ry = np.random.randint(0,segments.shape[1]-448)
        tregion = segments[rx:rx+448,ry:ry+448]
        sregions = regionprops(tregion)
        tmp_count +=1 
        if tmp_count> 30:
            return None,None
        
        
    fregions = refine_mask(tregion)
    fregions = refine_mask(fregions)
    fregions = refine_mask(fregions)
    return tregion,fregions


def check_mask_again(tmask,min_mask_thresh=0.2):
    if tmask is None:
        return False

    else:
        tregions = regionprops(tmask)
        for treg in tregions:
            if treg.area/(448*448*1.0) <min_mask_thresh:
                return False
        return True
    
tnum = 0

for fname in flist:
    img = plt.imread(PATH+fname)
    segments = slic(img[:,:,:3], n_segments = 100, sigma = 5)

    tcount = 0
    for _ in range(80):
    	tmask,fmask = get_masks(segments)
    	if check_mask_again(tmask):
    		cv2.imwrite('/mnt/DATA_CRLM/Patches/Patches_Segment/Patches_Syth_mask_448/mask_%04d.png'%tnum,fmask)
    		tnum += 1
        else:
            tcount +=1
            if tcount >30:
                break

    print fname,"End"
