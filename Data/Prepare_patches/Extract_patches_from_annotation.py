import sys
import os
import pylab as plt
import numpy as np
sys.path.append('../')
from CRLM import CRLM

label_dict = {'H':'H','N':'N', 'F':'F', 'T':'T', 'I':'I', 'M':'M', 'B':'B', 'FB':'D' ,'MF':'C', 'G':'G','BD':'Y'}

def extract_perannotation(slide_index=41,annotation_index=0,\
                       patch_size=448,step = 128,max_patches= None,\
                       level=0,count_offset = 0,\
                       save_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_448/Eval/',
                       annotation_root = os.path.expanduser('~')+'/DATA_CRLM/CRLM/ndpa_bak/'):
    """
    Extract patches from annotations
    slide_index
    annotation_index 
    patch_size:
    step: step_size
    max_patches:  if not None, change step to get patches less than max
    """
    offsetx = 128
    offsety = 128
    centroid_region = int(patch_size/2)
    ratio = 0.9
    
    tc = CRLM(slide_index,annotation_root=annotation_root)
    label,img,mask = tc.ExtractAnnotationImage(annotation_index)
    ref_x,xa,ref_y,ya = tc.AnnotationBbox(annotation_index)
    tim = img
    
    def get_current_patch_nums(tstep):
        tcount = 0
        for ix in range(offsetx,tim.shape[0]-tstep,tstep):
            for iy in range(offsety,tim.shape[1]-tstep,tstep):
                if np.sum(mask[ix:ix+centroid_region,iy:iy+centroid_region]) > centroid_region*centroid_region*ratio:
                    tcount +=1
        return tcount
    
    if max_patches is not None:
        t_num_patches = get_current_patch_nums(step)
        while(t_num_patches>max_patches):
            step = step + 32
            t_num_patches = get_current_patch_nums(step)
    
    count = 0
    for ix in range(offsetx,tim.shape[0]-step,step):
        for iy in range(offsety,tim.shape[1]-step,step):
            if np.sum(mask[ix:ix+centroid_region,iy:iy+centroid_region]) > centroid_region*centroid_region*0.9:
                tim2 = np.array(tc.img.read_region(location=(ref_x+iy-48,ref_y+ix-48),level=level,size=(patch_size,patch_size)))
                plt.imsave(save_path+label_dict[label]+'_%03d_%04d.png'%(slide_index,count+count_offset),tim2)
                count +=1
            #else:
                #print(np.sum(mask[ix:ix+centroid_region,iy:iy+centroid_region]),patch_size**2/0.9)
        
    return count
    
def extract_perslide(slide_index=41,patch_size=448,step = 128,level=0,\
                       save_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_448/Eval/',
                       annotation_root = os.path.expanduser('~')+'/DATA_CRLM/CRLM/ndpa_bak/'):
    
    count_offset = 0
    for annotation_index in range((tc.root_len)):
        count =extract_perannotation(slide_index,annotation_index,\
                       patch_size,step,level,count_offset,\
                       save_path,annotation_root)
        count_offset += count
        print("Extracted % patches from annotation index "%(count,annotation_index))
    return count_offset
    
if __name__=='__main__':
    extract_perslide(slide_index=41,patch_size=448,step = 128,level=0,\
                       save_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_448/Eval/',
                       annotation_root = os.path.expanduser('~')+'/DATA_CRLM/CRLM/ndpa_bak/')
    