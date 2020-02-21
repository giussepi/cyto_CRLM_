from CRLM import CRLM,ExtractImage

import mxnet as mx
import os
import numpy as np
from skimage import transform
import pylab as plt

fpre = '/mnt/DATA/'

class ROI_Mask():
    def __init__(self,index = 13):
        self.tc = CRLM(index)
        self.roi_list = self.Find_ROI()
        self.ldict = {'H':8, 'N':1, 'F':2, 'T':3, 'I':4, 'M':5, 'B':6, 'FB':7 }

    def Find_ROI(self):
        roi_list = []
        for i in range(self.tc.root_len):
            if self.tc.Get_Aname(i) == 'ROI':
                roi_list.append(i)
        return roi_list
    def Find_Contours_in_ROI(self,index):
        rxi,rxa,ryi,rya = self.Get_BPoint(index)
        nxi,nxa,nyi,nya = self.Get_BPoint(index)
        cin_list = []    
        for i in range(self.tc.root_len):
            if self.tc.Get_Aname(i) not in ['H','N','F','T','I','M','B','FB']:
                pass
            else:
                xi,xa,yi,ya = self.Get_BPoint(i)
                if  xi > rxa or xa< rxi or yi >rya and ya < ryi:
                    pass
                else:
                    cin_list.append(i)
                

                if xi < nxi:
                    nxi = xi
                if xa > nxa:
                    nxa = xa
                if yi < nyi:
                    nyi = yi
                if ya > nya:
                    nya = ya
        return cin_list, [nxi,nxa,nyi,nya]

    def Get_BPoint(self,index):
        pl_arr = self.tc.AnnotationDots(index)
        rxi = pl_arr[:,0].min()
        rxa = pl_arr[:,0].max()
        ryi = pl_arr[:,1].min()
        rya = pl_arr[:,1].max()
        return rxi,rxa,ryi,rya
    
    def Get_ROI_image(self,index,level=5,offset=32):
        rxi,rxa,ryi,rya = self.Get_BPoint(index)
        img = self.tc.img.read_region((rxi-offset,ryi-offset),level=level,size=((rxa-rxi)/2**level,(rya-ryi)/2**level))
        return np.array(img)
    
    def Make_ROI_Mask(self,index,level=5):
        #img.shape has problem !!!!!! carefull
        index =0
        level =5
        rxi,rxa,ryi,rya = self.Get_BPoint(index)
        cin_list,[nxi,nxa,nyi,nya] = self.Find_Contours_in_ROI(index)
        offsetx = (rxi-nxi)/2**level
        offsety = (ryi-nyi)/2**level
        nxa = nxa+offsetx*2**level
        nya = nya+128*2**level
        #_,img,_ = ExtractImage(roi_mask.tc,index=index,level=level)
        img = self.tc.img.read_region((nxi,nyi),level=level,size=((nxa-nxi)/2**level,(nya-nyi)/2**level))
        img = np.array(img)
        fmask = np.zeros((img.shape[1]+offsetx,img.shape[0]+offsety))*10

        print("Image shape ",img.shape[0], img.shape[1])

        for cindex in cin_list:
            print "process", cindex, "total is: ",len(cin_list)
            xi,xa,yi,ya = self.Get_BPoint(cindex)
            name,_,tmask = ExtractImage(self.tc,index=cindex,level=level)
            tmask = np.array(tmask)* self.ldict[name]
            tmask = tmask.T
            sx = (xi-nxi)/(2**level)
            sy = (yi-nyi)/(2**level)
            print sx, sy ,fmask[sx:sx+tmask.shape[0],sy:sy+tmask.shape[1]].shape
            fmask[sx:sx+tmask.shape[0],sy:sy+tmask.shape[1]] += tmask

        
        lx = (rxa-rxi)/2**level
        ly  =(rya-ryi)/2**level

        return img[offsetx:offsetx+lx,offsety:offsety+ly,], fmask[offsetx:offsetx+lx,offsety:offsety+ly,].T
  
if __name__ == '__main__':
    roi_mask = ROI_Mask(13)
    img,mask = roi_mask.Make_ROI_Mask(roi_mask.roi_list[0])


    offsetx = (rxi-nxi)/2**level
    offsety = (ryi-nyi)/2**level
    lx = (rxa-rxi)/2**level
    ly  =(rya-ryi)/2**level
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(mask)