import os
import openslide as ops
import numpy as np
import pylab as plt
from skimage import measure
from xml.etree import ElementTree
import cv2
import skimage.measure as measure
import mxnet as mx
from skimage.transform import resize



ldic = {'H':0, 'N':1, 'F':2, 'T':3, 'L':4, 'M':5, 'B':6, 'GC':7,'CP':8,'C':9,'SM':10,'FT':11}
label_names = ['histiocytes','Necrosis','Fibrosis','Tumour','lymphocytes','Mucin','Blood','Germinala Centra','Capillary','Capsule','Smooth Muscle','Tumour Fibrosis']

Aname_list = ldic.keys()

class CRC_LND():
    def __init__(self,
                 index=1,
                 level=5,
                 fileroot = '/mnt/DATA_CRLM/CRC_LND/',
                 prefix = 'CRC_LND_'
                 ):
        self.index = index
        self.level = level
        self.fileroot = fileroot
        self.GTroot = fileroot
        self.finame= prefix+'%04d' % self.index + '.ndpi'
        self.img = ops.open_slide(self.fileroot+self.finame)
        self.root = self.ExtractRoot(self.index)
        self.root_len = self.root.__len__()
    
    def ExtractRoot(self,index,rootname=None):
        if rootname ==None:
            rootname2 = self.GTroot+self.finame+'.ndpa'
        else:
            rootname2 = self.fdname+rootname+'.ndpa'
        with open(rootname2) as f:
            tree = ElementTree.parse(f)
        return tree.getroot()  

    
    def ConvertXY(self,ax,by):
        xmpp = float(self.img.properties['openslide.mpp-x'])
        xoff = float(self.img.properties['hamamatsu.XOffsetFromSlideCentre'])
        ympp = float(self.img.properties['openslide.mpp-y'])
        yoff = float(self.img.properties['hamamatsu.YOffsetFromSlideCentre'])
        ld = self.img.dimensions
        nax = int((ax - xoff)/(xmpp*1000.0)+ld[0]/2.0)
        nby = int((by - yoff)/(ympp*1000.0)+ld[1]/2.0)
        return nax,nby        
        

    def Get_Aname(self,index):
        if self.root[index].__len__() == 0:
            return 'O'
        if self.root[index][0].text != None and self.root[index][0].text.lower()!='roi':
            if self.root[index][0].text.__len__() == 1:
                return self.root[index][0].text.upper()
            else:
                tindexname = self.root[index][0].text.strip()
                if tindexname.upper() in Aname_list:
                    return tindexname.upper()
                # John:  need to change later accordingly
                elif tindexname.upper() in ['TF','FT']:
                    return 'FT' 
                else:
                    return tindexname[0].upper()
        elif self.root[index][0].text == None:
            return 'O'
        elif self.root[index][0].text.lower() == 'roi':
            return 'ROI'
        else:
            return 'O'

    def AnnotationDots(self,index):
        '''
        Transform the dots in the pixel coordinate
        '''
        if self.root[index].__len__() == 0:
            return np.array([[0,0],[0,0]])
        elif self.root[index][-1][-1].tag == 'pointlist':
            tplist = self.root[index][-1][-1]
        elif self.root[index][-1][-1].tag == 'specialtype':
            if self.root[index][-1][-3].tag == 'pointlist':
                tplist = self.root[index][-1][-3] 
            elif self.root[index][-1][-2].tag == 'pointlist':
                tplist = self.root[index][-1][-2]
            else:
                return np.array([])     
        else:
            return np.array([])
        Plist1 = []
        for i in range(tplist.__len__()): 
            Plist1.append(self.ConvertXY(int(tplist[i][0].text),int(tplist[i][1].text)))
        return np.array(Plist1)
    
        
    
    def Extract_WSI(self,level=7,offsetx=128,offsety=128):
        im = self.img.read_region((0-offsetx, \
                                  0-offsety),\
                                   level, \
                                   (self.img.dimensions[0] / (2**level), \
                                    self.img.dimensions[1] / (2**level)))
        return np.array(im)
    


    def ExtractImageBlock(self, startx=0, starty=0, sizex=100, sizey=100, level=None):
        if level == None:
            level = self.level
        im = self.img.read_region((startx, starty), level, (sizex, sizey))
        return np.array(im)[:, :, :3]

    def Plot_Annotation(self,index,level=6):
        offsetx = 128
        offsety = 128
        name = self.Get_Aname(index)
        pl_arr = self.AnnotationDots(index)
        pl_arr = np.vstack((pl_arr,pl_arr[0]))
        xi = pl_arr[:,0].min()
        xa = pl_arr[:,0].max()
        yi = pl_arr[:,1].min()
        ya = pl_arr[:,1].max()

        tmpim = self.img.read_region((xi-offsetx,yi-offsety),level,((xa-xi+2*offsetx)/2**level,(ya-yi+2*offsety)/2**level))
        plt.imshow(tmpim)
        return tmpim

    # Plot the overall view
    def PlotSlide(self,level=None):
        if level == None:
            level = self.level
        
        im = self.img.read_region((self.img.dimensions[0] / (2**level), self.img.dimensions[1] / (
            2**level)), level, (self.img.dimensions[0] / (2**level), self.img.dimensions[1] / (2**level)))
        
        im_arr = np.array(im)

        plt.imshow(im_arr[:, :, 0:3])


def ExtractImage(tc,index =1, offsetx=128,offsety =128,  level = 0):
    offsetx = 128
    offsety = 128
    name = tc.Get_Aname(index)
    pl_arr = tc.AnnotationDots(index)
    pl_arr = np.vstack((pl_arr,pl_arr[0]))
    xi = pl_arr[:,0].min()
    xa = pl_arr[:,0].max()
    yi = pl_arr[:,1].min()
    ya = pl_arr[:,1].max()

    tmpim = tc.img.read_region((xi-offsetx,yi-offsety),level,((xa-xi+2*offsetx)/2**level,(ya-yi+2*offsety)/2**level))
    
    #fig,ax = plt.subplots()
    #ax.imshow(tmpim)    
    #ax.axis('image')
    #ax.axis('off')
    #ax.plot((pl_arr[:,0]-xi+offsetx)/2**level,(pl_arr[:,1]-yi+offsety)/2**level,'y-',linewidth =3)
    
    pl_arr2 = pl_arr
    pl_arr2[:,0] = pl_arr[:,0]-xi+offsetx
    pl_arr2[:,1] = pl_arr[:,1]-yi+offsety
    
    maskx = pl_arr2[:,0].max() + offsetx
    masky = pl_arr2[:,1].max() + offsety
    maskp = [(x,y) for x in range(maskx) for y in range(masky)]
    mask2 = measure.points_in_poly(maskp,pl_arr2)
    mask = np.reshape(mask2,(maskx,masky)).T
    mask = resize(mask, (np.array(tmpim).shape[0],np.array(tmpim).shape[1]))>0

    #plt.figure()
    #plt.imshow(mask)
    return name,np.array(tmpim),mask



if __name__ == "__main__":
    test_wsi = CRC_LND(index=0)
