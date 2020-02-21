#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 16:37:59 2016

@author: john
"""
import os
import openslide as ops
import numpy as np
import pylab as plt
from skimage import measure
from xml.etree import ElementTree
import cv2
import skimage.measure as measure
from skimage.transform import resize


fpre = os.path.expanduser('~')+'/DATA_CRLM/'


Aname_list = ['F','H','T','N','M','I','B','FB','MF','BD'] # Annotation name list

label_name_list =['H', 'N', 'F', 'T', 'I', 'M', 'B', 'D' ,'C', 'G','Y']

class CRLM():
    """
    index of image,
    level is magnification level
    fileroot is the folder for containing the files
    annotation_root is the folder for containing the annotations
    """
    def __init__(self,
                 index=1,
                 level=5,
                 fileroot = fpre+'CRLM/Original/',
                 annotation_root = None,
                 ):
        self.index = index
        self.level = level
        self.fileroot = fileroot
        self.GTroot = fileroot
        self.finame= 'CRLM_' + ('%03d' % self.index) + '.ndpi'
        self.img = ops.open_slide(self.fileroot+self.finame)
        self.root = self.ExtractRoot(self.index,rootname=annotation_root)
        self.root_len = self.root.__len__()
    
    def ExtractRoot(self,index,rootname=None):
        if rootname ==None:
            rootname2 = self.GTroot+self.finame+'.ndpa'
        else:
            rootname2 = rootname+self.finame+'.ndpa'
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
    
    def AnnotationBbox(self,index):
        pl_arr = self.AnnotationDots(index)
        pl_arr = np.vstack((pl_arr,pl_arr[0]))
        xi = pl_arr[:,0].min()
        xa = pl_arr[:,0].max()
        yi = pl_arr[:,1].min()
        ya = pl_arr[:,1].max() 
        return xi,xa,yi,ya
    
    
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

    def ExtractAnnotationImage(self,index =1, offsetx=128,offsety =128,  level = 0):
        """
        tc : CRLM instance
        index: index of the annotation
        offsetx,offsety:  offset for extending the anontation area
        level: magnification level for extaction
        """
        name = self.Get_Aname(index)
        pl_arr = self.AnnotationDots(index)
        pl_arr = np.vstack((pl_arr,pl_arr[0]))
        xi = pl_arr[:,0].min()
        xa = pl_arr[:,0].max()
        yi = pl_arr[:,1].min()
        ya = pl_arr[:,1].max()

        tmpim = self.img.read_region((xi-offsetx,yi-offsety),level,\
                                   (int((xa-xi+2*offsetx)/2**level),int((ya-yi+2*offsety)/2**level)))

        pl_arr2 = pl_arr
        pl_arr2[:,0] = pl_arr[:,0]-xi+offsetx
        pl_arr2[:,1] = pl_arr[:,1]-yi+offsety

        maskx = pl_arr2[:,0].max() + offsetx
        masky = pl_arr2[:,1].max() + offsety
        maskp = [(x,y) for x in range(maskx) for y in range(masky)]
        mask2 = measure.points_in_poly(maskp,pl_arr2)
        mask = np.reshape(mask2,(maskx,masky)).T
        mask = resize(mask, (np.array(tmpim).shape[0],np.array(tmpim).shape[1]))>0
        return name,np.array(tmpim),mask

def ExtractImage(tc,index =1, offsetx=128,offsety =128,  level = 0):
    """
    tc : CRLM instance
    index: index of the annotation
    offsetx,offsety:  offset for extending the anontation area
    level: magnification level for extaction
    """
    name = tc.Get_Aname(index)
    pl_arr = tc.AnnotationDots(index)
    pl_arr = np.vstack((pl_arr,pl_arr[0]))
    xi = pl_arr[:,0].min()
    xa = pl_arr[:,0].max()
    yi = pl_arr[:,1].min()
    ya = pl_arr[:,1].max()

    #print((xi-offsetx,yi-offsety),level,((xa-xi+2*offsetx)/2**level,(ya-yi+2*offsety)/2**level))
    tmpim = tc.img.read_region((xi-offsetx,yi-offsety),level,\
                               (int((xa-xi+2*offsetx)/2**level),int((ya-yi+2*offsety)/2**level)))
    
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


def extract_single(slide_index,level=2,ex_roi_list=[]):
    count =0 
    tc = CRLM(slide_index)
    def get_corner(index):
        tpl_arr = tc.AnnotationDots(index)
        xi = tpl_arr[:,0].min()
        xa = tpl_arr[:,0].max()
        yi = tpl_arr[:,1].min()
        ya = tpl_arr[:,1].max()
        return xi,yi,xa,ya

    def check_hole_in_Tumour(patch):
        bimg = color.rgb2gray(patch)
        if np.sum(bimg[48:176,48:176]>0.8) > 128*128/2:
            return True
        else:
            return False
    def Flip_Recolor_mix(img):
        di_dist = [None,'down','up','r90','r180','r270']
        direction = di_dist[np.random.randint(0,6)]
        if np.random.randint(0,10) <7:
            rand_color = (np.random.random_sample()*0.4+0.7,
                                np.random.random_sample()*0.4+0.7,
                                np.random.random_sample()*0.4+0.7,
                                np.random.random_sample()*0.4-0.2
                                )
        else:
            rand_color = None
        img = np.array(img,dtype = np.float)
        if direction=='down':
            timg = np.flipud(img)
        elif direction==None:
            timg = img
        elif direction=='r90':
            timg = np.array(rotate(img,90))
        elif direction=='r180':
            timg = np.array(rotate(img,180))
        elif direction=='r270':
            timg = np.array(rotate(img,270))
        else:
            timg = np.fliplr(img)

        if rand_color != None:
            timg[:,:,0] = timg[:,:,0]*rand_color[0]
            timg[:,:,2] = timg[:,:,2]*rand_color[1]
            #timg = timg*rand_color[2]
            timg = timg + rand_color[3]*160
            timg[timg>255] =255
            timg[timg<0] = 0
        return np.array(timg,dtype = np.uint8)

    ex_roi_cor_list = []
    if len(ex_roi_list)!=0:
        for roi_index in ex_roi_list:
            xi,yi,xa,ya = get_corner(roi_index)
            ex_roi_cor_list.append([xi,yi,xa,ya])

    def Check_ROI(index):
        #if the annotation is inside the ROI
        if len(ex_roi_list)==0:
            return False
        else:
            axi,ayi,axa,aya = get_corner(index)
            for t in ex_roi_cor_list:
                if axa <=t[0] or axi >= t[1] or aya <=t[2] or ayi>=t[3]: 
                    return False
                else:
                    print(axi,ayi,axa,aya , t)
                    return True

    for an_index in range(tc.root_len):
    #for an_index in range(10):
        lname =tc.Get_Aname(an_index)
        print(an_index, lname, tc.root_len)
        if lname == 'ROI' or  lname =='O':
            pass
        elif lname in ldic.keys():
            label = ldic[lname]
            _, tim, mask = ExtractImage(tc=tc,index=an_index,level=level)
            tmp_ref = get_corner(an_index)
            ref_x = tmp_ref[0]-128
            ref_y = tmp_ref[1]-128


            step = 96
            if tim.shape[0]*tim.shape[1] >1000000:
                step = 128
            elif tim.shape[0]*tim.shape[1] >9000000:
                step = 256  
            else:
                step = 96

            if  tim.shape[0] <224 or tim.shape[1] <224:
                print("Annotaion too small")
                pass
            elif Check_ROI(an_index) == True:
                print(an_index, 'inside the ROI')
                pass
            elif label=='T' and check_hole_in_Tumour(tim)==True:
                print("This is background")
                pass
            else:
                for ix in range(step*2,tim.shape[0]-step*2,step):
                    for iy in range(step*2,tim.shape[1]-step*2,step):
                        if np.sum(mask[ix:ix+128,iy:iy+128]) > 128*128*0.9:
                            # For Level 2 the offset is 336 + 48
                            if level ==2:
                                tim2 = np.array(tc.img.read_region(location=(ref_x+iy-336-48,ref_y+ix-336-48),level=level,size=(224,224)))
                            elif level ==1:
                                tim2 = np.array(tc.img.read_region(location=(ref_x+iy-112-48,ref_y+ix-112-48),level=level,size=(224,224)))
                            elif level ==0:
                                tim2 = np.array(tc.img.read_region(location=(ref_x+iy-48,ref_y+ix-48),level=level,size=(224,224)))
                            else:
                                tim2 = np.array(tc.img.read_region(location=(ref_x+iy-48,ref_y+ix-48),level=level,size=(224,224)))

                            cv2.imwrite(prefix+name_list[label]+'%02d_%05d'%(slide_index,count)+'.jpg',Flip_Recolor_mix(tim2))
                            count +=1

                            # Increase the samples of the other categories
                            if lname in ['I', 'M', 'B', 'FB' ,'MF','BD']:
                                cv2.imwrite(prefix+name_list[label]+'%02d_%05d'%(slide_index,count)+'.jpg',Flip_Recolor_mix(tim2))
                                count +=1

        else:
            pass
        if count%100 ==0:
            print(count)

if __name__ == "__main__":
    tc = CRLM(index=1)
