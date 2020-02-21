# -*- coding: utf-8 -*-
"""
Created on Mon Mar 07 22:16:52 2016

@author: JOHN
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:31:40 2016

@author: JOHN
"""

from xml.etree import ElementTree
import openslide as ops
import numpy as np
import os
from piwigotools import Piwigo
from piwigotools.interface import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')



class Histimage():
    def __init__(self,fdname,finame):
        self.fdname = fdname
        self.finame = finame
        self.im = ops.OpenSlide(fdname+finame+'.ndpi')
        self.root = self.ExtractRoot()
    
    def ExtractRoot(self):
        with open(self.fdname+self.finame+'.ndpi.ndpa') as f:
            tree = ElementTree.parse(f)
        return tree.getroot()  
    
    def ConvertXY(self,ax,by):
        xmpp = float(self.im.properties['openslide.mpp-x'])
        xoff = float(self.im.properties['hamamatsu.XOffsetFromSlideCentre'])
        ympp = float(self.im.properties['openslide.mpp-y'])
        yoff = float(self.im.properties['hamamatsu.YOffsetFromSlideCentre'])
        ld = self.im.dimensions
        nax = int((ax - xoff)/(xmpp*1000.0)+ld[0]/2.0)
        nby = int((by - yoff)/(ympp*1000.0)+ld[1]/2.0)
        return nax,nby        
        
    def NameAnnotation(self,index):
        if self.root[index][0].text != None and self.root[index][0].text != 'roi' :
                name = 'ID-'+self.root[index].attrib['id']+'-'+self.root[index][0].text
        elif self.root[index][0].text == 'roi'or self.root[index][0].text == 'ROI':
            name = 'ID-'+self.root[index].attrib['id']+'-'+'R'
        else:
            name = 'ID-'+self.root[index].attrib['id']+'-'+'O'
        return name
    
    
    def AnnotationDots(self,index):
        if self.root[index][-1][-1].tag == 'pointlist':
            tplist = self.root[index][-1][-1]
        elif self.root[index][-1][-1].tag == 'specialtype' and self.root[index][-1][-3].tag == 'pointlist':
            tplist = self.root[index][-1][-3] 
        else:
            return np.array([])
        Plist1 = []
        for i in range(tplist.__len__()): 
            Plist1.append(self.ConvertXY(int(tplist[i][0].text),int(tplist[i][1].text)))
        return np.array(Plist1)
    
        
        
    def ExtractImage(self,index):
        name = self.NameAnnotation(index)
        pl_arr = self.AnnotationDots(index)
        pl_arr = np.vstack((pl_arr,pl_arr[0]))
        xi = pl_arr[:,0].min()
        xa = pl_arr[:,0].max()
        yi = pl_arr[:,1].min()
        ya = pl_arr[:,1].max()
        if (xa-xi)*(ya-yi) >1000*1000:
            level = 2
        else:
            level = 0
        tmpim = self.im.read_region((xi-100,yi-100),level,((xa-xi+200)/2**level,(ya-yi+200)/2**level))
        fig,ax = plt.subplots()
        ax.imshow(tmpim)    
        ax.axis('image')
        ax.axis('off')
        ax.plot((pl_arr[:,0]-xi+100.0)/2**level,(pl_arr[:,1]-yi+100.0)/2**level,'y-',linewidth =3)
        #plt.savefig(name,dpi=fig.dpi )
        self.SaveFigureAsImage(name,fig = fig, orig_size=(np.array(tmpim).shape[0],np.array(tmpim).shape[1]))
        plt.close(fig)
        return name

    def SaveFigureAsImage(self,fileName,fig=None,**kwargs):
        ''' Save a Matplotlib figure as an image without borders or frames.
           Args:
                fileName (str): String that ends in .png etc.
     
                fig (Matplotlib figure instance): figure you want to save as the image
            Keyword Args:
                orig_size (tuple): width, height of the original image used to maintain 
                aspect ratio.
        '''
        fig_size = fig.get_size_inches()
        w,h = fig_size[0], fig_size[1]
        fig.patch.set_alpha(0)
        if kwargs.has_key('orig_size'): # Aspect ratio scaling if required
            w,h = kwargs['orig_size']
            w2,h2 = fig_size[0],fig_size[1]
            fig.set_size_inches([(w2/w)*w,(w2/w)*h])
            fig.set_dpi((w2/w)*fig.get_dpi())
        a=fig.gca()
        a.set_frame_on(False)
        a.set_xticks([]); a.set_yticks([])
        plt.axis('off')
        plt.xlim(0,h); plt.ylim(w,0)
        fig.savefig(fileName, transparent=True,bbox_inches='tight', pad_inches=0)
        

class HistUpload():
    def __init__(self,finame):
        self.mysite = Piwigo(url='http://qmul.piwigo.com/')
        self.mysite.login(username="qmul", password="xuzhaoyang")       
        self.webfdlist = self.mysite.plan     
        self.webparent = r'/Test2/'
        self.finame = finame
        if self.webfdlist.has_key(self.webparent+self.finame) == False:
            self.mysite.mkdir(self.webparent+self.finame)
            
    def __del__(self):
        self.mysite.logout()
        
    def Upload(self,imgname):
        fdpath = self.webparent+self.finame+r'/'+imgname[-1].upper()
        if self.webfdlist.has_key(fdpath) == False:
            self.mysite.mkdir(fdpath)
        try:
            self.mysite.upload(imgname+'.png',fdpath)     
        except Exception as e:
            print imgname
            print e
            self.mysite.upload(imgname+'.png',fdpath)
        
        os.remove(imgname+'.png')
    
    def Logout(self):
        self.mysite.logout()


if __name__ == '__main__':
    #testhist = Histimage(r'D:/DATASET/CLRM/',r'2016-01-13 16.44.33_1_Carlos')
    testhist = Histimage(r'N:/CLRM/',r'2016-01-13 16.44.33_1_Carlos')
    testupload = HistUpload(testhist.finame)
    for index in range(testhist.root.__len__()):
        testupload.Upload(testhist.ExtractImage(index))
        print index,'Uploaded'
    testupload.Logout()





        
        
