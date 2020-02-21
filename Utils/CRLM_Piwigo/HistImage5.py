
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:31:40 2016

HistImage works well to extract all the annotations
@author: JOHN
"""

from xml.etree import ElementTree
import openslide as ops
import numpy as np
import os
from piwigotools import Piwigo
from piwigotools.interface import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


tfprefix = '/tmp/'
Abr_list = ['F','H','T','N','M','I','B','FB','MF']

class HistImage():
    def __init__(self,fdname,finame,rootname = None):
        self.fdname = fdname
        if finame.split('.')[-1] =='ndpi':
            self.finame = finame.split('.')[0]
        else:
            self.finame = finame
        self.rootname = rootname
        self.im = ops.OpenSlide(self.fdname+self.finame+'.ndpi')
        self.root = self.ExtractRoot()
    
    def ExtractRoot(self):
        if self.rootname ==None:
            rootname2 = self.fdname+self.finame+'.ndpi.ndpa'
        else:
            rootname2 = self.fdname+self.rootname+'.ndpi.ndpa'
        with open(rootname2) as f:
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
        
    def NameAnnotation_old(self,index):
        if self.root[index][0].text != None and self.root[index][0].text.lower()!='roi':
            if self.root[index][0].text.__len__() == 1:
                name = 'ID-'+self.root[index].attrib['id']+'-'+self.root[index][0].text.upper()
            else:
                tindexname = self.root[index][0].text.strip()
                if tindexname[0].upper() in ['F','H','T','N']:
                    name = 'ID-'+self.root[index].attrib['id']+'-'+tindexname[0].upper()
                else:
                    name = 'ID-'+self.root[index].attrib['id']+'-'+'O'
        elif self.root[index][0].text.lower() == 'roi':
            name = 'ID-'+self.root[index].attrib['id']+'-'+'u'
        else:
            name = 'ID-'+self.root[index].attrib['id']+'-'+'O'
        return name

    def NameAnnotation(self,index):
        try:        # Exception attrib['id'] No id found
            Oname = 'ID-'+self.root[index].attrib['id']+'-'+'O'
        except Exception as e:
            print e
            return None
        if self.root[index].__len__() == 0:
            return Oname
        if self.root[index][0].text != None and self.root[index][0].text.lower()!='roi':
            if self.root[index][0].text.__len__() == 1:
                tindexname = self.root[index][0].text.strip()
                if tindexname.upper() in Abr_list:
                    name = 'ID-'+self.root[index].attrib['id']+'-'+tindexname.upper()
                    return name
                else:
                    return Oname
            else:
                tindexname = self.root[index][0].text.strip()
                if tindexname.upper() in Abr_list:
                    name = 'ID-'+self.root[index].attrib['id']+'-'+tindexname.upper()
                    return name
                else:
                    return Oname
        elif self.root[index][0].text == None:
            return Oname
        elif self.root[index][0].text.lower() == 'roi':
            name = 'ID-'+self.root[index].attrib['id']+'-'+'u'
            return name 
        else:
            return Oname
    
    
    def AnnotationDots(self,index):
        '''
        Transform the dots in the pixel coordinate
        '''
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
        '''
        extract the image from the raw format and save it
        '''
        offsetx = 0
        offsety = 0
        name = self.NameAnnotation(index)
        if name == None:
            return None
        pl_arr = self.AnnotationDots(index)
        if len(pl_arr)<1:
            return None
        pl_arr = np.vstack((pl_arr,pl_arr[0]))
        xi = pl_arr[:,0].min()
        xa = pl_arr[:,0].max()
        yi = pl_arr[:,1].min()
        ya = pl_arr[:,1].max()

        if (xa-xi)*(ya-yi) >10000*10000:
            level = 2
        elif(xa-xi)*(ya-yi) >3000*3000:
            level = 1
        else:
            level = 0
        tmpim = self.im.read_region((xi-offsetx-100,yi-offsety-100),level,((xa-xi+200)/2**level,(ya-yi+200)/2**level))
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
            #fig.set_dpi((w2/w)*fig.get_dpi())
            fig.set_dpi(100)
        a=fig.gca()
        a.set_frame_on(False)
        a.set_xticks([]); a.set_yticks([])
        plt.axis('off')
        plt.xlim(0,h); plt.ylim(w,0)
        fig.savefig(tfprefix+fileName, transparent=True,bbox_inches='tight', pad_inches=0)
        

class HistUpload():
    def __init__(self,hist):
        self.mysite = Piwigo(url='http://qmul.piwigo.com/')
        self.mysite.login(username="qmul", password="xuzhaoyang")       
        self.webfdlist = self.mysite.plan     
        self.webparent = u'CRLM_V2 / '  ## " " space is necessary
        self.hist = hist
        self.finame = hist.finame
        if self.webfdlist.has_key(self.webparent+self.finame) == False:
            self.mysite.mkdir(self.webparent+self.finame)
            
    def __del__(self):
        self.mysite.logout()
        
    def Upload(self,imgname):
        if imgname == None:
            pass
        else:
            fdpath = self.webparent+self.finame+u' / '+imgname[-1].upper()
            self.webfdlist = self.mysite.plan
            if self.webfdlist.has_key(fdpath) == False:
                self.mysite.mkdir(fdpath)
            try:
                self.mysite.upload(tfprefix+imgname+'.png',fdpath)     
            except Exception as e:
                print imgname
                print e
                self.mysite.upload(tfprefix+imgname+'.png',fdpath)
            
            os.remove(tfprefix+imgname+'.png')
    
    def ExistID(self):
        existID = []
        for cat in Abr_list:
            fdpath =  self.webparent+self.finame+u' / '+cat
            if self.webfdlist.has_key(fdpath):
                webimlist = self.mysite.pwg.categories.getImages(cat_id=self.mysite.plan[fdpath],per_page = 1000)
                for webim in webimlist['images']:
                    existID.append([int(webim['name'].split('-')[1]),webim['id']])
        existID.sort()
        return existID
    def Update(self):
        existID = self.ExistID()
        webindex = 0
        localindex =0
        while(localindex<self.hist.root.__len__()):
        #for index in range(hist.root.__len__()):
            tmpid = int(self.hist.root[localindex].attrib['id'])
            if existID==[]:
                self.Upload(self.hist.ExtractImage(localindex))
                localindex +=1
                print localindex,' Uploaded'
            elif tmpid>existID[-1][0] or webindex>existID.__len__():
                self.Upload(self.hist.ExtractImage(localindex))
                localindex +=1
                print localindex,' Uploaded'
            elif tmpid>existID[webindex][0]:
                self.mysite.pwg.images.delete(image_id=existID[webindex][1],pwg_token = self.mysite.token)
                webindex +=1
                print 'ID-',webindex,'Deleted'
            elif tmpid == existID[webindex][0]:
                webindex +=1
                localindex+=1
            else:
                self.Upload(self.hist.ExtractImage(localindex))
                localindex +=1
                print localindex,' Uploaded'
                
            
    def Logout(self):
        self.mysite.logout()
        



if __name__ == '__main__':
    fdname = u'/mnt/DATA/CLRM/Original/'
    filist = []
    for tn in os.listdir(fdname):
        if tn.split('.')[-1] == 'ndpi':
            filist.append(tn)
    
    for i in range(18,28):
        finame = u'Test_%03d.ndpi'%i
        testhist = HistImage(fdname = fdname,finame = finame)

        testupload = HistUpload(testhist)
        for index in range(testhist.root.__len__()):
            testupload.Upload(testhist.ExtractImage(index))
            print index,'Uploaded'
        del(testhist)

    testupload.Logout()
