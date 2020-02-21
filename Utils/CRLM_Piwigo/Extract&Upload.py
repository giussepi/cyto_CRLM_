

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:31:40 2016

@author: JOHN
"""

from xml.etree import ElementTree
import openslide as ops
import cv2
#import pylab as plt
import numpy as np
import os
from piwigotools import Piwigo
from piwigo.ws import Ws
from piwigotools.interface import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')



#filename = r'2016-01-13 16.36.58_2_Danyil'
filename = r'2016-01-13 16.44.33_1_Carlos'
#im = ops.OpenSlide(r'D:/DATASET/CLRM/T12534-15-6E.ndpi')

im = ops.OpenSlide(r'D:/DATASET/CLRM/'+filename+'.ndpi')
fdname = filename


ld = im.level_dimensions
#'''
#with open(r'D:/DATASET/CLRM/T12534-15-6E.ndpi - Copy.xml') as f:
with open(r'D:/DATASET/CLRM/'+filename+'.ndpi.ndpa') as f:
    tree = ElementTree.parse(f)
root = tree.getroot()


level = 6
img = im.read_region((0,0),level,ld[level])

xmpp = float(im.properties['openslide.mpp-x'])
xoff = float(im.properties['hamamatsu.XOffsetFromSlideCentre'])
ympp = float(im.properties['openslide.mpp-y'])
yoff = float(im.properties['hamamatsu.YOffsetFromSlideCentre'])

def Mkfolders(parent,fdname):
    os.mkdir(parent+fdname)
    os.mkdir(parent+fdname+r'/'+'T')
    os.mkdir(parent+fdname+r'/'+'H')
    os.mkdir(parent+fdname+r'/'+'M')
    os.mkdir(parent+fdname+r'/'+'F')
    os.mkdir(parent+fdname+r'/'+'N')
    os.mkdir(parent+fdname+r'/'+'O')


def ConvertXY(ax,by):
    nax = int((ax - xoff)/(xmpp*1000.0)+ld[0][0]/2.0)
    nby = int((by - yoff)/(ympp*1000.0)+ld[0][1]/2.0)
    return nax,nby
    
'''#Make the folder first ''' 
Mkfolders('Samples/',fdname)


for index in range(root.__len__()):
#for index in range(2):
    if root[index][0].text:
        name = 'ID-'+root[index].attrib['id']+'-'+root[index][0].text
    else:
        name = 'ID-'+root[index].attrib['id']+'-'+'O'
        
    tplist = root[index][-1][-1]
    
    Plist1 = []
    for i in range(tplist.__len__()): 
        Plist1.append(ConvertXY(int(tplist[i][0].text),int(tplist[i][1].text)))
    
    pl_arr = np.array(Plist1)
    xi = pl_arr[:,0].min()
    xa = pl_arr[:,0].max()
    yi = pl_arr[:,1].min()
    ya = pl_arr[:,1].max()

    im3 = im.read_region((xi-100,yi-100),0,(xa-xi+200,ya-yi+200))
    #plt.imshow(im3)
    
    fig,ax = plt.subplots()
    #ax.set_visible(False)
    ax.imshow(im3)    
    ax.axis('image')
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    contours = np.array(Plist1)
    contours[:,0]= contours[:,0]-xi+100
    contours[:,1]= contours[:,1]-yi+100
    #cv2.drawContours(np.array(im3),[contours],-1,(255,255,0),3)
    #for i in Plist1:
    ax.plot(contours[:,0],contours[:,1],'y-',linewidth =3)
        
    fig.savefig('Samples/'+fdname+r'/'+name[-1]+r'/'+name,dpi=fig.dpi )
    
    plt.close(fig)


mysite = Piwigo(url='http://qmul.piwigo.com/')
mysite.login(username="qmul", password="xuzhaoyang")

imlist = mysite.images(mysite.plan.keys()[0])

def UploadFolders(fp):
    mysite.mkdir('/Test1/'+fp)
    for category in ['T','M','F','H']:
        pathid = mysite.mkdir('/Test1/'+fp+r'/'+category)
        fdpath = 'Samples/'+fp+r'/'+category+r'/'
        flist = os.listdir(fdpath)
        for im in flist:
            mysite.upload(fdpath+im,mysite.plan.keys()[mysite.plan.values().index(pathid)])

UploadFolders(r'Samples'+fdname)

