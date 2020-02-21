
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:31:40 2016

@author: JOHN
"""

from xml.etree import ElementTree
import openslide as ops
import cv2
import pylab as plt
import numpy as np
import os

#im = ops.OpenSlide(r'D:/DATASET/CLRM/T12534-15-6E.ndpi')
im = ops.OpenSlide(r'D:/DATASET/CLRM/2016-01-13 16.36.58_2_Danyil.ndpi')
fdname = r'2016-01-13 16.36.58_2_Danyil'

ld = im.level_dimensions
#'''
#with open(r'D:/DATASET/CLRM/T12534-15-6E.ndpi - Copy.xml') as f:
with open(r'D:/DATASET/CLRM/2016-01-13 16.36.58_2_Danyil.ndpi.ndpa') as f:
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
    


def ConvertXY(ax,by):
    nax = int((ax - xoff)/(xmpp*1000.0)+ld[0][0]/2.0)
    nby = int((by - yoff)/(ympp*1000.0)+ld[0][1]/2.0)
    return nax,nby
'''#Make the folder first ''' 
Mkfolders('Samples/',fdname)


for index in range(root.__len__()):
#for index in range(2):
    name = 'ID-'+root[index].attrib['id']+'-'+root[index][0].text
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
    ax.set_xticks([])
    ax.set_yticks([])
    for i in Plist1:
        ax.plot(i[0]-xi+100,i[1]-yi+100,'y-*')
        
    fig.savefig('Samples/'+fdname+r'/'+name[-1]+r'/'+name)
    
    
    plt.close(fig)
    #plt.waitforbuttonpress()    
    #plt.clf()
    
