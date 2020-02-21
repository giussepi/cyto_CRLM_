
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
with open(r'D:/DATASET/CLRM/2016-01-13 16.36.58_2_Danyil.xml') as f:
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
#Mkfolders('Samples/',fdname)



def SaveFigureAsImage(fileName,fig=None,**kwargs):
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
    fig.savefig(fileName, transparent=True, bbox_inches='tight', pad_inches=0)
    


#for index in range(root.__len__()):
for index in range(2):
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
        ax.plot(i[0]-xi+100,i[1]-yi+100,'y-',linewidth = 3)
        
    #fig.savefig('Samples/'+fdname+r'/'+name[-1]+r'/'+name)
    SaveFigureAsImage(r'./'+name,fig = fig,orig_size = (np.array(im3).shape[0],np.array(im3).shape[1]))
    
    plt.close(fig)
    #plt.waitforbuttonpress()    
    #plt.clf()

