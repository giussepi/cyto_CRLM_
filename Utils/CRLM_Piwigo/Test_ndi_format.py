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


im = ops.OpenSlide(r'D:/DATASET/CLRM/T12534-15-6E.ndpi')
ld = im.level_dimensions
#'''
with open(r'D:/DATASET/CLRM/T12534-15-6E.ndpi - Copy.xml') as f:
    tree = ElementTree.parse(f)
root = tree.getroot()


level = 6
img = im.read_region((0,0),level,ld[level])

img = np.array(img)
plt.imshow(img)
#img2 = cv2.rectangle(img,(3883018/2**4,1630488/2**4),(5012179/2**5,2307964/2**5),(255,0,0),3)
#plt.imshow(img2)
#'''
#img3 = im.read_region((3883018L,1630488L),0,(5012179L-3883018L,2307964L-1630488L))
#plt.imshow(img3)


ax = 3883018L
#17153
by = 1630488L
#7203
cx = 5012179L
#
dy = 2307964L
#


xmpp = float(im.properties['openslide.mpp-x'])

xoff = float(im.properties['hamamatsu.XOffsetFromSlideCentre'])
#10164335
#44901
ympp = float(im.properties['openslide.mpp-y'])

yoff = float(im.properties['hamamatsu.YOffsetFromSlideCentre'])
#2613402
#11545

nax = int((ax - xoff)/(xmpp*1000.0)+ld[0][0]/2.0)
nby = int((by - yoff)/(ympp*1000.0)+ld[0][1]/2.0)
ncx = int((cx - xoff)/(xmpp*1000.0)+ld[0][0]/2.0)
ndy = int((dy - yoff)/(ympp*1000.0)+ld[0][1]/2.0)


'''
nax = int((ax )/(xmpp*1000.0))
nby = int((by )/(ympp*1000.0))
ncx = int((cx )/(xmpp*1000.0))
ndy = int((dy )/(ympp*1000.0))
'''

#img3 = im.read_region((3883018L,1630488L),0,(5012179L-3883018L,2307964L-1630488L))
#plt.imshow(img3)


plt.plot(nax/2**level,nby/2**level,'r*')

plt.plot(ncx/2**level,nby/2**level,'r*')
plt.plot(nax/2**level,ndy/2**level,'r*')
plt.plot(ncx/2**level,ndy/2**level,'r*')
'''
plt.plot(19648/2**level,8661/2**level,'r*')
'''