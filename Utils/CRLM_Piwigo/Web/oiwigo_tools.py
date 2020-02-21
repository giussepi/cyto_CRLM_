# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 22:18:47 2016

@author: john
"""

from piwigotools import Piwigo
from piwigo.ws import Ws
from piwigotools.interface import *
import os 



mysite = Piwigo(url='http://qmul.piwigo.com/')
mysite.login(username="qmul", password="xuzhaoyang")

#mysite.mkdir(r'/Test1/2016-1-1/T')

#print mysite.plan
#print mysite.idcategory(mysite.plan.keys()[0])

#print mysite.images(mysite.plan.keys()[-1])


#mysite.removedirs('/Test3')

flist = os.listdir('../Samples')
'''
for tmp in flist:
    if tmp[-3:] != 'png':
        flist.remove(tmp)
for im in flist:
    mysite.upload(im,mysite.plan.keys()[0])
'''  
imlist = mysite.images(mysite.plan.keys()[0])

def UploadFolders(fp):
    mysite.mkdir('/Test2/'+fp)
    for category in ['T','M','F','H']:
        pathid = mysite.mkdir('/Test2/'+fp+r'/'+category)
        fdpath = '../Samples/'+fp+r'/'+category+r'/'
        flist = os.listdir(fdpath)
        for im in flist:
            # Return the image ID
            mysite.upload(fdpath+im,mysite.plan.keys()[mysite.plan.values().index(pathid)])
            

UploadFolders(flist[0])

'''
#pwg from api tools
#http://piwigo.org/demo/tools/ws.htm#top
'''
#mysite.pwg.images.rate(image_id = 8,rate=4)
#b = mysite.pwg.images.getInfo(image_id = 8)
