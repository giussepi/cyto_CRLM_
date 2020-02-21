import numpy as np
import os
from piwigotools import Piwigo
from piwigotools.interface import *


class HistUpload():
    """
    finame : category name 
    """
    def __init__(self,webparent=u'LYMPH_NODE_METASTASES_V2 / '):
        self.mysite = Piwigo(url='http://qmul.piwigo.com/')
        self.mysite.login(username="qmul", password="xuzhaoyang")       
        self.webfdlist = self.mysite.plan     
        self.webparent = webparent # u'LYMPH_NODE_METASTASES_V2 / '  ## " " space is necessary
            
    def __del__(self):
        self.mysite.logout()
        
    def Upload(self,image_path,subfolder=None,name=None,comment=None):
        """
        if multi-layer subfolder : 'Test / Test_v1 / Test_v2'
        """
        if subfolder is not None:
            if self.webfdlist.has_key(self.webparent+ subfolder) == False:
                self.mysite.mkdir(self.webparent+ subfolder)
                
        fdpath = self.webparent+subfolder
        try:
            if name is not None:
                self.mysite.upload(image_path,fdpath,name=name,comment=comment)     
            else:
                self.mysite.upload(image_path,fdpath,comment=comment)     
        except Exception as e:
            print e
            if name is not None:
                self.mysite.upload(image_path,fdpath,name=name,comment=comment)     
            else:
                self.mysite.upload(image_path,fdpath,comment=comment)  

        os.remove(image_path)
    
    
    def Check_exist(self,image_path,subfolder=None,name=None,comment=None):
        fdpath = self.webparent+subfolder
        tname = image_path.split('/')[-1].split('.')[0] if name is None else name
        if self.webfdlist.has_key(fdpath):
            #int(webimlist['paging']['count'])/500  
            #--todo
            webimlist = self.mysite.pwg.categories.getImages(cat_id=self.mysite.plan[fdpath],per_page = 1000)
            for webim in webimlist['images']:
                #existID.append([int(webim['name'].split('-')[1]),webim['id']]) 
                if webim['name'] == tname:
                    return webim['id'] 
        return False
                
    def Delete(self,image_path,subfolder=None,name=None,comment=None):
        tid = Check_exist(image_path,subfolder,name,comment)
        self.mysite.pwg.images.delete(image_id=tid,pwg_token=test_histo.mysite.token)
            
    def Logout(self):
        self.mysite.logout()
        