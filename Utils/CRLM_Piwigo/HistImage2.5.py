# -*- coding: utf-8 -*-
"""
Created on Mon Mar 07 22:40:12 2016

@author: JOHN
"""



fdname=r'D:/DATASET/CLRM/'
finame=r'2016-01-13 16.44.33_1_Carlos'

webparent = r'/Test2/'
im = ops.OpenSlide(fdname+finame+'.ndpi')

with open(fdname+finame+'.ndpi.ndpa') as f:
    tree = ElementTree.parse(f)
root = tree.getroot()  
mysite = Piwigo(url='http://qmul.piwigo.com/')
mysite.login(username="qmul", password="xuzhaoyang")       
webfdlist = mysite.plan      

if webfdlist.has_key(webparent+finame) == False:
    mysite.mkdir(webparent+finame)

    
def ConvertXY(ax,by):
    xmpp = float(im.properties['openslide.mpp-x'])
    xoff = float(im.properties['hamamatsu.XOffsetFromSlideCentre'])
    ympp = float(im.properties['openslide.mpp-y'])
    yoff = float(im.properties['hamamatsu.YOffsetFromSlideCentre'])
    ld = im.dimensions
    nax = int((ax - xoff)/(xmpp*1000.0)+ld[0]/2.0)
    nby = int((by - yoff)/(ympp*1000.0)+ld[1]/2.0)
    return nax,nby


def Upload(imgname):
    fdpath = webparent+finame+r'/'+imgname[-1].upper()
    if webfdlist.has_key(fdpath) == False:
        mysite.mkdir(fdpath)
    try:
        mysite.upload(imgname+'.png',fdpath)     
    except Exception as e:
        print imgname
        print e
        mysite.upload(imgname+'.png',fdpath)
    
    os.remove(imgname+'.png')
    
def NameAnnotation(index):
    if root[index][0].text != None and root[index][0].text != 'roi' :
            name = 'ID-'+root[index].attrib['id']+'-'+root[index][0].text
    elif root[index][0].text == 'roi'or root[index][0].text == 'ROI':
        name = 'ID-'+root[index].attrib['id']+'-'+'R'
    else:
        name = 'ID-'+root[index].attrib['id']+'-'+'O'
    return name


def AnnotationDots(index):
    if root[index][-1][-1].tag == 'pointlist':
        tplist = root[index][-1][-1]
    elif root[index][-1][-1].tag == 'specialtype' and root[index][-1][-3].tag == 'pointlist':
        tplist = root[index][-1][-3] 
    else:
        return np.array([])
    Plist1 = []
    for i in range(tplist.__len__()): 
        Plist1.append(ConvertXY(int(tplist[i][0].text),int(tplist[i][1].text)))
    return np.array(Plist1)

    
    
def ExtractImage(index):
    name = NameAnnotation(index)
    pl_arr = AnnotationDots(index)
    pl_arr = np.vstack((pl_arr,pl_arr[0]))
    xi = pl_arr[:,0].min()
    xa = pl_arr[:,0].max()
    yi = pl_arr[:,1].min()
    ya = pl_arr[:,1].max()
    if (xa-xi)*(ya-yi) >1000*1000:
        level = 2
    else:
        level = 0
    tmpim = im.read_region((xi-100,yi-100),level,((xa-xi+200)/2**level,(ya-yi+200)/2**level))
    fig,ax = plt.subplots()
    ax.imshow(tmpim)    
    ax.axis('image')
    ax.axis('off')
    ax.plot((pl_arr[:,0]-xi+100.0)/2**level,(pl_arr[:,1]-yi+100.0)/2**level,'y-',linewidth =3)
    #plt.savefig(name,dpi=fig.dpi )
    SaveFigureAsImage(name,fig = fig, orig_size=(np.array(tmpim).shape[0],np.array(tmpim).shape[1]))
    plt.close(fig)
    return name

def ExtractandUpload(index):
    name = ExtractImage(index)
    Upload(name)      
        

def ExtractandUploadAll():
    print 'uploading started'
    for index in range(root.__len__()):
        #if root[index][0].text == 'roi' or root[index][0].text == 'ROI' :
            #continue
        name = ExtractImage(index)
        Upload(name)
        print index+'Uploaded'
        
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
    fig.savefig(fileName, transparent=True,bbox_inches='tight', pad_inches=0)
 


#Upload(ExtractImage(1))
ExtractandUploadAll()
mysite.logout()


    