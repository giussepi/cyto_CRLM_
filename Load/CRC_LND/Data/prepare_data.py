from CRC_LND import CRC_LND as CRLM
import mxnet as mx
import os
import numpy as np
import skimage.measure as measure
from skimage.transform import resize
import cv2
from skimage.transform import rotate

if os.uname()[1] == 'Camelyon':
    prefix = '/mnt/DATA_CRLM/Patches/Patches_CRCLND/'
elif os.uname()[1] == 'TITANX-SERVER':
    #prefix = '/mnt/DATA/CLRM/Patches/'
    prefix = '/mnt/DATA_CRLM/Patches/Patches_CRCLND/'

# ldic = {'H':0, 'N':1, 'F':2, 'T':3, 'I':4, 'M':5, 'B':6, 'FB':7 ,'MF':8,'BD':9}
# #ldic = {'H':0, 'N':1, 'F':2, 'T':3, 'I':2, 'M':2, 'B':2, 'FB':3 ,'MF':2}
# name_list = ['H', 'N', 'F', 'T', 'I', 'M', 'B', 'D' ,'C','Y','G']
# # Hepatocyte Necrosis, Fibrosis, Tumour, Inflamnation,Macrophage, 
# #                                Blood, Foregin Blood, Mucin, BD, Backgroud 



ldic = {'H':0, 'N':1, 'F':2, 'T':3, 'L':4, 'M':5, 'B':6, 'GC':7,'CP':8,'C':9,'SM':10,'FT':11}
label_names = ['histiocytes','Necrosis','Fibrosis','Tumour','lymphocytes','Mucin','Blood','Germinala Centra','Capillary','Capsule','Smooth Muscle','Tumour Fibrosis']

name_list = ['H', 'N', 'F', 'T', 'L', 'M', 'B', 'D' ,'P','C','S','Y','G']

def ExtractImage(tc,index =1, offsetx=128,offsety =128,  level = 0):
    offsetx = 128
    offsety = 128
    name = tc.Get_Aname(index)
    pl_arr = tc.AnnotationDots(index)
    pl_arr = np.vstack((pl_arr,pl_arr[0]))
    xi = pl_arr[:,0].min()
    xa = pl_arr[:,0].max()
    yi = pl_arr[:,1].min()
    ya = pl_arr[:,1].max()

    tmpim = tc.img.read_region((xi-offsetx,yi-offsety),level,((xa-xi+2*offsetx)/2**level,(ya-yi+2*offsety)/2**level))
    
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
                    print axi,ayi,axa,aya , t
                    return True

    for an_index in range(tc.root_len):
    #for an_index in range(10):
        lname =tc.Get_Aname(an_index)
        print an_index, lname, tc.root_len
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
                print "Annotaion too small"
                pass
            elif Check_ROI(an_index) == True:
                print an_index, 'inside the ROI'
                pass
            elif label=='T' and check_hole_in_Tumour(tim)==True:
                print "This is background"
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

                            # # Increase the samples of the other categories
                            # if lname in ['I', 'M', 'B', 'FB' ,'MF','BD']:
                            #     cv2.imwrite(prefix+name_list[label]+'%02d_%05d'%(slide_index,count)+'.jpg',Flip_Recolor_mix(tim2))
                            #     count +=1

        else:
            pass
        if count%100 ==0:
            print count


def extract_single_to_multi(slide_index,level=0,ex_roi_list=[]):
    """
    Extract from single CRLM file  to different sizes
    224 -  448 -- 896
    """
    prefix_224 =  prefix+'Patches_Level0/Patches_224/All/'
    prefix_448 = prefix+'Patches_Level0/Patches_448/All/'
    prefix_896 = prefix+'Patches_Level0/Patches_896/All/'

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
                    print axi,ayi,axa,aya , t
                    return True

    for an_index in range(tc.root_len):
    #for an_index in range(10):
        lname =tc.Get_Aname(an_index)
        print an_index, lname, tc.root_len
        if lname == 'ROI' or  lname =='O':
            pass
        elif lname in ldic.keys():
            label = ldic[lname]
            try:
                _, tim, mask = ExtractImage(tc=tc,index=an_index,level=level)
            except Exception as e:
                print e
                continue
            tmp_ref = get_corner(an_index)
            ref_x = tmp_ref[0]-128
            ref_y = tmp_ref[1]-128


            step = 224
            if tim.shape[0]*tim.shape[1] >1000000:
                step = 336
            elif tim.shape[0]*tim.shape[1] >9000000:
                step = 448  
            else:
                step = 224

            if  tim.shape[0] <224 or tim.shape[1] <224:
                print "Annotaion too small"
                pass
            elif Check_ROI(an_index) == True:
                print an_index, 'inside the ROI'
                pass
            elif label=='T' and check_hole_in_Tumour(tim)==True:
                print "This is background"
                pass
            else:
                for ix in range(step,tim.shape[0]-step,step):
                    for iy in range(step,tim.shape[1]-step,step):
                        if np.sum(mask[ix:ix+128,iy:iy+128]) > 128*128*0.9:
                            # For Level 2 the offset is 336 + 48
                            #if level ==2:
                            tim2_896 = np.array(tc.img.read_region(location=(ref_x+iy-336-48,ref_y+ix-336-48),level=0,size=(896,896)))
                            #elif level ==1:
                            tim2_448 = np.array(tc.img.read_region(location=(ref_x+iy-112-48,ref_y+ix-112-48),level=0,size=(448,448)))
                            #elif level ==0:
                            tim2_224 = np.array(tc.img.read_region(location=(ref_x+iy-48,ref_y+ix-48),level=0,size=(224,224)))
                            
                            cv2.imwrite(prefix_224+name_list[label]+'%02d_%05d'%(slide_index,count)+'.jpg',tim2_224)
                            cv2.imwrite(prefix_448+name_list[label]+'%02d_%05d'%(slide_index,count)+'.jpg',tim2_448)
                            cv2.imwrite(prefix_896+name_list[label]+'%02d_%05d'%(slide_index,count)+'.jpg',tim2_896)
                            
                            count +=1

                            # Increase the samples of the other categories
                            # if lname in ['I', 'M', 'B', 'FB' ,'MF','BD']:
                            #     cv2.imwrite(prefix+name_list[label]+'%02d_%05d'%(slide_index,count)+'.jpg',Flip_Recolor_mix(tim2))
                            #     count +=1

        else:
            pass
        if count%100 ==0:
            print count

def extract_all(slide_list,level,ex_roi_list):
    for ti, index in enumerate(slide_list):
        try:
            print "****************************************"
            print "******Start to Extract %03d*******"%index
            print "*************************************"
            extract_single(index,level,ex_roi_list[ti])

        except Exception as e:
            print "*************************************"
            print "*************************************"
            print e 
            print "*************************************"
            print "*************************************"

def  func_wrapper(args):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return extract_all(*args)

if __name__ == '__main__':
    #slide_indexes  = list(range(1,41))+[88,89,90,91,92,101,102,103,104]
    slide_indexes = list(range(73,144))
    for slide_index in slide_indexes:
        print "Processing ",slide_index
        extract_single_to_multi(slide_index,ex_roi_list=[])

    # from multiprocessing import Pool, freeze_support
    # freeze_support()
    # level = 1
    # pool = Pool(6)
    # pool.map(func_wrapper, 
    #         [
    #             [[1,2,3,4,5,31],level,[[],[],[],[],[],[]]],
    #             [[6,7,8,9,10,32],level,[[],[],[],[],[],[]]],
    #             [[11,12,13,14,15,33],level,[[],[],[],[],[],[]]],
    #             [[16,17,18,19,20],level,[[],[],[],[],[]]],
    #             [[21,22,23,24,25],level,[[],[],[],[],[]]],
    #             [[26,27,28,29,30],level,[[],[],[],[],[]]],                
    #         ])

    # for slide_index in range(0,34):
    #     print "****************************************"
    #     print "******Start to Extract %03d*******"%slide_index
    #     print "*************************************"
    #     #extract_single(slide_index,level=1,ex_roi_list=[])
    #     extract_single(slide_index=slide_index,level=1,ex_roi_list=[])

