from cytomine import Cytomine
import cytomine.models
import time
import shapely
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import pylab as plt

import openslide as ops
from fig2img import fig2img,SaveFigureAsImage
from Histupload import HistUpload

#Replace XXX values by your values
# the web url of cytomine instance, always without the  protocol
# your public & private keys of your account on Cytomine (can be found on your Account details on Cytomine)
protocol = 'http://'
cytomine_core_path="robbin.eecs.qmul.ac.uk"
cytomine_public_key="2f4648cb-a3bb-4d0a-9d98-bfcf1d8e2854"
cytomine_private_key="8a6c2440-0964-4948-9bad-4c9dd5d1edf6"


test_histo = HistUpload()


# check connection to the Cytomine instance
core_conn = Cytomine(cytomine_core_path,cytomine_public_key,cytomine_private_key, verbose= False)
# check that the storage exists
tproject = core_conn.get_projects().data()  # Targeted_project_id = 28282


project_id = 28282
test_instance = core_conn.get_project_image_instances(project_id)

tonto = core_conn.get_terms(id_ontology=27995)
terms_dict = {}
for tterm in tonto.data():
    terms_dict[tterm.id] = tterm.name


#image_index = 0
for image_index in range(len(test_instance)):
    timage = test_instance[image_index]
    image_id = timage.id
    tannotations = core_conn.get_annotations(id_project=project_id,id_image=image_id)



    #anno_index = 0

    tannos = tannotations.data()

    print "working on %s"%timage.fullPath+'number of annotaions in %d'%len(tannos)

    for anno_index in range(len(tannos)):
        try:
            tanno_id = tannos[anno_index].id
            #print tanno_id
            tannotation = core_conn.get_annotation(id_annotation=tanno_id)
            tlocation = tannotation.location


            g1 = shapely.wkt.loads(tlocation)
            polygons =[g1]
            int_coords = lambda x: np.array(x).round().astype(np.int32)
            exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
            interiors = [int_coords(pi.coords) for poly in polygons
                         for pi in poly.interiors]

            sx,sy,w,h = exteriors[0][:,0].min(),exteriors[0][:,1].max(),exteriors[0][:,0].max()-exteriors[0][:,0].min(),exteriors[0][:,1].max()-exteriors[0][:,1].min(),

            wsi = ops.open_slide(timage.fullPath.replace('DATA_CYTO','usbflash'))



            wx,wy = wsi.dimensions


            if w*h >10000*10000:
                level = 2
            elif w*h >3000*3000:
                level = 1
            else:
                level = 0

            ttimage = wsi.read_region((sx,wy-sy),level,(w/(2**level),h/(2**level)))

            #sx,sy,w,h =(xx.min(),yy.max(),xx.max()-xx.min(),yy.max()-yy.min())
            fig,ax = plt.subplots()
            ax.imshow(ttimage)    
            ax.axis('image')
            ax.axis('off')
            ax.plot( (exteriors[0][:,0]-sx)/(2**level),(sy-exteriors[0][:,1])/(2**level),'y-',linewidth =5)
            for tinter_index in range(len(interiors)):
                ax.plot((interiors[tinter_index][:,0]-sx)/(2**level),(sy-interiors[tinter_index][:,1])/(2**level),'y-',linewidth =5)

            #plt.axis('off')
            #plt.title('%d-%d-%d'%(project_id,image_id,tanno_id))
            #print(terms_dict[tannos[0].term[0]])
            image_path = '/tmp/%d-%d-%d.png'%(project_id,image_id,tanno_id)
            img = fig2img(fig)
            img.save(image_path)
            
            #SaveFigureAsImage(fileName=image_path,fig=fig)
            plt.close(fig)

            test_histo.Upload(image_path=image_path,\
                              subfolder=terms_dict[tannos[anno_index].term[0]],\
                              comment=u'robbin.eecs.qmul.ac.uk/#tabs-image-%d-%d-%d'%(project_id,image_id,tanno_id))


        except Exception as e:
            print e