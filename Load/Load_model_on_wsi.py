# -*- coding: utf-8 -*-
""" Load/Load_model_on_wsi """

import os
from pathlib import Path
# from random import randrange
import time

import cv2
# import imutils
from matplotlib.colors import ListedColormap
import numpy as np
import openslide as ops
import pylab as plt
# from scipy import ndimage
# from skimage.feature import peak_local_max
# from skimage.morphology import watershed
from skimage.transform import resize
import torch

# from Data.CRLM import CRLM
from constants import Label
from Model.PatchCNN import PatchCNN
from settings import LOAD_STATE_DICT_PATH, LOAD_SLIDE_PATH, LOAD_SAVE_PATH, LOAD_SAVE_IMAGE_PATH,\
    LOAD_SAVE_IMAGE_PATH_2, ANNOTATION_MIN_AREA, RELEVANT_LABELS, DEVICE_IDS
from utils.utils import CNN_Superpixels, remove_repeated_coords, scale_adapt,\
    smooth_simplify_polygons, generate_result_full_path
from Utils.fig2img import fig2img


def process_large_image(model, input_patch, step=28, out_scale=4, num_classes=11, patch_size=448, show=False, cuda_size=None, use_cuda=True):
    """
    model pytorch model
    input_patch  nd_array
    step = 28   #sliding window step size
    out_scale =4   # the output shape of the network
    num_classes= 11   # the number of classes
    patch_size = 448     # the patch size of the
    cuda_size = None  # the batchsize for feeding cuda
    """
    if input_patch.max() > 2:
        test_img = input_patch/255.0
    else:
        test_img = input_patch
    tt = torch.from_numpy(((test_img[:, :, (2, 1, 0)]-np.array([0.485, 0.456, 0.406])) /
                           np.array([0.229, 0.224, 0.225])).transpose(2, 0, 1)).float()

    ta = tt.unfold(2, patch_size, step)
    # print(ta.size())
    tb = ta.unfold(1, patch_size, step)
    # print(tb.size())
    tc = tb.permute((1, 2, 0, 3, 4))
    # print(tc.shape)
    td = tc.reshape(-1, 3, patch_size, patch_size)
    # print(td.shape)
    nx = len(range(0, test_img.shape[0]-patch_size+1, step))
    ny = len(range(0, test_img.shape[1]-patch_size+1, step))

    final_tensor = td
    final_result2 = []
    if cuda_size == None:
        for i in range(0, nx):
            test_tensor = final_tensor[i*ny:i*ny+ny]

            with torch.no_grad():
                if use_cuda:
                    out = model(test_tensor.cuda())
                else:
                    out = model(test_tensor.cpu())

            softmax = torch.nn.Softmax2d()
            out2 = softmax(out)
            final_result2.append(out2.detach().cpu().numpy())
            # torch.cuda.empty_cache()

        t = np.array(final_result2).transpose(0, 3, 1, 4, 2).reshape(nx*out_scale, ny*out_scale, num_classes)

    else:
        for i in range(0, nx*ny, cuda_size):
            test_tensor = final_tensor[i:i+cuda_size]

            with torch.no_grad:
                if use_cuda:
                    out = model(test_tensor.cuda())
                else:
                    out = model(test_tensor.cpu())

            softmax = torch.nn.Softmax2d()
            out2 = softmax(out)
            if out2.size(0) != cuda_size:
                tout2 = np.zeros((cuda_size, out2.size(1), out2.size(2), out2.size(3)))
                tout2[:out2.size(0)] = out2.detach().cpu().numpy()
                final_result2.append(tout2)
            else:
                final_result2.append(out2.detach().cpu().numpy())
            # torch.cuda.empty_cache()

        tt = np.array(final_result2)
        s = tt.shape
        tt = tt.reshape(-1, s[2], s[3], s[4])[:nx*ny].reshape(nx, ny, s[2], s[3], s[4])
        t = tt.transpose(0, 3, 1, 4, 2).reshape(nx*out_scale, ny*out_scale, num_classes)

    if show:
        plt.subplot(1, 2, 1)
        plt.imshow(test_img)
        plt.subplot(1, 2, 2)
        plt.imshow(np.argmax(t[:, :, :], 2))
    return t


def process_wsi(
        model, slide_path, small_step=56, small_patch_size=448, out_scale=4, num_classes=11,
        use_cuda=True
):
    sl = ops.OpenSlide(slide_path)

    # small_step = 224    # should be times of 28
    large_patch_size = small_patch_size*(small_step//56)*4
    large_step = large_patch_size-small_patch_size+small_step
    patch_result_size = (large_patch_size-small_patch_size)//small_step+1

    lnx = len(range(0, sl.dimensions[0]-large_patch_size+1, large_step))+1
    lny = len(range(0, sl.dimensions[1]-large_patch_size+1, large_step))+1

    result_array = np.zeros((lnx*patch_result_size*out_scale, lny*patch_result_size*out_scale, num_classes))

    for i in range(lnx):
        for j in range(lny):
            try:
                tpatch = np.array(sl.read_region(
                    (i*large_step, j*large_step),
                    0, (large_patch_size, large_patch_size)))
                tpatch_result = process_large_image(model, input_patch=tpatch, out_scale=out_scale, step=small_step,
                                                    num_classes=num_classes, patch_size=small_patch_size, use_cuda=use_cuda)
                mask448 = CNN_Superpixels(tpatch, tpatch_result)
                result_array[i*patch_result_size*out_scale:(i+1)*patch_result_size*out_scale,
                             j*patch_result_size*out_scale:(j+1)*patch_result_size*out_scale, :] = \
                    np.array(
                    resize(np.array(mask448, dtype=np.int),
                           tpatch_result.shape[:2], preserve_range=True),
                    dtype=np.int
                ).transpose(1, 0, 2)

            except Exception as e:
                print(e)

    return result_array[:sl.dimensions[0]//small_step*out_scale, :sl.dimensions[1]//small_step*out_scale, :]


def process_wsi_and_save(wsi_path=LOAD_SLIDE_PATH, overwrite=False, use_cuda=True):
    """
    Processes the WSI (.ndpi) and saves the result at save_path with the
    format <wsi_filename>.npy

    Args:
        wsi_path   (str)  : path to ndpi file
        overwrite  (bool) : If true recalculates the result, otherwise nothing is executed
        use_cuda   (bool) : If true will try to use use cuda; otherwise, cpu
    """
    # TODO: TO BE TESTED ON titanX SERVER
    assert wsi_path.endswith('.ndpi')
    assert Path(wsi_path).is_file()
    assert isinstance(use_cuda, bool)

    save_full_path = generate_result_full_path(wsi_path)

    if Path(save_full_path).is_file() and not overwrite:
        return

    if use_cuda:
        device = torch.device('cuda:{}'.format(','.join([str(i) for i in DEVICE_IDS]))
                              if torch.cuda.device_count() > 0 else torch.device('cpu'))
    else:
        device = torch.device("cpu")

    num_layers = [3, 4, 6, 3]  # res34
    # num_layers = [2,2,2,2] # res18
    dropout_rate = 0
    model = PatchCNN(layers=num_layers, dropout_rate=dropout_rate)
    state_dict = torch.load(LOAD_STATE_DICT_PATH)
    new_state_dict = {}

    for key in model.state_dict():
        # workaround to avoid issues with keys
        if key in ('down.3.weight', 'down.3.bias'):
            key_ = key.replace('.3.', '.4.')
        else:
            key_ = key

        new_state_dict[key] = state_dict['module.'+key_].double()

    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(device)

    print("Start Processing")
    print(time.ctime())
    result_array = process_wsi(model, wsi_path, small_step=112, use_cuda=use_cuda)
    os.makedirs(LOAD_SAVE_PATH, exist_ok=True)
    np.save(save_full_path, result_array)
    print("Process Done")
    print(time.ctime())


def plot():
    """ Plots the original image and the model's results """
    tresult = np.array(np.load(LOAD_SAVE_PATH), dtype=np.int)[:, :, 0].T

    fig = plt.figure(figsize=(18, 18))
    plt.subplot(1, 2, 1)
    sl = ops.OpenSlide(LOAD_SLIDE_PATH)
    level = 7
    ttimage = sl.read_region((0, 0), level, (sl.dimensions[0]//2**level, sl.dimensions[1]//2**level))
    plt.imshow(ttimage)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(np.array(tresult == 1, dtype=np.int)+np.array(tresult == 2, dtype=np.int)*2
               + np.array(tresult == 3, dtype=np.int)*3+np.array(tresult == 4, dtype=np.int)*4,
               cmap=Label.short_cmap)
    plt.axis('off')

    os.makedirs(os.path.dirname(LOAD_SAVE_IMAGE_PATH), exist_ok=True)
    plt.savefig(LOAD_SAVE_IMAGE_PATH, dpi=300)

    img = fig2img(fig)
    os.makedirs(os.path.dirname(LOAD_SAVE_IMAGE_PATH_2), exist_ok=True)
    img.save(LOAD_SAVE_IMAGE_PATH_2)

    plt.show()


def generate_polygons(
        wsi_path=LOAD_SLIDE_PATH, generate_image=False, annotated_image_pattern='annotated_image',
        adapt_to_cytomine=False, delete_results_file=False
):
    """
    Generates and returns the polygons annotations rescaled to the original dimension.
    Optionally, it can remove the <wsi_filename>.npy file (if delete_results_file=True)

    Args:
        wsi_path                (str)  : path to the ndpi image
        generate_image          (bool) : boolean to plot and save the annotations
        annotated_image_pattern (str)  : name pattren for the generated annotated images
        delete_results_file     (bool) : If set to true, removes the <wsi_filename>.npy file

    Returns:
        Dictionary of rescaled contours
        {'label1': [annotations1], 'label2': [annotations2], ...}
    """
    assert isinstance(wsi_path, str)
    assert isinstance(generate_image, bool)
    assert isinstance(annotated_image_pattern, str)
    assert Path(wsi_path).is_file()
    assert isinstance(adapt_to_cytomine, bool)
    assert isinstance(delete_results_file, bool)

    # TODO: TO BE TESTED ON TITANX server
    wsi_result_full_path = generate_result_full_path(wsi_path)
    tresult = np.array(np.load(wsi_result_full_path), dtype=np.int)[:, :, 0].T
    annotations = {}

    for label_id in RELEVANT_LABELS:
        # options for cv2.findContours
        # dtype np.unit8 for cv2.RETR_CCOMP, cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_TREE
        # dtype np.int for cv2.RETR_FLOODFILL
        # cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE
        contours = cv2.findContours(
            np.array(tresult == label_id, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Removing tiny contours based on their area
        annotations[label_id] = list(filter(
            lambda x: cv2.contourArea(x) >= ANNOTATION_MIN_AREA, contours[0]))

        annotations[label_id] = smooth_simplify_polygons(annotations, label_id)

        if generate_image:
            # Drawing contours and saving image into a file
            canvas = np.zeros((tresult.shape[0], tresult.shape[1], 3))
            # NOTE: cv2.drawContours cannot deal with non-integer contours so when
            #       drawing contours we're casting the coordinated to integers.
            #       Fortunately, Cytomine is be able to handle floats properly
            cv2.drawContours(
                canvas, [ann.astype(int) for ann in annotations[label_id]], -1, (0, 255, 0), 1)
            cv2.imwrite('{}_{}.png'.format(annotated_image_pattern, label_id), canvas)

    rescaled_annotations = scale_adapt(wsi_path, tresult, annotations, adapt_to_cytomine)

    # NOTE: Results file is removed only after all its annotations has been uploaded
    # So if there's and issue, we can re-run the command without having to recalculated
    # the results file (thus, we're saving some processing hours)
    if delete_results_file:
        os.remove(wsi_result_full_path)

    return rescaled_annotations
