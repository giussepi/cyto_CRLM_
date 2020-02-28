# -*- coding: utf-8 -*-
""" utils/utils """

from collections import Counter
import os
from pathlib import Path

from cytomine.models import ImageInstance
import numpy as np
import openslide as ops
from shapely.geometry import Polygon, MultiPolygon
from skimage.measure import regionprops
from skimage.segmentation import slic
from skimage.transform import resize

from settings import ANNOTATION_DISPLACEMENT_X, ANNOTATION_DISPLACEMENT_Y, LOAD_SAVE_PATH,\
    LOAD_SLIDE_PATH, Label, APP_CONTAINER_DOWNLOAD_PATH


def CNN_Superpixels(im, label_tissue):
    """
    label_tissue (np.array) is feature map of the tissue level results
    """
    segments_slic = slic(im[:, :, :3], n_segments=1000, compactness=10, sigma=1)
    regions = regionprops(segments_slic)

    if len(label_tissue.shape) == 2:
        # mask2 = resize(label_image,segments_slic.shape)
        mask2 = np.array(resize(np.array(label_tissue, dtype=np.int), im.shape[:2], preserve_range=True), dtype=np.int)
        mask3 = np.zeros_like(mask2)

        for i in range(len(regions)):
            region = regions[i]
            a, b, c, d = region.bbox

            list_of_integers = mask2[a:c, b:d][region.filled_image].flatten()
            cnt = Counter(list_of_integers)
            tlabel = cnt.most_common(1)[0][0]
            mask3[a:c, b:d][region.filled_image] = tlabel
    elif len(label_tissue.shape) == 3:
        mask2 = np.array(resize(np.array(label_tissue), im.shape[:2], preserve_range=True))
        mask3 = np.zeros_like(mask2)
        for i in range(len(regions)):
            region = regions[i]
            a, b, c, d = region.bbox

            list_of_integers = mask2[a:c, b:d][region.filled_image].argmax(1).flatten()
            cnt = Counter(list_of_integers)
            tlabel = cnt.most_common(1)[0][0]
            mask3[a:c, b:d][region.filled_image] = tlabel
    else:
        raise Exception("input features dimension incompatible")
    return mask3


def remove_repeated_coords(sequence):
    """
    Removes coordinates repeated on a sequence

    Args:
        sequence (list): list of tuples

    Returns:
        List of unique tuples
    """
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def chunks(lst, n):
    """
    Yields successive n-sized chunks from lst.

    Source: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks#answer-312464

    Args:
        lst (list): list object
        n   (int): Size of the sub-lists

    Returns:
        Sub-lists of size n

    """
    assert isinstance(lst, list)
    assert isinstance(n, int)

    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def scale_adapt(wsi_path, tresult, annotations, adapt_to_cytomine=True):
    """
    Scales the annotations to its original dimentions and (optionally) adapts them
    to match Cytomine coordiante system

    Args:
        wsi_path             (str) : Path to the ndpi image
        tresults      (np.ndarray) : Numpy matrix containing the mask
        annotations          (dic) : Dictionary contanint image annotations
        adapt_to_cytomine   (bool) : Boolean to adapt or not to cytomine coordiante system

    Returns:
        List of rescaled annotations
    """

    assert isinstance(wsi_path, str)
    assert Path(wsi_path).is_file()
    assert isinstance(tresult, np.ndarray)
    assert isinstance(annotations, dict)
    assert isinstance(adapt_to_cytomine, bool)

    # Rescaling contours to original dimensions
    wsi_dim_x, wsi_dim_y = ops.OpenSlide(wsi_path).dimensions
    rescaling_factor = wsi_dim_x/tresult.shape[1]
    rescaled_annotations = {}

    for key, values in annotations.items():
        rescaled_annotations[key] = list(map(lambda item: item*rescaling_factor, values))

    # Adapting coordinates to cytomine coordinates system
    if adapt_to_cytomine:
        for key, values in rescaled_annotations.items():
            rescaled_annotations[key] = list(map(
                lambda item: np.absolute(item - np.full(item.shape, (0, wsi_dim_y), )) +
                np.full(item.shape, (ANNOTATION_DISPLACEMENT_X, ANNOTATION_DISPLACEMENT_Y),),
                values
            ))

    return rescaled_annotations


def smooth_simplify_polygons(annotations, label_id):
    """
    Smooths and simplifies annotations polygons to avoid issues when drawing
    then on Cytomine

    Args:
        annotations  (dic) : Dictionary contanint image annotations
        label_id     (int) : Model label id

    Returns:
        List of smoothed and simplified annotations
    """

    def handle_polygon(pol, result_list):
        """
        Removes repeated coordinates, cast values to integer and appends the
        numpy array outcome to result_list.

        Args:
            pol         (Polygon) : Polygon intance
            result_list (list)    : List to append the results
        """
        assert isinstance(pol, Polygon)
        assert isinstance(result_list, list)

        if len(pol.exterior.coords) != 0:
            new_pol = list(zip(*pol.exterior.coords.xy))
            int_list = new_pol
            cleaned_np = np.array(remove_repeated_coords(int_list))
            if cleaned_np.shape[0] >= 3:
                result_list.append(
                    cleaned_np.reshape(cleaned_np.shape[0], 1, cleaned_np.shape[1])
                )

    cleaned_polygons = list()

    for np_array in annotations[label_id]:
        # NOTE: The smoothing and simpligying process seems to work properly so far.
        #       However, for new weird/complex annotations/polygons new
        #       configuration could be required.
        pol = Polygon(np_array.reshape(np_array.shape[0], np_array.shape[2]))\
            .buffer(0, cap_style=3, join_style=2, mitre_limit=5.0)\
            .simplify(.2, True)

        if isinstance(pol, MultiPolygon):
            for sub_pol in pol:
                handle_polygon(sub_pol, cleaned_polygons)
        else:
            handle_polygon(pol, cleaned_polygons)

    return cleaned_polygons


def generate_result_full_path(wsi_path):
    """
    Generates and returns the path of the result file <wsi_filename>.npy
    """
    assert Path(wsi_path).is_file()
    assert wsi_path.endswith('.ndpi')
    return os.path.join(LOAD_SAVE_PATH, os.path.basename(wsi_path).replace('.ndpi', '.npy'))


def get_pie_chart_data(wsi_path=LOAD_SLIDE_PATH, delete_results_file=False):
    """
    Returns a quantitaive analysis that can be used to generate a pie chart
    Optionally, it can remove the <wsi_filename>.npy file (if delete_results_file=True)

    Args:
        wsi_path                (str)  : path to the ndpi image
        delete_results_file     (bool)  : If set to true, removes the <wsi_filename>.npy file

    Returns:
        The List of percentages of pixels per class, sorted following the order from Label.ids
        [class_0_percentage, class_1_percentage, class_2_percentage, ...]
    """
    assert isinstance(wsi_path, str)
    assert Path(wsi_path).is_file()
    assert isinstance(delete_results_file, bool)

    # NOTE: we need to use tresult mask because the polygons present overlapping;
    #       in this regard, using the mask to calculate the percentages will provide
    #       accurate results (otherwise, using the polygons areas will sum more than
    #       the total area)
    wsi_result_full_path = generate_result_full_path(wsi_path)
    tresult = np.array(np.load(wsi_result_full_path), dtype=np.int)[:, :, 0].T
    total = tresult.flatten().shape[0]

    # adding labels values not found on tresult mask
    label_num_pixels = dict((
        label, np.count_nonzero(tresult == label)) for label in np.unique(tresult))
    difference = dict((
        label, 0) for label in set(Label.ids).difference(label_num_pixels.keys()))
    label_num_pixels.update(difference)

    if delete_results_file:
        os.remove(wsi_result_full_path)

    return [round(label_num_pixels[label]*100/total, 2) for label in Label.ids]


def get_container_image_path(img):
    """
    Returns the path to the file image 'img' in the application docker container

    Args:
        img (ImageInstance) : Cytomine ImageInstance

    Returns:
        string
    """
    assert isinstance(img, ImageInstance)

    return os.path.join(APP_CONTAINER_DOWNLOAD_PATH, img.originalFilename)


def download_image(img):
    """
    Downloads the img at APP_CONTAINER_DOWNLOAD_PATH in the docker container and replaces
    the extendion .jpg (set by default by ImageInstance.download) by the right one .ndpi

    Args:
        img (ImageInstance) : Cytomine ImageInstance
    """
    assert isinstance(img, ImageInstance)

    img.download(os.path.join(APP_CONTAINER_DOWNLOAD_PATH, '{originalFilename}'))
    download_name = get_container_image_path(img).replace('.ndpi', '.jpg')

    assert Path(download_name).is_file(),\
        "{} could not be downloaded".format(download_name)

    os.rename(download_name, get_container_image_path(img))


def remove_image_local_copy(img):
    """
    Removes local image copy of the image (if it exists)

    Args:
        img (ImageInstance) : Cytomine ImageInstance
    """

    assert isinstance(img, ImageInstance)

    if Path(get_container_image_path(img)).is_file():
        os.remove(get_container_image_path(img))
