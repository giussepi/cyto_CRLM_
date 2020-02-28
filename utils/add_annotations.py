# -*- coding: utf-8 -*-
""" utils/add_annotation """

from argparse import ArgumentParser
import sys
import time

from cytomine import CytomineJob
from cytomine.models import ImageInstance, Property, Annotation, AnnotationCollection
import pylab as plt
from shapely.geometry import Polygon

from constants import Label
from Load.Load_model_on_wsi import generate_polygons, process_wsi_and_save
from settings import CYTOMINE_LABELS, ANNOTATION_BATCH, ANNOTATION_SLEEP_TIME, \
    APP_CONTAINER_DOWNLOAD_PATH
from utils.utils import chunks, get_pie_chart_data, download_image, \
    get_container_image_path, remove_image_local_copy


def run(debug=False):
    """
    Gets project image from cytomine

    Args:
        debug (bool): If true will save annotations individually and plot any error

    Example:
      python main.py --cytomine_host 'localhost-core' --cytomine_public_key 'dadb7d7a-5822-48f7-ab42-59bce27750ae' --cytomine_private_key 'd73f4602-51d2-4d15-91e4-d4cc175d65fd' --cytomine_id_project 187 --cytomine_id_image_instance 375 --cytomine_id_software 228848

      python main.py --cytomine_host 'localhost-core' --cytomine_public_key 'b6ebb23c-00ff-427b-be24-87b2a82490df' --cytomine_private_key '6812f09b-3f33-4938-82ca-b23032d377fd' --cytomine_id_project 154 --cytomine_id_image_instance 3643

      python main.py --cytomine_host 'localhost-core' --cytomine_public_key 'd2be8bd7-2b0b-40c3-9e81-5ad5765568f3' --cytomine_private_key '6dfe27d7-2ad1-4ca2-8ee9-6321ec3f1318' --cytomine_id_project 197 --cytomine_id_image_instance 2140 --cytomine_id_software 2633

      docker run --gpus all -it --rm --mount type=bind,source=/home/giussepi/Public/environments/Cytomine/cyto_CRLM/,target=/CRLM,bind-propagation=private --network=host ttt --cytomine_host 'localhost-core' --cytomine_public_key 'd2be8bd7-2b0b-40c3-9e81-5ad5765568f3' --cytomine_private_key '6dfe27d7-2ad1-4ca2-8ee9-6321ec3f1318' --cytomine_id_project 197 --cytomine_id_image_instance 31296 --cytomine_id_software 79732
    """

    parser = ArgumentParser(prog="Cytomine Python client example")

    # Cytomine connection parameters
    parser.add_argument('--cytomine_host', dest='host',
                        default='demo.cytomine.be', help="The Cytomine host")
    parser.add_argument('--cytomine_public_key', dest='public_key',
                        help="The Cytomine public key")
    parser.add_argument('--cytomine_private_key', dest='private_key',
                        help="The Cytomine private key")
    parser.add_argument('--cytomine_id_project', dest='id_project',
                        help="The project from which we want the images")
    parser.add_argument('--cytomine_id_software', dest='id_software',
                        help="The software to be used to process the image")
    parser.add_argument('--cytomine_id_image_instance', dest='id_image_instance',
                        help="The image to which the annotation will be added")

    params, _ = parser.parse_known_args(sys.argv[1:])

    with CytomineJob.from_cli(sys.argv[1:]) as cytomine:
        # TODO: To be tested on TITANx
        img = ImageInstance().fetch(params.id_image_instance)
        download_image(img)
        process_wsi_and_save(get_container_image_path(img))
        new_annotations = generate_polygons(get_container_image_path(img), adapt_to_cytomine=True)
        annotation_collection = None

        for label_key in new_annotations:
            # Sending annotation batches to the server
            for sub_list in chunks(new_annotations[label_key], ANNOTATION_BATCH):
                if not debug:
                    annotation_collection = AnnotationCollection()

                for exterior_points in sub_list:
                    if debug:
                        annotation_collection = AnnotationCollection()

                    annotation_collection.append(Annotation(
                        location=Polygon(exterior_points.astype(int).reshape(
                            exterior_points.shape[0], exterior_points.shape[2]).tolist()).wkt,
                        id_image=params.id_image_instance,
                        id_project=params.id_project,
                        id_terms=[CYTOMINE_LABELS[label_key]]
                    ))

                    if debug:
                        try:
                            annotation_collection.save()
                        except Exception as e:
                            print(exterior_points.astype(int).reshape(
                                exterior_points.shape[0], exterior_points.shape[2]).tolist())
                            plt.plot(*Polygon(exterior_points.astype(int).reshape(
                                exterior_points.shape[0], exterior_points.shape[2])).exterior.coords.xy)
                            plt.show()
                            # raise(e)
                            print(e)
                        finally:
                            time.sleep(1)

                if not debug:
                    annotation_collection.save()
                    time.sleep(ANNOTATION_SLEEP_TIME)

        # Adding pie chart labels data as image property
        # TODO: Change delete_results_file to True for final test on titanX
        num_pixels_per_label = get_pie_chart_data(
            get_container_image_path(img), delete_results_file=False)

        for percentage, label_ in zip(num_pixels_per_label, Label.names):
            Property(img, key=label_, value='{}%'.format(percentage)).save()

        remove_image_local_copy(img)

        cytomine.job.update(statusComment="Finished.")
