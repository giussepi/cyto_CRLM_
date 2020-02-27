# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2018. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys
from argparse import ArgumentParser

from cytomine import Cytomine
from cytomine.models import AnnotationCollection

__author__ = "Rubens Ulysse <urubens@uliege.be>"


def run():
    """
    Deletes all the annotations from an image, but those created by a software

    Example:
      python main.py --cytomine_host 'localhost-core' --cytomine_public_key 'b6ebb23c-00ff-427b-be24-87b2a82490df' --cytomine_private_key '6812f09b-3f33-4938-82ca-b23032d377fd' --cytomine_id_image_instance 347 --cytomine_id_user 61 --cytomine_id_project 154

      python main.py --cytomine_host 'localhost-core' --cytomine_public_key 'b6ebb23c-00ff-427b-be24-87b2a82490df' --cytomine_private_key '6812f09b-3f33-4938-82ca-b23032d377fd' --cytomine_id_image_instance 3643 --cytomine_id_user 61 --cytomine_id_project 154

      python main.py --cytomine_host 'localhost-core' --cytomine_public_key 'd2be8bd7-2b0b-40c3-9e81-5ad5765568f3' --cytomine_private_key '6dfe27d7-2ad1-4ca2-8ee9-6321ec3f1318' --cytomine_id_image_instance 2140 --cytomine_id_user 58 --cytomine_id_project 197
    """
    parser = ArgumentParser(prog="Cytomine Python client example")

    # Cytomine
    parser.add_argument('--cytomine_host', dest='host',
                        default='demo.cytomine.be', help="The Cytomine host")
    parser.add_argument('--cytomine_public_key', dest='public_key',
                        help="The Cytomine public key")
    parser.add_argument('--cytomine_private_key', dest='private_key',
                        help="The Cytomine private key")

    parser.add_argument('--cytomine_id_image_instance', dest='id_image_instance',
                        help="The image with annotations to delete")
    parser.add_argument('--cytomine_id_user', dest='id_user',
                        help="The user with annotations to delete")
    parser.add_argument('--cytomine_id_project', dest='id_project',
                        help="The project with annotations to delete")
    params, _ = parser.parse_known_args(sys.argv[1:])

    with Cytomine(host=params.host, public_key=params.public_key, private_key=params.private_key,
                  verbose=logging.INFO) as cytomine:
        # Get the list of annotations
        annotations = AnnotationCollection()
        annotations.image = params.id_image_instance
        # NOTE: use userjob id to retrieve annotations from the job. However, they
        # cannot be deleted.
        annotations.user = params.id_user
        annotations.project = params.id_project
        annotations.fetch()
        print(annotations)

        for annotation in annotations:
            annotation.delete()
