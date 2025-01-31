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

import os

from cytomine import Cytomine
from cytomine.models import StorageCollection, Project


__author__ = "Rubens Ulysse <urubens@uliege.be>"


def run():
    """
    Uploads a fixed image to cytomine

    Example:
        python main.py --cytomine_host 'localhost-core' --cytomine_public_key 'b6ebb23c-00ff-427b-be24-87b2a82490df' --cytomine_private_key '6812f09b-3f33-4938-82ca-b23032d377fd' --cytomine_upload_host 'localhost-upload' --cytomine_id_project 154 --filepath /media/giussepi/xingru_dev/CRLM/images/raw_images/CRLM_042.ndpi

        python main.py --cytomine_host 'localhost-core' --cytomine_public_key 'dadb7d7a-5822-48f7-ab42-59bce27750ae' --cytomine_private_key 'd73f4602-51d2-4d15-91e4-d4cc175d65fd' --cytomine_upload_host 'localhost-upload' --cytomine_id_project 187 --filepath /home/giussepi/Public/link/environments/CRLM/images/raw_images/CRLM_042.ndpi
    """
    parser = ArgumentParser(prog="Cytomine Python client example")

    # Cytomine
    parser.add_argument('--cytomine_host', dest='host',
                        default='demo.cytomine.be', help="The Cytomine host")
    parser.add_argument('--cytomine_public_key', dest='public_key',
                        help="The Cytomine public key")
    parser.add_argument('--cytomine_private_key', dest='private_key',
                        help="The Cytomine private key")
    parser.add_argument('--cytomine_upload_host', dest='upload_host',
                        default='demo-upload.cytomine.be', help="The Cytomine upload host")
    parser.add_argument('--cytomine_id_project', dest='id_project', required=False,
                        help="The project from which we want the images (optional)")
    parser.add_argument('--filepath', dest='filepath',
                        help="The filepath (on your file system) of the file you want to upload")
    params, other = parser.parse_known_args(sys.argv[1:])

    with Cytomine(host=params.host, public_key=params.public_key, private_key=params.private_key,
                  verbose=logging.INFO) as cytomine:

        print(cytomine.current_user)

        # Check that the file exists on your file system
        if not os.path.exists(params.filepath):
            raise ValueError("The file you want to upload does not exist")

        # Check that the given project exists
        if params.id_project:
            project = Project().fetch(params.id_project)

            if not project:
                raise ValueError("Project not found")

        # To upload the image, we need to know the ID of your Cytomine storage.
        storages = StorageCollection().fetch()
        my_storage = next(filter(lambda storage: storage.user == cytomine.current_user.id, storages))
        if not my_storage:
            raise ValueError("Storage not found")

        # there is a keyerror after running this line. Despite of this issue, it uploads
        # the file properly.
        uploaded_file = cytomine.upload_image(upload_host=params.upload_host,
                                              filename=params.filepath,
                                              id_storage=my_storage.id,
                                              id_project=params.id_project)

        print(uploaded_file)
