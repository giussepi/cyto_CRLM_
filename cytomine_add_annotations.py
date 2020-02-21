# -*- coding: utf-8 -*-
"""
cytomine_add_annotations

Script to be run by cytomine server
"""

from utils.add_annotations import run


def main():
    """ Runs utils.add_annotations method on cytomine processing server """
    run()


if __name__ == "__main__":
    main()
