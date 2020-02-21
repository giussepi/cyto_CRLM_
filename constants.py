# -*- coding: utf-8 -*-
""" constants """

from collections import namedtuple

from matplotlib.colors import ListedColormap


Detail = namedtuple('Detail', ['colour', 'id', 'name'])


class Label:
    """ Holds tissue colours, ids and label names """
    hepatocyte = Detail('pink', 0, 'Hepatocyte')
    necrosis = Detail('c', 1, 'Necrosis')
    fibrosis = Detail('lightgrey', 2, 'Fibrosis')
    tumour = Detail('saddlebrown', 3, 'Tumour')
    inflammation = Detail('g', 4, 'Inflammation')
    macrophage = Detail('r', 5, 'Macrophage')
    blood = Detail('purple', 6, 'Blood')
    other_tissue = Detail('k', 7, 'Other tissue')     # fb foreign blood reaction
    mucin = Detail('royalblue', 8, 'Mucin')
    bile_duct = Detail('gold', 9, 'Bile Duct')
    background = Detail('white', 10, 'Background')

    # CMAP = ListedColormap(['pink', 'c', 'lightgrey', 'saddlebrown', 'g', 'r', 'purple', 'royalblue', 'k', 'white', 'gold'])
    cmap = ListedColormap([
        hepatocyte.colour, necrosis.colour, fibrosis.colour, tumour.colour,
        inflammation.colour, macrophage.colour, blood.colour, other_tissue.colour,
        mucin.colour, bile_duct.colour, background.colour
    ])

    # 'white', 'c', 'lightgrey', 'saddlebrown', 'g'
    short_cmap = ListedColormap([
        background.colour, necrosis.colour, fibrosis.colour, tumour.colour,
        inflammation.colour
    ])

    ids = (hepatocyte.id, necrosis.id, fibrosis.id, tumour.id,
           inflammation.id, macrophage.id, blood.id, other_tissue.id,
           mucin.id, bile_duct.id, background.id)

    names = (hepatocyte.name, necrosis.name, fibrosis.name, tumour.name,
             inflammation.name, macrophage.name, blood.name, other_tissue.name,
             mucin.name, bile_duct.name, background.name)
