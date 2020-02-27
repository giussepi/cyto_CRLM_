# -*- coding: utf-8 -*-
""" main.py """

# import matplotlib

# from Load.Load_model_on_wsi import process_wsi_and_save, plot, generate_polygons
# from Train.Train_patchcnn_448 import run
from utils.add_annotations import run
# from utils.delete_annotations import run
# from utils.upload_image import run


# matplotlib.use('qt5agg')


def main():
    run()
    # plot()
    # generate_polygons(generate_image=True)


if __name__ == '__main__':
    main()
