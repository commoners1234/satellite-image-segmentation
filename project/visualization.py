import os

import json
import rasterio
import pathlib
import numpy as np
import pandas as pd

from utils import get_config_yaml, create_false_color_composite
from dataset import get_test_dataloader, read_img, transform_data
from tensorflow.keras.models import load_model
import earthpy.plot as ep
import earthpy.spatial as es
from matplotlib import pyplot as plt


# setup gpu
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def display_all(data):
    """
    Summary:
        save all images into single figure
    Arguments:
        data : data file holding images path
        directory (str) : path to save images
    Return:
        save images figure into directory
    """
    
    pathlib.Path((config['visualization_dir']+'display')).mkdir(parents = True, exist_ok = True)

    for i in range(len(data)):
        with rasterio.open((data.feature_ids.values[i]+"_vv.tif")) as vv:
            vv_img = vv.read(1)
        with rasterio.open((data.feature_ids.values[i]+"_vh.tif")) as vh:
            vh_img = vh.read(1)
        with rasterio.open((data.feature_ids.values[i]+"_nasadem.tif")) as dem:
            dem_img = dem.read(1)
        with rasterio.open((data.masks.values[i])) as l:
            lp_img = l.read(1)
            lp_img[lp_img==255]=0
        id = data.feature_ids.values[i].split("/")[-1]
        display_list = {
                     "vv":vv_img,
                     "vh":vh_img,
                     "dem":dem_img,
                     "label":lp_img}


        plt.figure(figsize=(12, 8))
        title = list(display_list.keys())

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            
            # plot dem channel using earthpy
            if title[i]=="dem":
                ax = plt.gca()
                hillshade = es.hillshade(display_list[title[i]], azimuth=180)
                ep.plot_bands(
                    display_list[title[i]],
                    cbar=False,
                    cmap="terrain",
                    title=title[i],
                    ax=ax
                )
                ax.imshow(hillshade, cmap="Greys", alpha=0.5)
            
            # gray image plot vv and vh channels
            elif title[i]=="vv" or title[i]=="vh":
                plt.title(title[i])
                plt.imshow((display_list[title[i]]), cmap="gray")
                plt.axis('off')
            else:
                plt.title(title[i])
                plt.imshow((display_list[title[i]]))
                plt.axis('off')

        prediction_name = "img_id_{}.png".format(id) # create file name to save
        plt.savefig(os.path.join((config['visualization_dir']+'display'), prediction_name), bbox_inches='tight', dpi=800)
        plt.clf()
        plt.cla()
        plt.close()


def class_balance_check(patchify, data_dir):
    """
    Summary:
        checking class percentage in full dataset
    Arguments:
        patchify (bool): TRUE if want to check class balance for patchify experiments
        data_dir (str): directory where data files save
    Return:
        class percentage
    """
    if patchify:
        with open(data_dir, 'r') as j:
            train_data = json.loads(j.read())
        labels = train_data['masks']
        patch_idx = train_data['patch_idx']
    else:
        train_data = pd.read_csv(data_dir)
        labels = train_data.masks.values
        patch_idx = None
    class_one_t = 0
    class_zero = 0
    total = 0

    for i in range(len(labels)):
        with rasterio.open(labels[i]) as l:
            mask = l.read(1)
        mask[mask == 255] = 0
        if patchify:
            idx = patch_idx[i]
            mask = mask[idx[0]:idx[1], idx[2]:idx[3]]
        total_pix = mask.shape[0]*mask.shape[1]
        total += total_pix
        class_one = np.sum(mask)
        class_one_t += class_one
        class_zero_p = total_pix-class_one
        class_zero += class_zero_p
    
    print("Water Class percentage in train after class balance: {}".format((class_one_t/total)*100))

def class_distribution(data):
    masks = data["masks"]
    pixels = {"Water":0, "NON-Water":0}
    for i in range(len(masks)):
        mask = read_img(masks[i], label=True)
        pixels["Water"] += np.sum(mask)
        pixels["NON-Water"] += (mask.shape[0]*mask.shape[1]) - np.sum(mask)
    return pixels


def display_color_composite(data):
    """
    Plots a 3-channel representation of VV/VH polarizations as a single chip (image 1).
    Overlays a chip's corresponding water label (image 2).

    Args:
        random_state (int): random seed used to select a chip

    Returns:
        plot.show(): chip and labels plotted with pyplot
    """
    
    pathlib.Path((config['visualization_dir']+'display_color_composite')).mkdir(parents = True, exist_ok = True)
    
    for i in range(len(data)):
        f, ax = plt.subplots(1, 2, figsize=(9, 9))
        with rasterio.open((data.feature_ids.values[i]+"_vv.tif")) as vv:
            vv_img = vv.read(1)
        with rasterio.open((data.feature_ids.values[i]+"_vh.tif")) as vh:
            vh_img = vh.read(1)
        with rasterio.open((data.feature_ids.values[i]+"_nasadem.tif")) as dem:
            dem_img = dem.read(1)
        with rasterio.open((data.masks.values[i])) as l:
            lp_img = l.read(1)
            lp_img[lp_img==255]=0

        # Create false color composite
        s1_img = create_false_color_composite(vv_img, vh_img)

        # Visualize features
        ax[0].imshow(s1_img)
        ax[0].set_title("Feature", fontsize=14)

        # Mask missing data and 0s for visualization
        label = np.ma.masked_where((lp_img == 0) | (lp_img == 255), lp_img)

        # Visualize water label
        ax[1].imshow(s1_img)
        ax[1].imshow(label, cmap="cool", alpha=1)
        ax[1].set_title("Feature with Water Label", fontsize=14)
        id = data.feature_ids.values[i].split("/")[-1]
        prediction_name = "img_id_{}.png".format(id) # create file name to save
        plt.savefig(os.path.join((config['visualization_dir']+'display_color_composite'), prediction_name), bbox_inches='tight', dpi=800)


if __name__=='__main__':
    
    config = get_config_yaml('project/config.yaml', {})
    
    pathlib.Path(config['visualization_dir']).mkdir(parents = True, exist_ok = True)

    # check class balance for patchify pass True and p_train_dir
    # check class balance for original pass False and train_dir
    class_balance_check(True, config["p_train_dir"])


    train_dir = pd.read_csv(config['train_dir'])
    print("Train examples: ", len(train_dir))
    print(class_distribution(train_dir))

    test_dir = pd.read_csv(config['test_dir'])
    print("Test examples: ", len(test_dir))
    print(class_distribution(test_dir))

    valid_dir = pd.read_csv(config['valid_dir'])
    print("Valid examples: ", len(valid_dir))
    print(class_distribution(valid_dir))
    
    print("Saving figures....")
    display_all(train_dir)
    display_all(valid_dir)
    display_all(test_dir)
    
    print("Saving color composite figures....")    
    display_color_composite(train_dir)
    display_color_composite(valid_dir)
    display_color_composite(test_dir)