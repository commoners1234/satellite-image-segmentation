import os
import json
import math
import yaml
import glob
import numpy as np
import pandas as pd
import pathlib
from loss import *
import tensorflow as tf
import earthpy.plot as ep
import earthpy.spatial as es
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt
from dataset import read_img, transform_data

class SelectCallbacks(keras.callbacks.Callback):
    def __init__(self, val_dataset, model, config):
        super(keras.callbacks.Callback, self).__init__()
        self.val_dataset = val_dataset
        self.model = model
        self.config = config
        self.callbacks = []

    def lr_scheduler(self, epoch):     
        drop = 0.5
        epoch_drop = self.config['epochs'] / 8.
        lr = self.config['learning_rate'] * math.pow(drop, math.floor((1 + epoch) / epoch_drop))
        return lr

    def on_epoch_end(self, epoch, logs={}):
        if (epoch % self.config['val_plot_epoch'] == 0):
            show_predictions(self.val_dataset, self.model, self.config, True)

    def get_callbacks(self, val_dataset, model):
        if self.config['csv']:
            self.callbacks.append(keras.callbacks.CSVLogger(os.path.join(self.config['csv_log_dir'], self.config['csv_log_name']), separator = ",", append = False))
        if self.config['checkpoint']:
            self.callbacks.append(keras.callbacks.ModelCheckpoint(os.path.join(self.config['checkpoint_dir'], self.config['checkpoint_name']), save_best_only = True))
        if self.config['tensorboard']:
            self.callbacks.append(keras.callbacks.TensorBoard(log_dir = os.path.join(self.config['tensorboard_log_dir'], self.config['tensorboard_log_name'])))
        if self.config['lr']:
            self.callbacks.append(keras.callbacks.LearningRateScheduler(schedule = self.lr_scheduler))
        if self.config['early_stop']:
            self.callbacks.append(keras.callbacks.EarlyStopping(monitor = 'my_mean_iou', patience = self.config['patience']))
        if self.config['val_pred_plot']:
            self.callbacks.append(SelectCallbacks(val_dataset, model, self.config))
        return self.callbacks

def create_mask(mask, pred_mask):
    mask = np.argmax(mask, axis = 3)
    pred_mask = np.argmax(pred_mask, axis = 3)
    return mask, pred_mask

def display(display_list, idx, directory, score, exp):
    plt.figure(figsize=(12, 8))
    title = list(display_list.keys())

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        if title[i]=="DEM":
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
        elif title[i]=="VV" or title[i]=="VH":
            plt.title(title[i])
            plt.imshow((display_list[title[i]]), cmap="gray")
            plt.axis('off')
        else:
            plt.title(title[i])
            plt.imshow((display_list[title[i]]))
            plt.axis('off')

    prediction_name = "{}_{}_mio_{:.4f}.png".format(exp, idx, score) # create file name to save
    plt.savefig(os.path.join(directory, prediction_name), bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()

def show_predictions(dataset, model, config, val=False):
    if val:
        directory = config['prediction_val_dir']
    else:
        directory = config['prediction_test_dir']

    if config['plot_single']:
        feature, mask, idx = dataset.get_random_data(config['index'])
        data = [(feature, mask)]
    else:
        data = dataset
        idx = 0

    for feature, mask in data:
        prediction = model.predict_on_batch(feature)
        mask, pred_mask = create_mask(mask, prediction)
        for i in range(len(feature)):
            m = keras.metrics.MeanIoU(num_classes=config['num_classes'])
            m.update_state(mask[i], pred_mask[i])
            score = m.result().numpy()
            display({"VV": feature[i][:,:,0],
                     "VH": feature[i][:,:,1],
                     "DEM": feature[i][:,:,2],
                      "Mask": mask[i],
                      "Prediction (MeanIOU_{:.4f})".format(score): pred_mask[i]
                      }, idx, directory, score, config['experiment'])
            idx += 1

def patch_show_predictions(dataset, model, config):
    with open(config['p_test_dir'], 'r') as j:
        patch_test_dir = json.loads(j.read())
    
    df = pd.DataFrame.from_dict(patch_test_dir)
    test_dir = pd.read_csv(config['test_dir'])
    total_score = 0.0

    for i in range(len(test_dir)):
        idx = df[df["masks"]==test_dir["masks"][i]].index
        
        pred_full_label = np.zeros((512,512), dtype=int)
        for j in idx:
            p_idx = patch_test_dir["patch_idx"][j]
            feature, mask, _ = dataset.get_random_data(j)
            pred_mask = model.predict(feature)
            pred_mask = np.argmax(pred_mask, axis = 3)
            pred_full_label[p_idx[0]:p_idx[1], p_idx[2]:p_idx[3]] = pred_mask[0]
        
        feature = read_img(test_dir["feature_ids"][i], in_channels=config['in_channels'])
        mask = transform_data(read_img(test_dir["masks"][i], label=True), config['num_classes'])
        
        m = keras.metrics.MeanIoU(num_classes=config['num_classes'])
        m.update_state(np.argmax([mask], axis = 3), [pred_full_label])
        score = m.result().numpy()
        total_score += score
        
        display({"VV": feature[:,:,0],
                    "VH": feature[:,:,1],
                    "DEM": feature[:,:,2],
                    "Mask": np.argmax([mask], axis = 3)[0],
                    "Prediction (MeanIOU_{:.4f})".format(score): pred_full_label
                    }, i, config['prediction_test_dir'], score, config['experiment'])

def set_gpu(gpus):
    gpus = gpus.split(",")
    if len(gpus)>1:
        print("MirroredStrategy Enable")
        GPUS = []
        for i in range(len(gpus)):
            GPUS.append("GPU:{}".format(gpus[i]))
        strategy = tf.distribute.MirroredStrategy(GPUS)
    else:
        print("OneDeviceStrategy Enable")
        GPUS = []
        for i in range(len(gpus)):
            GPUS.append("GPU:{}".format(gpus[i]))
        strategy = tf.distribute.OneDeviceStrategy(GPUS[0])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)
    
    return strategy

def create_paths(config, test=False):
    if test:
        pathlib.Path(config['prediction_test_dir']).mkdir(parents = True, exist_ok = True)
    else:
        pathlib.Path(config['csv_log_dir']).mkdir(parents = True, exist_ok = True)
        pathlib.Path(config['tensorboard_log_dir']).mkdir(parents = True, exist_ok = True)
        pathlib.Path(config['checkpoint_dir']).mkdir(parents = True, exist_ok = True)
        pathlib.Path(config['prediction_val_dir']).mkdir(parents = True, exist_ok = True)

def get_config_yaml(path, args):
    with open(path, "r") as f:
      config = yaml.safe_load(f)
    
    for key in args.keys():
        if args[key] != None:
            config[key] = args[key]  
    config['height'] = config['patch_size']
    config['width'] = config['patch_size']
    config['train_dir'] = config['dataset_dir']+config['train_dir']
    config['valid_dir'] = config['dataset_dir']+config['valid_dir']
    config['test_dir'] = config['dataset_dir']+config['test_dir']
    config['p_train_dir'] = config['dataset_dir']+config['p_train_dir']
    config['p_valid_dir'] = config['dataset_dir']+config['p_valid_dir']
    config['p_test_dir'] = config['dataset_dir']+config['p_test_dir']
    config['tensorboard_log_name'] = "{}_{}_ep_{}_{}".format(config['model_name'],config['experiment'],config['epochs'],datetime.now().strftime("%d-%b-%y"))
    config['tensorboard_log_dir'] = config['root_dir']+'/logs/'+config['model_name']+'/'
    config['csv_log_name'] = "{}_{}_ep_{}_{}.csv".format(config['model_name'],config['experiment'],config['epochs'],datetime.now().strftime("%d-%b-%y"))
    config['csv_log_dir'] = config['root_dir']+'/csv_logger/'+config['model_name']+'/'
    config['checkpoint_name'] = "{}_{}_ep_{}_{}.hdf5".format(config['model_name'],config['experiment'],config['epochs'],datetime.now().strftime("%d-%b-%y"))
    config['checkpoint_dir'] = config['root_dir']+'/model/'+config['model_name']+'/'
    if config['load_model_dir']=='None':
        config['load_model_dir'] = config['root_dir']+'/model/'+config['model_name']+'/'

    config['prediction_test_dir'] = config['root_dir']+'/prediction/'+config['model_name']+'/test/'+config['experiment']+'/'
    config['prediction_val_dir'] = config['root_dir']+'/prediction/'+config['model_name']+'/validation/'+config['experiment']+'/'
    config['visualization_dir'] = config['root_dir']+'/visualization/'
    return config

def scale_img(matrix):
    min_values = np.array([[-23, -28, 0.2]])
    max_values = np.array([[0, -5, 1]])
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    matrix = (matrix - min_values) / (
        max_values - min_values
    )
    matrix = np.reshape(matrix, [w, h, d])
    return matrix.clip(0, 1)

def create_false_color_composite(vv_img, vh_img):
    s1_img = np.stack((vv_img, vh_img), axis=-1)
    img = np.zeros((512, 512, 3), dtype=np.float32)
    img[:,:,:2] = s1_img.copy()
    img[:, :, 2] = (s1_img[:, :, 0]*s1_img[:, :, 1])
    return scale_img(img)
