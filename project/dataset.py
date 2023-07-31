import os
import math
import json
import rasterio
import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as A
import matplotlib
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, Sequence
matplotlib.use('Agg')

# labels normalization values       
label_norm = {0:["_vv.tif", -17.54, 5.15],
                1:["_vh.tif",-10.68, 4.62],
                2:["_nasadem.tif",166.47, 178.47],
                3:["_jrc-gsw-change.tif", 238.76, 5.15],
                4:["_jrc-gsw-extent.tif", 2.15, 22.71],
                5:["_jrc-gsw-occurrence.tif", 6.50, 29.06],
                6:["_jrc-gsw-recurrence.tif", 10.04, 33.21],
                7:["_jrc-gsw-seasonality.tif", 2.60, 22.79],
                8:["_jrc-gsw-transitions.tif", 0.55, 1.94]}


def transform_data(label, num_classes):
    return to_categorical(label, num_classes = num_classes)


def read_img(directory, in_channels=None, label=False, patch_idx=None, height=512, width=512):
    if label:
        with rasterio.open(directory) as fmask: 
            mask = fmask.read(1)    
            mask[mask == 255] = 0
            
            if patch_idx:  
                return mask[patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3]] 
            else:
                return mask
    
    else:
        X = np.zeros((height,width, in_channels)) 
        
        
        for i in range(in_channels):
            tmp_ext = label_norm[i][0] 
            with rasterio.open((directory+tmp_ext)) as f:
                fea = f.read(1)
            
            X[:,:,i] = (fea - label_norm[i][1]) / label_norm[i][2]
         
        if patch_idx:
            return X[patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3],:]
        else:
            return X


def data_split(images, masks, config):
    x_train, x_rem, y_train, y_rem = train_test_split(images, masks, train_size = config['train_size'], random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size = 0.5, random_state=42)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def save_csv(dictionary, config, name):
    df = pd.DataFrame.from_dict(dictionary) 
    df.to_csv((config['dataset_dir']+name), index=False, header=True)


def data_path_split(config):
    paths = pd.read_csv((config['dataset_dir']+"flood-training-metadata.csv")) 
    paths = paths.drop_duplicates('chip_id').reset_index(drop=True)
    ids = list(paths.chip_id.values)

    masks = []
    images = []
    
    for i in range(len(ids)):
        masks.append(config['dataset_dir']+"train_labels/"+ids[i]+".tif")
        images.append(config['dataset_dir']+"train_features/"+ids[i])
    
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_split(images, masks, config)
    
    train = {'feature_ids': x_train, 'masks': y_train}
    valid = {'feature_ids': x_valid, 'masks': y_valid}
    test = {'feature_ids': x_test, 'masks': y_test}
    
    save_csv(train, config, "train.csv")
    save_csv(valid, config, "valid.csv")
    save_csv(test, config, "test.csv")


def class_percentage_check(label):
        
    total_pix = label.shape[0]*label.shape[0]
    class_one = np.sum(label)
    class_zero_p = total_pix-class_one
    
    return {"zero_class":((class_zero_p/total_pix)*100), 
            "one_class":((class_one/total_pix)*100)
    }


def save_patch_idx(path, patch_size=256, stride=8, test=None, patch_class_balance=None):
    with rasterio.open(path) as t:
        img = t.read(1)
        img[img == 255] = 0
    
    patch_height = int((img.shape[0]-patch_size)/stride)+1
    patch_weight = int((img.shape[1]-patch_size)/stride)+1
    
    patch_idx = []
    
    for i in range(patch_height):
        s_row = i * stride
        e_row = s_row + patch_size
        if e_row <= img.shape[0]:
            for j in range(patch_weight):
                start = (j*stride)
                end = start+patch_size
                if end <= img.shape[1]:
                    tmp = img[s_row:e_row, start:end]
                    percen = class_percentage_check(tmp)
                    if not patch_class_balance or test=='test':
                        patch_idx.append([s_row, e_row, start, end])
                    else:
                        if percen["one_class"]>19.0:
                            patch_idx.append([s_row, e_row, start, end])
    return  patch_idx

def write_json(target_path, target_file, data):
    if not os.path.exists(target_path): 
        try:
            os.makedirs(target_path)
        except Exception as e:
            print(e)
            raise
    
    with open(os.path.join(target_path, target_file), 'w') as f:
        json.dump(data, f) 


def patch_images(data, config, name):
    img_dirs = []
    masks_dirs = []
    all_patch = []
    for i in range(len(data)):  
        
        patches = save_patch_idx(data.masks.values[i], patch_size=config['patch_size'], stride=config['stride'], test=name.split("_")[0], patch_class_balance=config['patch_class_balance'])    # fetching patch indices
        
        for patch in patches:
            img_dirs.append(data.feature_ids.values[i])
            masks_dirs.append(data.masks.values[i])
            all_patch.append(patch)
            
    temp = {'feature_ids': img_dirs, 'masks': masks_dirs, 'patch_idx':all_patch}
    
    write_json((config['dataset_dir']+"json/"), (name+str(config['patch_size'])+'.json'), temp)


class Augment:
    def __init__(self, batch_size, channels, ratio=0.3, seed=42):
        super().__init__()
        self.ratio=ratio
        self.channels= channels
        self.aug_img_batch = math.ceil(batch_size*ratio)
        self.aug = A.Compose([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Blur(p=0.5),])

    def call(self, feature_dir, label_dir, patch_idx=None):
        aug_idx = np.random.randint(0, len(feature_dir), self.aug_img_batch)
        features = []
        labels = []
        
        for i in aug_idx:
            
            if patch_idx:
                img = read_img(feature_dir[i], in_channels = self.channels, patch_idx=patch_idx[i])
                mask = read_img(label_dir[i], label=True,patch_idx=patch_idx[i])
                
            else:
                img = read_img(feature_dir[i], in_channels = self.channels)
                mask = read_img(label_dir[i], label=True)
                
            augmented = self.aug(image=img, mask=mask)
            features.append(augmented['image'])
            labels.append(augmented['mask'])
        return features, labels

class MyDataset(Sequence):

    def __init__(self, img_dir, tgt_dir, in_channels, 
                 batch_size, num_class, patchify,
                 transform_fn=None, augment=None, weights=None, patch_idx=None):

        self.img_dir = img_dir
        self.tgt_dir = tgt_dir
        self.patch_idx = patch_idx
        self.patchify = patchify
        self.in_channels = in_channels
        self.transform_fn = transform_fn
        self.batch_size = batch_size
        self.num_class = num_class
        self.augment = augment
        self.weights = weights

    def __len__(self):
        return math.ceil(len(self.img_dir) // self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.img_dir[idx * self.batch_size:(idx + 1) *self.batch_size]
        batch_y = self.tgt_dir[idx * self.batch_size:(idx + 1) *self.batch_size]
        
        if self.patchify: 
            batch_patch = self.patch_idx[idx * self.batch_size:(idx + 1) *self.batch_size]
        
        imgs = []
        tgts = []
        
        for i in range(len(batch_x)):
            if self.patchify:
                imgs.append(read_img(batch_x[i], in_channels = self.in_channels, patch_idx = batch_patch[i]))
                
                if self.transform_fn:
                    tgts.append(self.transform_fn(read_img(batch_y[i], label=True, patch_idx = batch_patch[i]), self.num_class))
                else:
                    tgts.append(read_img(batch_y[i], label=True,patch_idx=batch_patch[i]))
            else:
                imgs.append(read_img(batch_x[i], in_channels = self.in_channels))
                
                if self.transform_fn:
                    tgts.append(self.transform_fn(read_img(batch_y[i], label=True), self.num_class))
                else:
                    tgts.append(read_img(batch_y[i], label=True))
        
        if self.augment:
            if self.patchify:
                aug_imgs, aug_masks = self.augment.call(self.img_dir, self.tgt_dir, self.patch_idx)
                imgs = imgs+aug_imgs
            else:
                aug_imgs, aug_masks = self.augment.call(self.img_dir, self.tgt_dir)
                imgs = imgs+aug_imgs

            if self.transform_fn:
                for i in range(len(aug_masks)):
                    tgts.append(self.transform_fn(aug_masks[i], self.num_class))
            else:
                tgts = tgts+aug_masks

        tgts = np.array(tgts)
        imgs = np.array(imgs)
        
        if self.weights != None:
            class_weights = tf.constant(self.weights)
            class_weights = class_weights/tf.reduce_sum(class_weights)
            
            y_weights = tf.gather(class_weights, indices=tf.cast(tgts, tf.int32))

            return tf.convert_to_tensor(imgs), y_weights
        
        return tf.convert_to_tensor(imgs), tf.convert_to_tensor(tgts)
    

    def get_random_data(self, idx=-1):
        if idx!=-1:
            idx = idx
        else:
            idx = np.random.randint(0, len(self.img_dir))
        
        imgs = []
        tgts = []
        if self.patchify:
            imgs.append(read_img(self.img_dir[idx], in_channels=self.in_channels, patch_idx=self.patch_idx[idx]))
            
            if self.transform_fn:
                tgts.append(self.transform_fn(read_img(self.tgt_dir[idx], label=True,patch_idx=self.patch_idx[idx]), self.num_class))
            else:
                tgts.append(read_img(self.tgt_dir[idx], label=True,patch_idx=self.patch_idx[idx]))
        else:
            imgs.append(read_img(self.img_dir[idx], in_channels=self.in_channels))
            
            if self.transform_fn:
                tgts.append(self.transform_fn(read_img(self.tgt_dir[idx], label=True), self.num_class))
            else:
                tgts.append(read_img(self.tgt_dir[idx], label=True))

        return tf.convert_to_tensor(imgs), tf.convert_to_tensor(tgts), idx


def get_train_val_dataloader(config):
    if not (os.path.exists(config['train_dir'])):
        data_path_split(config)
    
    if not (os.path.exists(config["p_train_dir"])) and config['patchify']:
        print("Saving patchify indices for train and test.....")
        data = pd.read_csv(config['train_dir'])
        if config["patch_class_balance"]:
            patch_images(data, config, "train_patch_WOC_")
        else:
            patch_images(data, config, "train_patch_")
        
        data = pd.read_csv(config['valid_dir'])
        if config["patch_class_balance"]:
            patch_images(data, config, "valid_patch_WOC_")
        else:
            patch_images(data, config, "valid_patch_")        
    
    if config['patchify']:
        print("Loading Patchified features and masks directories.....")
        with open(config['p_train_dir'], 'r') as j:
            train_dir = json.loads(j.read())
        with open(config['p_valid_dir'], 'r') as j:
            valid_dir = json.loads(j.read())
        train_features = train_dir['feature_ids']
        train_masks = train_dir['masks']
        valid_features = valid_dir['feature_ids']
        valid_masks = valid_dir['masks']
        train_idx = train_dir['patch_idx']
        valid_idx = valid_dir['patch_idx']
    
    else:
        print("Loading features and masks directories.....")
        train_dir = pd.read_csv(config['train_dir'])
        valid_dir = pd.read_csv(config['valid_dir'])
        train_features = train_dir.feature_ids.values
        train_masks = train_dir.masks.values
        valid_features = valid_dir.feature_ids.values
        valid_masks = valid_dir.masks.values
        train_idx = None
        valid_idx = None

    print("train Example : {}".format(len(train_features)))
    print("valid Example : {}".format(len(valid_features)))
    
    if config['augment'] and config['batch_size']>1:
        augment_obj = Augment(config['batch_size'], config['in_channels'])
        n_batch_size = config['batch_size']-augment_obj.aug_img_batch
    else:
        n_batch_size = config['batch_size']
        augment_obj = None

    if config['weights']:
        weights=tf.constant(config['balance_weights'])
    else:
        weights = None
    
    train_dataset = MyDataset(train_features,
                              train_masks,
                              in_channels = config['in_channels'],
                              patchify = config['patchify'],
                              batch_size = n_batch_size,
                              transform_fn = transform_data,
                              num_class = config['num_classes'],
                              augment = augment_obj,
                              weights = weights,
                              patch_idx = train_idx)

    val_dataset = MyDataset(valid_features, valid_masks,
                            in_channels = config['in_channels'],
                            patchify = config['patchify'],
                            batch_size = config['batch_size'], transform_fn=transform_data, 
                            num_class=config['num_classes'],patch_idx=valid_idx)
    
    return train_dataset, val_dataset


def get_test_dataloader(config):
    if not (os.path.exists(config['test_dir'])):
        data_path_split(config)
    
    if not (os.path.exists(config["p_test_dir"])) and config['patchify']:
        print("Saving patchify indices for test.....")
        data = pd.read_csv(config['test_dir'])
        patch_images(data, config, "test_patch_")
    
    if config['patchify']:
        print("Loading Patchified features and masks directories.....")
        with open(config['p_test_dir'], 'r') as j:
            test_dir = json.loads(j.read())
        test_features = test_dir['feature_ids']
        test_masks = test_dir['masks']
        test_idx = test_dir['patch_idx']
    
    else:
        print("Loading features and masks directories.....")
        test_dir = pd.read_csv(config['test_dir'])
        test_features = test_dir.feature_ids.values
        test_masks = test_dir.masks.values
        test_idx = None

    print("test Example : {}".format(len(test_features)))
    test_dataset = MyDataset(test_features, test_masks,
                            in_channels=config['in_channels'],patchify=config['patchify'],
                            batch_size=config['batch_size'], transform_fn=transform_data, 
                            num_class=config['num_classes'],patch_idx=test_idx)
    
    return test_dataset