import os
import time
import argparse
from loss import *
from metrics import get_metrics
from model import get_model, get_model_transfer_lr
from tensorflow import keras
from utils import set_gpu, SelectCallbacks, get_config_yaml, create_paths
from dataset import get_train_val_dataloader
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa

tf.config.optimizer.set_jit("True")
parser = argparse.ArgumentParser()

parser.add_argument("--root_dir")
parser.add_argument("--dataset_dir")
parser.add_argument("--model_name")
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--index", type=int)
parser.add_argument("--experiment")
parser.add_argument("--patchify")
parser.add_argument("--patch_size", type=int)
parser.add_argument("--weights")
parser.add_argument("--patch_class_balance")
args = parser.parse_args()
config = get_config_yaml('project/config.yaml', vars(args))
create_paths(config)
print("Model = {}".format(config['model_name']))
print("Epochs = {}".format(config['epochs']))
print("Batch Size = {}".format(config['batch_size']))
print("Preprocessed Data = {}".format(os.path.exists(config['train_dir'])))
print("Class Weigth = {}".format(str(config['weights'])))
print("Experiment = {}".format(str(config['experiment'])))
train_dataset, val_dataset = get_train_val_dataloader(config)
metrics = list(get_metrics(config).values())
custom_obj = get_metrics(config)
learning_rate = 0.001
weight_decay = 0.0001
adam = tfa.optimizers.AdamW(learning_rate = learning_rate, weight_decay = weight_decay)
custom_obj['loss'] = focal_loss()
if (os.path.exists(os.path.join(config['load_model_dir'], config['load_model_name']))) and config['transfer_lr']: 
    print("Build model for transfer learning..")
    model = load_model(os.path.join(config['load_model_dir'], config['load_model_name']), custom_objects = custom_obj, compile = True)
    model = get_model_transfer_lr(model, config['num_classes'])
    model.compile(optimizer = adam, loss = loss, metrics = metrics)

else:
    if (os.path.exists(os.path.join(config['load_model_dir'], config['load_model_name']))):
        print("Resume training from model checkpoint {}...".format(config['load_model_name']))
        model = load_model(os.path.join(config['load_model_dir'], config['load_model_name']), custom_objects = custom_obj, compile = True)
    else:
        model = get_model(config)
        model.compile(optimizer = adam, loss = bce_jaccard_loss, metrics = metrics)

loggers = SelectCallbacks(val_dataset, model, config)
model.summary()
t0 = time.time()
history = model.fit(train_dataset,
                    verbose = 1, 
                    epochs = config['epochs'],
                    validation_data = val_dataset, 
                    shuffle = False,
                    callbacks = loggers.get_callbacks(val_dataset, model),
                    )
print("training time minute: {}".format((time.time()-t0)/60))