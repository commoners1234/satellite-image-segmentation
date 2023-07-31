# Satellite Image Segmentation

## Setup

First clone the github repo in your local or server machine by following:

```
git clone https://github.com/commoners1234/satellite-image-segmentation.git
```

Use Pip to create a new environment and install dependency from `requirement.txt` file. The following command will install the packages according to the configuration file `requirement.txt`.

```
pip install -r requirements.txt
```

Before start training check the variable inside config.yaml i.e. `height`, `in_channels`. Keep the above mention dataset in the data folder that give you following structure:

```
--data
    --train_features
        --image_id_vv.tif
        --image_id_vh.tif
            ..
    --train_labels
        --image_id.tif
        --image_id.tif
            ..
    flood-training-metadata.csv
```

## Experiments

After setup the required folders and package run one of the following experiment. There are four experiments based on combination of parameters passing through `argparse` and `config.yaml`. Combination of each experiments given below.

When you run the following code based on different experiments, some new directories will be created;

1. csv_logger (save all evaluation result in csv format)
2. logs (tensorboard logger)
3. model (save model checkpoint)
4. prediction (validation and test prediction png format)


In this experiment we take a threshold value (19%) of water class and remove the patch images for each chip that are less than threshold value.

```
python train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 300 \
    --batch_size 10 \
    --index -1 \
    --experiment tvl \
    --patchify True \
    --patch_size 256 \
    --weights False \
    --patch_class_balance True
```

## Testing

Run following model for evaluating train model on test dataset.

```
python project/test.py \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --load_model_name my_model.hdf5 \
    --plot_single False \
    --index -1 \
    --patchify True \
    --patch_size 256 \
    --experiment tvl \
```
