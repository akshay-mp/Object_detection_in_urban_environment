import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import shutil


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    path=glob.glob(data_dir+'/*.tfrecord')
    
    new_files=[data_file for data_file in path]
       
      
    np.random.shuffle(new_files)
    
    train_set, val_set = np.split(new_files, [int(.8*len(new_files))])
    
    for file in train_set:
        shutil.move(file, 'data/waymo/train')
    
    for file in val_set:
        shutil.move(file, 'data/waymo/val')

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)