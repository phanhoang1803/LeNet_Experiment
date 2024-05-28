# src/data/data_loader.py
# Script for loading and preprocessing data

import tensorflow as tf
import tensorflow_datasets as tfds
import os
import shutil

data_dir = '../data'

def load_processed_caltech_data(processed_dir):
    """
    Load the processed data from the given processed directory.
    
    Args:
        processed_dir (str): Path to the directory containing the processed data.
        
    Returns:
        train_ds (tf.data.Dataset): Training dataset.
        valid_ds (tf.data.Dataset): Validation dataset.
        test_ds (tf.data.Dataset): Test dataset.
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of classes in the dataset.
    """
    print('Loading processed Caltech dataset')
    
    train_dir = os.path.join(processed_dir, 'train')
    valid_dir = os.path.join(processed_dir, 'valid')
    test_dir = os.path.join(processed_dir, 'test')
    
    train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, 
                                                           label_mode='int',
                                                           image_size=(224, 224),
                                                           batch_size=None,
                                                           verbose=True)
    valid_ds = tf.keras.utils.image_dataset_from_directory(valid_dir,
                                                           label_mode='int',
                                                           image_size=(224, 224),
                                                           batch_size=None,
                                                           verbose=True)
    test_ds = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                           label_mode='int',
                                                           image_size=(224, 224),
                                                           batch_size=None,
                                                           verbose=True)
    
    dataset = {
            'train': train_ds,
            'valid': valid_ds,
            'test': test_ds,
            'input_shape': (224, 224, 3),
            'num_classes': len([name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))]),
        }
    
    return dataset

def create_processed_caltech_dataset(raw_dir, processed_dir):
    """
    This function takes the raw dataset directory as input and processes it by creating separate directories
    for the train, validation, and test sets. It then copies the images into their respective directories.
    
    Args:
        raw_dir (str): Path to the directory containing the raw data.
        processed_dir (str): Path to the directory where the processed data will be saved.
    
    Returns:
        train_ds (tf.data.Dataset): Training dataset.
        valid_ds (tf.data.Dataset): Validation dataset.
        test_ds (tf.data.Dataset): Test dataset.
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of classes in the dataset.
    """
    print('Creating processed Caltech dataset')
    
    train_dir = os.path.join(processed_dir, 'train')
    valid_dir = os.path.join(processed_dir, 'valid')
    test_dir = os.path.join(processed_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get list of categories
    categories = [f for f in sorted(os.listdir(raw_dir))]
    if 'BACKGROUND_Google' in categories:
        categories.remove('BACKGROUND_Google') 
    for directory in [train_dir, valid_dir, test_dir]:
        for category in categories:
            os.makedirs(os.path.join(directory, category))
    
    for i, category in enumerate(categories):
        category_dir = os.path.join(raw_dir, category)
        no_images_per_category = len(os.listdir(category_dir))
        
        no_train = int(0.7 * no_images_per_category)
        no_valid = int(0.2 * no_images_per_category)
        # no_test = no_images_per_category - no_train - no_valid
        
        fnames = [fname for fname in sorted(os.listdir(category_dir))]
        for fname in fnames[:no_train]:
            src = os.path.join(category_dir, fname)
            dst = os.path.join(train_dir, category, fname)
            shutil.copyfile(src, dst)
        
        for fname in fnames[no_train:no_train+no_valid]:
            src = os.path.join(category_dir, fname)
            dst = os.path.join(valid_dir, category, fname)
            shutil.copyfile(src, dst)
        
        for fname in fnames[no_train+no_valid:]:
            src = os.path.join(category_dir, fname)
            dst = os.path.join(test_dir, category, fname)
            shutil.copyfile(src, dst)

    return load_processed_caltech_data(processed_dir)

def load_data(dataset_name, raw_dir=None):
    """
    Load the dataset with the given name.
    
    Args:
        dataset_name (str): Name of the dataset.
        raw_dir (str): Path to the directory containing the raw data (Optional).
        
    Returns:
        dataset (dict): Dictionary containing the dataset.
    """
    
    # Load MNIST dataset
    if dataset_name == 'mnist':
        print('Loading MNIST dataset')
        train_ds, valid_ds, test_ds = tfds.load('mnist', split=['train[:80%]', 'train[80%:]', 'test'], as_supervised=True)
        
        dataset = {
            'train': train_ds,
            'valid': valid_ds,
            'test': test_ds,
            'input_shape': (28, 28, 1),
            'num_classes': 10,
        }
        
        return dataset
    
    # Load Fashion MNIST dataset
    elif dataset_name == 'fmnist':
        print('Loading Fashion MNIST dataset')
        train_ds, valid_ds, test_ds = tfds.load('fashion_mnist', split=['train[:80%]', 'train[80%:]', 'test'], as_supervised=True)
        
        dataset = {
            'train': train_ds,
            'valid': valid_ds,
            'test': test_ds,
            'input_shape': (28, 28, 1),
            'num_classes': 10,
        }
        
        return dataset
    
    # Load Caltech 101 dataset
    elif dataset_name == 'caltech101':
        print('Loading Caltech 101 dataset')
        processed_dir = os.path.join(data_dir, 'processed', 'caltech101')
        train_dir = os.path.join(processed_dir, 'train')
        valid_dir = os.path.join(processed_dir, 'valid')
        test_dir = os.path.join(processed_dir, 'test')
        
        # If the dataset is already processed, load it
        if os.path.exists(train_dir) and os.path.exists(valid_dir) and os.path.exists(test_dir):
            return load_processed_caltech_data(processed_dir=processed_dir)
        else:
            # If the dataset is not processed, process it
            assert raw_dir
            assert os.path.exists(raw_dir)
            return create_processed_caltech_dataset(raw_dir=raw_dir, processed_dir=processed_dir)
        
    # Load Caltech 256 dataset
    elif dataset_name == 'caltech256':
        print('Loading Caltech 256 dataset')
        processed_dir = os.path.join(data_dir, 'processed', 'caltech256')
        train_dir = os.path.join(processed_dir, 'train')
        valid_dir = os.path.join(processed_dir, 'valid')
        test_dir = os.path.join(processed_dir, 'test')
        
        # If the dataset is already processed, load it
        if os.path.exists(train_dir) and os.path.exists(valid_dir) and os.path.exists(test_dir):
            return load_processed_caltech_data(processed_dir=processed_dir)
        else:
            # If the dataset is not processed, process it
            assert raw_dir
            assert os.path.exists(raw_dir)
            return create_processed_caltech_dataset(raw_dir=raw_dir, processed_dir=processed_dir)
        
    else:
        raise ValueError('Invalid dataset name: {}'.format(dataset_name))
