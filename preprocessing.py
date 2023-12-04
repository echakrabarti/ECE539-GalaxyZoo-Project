import tensorflow as tf
from tensorflow.io import decode_image, read_file
from tensorflow.image import resize_with_crop_or_pad, resize
from tensorflow.keras.layers import RandomRotation, RandomTranslation, RandomZoom, RandomFlip, RandomBrightness, RandomContrast
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import keras

def test():
    print("Hello World!")

# returns a cropped and resized 224x224 image
def get_image(image_path):
    image = decode_image(read_file(image_path), channels=3)
    image = resize_with_crop_or_pad(image, 207, 207)
    image = resize(image, (69, 69))
    return image

# converts X, y, and a batch size into a tensorflow dataset
def get_train_dataset(X, y, batch_size):    
    path_dataset = tf.data.Dataset.from_tensor_slices(X)
    image_dataset = path_dataset.map(get_image, num_parallel_calls=tf.data.AUTOTUNE)

    label_dataset = tf.data.Dataset.from_tensor_slices(tf.cast(y, tf.float32))

    dataset = tf.data.Dataset.zip(image_dataset, label_dataset)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# converts X into a tensorflow dataset
def get_test_dataset(X, batch_size):
    path_dataset = tf.data.Dataset.from_tensor_slices(X)
    dataset = path_dataset.map(get_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset

# generates the training, validation, and testing datasets
def get_datasets(batch_size, augment=True, preprocess_input=None):
    labels_df = pd.read_csv('./dataset/training_solutions_rev1.csv')
    
    train_paths = os.listdir('./dataset/images_training_rev1/')
    train_paths = ['./dataset/images_training_rev1/' + path for path in train_paths]
    train_paths.sort()
    features = labels_df.values[:, 1:]
    X_train, X_val, y_train, y_val = train_test_split(train_paths, features, test_size=0.2, random_state=0)
    train_dataset = get_train_dataset(X_train, y_train, batch_size)
    val_dataset = get_train_dataset(X_val, y_val, batch_size)   
    
    test_paths = os.listdir('./dataset/images_test_rev1/')
    X_test = ['./dataset/images_test_rev1/' + path for path in test_paths]
    test_dataset = get_test_dataset(X_test, batch_size)
    
    if augment:
        train_dataset = augment_dataset(train_dataset)
    if preprocess_input:
        train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        test_dataset = test_dataset.map(lambda x: preprocess_input(x), num_parallel_calls=tf.data.AUTOTUNE)
    return train_dataset, val_dataset, test_dataset

# augments the dataset
def augment_dataset(dataset):
    data_augmentation = tf.keras.Sequential([
        RandomRotation(0.5), # 180 degrees each way (360 degrees total))
        RandomTranslation(0.06, 0.06), # roughly 4 pixels each way
        RandomZoom(0.3), # 0.7x to 1.3x
        RandomFlip('horizontal_and_vertical'), 
        RandomBrightness(0.2), # 0.8x to 1.2x
        RandomContrast(0.2), # 0.8x to 1.2x
    ])
    
    # map the data_augmentation function to the dataset
    augmented_dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                                    num_parallel_calls=tf.data.AUTOTUNE)
    return augmented_dataset