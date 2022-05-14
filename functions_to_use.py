import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import keras
import os
import splitfolders
from pathlib import Path
import imghdr

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Resizing,
    Rescaling,
    Lambda,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
    MaxPooling2D,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    Dropout, Input
)

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

import warnings
warnings.filterwarnings('ignore')

sns.set_style('darkgrid')


def checking_extensions(data_path):

    image_extensions = [".png", ".jpg", ".jpeg"]
    files_with_wrong_extension = []

    for filepath in Path(data_path).rglob("*"):
        if os.path.isfile(filepath):

            if not filepath.suffix.lower() in image_extensions:
                files_with_wrong_extension += [str(filepath)]
                print(filepath)

    return files_with_wrong_extension

def checking_if_images_are_valid(path_to_data_dir):

    image_extensions = [".png", ".jpg", ".jpeg"]
    img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
    invalid_images = []

    for filepath in Path(path_to_data_dir).rglob("*"):

        if filepath.suffix.lower() in image_extensions:
            img_type = imghdr.what(filepath)
            
            if img_type is None:
                print(f"{filepath} is not an image")
                invalid_images += [filepath]

            elif img_type not in img_type_accepted_by_tf:
                print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
                invalid_images += [filepath]

    return invalid_images

def displaying_random_images(batched_dataset, class_names):

    plt.figure(figsize=(20, 20))

    for images, labels in batched_dataset.take(np.random.randint(1, 4)):

        for i in range(16):
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]], fontsize=25)
            plt.axis("off")

    plt.show()

def data_augmentation_from_keras():

    data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"), 
    RandomRotation(0.2),
    RandomZoom(0.1),
    RandomTranslation(0.1, 0.1)
    ])

    return data_augmentation

def augmented_sample(train_dataset):

    augmented_photos= []
    data_augmentation = data_augmentation_from_keras()

    for images, _ in train_dataset.take(1):
        for i in range(9):

            augmented_images = data_augmentation(images)
            augmented_photos += [augmented_images[0]]

    return augmented_photos

def displaying_augmented_image(augmented_images):

    plt.figure(figsize=(15, 15))

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[i].numpy().astype("uint8"))
        plt.axis("off")

    plt.suptitle('Sample of augmented data', fontsize=25)
    plt.tight_layout()
    plt.show()

def creating_convolutional_neural_network(num_classes, image_size):

    data_augmentation = data_augmentation_from_keras()
        
    model = Sequential([
        Rescaling(1. / 255),
        data_augmentation,
        # Conv2D(64, 3, padding='same', activation='relu'),
        # MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(8, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(num_classes)
    ])

    return model

def plotting_learing_history(history):

    if isinstance(history, pd.DataFrame):
        accuracy = history['accuracy']
        validation_accuracy = history['val_accuracy']

        loss = history['loss']
        validation_loss = history['val_loss']

    else:
        accuracy = history.history['accuracy']
        validation_accuracy = history.history['val_accuracy']

        loss = history.history['loss']
        validation_loss = history.history['val_loss']

    fig, axes = plt.subplots(2, 1, figsize=(18, 12))

    axes[0].plot(accuracy, linewidth=2, label="Training accuracy")
    axes[0].plot(validation_accuracy, linewidth=2, label="Validation accuracy")
    axes[0].grid(True)
    axes[0].set_title('Training and validation accuracy', fontsize=20)
    axes[0].set_xlabel('Epoch', fontsize=15)
    axes[0].set_ylabel('Accuracy', fontsize=15)
    axes[0].legend(fontsize=16)

    axes[1].plot(loss, linewidth=2, label="Loss function value for training set")
    axes[1].plot(validation_loss, linewidth=2, label="Loss function value for validation set")
    axes[1].grid(True)
    axes[1].set_title('Training and validation loss', fontsize=20)
    axes[1].set_xlabel('Epoch', fontsize=15)
    axes[1].set_ylabel('Loss', fontsize=15)
    axes[1].legend(fontsize=16)

    fig.tight_layout()
    plt.show()

def max_probality(preds):
    return np.argmax(preds)

def comparing_true_and_predicted(list_with_tensors, new_real_labels, new_preds, class_names):

    plt.figure(figsize=(25, 25))

    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        index1, index2 = np.random.randint(0, 8), np.random.randint(0, 128)

        if index1 == 7 and index2 == 127:
                index2 = 126

        plt.imshow(list_with_tensors[index1][index2].astype("uint8"))
        plt.title(f'Real label: {class_names[new_real_labels[index1][index2]]}\nPredicted: {class_names[new_preds[index1][index2]]}', fontsize=25)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
