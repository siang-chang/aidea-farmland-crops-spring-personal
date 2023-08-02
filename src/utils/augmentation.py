#!/usr/bin/env python
# coding: utf-8

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def data_generator():
    """
    Returns:
    --------
        ImageDataGenerator
    """
    # Add data augmentation with its sample data
    
    return ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=True,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=True,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=15,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.1,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # brightness (min, max)
        brightness_range=(0.9, 1.1),
        # set range for random channel shifts
        channel_shift_range=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        # "channels_last" (samples, height, width, channels)
        # "channels_first" (samples, channels, height, width)
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0
    )


# %%
if __name__ == "__main__":
    import os
    import numpy as np
    from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
    
    read_from_dir = '../data/sample/'
    save_to_dir = '../data/result/'
    
    if not os.path.exists(save_to_dir):
        os.mkdir(save_to_dir)
    
    x_train = []
    for file in os.listdir(read_from_dir):
        image = load_img(read_from_dir+file, target_size=(512, 512))
        image = img_to_array(image)
        x_train.append(image)
    x_train = np.array(x_train)
    
    datagen = data_generator()
    datagen.fit(x_train)
    
    i = 0
    for batch in datagen.flow(
        x_train, batch_size=18,
        save_to_dir=save_to_dir, save_format="jpg"
    ):
        i += 1
        if i > 3:
            break


# %%
