#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np

def load_image_and_resize(path, new_size=224, resize_method="resize", padding_method="wrap"):
    """Use CV2 to read image, and resize the image to make the aspect ratio.

    Arguments:
    ----------
        path: string, default=None
            image path. 

        new_size: number, default=500
            desired image size.

        resize_method: 'resize' or 'padding', default='resize'
            resize: only resize.
            padding: padding and resize.

        padding_method: 'wrap' or 'constant'
            wrap: take mirrored pixel padding.
            constant: border padding to a fixed value.

    Returns:
    --------
        image_rgb: 3d-array
            image vector converted to rgb format.
    """
    image = cv2.imread(path)
    height, width, channel = image.shape

    if resize_method == "padding":
        # resize by ratio
        ratio = new_size/max(height, width)
        new_height, new_width = int(ratio * height), int(ratio * width)
        image = cv2.resize(image, (new_width, new_height))

        # calculate boundaries
        top, bottom = (new_size-new_height)//2, (new_size-new_height)//2
        if top + bottom + height < new_size:
            bottom += 1
        left, right = (new_size-new_width)//2, (new_size-new_width)//2
        if left + right + width < new_size:
            right += 1

        # padding
        if padding_method == "wrap":
            image = cv2.copyMakeBorder(
                image, top, bottom, left, right, cv2.BORDER_WRAP)
        elif padding_method == "constant":
            image = cv2.copyMakeBorder(
                image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    elif resize_method == "resize":
        image = cv2.resize(image, (new_size, new_size),
                           interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def resize_with_aspect_ratio(
    image: np.ndarray, width: int = None, height: int = None, inter: int = cv2.INTER_AREA
) -> np.ndarray:
    """Resize a image with aspect ratio.

    Args:
        image (np.ndarray): The image data.
        width (int, optional): The fixed width. Defaults to None.
        height (int, optional): The fixed height. Defaults to None.
        inter (int, optional): The resize interpolation. Defaults to cv2.INTER_AREA.

    Returns:
        np.ndarray: The resized image data.
    """
    dim = None
    h, w = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)
