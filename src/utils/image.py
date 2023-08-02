#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np


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
