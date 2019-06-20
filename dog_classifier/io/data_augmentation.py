import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def crop_range(img_size, bbox, rescale_size):
    '''Function to estimate the maximal x and y value for cropping.

    :param img_size: Size of the original scaled image
    :param bbox: Unscaled boundinx box of the image
    :param rescale_size: Rescale size on which the original img will be scaled
    :return trans_limits: Maximal values for linear translations
                          (left -> right), (bottom -> top) and vice versa. For
                          some directions may achieve higher. For some
                          translation directions higher translation values can
                          be achieved. 
    '''
    rescaled_bbox = resize_bbox(img_shape, bbox, rescale_size)

    diff_bbox_right_edge = min(rescale_size[1] - rescaled_bbox[::2])
    diff_bbox_left_edge = min(rescaled_bbox[::2])

    diff_bbox_top_edge = min(rescale_size[0] - rescaled_bbox[1::2])
    diff_bbox_lower_edge = min(rescaled_bbox[1::2])

    max_x_translation = diff_bbox_right_edge if diff_bbox_right_edge < diff_bbox_left_edge else diff_bbox_left_edge
    max_y_translation = diff_bbox_top_edge if diff_bbox_top_edge < diff_bbox_lower_edge else diff_bbox_lower_edge

    max_x_zoom = max_x_translation / rescale_size[1]  # in percent
    max_y_zoom = max_y_translation / rescale_size[0]  # in percent

    trans_limits = (max_x_translation, max_y_translation)
    zoom_limits = (max_x_zoom, max_y_zoom)

    return trans_limits, zoom_limits


def resize_bbox(img_size, bbox, rescale_size):
    ''' Function to transform the bounding box to fit the resized image
    '''
    bbox[::2] = bbox[::2] * rescaled_size[1] / img_size[1]
    bbox[1::2] = bbox[1::2] * rescaled_size[0] / img_size[0]
    return resized_bbox
