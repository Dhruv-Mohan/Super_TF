import tensorflow as tf

class Dataset_config_segmentation(object):
    """shared config for segmentation Dataset"""


    def __init__(self):
        self._Image_handle     = 'image_raw'
        self._Height_handle    = 'height'
        self._Width_handle     = 'width'
        self._Depth_handle     = 'depth'
        self._Image_name       = 'image_name'
        self._Image_mask       = 'image_mask'
        self._Mask_weights     = 'mask_weights'
        self._Feature_dict     =  {\
            self._Image_handle  : tf.FixedLenFeature([], tf.string),\
            self._Height_handle : tf.FixedLenFeature([], tf.int64),\
            self._Width_handle  : tf.FixedLenFeature([], tf.int64),\
            self._Depth_handle  : tf.FixedLenFeature([], tf.int64),\
            self._Image_name    : tf.FixedLenFeature([], tf.string),\
            self._Image_mask    : tf.FixedLenFeature([], tf.string),\
            self._Mask_weights  : tf.FixedLenFeature([], tf.string)}