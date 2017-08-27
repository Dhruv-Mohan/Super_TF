import tensorflow as tf


class Dataset_conifg_classification(object):
    """Shared config for classification Dataset """

    def __init__(self):
        self._Image_handle     = 'image_raw'
        self._Label_handle     = 'label'
        self._Height_handle    = 'height'
        self._Width_handle     = 'width'
        self._Depth_handle     = 'depth'
        self._Feature_dict     =  {\
            self._Label_handle  : tf.FixedLenFeature([], tf.int64),\
            self._Image_handle  : tf.FixedLenFeature([], tf.string),\
            self._Height_handle : tf.FixedLenFeature([], tf.int64),\
            self._Width_handle  : tf.FixedLenFeature([], tf.int64),\
            self._Depth_handle  : tf.FixedLenFeature([], tf.int64),\
            self._Image_name    : tf.FixedLenFeature([], tf.string)}