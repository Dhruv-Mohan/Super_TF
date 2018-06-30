import tensorflow as tf

class Dataset_config_Facelandmarks(object):
    """description of class"""

    def __init__(self):
        self._Image_handle     = 'image_raw'
        self._Landmarks_GT     = 'landmarkgt'
        self._Landmarks_init        = 'landmarkinit'
        self._Height_handle    = 'height'
        self._Width_handle     = 'width'
        self._Depth_handle     = 'depth'
        self._Image_name       = 'image_name'
        self._Feature_dict     =  {\
            self._Landmarks_init  : tf.FixedLenFeature([], tf.string),\
            self._Landmarks_GT  : tf.FixedLenFeature([], tf.string),\
            self._Image_handle  : tf.FixedLenFeature([], tf.string),\
            self._Height_handle : tf.FixedLenFeature([], tf.int64),\
            self._Width_handle  : tf.FixedLenFeature([], tf.int64),\
            self._Depth_handle  : tf.FixedLenFeature([], tf.int64),\
            self._Image_name    : tf.FixedLenFeature([], tf.string)}
