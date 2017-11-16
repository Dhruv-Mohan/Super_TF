import tensorflow as tf


class Dataset_conifg_ImageSeqGen(object):
    """Shared config for ImageSeqGen Dataset """

    def __init__(self):
        self._Image_handle     = 'image_raw'
        self._Seq_handle     = 'sequence'
        self._Seq_mask        = 'seq_mask'
        self._Height_handle    = 'height'
        self._Width_handle     = 'width'
        self._Depth_handle     = 'depth'
        self._Image_name       = 'image_name'
        self._Feature_dict     =  {\
            self._Seq_handle  : tf.FixedLenFeature([], tf.string),\
            self._Seq_mask  : tf.FixedLenFeature([], tf.string),\
            self._Image_handle  : tf.FixedLenFeature([], tf.string),\
            self._Height_handle : tf.FixedLenFeature([], tf.int64),\
            self._Width_handle  : tf.FixedLenFeature([], tf.int64),\
            self._Depth_handle  : tf.FixedLenFeature([], tf.int64),\
            self._Image_name    : tf.FixedLenFeature([], tf.string)}