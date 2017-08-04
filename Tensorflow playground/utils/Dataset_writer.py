import tensorflow as tf

class Dataset_writer(object):
    """description of class"""

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    def construct_writer(self,record_writer):
                self._Writer = tf.python_io.TFRecordWriter(record_writer)

    def filename_constructor(self, filename_path): {}


    def write_record(self):{}
