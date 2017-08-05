from utils.Dataset_writer import Dataset_writer
from Dataset_IO.Dataset_conifg_classification import Dataset_conifg_classification
import tensorflow as tf
import os

class Dataset_writer_classification(Dataset_conifg_classification,Dataset_writer):
    """Implementation of Dataset writer for classification"""

    def __init__(self, Dataset_filename, image_shape=[]):
        super().__init__()
        self.construct_writer(Dataset_filename)
        self.Param_dict = {\
            self._Height_handle : self._int64_feature(image_shape[0]),\
            self._Width_handle  : self._int64_feature(image_shape[1]),\
            self._Depth_handle  : self._int64_feature(image_shape[2]),\
            self._Label_handle  : None,\
            self._Image_handle  : None}


    def filename_constructor(self, filename_path):
        self.image_list=[]
        with os.scandir(filename_path) as it:
            for entry in it:
                if entry.is_dir():
                    images=os.listdir(entry.path)
                    images_full_path = list(map(lambda x: entry.path +'/' + x, images))
                    self.image_list.append(images_full_path)

    def write_record(self, sess=None):
        if sess is None:
            self.sess = tf.get_default_session()
        else:
            self.sess = sess
        for lab in range(len(self.image_list)):
            for entry in self.image_list[lab]:
                image_raw = tf.read_file(entry).eval()
                self.Param_dict[self._Label_handle] = self._int64_feature(lab)
                self.Param_dict[self._Image_handle] = self._bytes_feature(image_raw)
                example = tf.train.Example(features=tf.train.Features(feature=self.Param_dict))
                self._Writer.write(example.SerializeToString())
        self._Writer.close()