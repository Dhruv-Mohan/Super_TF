from utils.Dataset_writer import Dataset_writer
from Dataset_IO.Facelandmarks.Dataset_config_Facelandmarks import Dataset_config_Facelandmarks
import Dataset_IO.Facelandmarks.Dataset_Facelandmarks_pb2 as proto
import tensorflow as tf
import os
import random

class label_helper(object):

    def __init__(self, image_path=None, landmark_gt = None, landmark_init = None):
        self.image_path=image_path

        with open(landmark_gt) as lgt:
            self.landmark_gt = lgt.read()

        with open(landmark_init) as linit:
            self.landmark_init = linit.read()

        print("IMAGE_PATH")
        print(image_path)





class Dataset_writer_Facelandmarks(Dataset_config_Facelandmarks, Dataset_writer):
    """description of class"""

    def __init__(self, Dataset_filename, image_shape=[]):
        self.dataset_name, _ =  os.path.splitext(Dataset_filename)
        with tf.name_scope('Dataset_ImageSeqGen_Writer') as scope:
            self.image_shape = image_shape
            super().__init__()
            self.construct_writer(Dataset_filename)

            self.Param_dict = {\
                self._Height_handle : self._int64_feature(image_shape[0]),\
                self._Width_handle  : self._int64_feature(image_shape[1]),\
                self._Depth_handle  : self._int64_feature(image_shape[2]),\
                self._Landmarks_GT  : None,\
                self._Landmarks_init  : None,\
                self._Image_handle  : None,\
                self._Image_name    : None}

            self.mean_header_proto = proto.Image_set()
            self.mean_header_proto.Image_headers.image_width = image_shape[1]
            self.mean_header_proto.Image_headers.image_height = image_shape[0]
            self.mean_header_proto.Image_headers.image_depth = image_shape[2]
            self.dataset_mean = tf.Variable(tf.zeros(shape=[self.image_shape[0], self.image_shape[1], self.image_shape[2]]))
            self.data_container = []

    def prune_data(self, image):
        _, ext = os.path.splitext(image)
        if ext in ['.jpg', '.png']:
            return True
        else:
            return False

    def __filepath_constructor(self, image_path, text_path, init_lp, text_suffix, label_extension='.txt'):
        images = os.listdir(image_path)
        init_lmpts = init_lp
        
        for image in images:
            image_name = os.path.splitext(image)[0]
            text_name = text_path + image_name + text_suffix + label_extension
            self.data_container.append(label_helper(image_path=image_path+image, landmark_gt=text_name, landmark_init=init_lp))

    def filename_constructor(self, image_path=None, text_path=None, init_lp=None, text_suffix='', label_extension= '.txt'):
        self.__filepath_constructor( image_path, text_path, init_lp, text_suffix, label_extension)
        

    def write_record(self, sess=None):
        with tf.name_scope('Dataset_ImageSeqGen_Writer') as scope:
            if sess is None:
                self.sess = tf.get_default_session()
            else:
                self.sess = sess

            im_pth = tf.placeholder(tf.string)
            image_raw = tf.read_file(im_pth)
            image_pix = tf.image.convert_image_dtype(tf.image.decode_image(image_raw), tf.float32)
            image_pix = tf.image.resize_bicubic(image_pix, size=self.image_shape)
            total_images = len(self.data_container)
            mean_assign = tf.assign(self.dataset_mean, self.dataset_mean + image_pix/total_images)
            print('\t\t Constructing Database')
            self.mean_header_proto.Image_headers.image_count = total_images

            for index , image_container in enumerate(self.data_container):
                print(total_images)
                printProgressBar(index+1, total_images)
                im_rw = self.sess.run([image_raw, mean_assign],feed_dict={im_pth: image_container.image_path})
                self.Param_dict[self._Landmarks_GT] = self._bytes_feature(str.encode(image_container.landmark_gt))
                self.Param_dict[self._Landmarks_init] = self._bytes_feature(str.encode(image_container.landmark_init))
                self.Param_dict[self._Image_handle] = self._bytes_feature(im_rw[0])
                self.Param_dict[self._Image_name]   = self._bytes_feature(str.encode(image_container.image_path))
                example = tf.train.Example(features=tf.train.Features(feature=self.Param_dict))
                self._Writer.write(example.SerializeToString())
            #ENCODE MEAN AND STORE IT
            mean_pix = tf.reduce_mean(self.dataset_mean, 0)
            pix = self.sess.run(mean_pix)
            with open('mean.txt', w) as f:
                f.write(str(pix))

            self.dataset_mean = tf.image.convert_image_dtype(self.dataset_mean, tf.uint8)
            encoded_mean = tf.image.encode_png(self.dataset_mean)
            self.mean_header_proto.mean_data = encoded_mean.eval()


            with open(self.dataset_name+'_mean.proto','wb') as mean_proto_file:
                mean_proto_file.write(self.mean_header_proto.SerializeToString())


            self.sess.run([tf.write_file(self.dataset_name+'_mean.png', encoded_mean.eval())])

            self._Writer.close()


#From: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('       Progress: |%s| %s%% %s' % (bar, percent, suffix), end="\r")
    # Print New Line on Complete