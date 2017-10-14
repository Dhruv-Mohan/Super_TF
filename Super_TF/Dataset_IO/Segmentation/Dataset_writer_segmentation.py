from utils.Dataset_writer import Dataset_writer
from Dataset_IO.Segmentation.Dataset_config_segmentation import Dataset_config_segmentation
import Dataset_IO.Segmentation.Dataset_segmentation_pb2 as proto
import tensorflow as tf
import os
import random
import cv2
import numpy as np

class label_helper(object):

    def __init__(self, image_path=None, mask_path=None):
        self.image_path = image_path
        self.mask_path = mask_path
        _, self.image_name=os.path.split(image_path)
        _, self.mask_name = os.path.split(mask_path)

class Dataset_writer_segmentation(Dataset_config_segmentation, Dataset_writer):
    """Implementation of Dataset writer for segmentation"""

    def __init__(self, Dataset_filename, image_shape=[]):
        self.dataset_name, _ =  os.path.splitext(Dataset_filename)
        self.weight=None
        with tf.name_scope('Dataset_Segmentation_Writer') as scope:
            self.image_shape = image_shape
            super().__init__()
            self.construct_writer(Dataset_filename)
            self.Param_dict = {\
                self._Height_handle : self._int64_feature(image_shape[0]),\
                self._Width_handle  : self._int64_feature(image_shape[1]),\
                self._Depth_handle  : self._int64_feature(image_shape[2]),\
                self._Image_handle  : None,\
                self._Image_name    : None,\
                self._Image_mask    : None,\
                self._Mask_weights  : None}

            self.mean_header_proto = proto.Image_set()
            self.mean_header_proto.Image_headers.image_width = image_shape[1]
            self.mean_header_proto.Image_headers.image_height = image_shape[0]
            self.mean_header_proto.Image_headers.image_depth = image_shape[2]

    def prune_data(self, image):
        _, ext = os.path.splitext(image)
        if ext in ['.jpg', '.png', '.gif']:
            return True
        else:
            return False

    def __shuffle_input(self, image_full_path, mask_full_path):
        self.shuffled_images = []
        for image,mask in zip(image_full_path,mask_full_path):
            self.shuffled_images.extend([label_helper(image_path=image, mask_path=mask)])

        random.shuffle(self.shuffled_images)

    def filename_constructor(self, image_path=None, mask_path=None, mask_extension='.gif', mask_suffix='_mask'):

        images = os.listdir(image_path)
        images[:] = [x for x in images if self.prune_data(x)]
        image_full_path = list(map(lambda x: image_path +'/' + x, images))
        names = list(map(lambda x: os.path.splitext(x)[0], images))
        mask_full_path = list(map(lambda x: mask_path +'/' + x + mask_suffix +mask_extension, names))
        self.__shuffle_input(image_full_path, mask_full_path)


    def getweight(self, mask_mat=None):
        gray_mask = cv2.cvtColor(mask_mat, cv2.COLOR_BGR2GRAY)

        ret, bin_mask = cv2.threshold(gray_mask,1,1,0)
        _, contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        weights = np.zeros_like(bin_mask, dtype=np.float)

        weights = cv2.drawContours(weights, contours, -1, (1), 5)
        weights = cv2.GaussianBlur(weights, (41,41), 1000)
        weights = np.multiply(weights,10)+0.6
        return weights


    def write_record(self, sess=None):
        with tf.name_scope('Dataset_Segmentation_Writer') as scope:
            if sess is None:
                self.sess = tf.get_default_session()
            else:
                self.sess = sess


            mk_pth = tf.placeholder(tf.string)
            mask_raw = tf.read_file(mk_pth)
            mask_pix = tf.image.convert_image_dtype(tf.image.decode_image(mask_raw), tf.uint8)

            im_pth = tf.placeholder(tf.string)
            image_raw = tf.read_file(im_pth)


            image_to_encode = tf.placeholder(tf.uint8, shape=[self.image_shape[1], self.image_shape[0], 1]) #subject to change
            encode_image = tf.image.encode_png(tf.image.convert_image_dtype(image_to_encode,tf.uint8))
            total_images = len(self.shuffled_images)

            print('\t\t Constructing Database')
            self.mean_header_proto.Image_headers.image_count = total_images

            for index , image_container in enumerate(self.shuffled_images):
                printProgressBar(index+1, total_images)
                im_rw = self.sess.run([image_raw],feed_dict={im_pth: image_container.image_path})
                [mask , mask_mat] = self.sess.run([mask_raw, mask_pix], feed_dict={mk_pth: image_container.mask_path})
                mask_weightraw = self.getweight(mask_mat[0])
                mask_weightraw =  mask_weightraw[:,:,np.newaxis]
                weight_raw = self.sess.run([encode_image], feed_dict={image_to_encode : mask_weightraw})
                self.Param_dict[self._Image_handle] = self._bytes_feature(im_rw[0])
                self.Param_dict[self._Image_mask] = self._bytes_feature(mask)
                self.Param_dict[self._Mask_weights] = self._bytes_feature(weight_raw[0])
                self.Param_dict[self._Image_name]   = self._bytes_feature(str.encode(image_container.image_name))
                example = tf.train.Example(features=tf.train.Features(feature=self.Param_dict))
                self._Writer.write(example.SerializeToString())
                #ADD TO MEAN IMAGE


            with open(self.dataset_name+'_header.proto','wb') as mean_proto_file:
                mean_proto_file.write(self.mean_header_proto.SerializeToString())

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
