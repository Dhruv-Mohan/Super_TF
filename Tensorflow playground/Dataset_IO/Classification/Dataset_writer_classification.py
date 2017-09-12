from utils.Dataset_writer import Dataset_writer
from Dataset_IO.Dataset_conifg_classification import Dataset_conifg_classification
import Dataset_IO.Dataset_classification_pb2 as proto
import tensorflow as tf
import os
import random



class label_helper(object):

    def __init__(self, image_path=None, image_data=None):
        self.image_path=image_path
        self.image_data=image_data
        _, self.image_name=os.path.split(image_path)


class Dataset_writer_classification(Dataset_conifg_classification,Dataset_writer):
    """Implementation of Dataset writer for classification"""

    def __init__(self, Dataset_filename, image_shape=[]):
        self.dataset_name, _ =  os.path.splitext(Dataset_filename)
        with tf.name_scope('Dataset_Classification_Writer') as scope:
            self.image_shape = image_shape
            super().__init__()
            self.construct_writer(Dataset_filename)
            self.Param_dict = {\
                self._Height_handle : self._int64_feature(image_shape[0]),\
                self._Width_handle  : self._int64_feature(image_shape[1]),\
                self._Depth_handle  : self._int64_feature(image_shape[2]),\
                self._Label_handle  : None,\
                self._Image_handle  : None,\
                self._Image_name    : None}
            self.mean_header_proto = proto.Image_set()
            self.mean_header_proto.Image_headers.image_width = image_shape[1]
            self.mean_header_proto.Image_headers.image_height = image_shape[0]
            self.mean_header_proto.Image_headers.image_depth = image_shape[2]
            self.dataset_mean = tf.Variable(tf.zeros(shape=[self.image_shape[0], self.image_shape[1], self.image_shape[2]]))
    def prune_data(self, image):
        _, ext = os.path.splitext(image)
        if ext in ['.jpg', '.png']:
            return True
        else:
            return False

    def __shuffle_input(self, image_list):
        self.shuffled_images = []
        for im_lab, im_paths in enumerate(image_list):
            for im_path in im_paths:
                self.shuffled_images.extend([label_helper(image_path=im_path, image_data=im_lab)])
        random.shuffle(self.shuffled_images)


    def __filepath_constructor(self, filename_path):
        image_list = []
        with os.scandir(filename_path) as it:
            for entry in it:
                if entry.is_dir():
                    images  = os.listdir(entry.path)
                    images[:] = [x for x in images if self.prune_data(x)]
                    images_full_path = list(map(lambda x: entry.path +'/' + x, images))
                    image_list.append(images_full_path)
        self.__shuffle_input(image_list)


    def __file_dict_constructor(self, file_dict):
        image_list = []
        for key in sorted(file_dict):
            image_list.append(file_dict[key])
        self.__shuffle_input(image_list)


    def filename_constructor(self, filename_path = None, file_dict= None):
        if filename_path is not None:
            self.__filepath_constructor(filename_path)
        elif file_dict is not None:
            self.__file_dict_constructor(file_dict)





    def write_record(self, sess=None):
        with tf.name_scope('Dataset_Classification_Writer') as scope:
            if sess is None:
                self.sess = tf.get_default_session()
            else:
                self.sess = sess

            im_pth = tf.placeholder(tf.string)
            image_raw = tf.read_file(im_pth)
            image_pix = tf.image.convert_image_dtype(tf.image.decode_image(image_raw), tf.float32)
            total_images = len(self.shuffled_images)
            mean_assign = tf.assign(self.dataset_mean, self.dataset_mean + image_pix/total_images)
            print('\t\t Constructing Database')
            self.mean_header_proto.Image_headers.image_count = total_images

            for index , image_container in enumerate(self.shuffled_images):
                printProgressBar(index+1, total_images)
                im_rw = self.sess.run([image_raw, mean_assign],feed_dict={im_pth: image_container.image_path})
                self.Param_dict[self._Label_handle] = self._int64_feature(image_container.image_data)
                self.Param_dict[self._Image_handle] = self._bytes_feature(im_rw[0])
                self.Param_dict[self._Image_name]   = self._bytes_feature(str.encode(image_container.image_name))
                example = tf.train.Example(features=tf.train.Features(feature=self.Param_dict))
                self._Writer.write(example.SerializeToString())
                #ADD TO MEAN IMAGE

            #ENCODE MEAN AND STORE IT
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
