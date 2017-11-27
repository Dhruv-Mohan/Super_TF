from utils.Dataset_reader import Dataset_reader
from Dataset_IO.Classification.Dataset_conifg_classification import Dataset_conifg_classification
import Dataset_IO.Classification.Dataset_classification_pb2 as proto
import tensorflow as tf
import os


#TODO: ADD TFRECORDS AND MEANPROTO READING CHECKS
class Dataset_reader_classification(Dataset_reader,Dataset_conifg_classification):
    """Implementation of Dataset reader for classification"""


    def __init__(self, filename=None, epochs=100, num_classes=18):

        super().__init__()
        with tf.name_scope('Dataset_Classification_Reader') as scope:
            self.batch_size = tf.placeholder(tf.int32, name='Dataset_batch_size')
            self.num_classes = num_classes
            self.open_dataset(filename=filename, epochs=epochs)
            self.mean_header_proto = proto.Image_set()
            dataset_path, dataset_name = os.path.split(filename)
            common_name, _ = os.path.splitext(dataset_name)
            mean_file_path = os.path.join(dataset_path,common_name +'_mean.proto')
            
            with open(mean_file_path,"rb") as mean_header_file:
                self.mean_header_proto.ParseFromString(mean_header_file.read())

            self.image_shape = [self.mean_header_proto.Image_headers.image_height, self.mean_header_proto.Image_headers.image_width, self.mean_header_proto.Image_headers.image_depth]
            mean_image_data = self.mean_header_proto.mean_data


            self.mean_image = tf.image.convert_image_dtype(tf.image.decode_image(mean_image_data), tf.float32)
            self.mean_image.set_shape(self.image_shape)
            self.images , self.one_hot_labels = self.batch_inputs()





    def single_read(self):
        features = tf.parse_single_example(self.serialized_example, features=self._Feature_dict)
        image = tf.image.decode_image(features[self._Image_handle])
        image.set_shape(self.image_shape)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = image - self.mean_image
        return image , features[self._Label_handle]


    def pre_process_image(self,pre_process_op):
        with tf.name_scope('Pre_Processing_op') as scope:
            self.images = pre_process_op(self.images)
        

    def batch_inputs(self):
        image , label = self.single_read()
        images , sparse_labels = tf.train.shuffle_batch([image , label], batch_size=self.batch_size, num_threads=8, capacity=5000+128, min_after_dequeue=5000)
        one_hot_labels = tf.one_hot(sparse_labels,self.num_classes)
        return images, one_hot_labels
        #TODO: CONFIGURABLE PARAMS


    def next_batch(self, batch_size=1, sess=None):
        with tf.name_scope('Batch_getter') as scope:
            if sess is None :
                self.sess = tf.get_default_session()
            else:
                self.sess = sess
            images , labels = self.sess.run([self.images , self.one_hot_labels], feed_dict={self.batch_size : batch_size})
            return images , labels 

