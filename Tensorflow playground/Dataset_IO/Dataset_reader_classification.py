from utils.Dataset_reader import Dataset_reader
from Dataset_IO.Dataset_conifg_classification import Dataset_conifg_classification
import tensorflow as tf


class Dataset_reader_classification(Dataset_reader,Dataset_conifg_classification):
    """Implementation of Dataset reader for classification"""


    def __init__(self, filename=None, epochs=100, image_shape=[], num_classes=10):

        super().__init__()
        with tf.name_scope('Dataset_Classification_Reader') as scope:
            self.batch_size = tf.placeholder(tf.int32, name='Dataset_batch_size')
            self.image_shape =  image_shape
            self.num_classes = num_classes
            self.open_dataset(filename=filename, epochs=epochs)
            self.images , self.one_hot_labels = self.batch_inputs()


    def single_read(self):
        features = tf.parse_single_example(self.serialized_example, features=self._Feature_dict)
        image = tf.image.decode_image(features[self._Image_handle])
        image.set_shape(self.image_shape)
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
        with tf.name_scope('Batch_geter') as scope:
            if sess is None :
                self.sess = tf.get_default_session()
            else:
                self.sess = sess
            images , labels = self.sess.run([self.images , self.one_hot_labels], feed_dict={self.batch_size : batch_size})
            return images , labels 

