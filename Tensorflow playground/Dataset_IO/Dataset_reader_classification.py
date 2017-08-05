from utils.Dataset_reader import Dataset_reader
from Dataset_IO.Dataset_conifg_classification import Dataset_conifg_classification
import tensorflow as tf


class Dataset_reader_classification(Dataset_reader,Dataset_conifg_classification):
    """Implementation of Dataset reader for classification"""


    def __init__(self,filename=None,epochs=100,image_shape=[]):

        super().__init__()
        self.batch_size = tf.placeholder(tf.int32, name='Dataset_batch_size')
        self.image_shape =  image_shape
        self.open_dataset(filename=filename, epochs=epochs)
        self.images , self.sparse_labels = self.batch_inputs()



    def single_read(self):
        features = tf.parse_single_example(self.serialized_example, features=self._Feature_dict)
        image = tf.image.decode_image(features[self._Image_handle])
        shape = tf.stack([features[self._Height_handle], features[self._Width_handle], features[self._Depth_handle]])
        image.set_shape(self.image_shape)
        return image , features[self._Label_handle]


    def batch_inputs(self):
        image , label = self.single_read()
        images , sparse_labels = tf.train.shuffle_batch([image , label], batch_size=self.batch_size, num_threads=4, capacity=1000 + 30, min_after_dequeue=1000)
        return images, sparse_labels
        #TODO: CONFIGURABLE PARAMS


    def get_next_batch(self, batch_size=1, sess=None):
        if sess is None :
            self.sess = tf.get_default_session()
        else:
            self.sess = sess

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        images , labels = self.sess.run([self.images , self.sparse_labels], feed_dict={self.batch_size : batch_size})
        coord.request_stop()
        coord.join(threads)
        return images , labels 