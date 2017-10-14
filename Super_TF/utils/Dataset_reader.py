import tensorflow as tf

class Dataset_reader(object):
    """Interface Class for reading Dataset objects"""

    def single_read(self):{}

    def open_dataset(self, filename, epochs):
       filename_queue = tf.train.string_input_producer([filename])
       reader = tf.TFRecordReader()
       _, self.serialized_example = reader.read(filename_queue)

    def batch_inputs(self):{}

    def next_batch(self):{}