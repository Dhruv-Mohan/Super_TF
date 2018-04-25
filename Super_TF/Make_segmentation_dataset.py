
from Model_interface.Model import Model
import tensorflow as tf
from Dataset_IO.Segmentation.Dataset_reader_segmentation import Dataset_reader_segmentation
from Dataset_IO.Segmentation.Dataset_writer_segmentation import Dataset_writer_segmentation
#from tensorflow.examples.tutorials.mnist import input_data
import os
import cv2
import numpy as np


_IMAGE_WIDTH_       = 256*2
_IMAGE_HEIGHT_      = 256*2
_IMAGE_CSPACE_      = 3
_DATA_PATH_ = 'G:/Datasets/MD/complete/'

def main():
    dummy_writer = Dataset_writer_segmentation(Dataset_filename='test.tfrecords', image_shape = [900,900,3])
    dummy_writer.filename_constructor(_DATA_PATH_ + 'img/',  _DATA_PATH_ + 'mask/', '.png', '_ODsegSoftmap')
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    dummy_reader = Dataset_reader_segmentation('test.tfrecords')
    with tf.Session() as session:
        session.run(init_op)
        dummy_writer.write_record()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        crap = dummy_reader.next_batch(1)
        cv2.imshow('crap', crap[0][0])
        cv2.waitKey(0)




if __name__ == "__main__":
    main()
