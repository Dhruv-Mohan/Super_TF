import tensorflow as tf
import os
import cv2
import numpy as np
from Dataset_IO.Facelandmarks.Dataset_writer_Facelandmarks import Dataset_writer_Facelandmarks
from Dataset_IO.Facelandmarks.Dataset_reader_Facelandmarks import Dataset_reader_Facelandmarks

'''
_IMAGE_PATH_ = 'G:\\Datasets\\Face_landmark\\images/'
_TXT_PATH_ = 'G:\\Datasets\\Face_landmark\\norrmlab/'
_MEAN_TXT_PATH_ = 'G:\\Datasets\\Face_landmark/mean.txt'
'''
_IMAGE_PATH_ = 'E:/Datasetdir/custom/images/'
_TXT_PATH_ = 'E:/Datasetdir/custom/labs/'
_MEAN_TXT_PATH_ = 'G:\\Datasets\\Face_landmark/mean.txt'

_IMAGE_WIDTH_ = 512
_IMAGE_HEIGHT_ = 512
_IMAGE_CSPACE = 3

def main():

    dummy_writer = Dataset_writer_Facelandmarks(Dataset_filename='facetest2.tfrecords', image_shape=[512,512,3])
    dummy_writer.filename_constructor(image_path=_IMAGE_PATH_, text_path=_TXT_PATH_, init_lp=_MEAN_TXT_PATH_,
                                     text_suffix='', label_extension='.pts')

    #dummy_reader = Dataset_reader_Facelandmarks('facetest.tfrecords')
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

    with tf.Session() as sess:
        init_op.run()
        '''
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        crap = dummy_reader.next_batch(1)
        cra = crap[0][0]
        print(cra.shape)
        print(crap[1][0])
        print(crap[1][0].shape)
        cv2.imshow('cra', cra)
        cv2.waitKey(0)
        print(cra)
        '''
        dummy_writer.write_record()


if __name__ == "__main__":
    main()
