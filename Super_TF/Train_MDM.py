import sys
from Model_interface.Model import Model
import tensorflow as tf
from Dataset_IO.Facelandmarks.Dataset_reader_Facelandmarks import Dataset_reader_Facelandmarks
import cv2
import numpy as np
import os

_SUMMARY_           = True
_BATCH_SIZE_        = 5
_IMAGE_WIDTH_       = 256*2
_IMAGE_HEIGHT_      = 256*2
_IMAGE_CSPACE_      = 3
_CLASSES_           = 1
_MODEL_NAME_        = 'MDM'
_ITERATIONS_        = 500000
_LEARNING_RATE_     = 0.005
_SAVE_DIR_          = 'G:\\TFmodels\\MDM'
_SAVE_INTERVAL_     = 5000
_RESTORE_           =  True
_TEST_              =   False
_RENORM_        = False
_DATASET_      = 'facetest.tfrecords'
_STATE_ = 'Train'
_DROPOUT_ = 0.6
_PATCHES_ = 106
_DATA_INPUT_FOLDER_ = 'G:\\Datasets\\Face_landmark\\Test'
_DATA_OUTPUT_FOLDER_ = 'G:\\Datasets\\Face_landmark\\Output/'

def writer_pre_proc(images):
    print('Adding image Preproc')
    resized_images = tf.image.resize_images(images, size=[_IMAGE_WIDTH_,_IMAGE_HEIGHT_])
    #resized_images = tf.image.random_brightness(resized_images, 0.2)
    #resized_images = tf.image.random_contrast(resized_images, 0.9,1.1)
    #resized_images = tf.image.random_saturation(resized_images,  0.9, 1.1)

    #reshaped_images = tf.cast(resized_images,tf.uint8)
    #rgb_image_float = tf.image.convert_image_dtype(resized_images, tf.float32) 
    #reshaped_images = tf.reshape(resized_images, [-1,_IMAGE_HEIGHT_*_IMAGE_WIDTH_*_IMAGE_CSPACE_])
    return resized_images

def writer_pre_proc_test(image, mean_image):
    print('Test_PP')
    recast_image = tf.image.convert_image_dtype(image,tf.float32)
    #normalized_image =tf.image.per_image_standardization(recast_image)
    normalized_image = recast_image - tf.reduce_mean(mean_image)
    resized_images = tf.image.resize_images(normalized_image, size=[_IMAGE_WIDTH_,_IMAGE_HEIGHT_])
    resized_images =tf.image.per_image_standardization(resized_images)
    #return recast_image - mean_image
    return resized_images, tf.image.resize_images(recast_image, size=[_IMAGE_WIDTH_,_IMAGE_HEIGHT_])

def main():
    dummy_reader = Dataset_reader_Facelandmarks(_DATASET_)
    dummy_reader.pre_process_image(writer_pre_proc)
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

    with tf.name_scope('MDM'):
        Simple_DNN = Model(Model_name=_MODEL_NAME_, Summary=_SUMMARY_, \
                Batch_size=_BATCH_SIZE_, Image_width=_IMAGE_WIDTH_, Image_height=_IMAGE_HEIGHT_,\
               Image_cspace=_IMAGE_CSPACE_, Classes=_CLASSES_, Save_dir=_SAVE_DIR_, \
               State=_STATE_, Dropout=_DROPOUT_, Renorm = True, Patches=_PATCHES_)

        if not _TEST_:
            with tf.name_scope('Train'):
                Optimizer_params = {'decay': 0.9, 'momentum':0.01, 'epsilon':1e-10, 'Nestrov':False}

                Optimizer_params_adam = {'beta1': 0.9, 'beta2':0.99, 'epsilon':0.1}
                Simple_DNN.Set_optimizer(starter_learning_rate= _LEARNING_RATE_, Optimizer='ADAM', decay_steps=200000, decay_rate=0.9)
        else:
            Simple_DNN.Place_arch()
            ti = tf.placeholder(tf.string)
            imf = tf.image.decode_image(tf.read_file(ti))
            imf.set_shape([512,512,3])
            ipp, itt = writer_pre_proc_test(imf, dummy_reader.mean_image); #got preproc function

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print('Constructing Writers')
        Simple_DNN.Construct_Writers()
        print('Writers Constructed')
        if not _TEST_:
            Simple_DNN.Train_Iter(iterations=_ITERATIONS_, save_iterations=_SAVE_INTERVAL_, data=dummy_reader, restore=_RESTORE_, log_iteration=5)
        else:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            Simple_DNN.Try_restore(session)
            mean_pts = dummy_reader.next_batch(1)[2]
            images = os.listdir(_DATA_INPUT_FOLDER_)
            for image in images:
                #image_path = input('enter image path')
                image_path = _DATA_INPUT_FOLDER_ +'/' + image
                image_togo, image_topaint = session.run([ipp, itt], feed_dict={ti:image_path})
                image_topaint = cv2.cvtColor(image_topaint, cv2.COLOR_BGR2RGB)
                output = Simple_DNN.Predict(Input_Im = np.expand_dims(image_togo, axis=0), Input_seq = mean_pts, session=session)[0][0]
                image_topaint *= 255
                for point in output:
                    #point += 0.5
                    point = [int(point[0]), int(point[1])]
                    cv2.circle(image_topaint, (point[0], point[1]), 2, (255,0,0), 2)
                cv2.imshow('testim', image_topaint/255)
                cv2.waitKey(1)
                cv2.imwrite(_DATA_OUTPUT_FOLDER_ + image, image_topaint)

             

if __name__ == "__main__":
    main()