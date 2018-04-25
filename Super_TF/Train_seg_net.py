from Model_interface.Model import Model
import tensorflow as tf
from Dataset_IO.Segmentation.Dataset_reader_segmentation import Dataset_reader_segmentation
import os
import cv2
import numpy as np

_SUMMARY_           = True
_BATCH_SIZE_        = 5
_IMAGE_WIDTH_       = 512
_IMAGE_HEIGHT_      = 512
_IMAGE_CSPACE_      = 3
_CLASSES_           = 1
_MODEL_NAME_        = 'FRRN_C'
_ITERATIONS_        = 50000
_LEARNING_RATE_     = 0.0001
_SAVE_DIR_          = 'G:\\TFmodels\\odFrrnc'
_SAVE_INTERVAL_     = 5000
_RESTORE_           =  False
_TEST_              =   False
#_DATASET_      = 'smallcompleted.tfrecords'
_DATASET_      = 'test.tfrecords'
#_TEST_PATH_ = 'G:/all-images (1)/Stare/'
_TEST_PATH_ = 'G:\\Datasets\\IDRiD\\Testing a and b/'
_OUTPUT_PATH_ = 'G:/idridout/'
_DROPOUT_ = 0.6
_STATE_         = 'Train'
_SAVE_ITER_     = 10000
_GRAD_NORM_     = 0.5
_RENORM_        = False


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

def writer_pre_proc_mask(images):
    print('Adding mask Preproc')
    resized_images = tf.image.resize_images(images, size=[_IMAGE_WIDTH_,_IMAGE_HEIGHT_])
    #reshaped_images = tf.cast(resized_images,tf.uint8)
    #rgb_image_float = tf.image.convert_image_dtype(resized_images, tf.float32) 
    gray_images = tf.image.rgb_to_grayscale(resized_images)
    #reshaped_images = tf.reshape(gray_images, [-1,_IMAGE_HEIGHT_*_IMAGE_WIDTH_])
    return gray_images

def writer_pre_proc_weight(images):
    print('Adding weight Preproc')
    resized_images = tf.image.resize_images(images, size=[_IMAGE_WIDTH_,_IMAGE_HEIGHT_])
    #reshaped_images = tf.cast(resized_images,tf.uint8)
    #rgb_image_float = tf.image.convert_image_dtype(resized_images, tf.float32) 
    #gray_images = (tf.image.rgb_to_grayscale(resized_images)+2)* 5
    #gray_images = gray_images/ tf.reduce_mean(gray_images)
    #reshaped_images = tf.reshape(gray_images, [-1,_IMAGE_HEIGHT_*_IMAGE_WIDTH_])
    return resized_images

def pre_process_image(input_image_path):
    image = cv2.imread(input_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (_IMAGE_HEIGHT_, _IMAGE_WIDTH_), 0)
    image = image.astype(np.float32)
    image = image/255
    cv2.imshow('Inputimage', image)
    image = np.expand_dims(image, axis=0)
    return image

def make_cent(image, cent):
    if image.shape[-1] is not 3:
        image = np.concatenate((image,image,image), axis=-1)
    cent_im = cv2.circle(image, (int(cent[0]*32), int(cent[1] *32)), 5, (255,0,255), -1)
    return cent_im


def main():
    dummy_reader = Dataset_reader_segmentation(_DATASET_)
    dummy_reader.pre_process_image(writer_pre_proc)
    dummy_reader.pre_process_mask(writer_pre_proc_mask)
    
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

    #Construct model
    with tf.name_scope('FRRN_C'):
        Simple_DNN = Model(Model_name=_MODEL_NAME_, Summary=_SUMMARY_, \
                Batch_size=_BATCH_SIZE_, Image_width=_IMAGE_WIDTH_, Image_height=_IMAGE_HEIGHT_,\
               Image_cspace=_IMAGE_CSPACE_, Classes=_CLASSES_, Save_dir=_SAVE_DIR_, \
               State=_STATE_, Dropout=_DROPOUT_, Grad_norm=_GRAD_NORM_, Renorm = True)


        if not _TEST_:
            with tf.name_scope('Train'):
                Optimizer_params = {'decay': 0.9, 'momentum':0.01, 'epsilon':1e-10, 'Nestrov':False}

                Optimizer_params_adam = {'beta1': 0.9, 'beta2':0.999, 'epsilon':0.1}
                Simple_DNN.Set_optimizer(starter_learning_rate= _LEARNING_RATE_, Optimizer='ADAM', Optimizer_params=Optimizer_params_adam, decay_steps=5000, decay_rate=0.8)

            #Construct op to check accuracy
            Simple_DNN.Construct_Accuracy_op()
        else:
            Simple_DNN.Place_arch()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print('Constructing Writers')
        Simple_DNN.Construct_Writers()
        print('Writers Constructed')
        
        if not _TEST_:
            Simple_DNN.Train_Iter(iterations=_ITERATIONS_, save_iterations=_SAVE_INTERVAL_, data=dummy_reader, restore=_RESTORE_, log_iteration=5)
        else:
            Simple_DNN.Try_restore()
            images = os.listdir(_TEST_PATH_)
            for input_image_path in images:
                image = Simple_DNN.Predict(Input_Im = pre_process_image(_TEST_PATH_ + input_image_path), session=session)
                cv2.imshow('predict image', image[0][0])
                cv2.waitKey(1)
                print(image[1], image[2], image[3], image[4])
                outim = cv2.resize(cv2.imread(_TEST_PATH_ + input_image_path), (256,256), 0)
                cent_image = make_cent(outim, [ image[1][0]+ image[3][0] +0.5,  image[2][0]+ image[4][0] +0.5])
                image_crap = os.path.splitext(input_image_path)[0]
                cv2.imwrite(_OUTPUT_PATH_ + image_crap +'.png', cent_image)
                cv2.imshow('cent_image', cent_image)
                cv2.waitKey(1)

if __name__ == "__main__":
    main()
