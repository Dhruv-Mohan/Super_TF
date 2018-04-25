import sys
from Model_interface.Model import Model
import tensorflow as tf
from Dataset_IO.Classification.Dataset_reader_classification import Dataset_reader_classification
# from tensorflow.examples.tutorials.mnist import input_data
import os
import cv2
import numpy as np
import csv

annotation_paths = 'experts_anotation/'
masks_path = 'masks/'

_SUMMARY_ = True
_BATCH_SIZE_ = 1
_IMAGE_WIDTH_ = 128
_IMAGE_HEIGHT_ = 128
_IMAGE_CSPACE_ = 3
_CLASSES_ = 2
_MODEL_NAME_ = 'Stargan'
_ITERATIONS_ = 100000
_LEARNING_RATE_ = 0.0001  # 0.05 0.08
_SAVE_DIR_ = 'G:\\Dhruv\\Projects\\Models\\Gan'
_SAVE_INTERVAL_ = 2500
_RESTORE_ = False
_TEST_ = False
_DATASET_ = 'Gan.tfrecords'
_TEST_PATH_ = 'G:/Datasets/MD/Test1/M/'
_DROPOUT_ = 0.7
_STATE_ = 'Train'
_SAVE_ITER_ = 2000
_GRAD_NORM_ = 0.5
_RENORM_ = False
path = 'G:/Tftestdata'
index = 0
Test_GT = 'G:/Datasets/MD/Test1/mask/'

_Densereg_output_ = 'G:/Datasets/Correct_Dreg/'


def writer_pre_proc_seg_test(images):
    print('Adding image Preproc')
    resized_images = tf.image.resize_images(images, size=[_IMAGE_WIDTH_, _IMAGE_HEIGHT_])
    #resized_images = tf.image.per_image_standardization(resized_images)
    # reshaped_images = tf.cast(resized_images,tf.uint8)
    # rgb_image_float = tf.image.convert_image_dtype(resized_images, tf.float32)
    reshaped_images = tf.reshape(resized_images, [-1, _IMAGE_HEIGHT_ * _IMAGE_WIDTH_ * _IMAGE_CSPACE_])
    return reshaped_images


def writer_pre_proc(images):
    print('Adding image Preproc')
    resized_images = tf.image.resize_images(images, size=[_IMAGE_WIDTH_, _IMAGE_HEIGHT_])
    # reshaped_images = tf.cast(resized_images,tf.uint8)
    # rgb_image_float = tf.image.convert_image_dtype(resized_images, tf.float32)
    # reshaped_images = tf.reshape(resized_images, [-1,_IMAGE_HEIGHT_*_IMAGE_WIDTH_*_IMAGE_CSPACE_])
    return resized_images


def writer_pre_proc_mask(images):
    print('Adding mask Preproc')
    resized_images = tf.image.resize_images(images, size=[_IMAGE_WIDTH_, _IMAGE_HEIGHT_])
    # reshaped_images = tf.cast(resized_images,tf.uint8)
    # rgb_image_float = tf.image.convert_image_dtype(resized_images, tf.float32)
    gray_images = tf.image.rgb_to_grayscale(resized_images)
    # reshaped_images = tf.reshape(gray_images, [-1,_IMAGE_HEIGHT_*_IMAGE_WIDTH_])
    return gray_images


def writer_pre_proc_weight(images):
    print('Adding weight Preproc')
    resized_images = tf.image.resize_images(images, size=[_IMAGE_WIDTH_, _IMAGE_HEIGHT_])
    # reshaped_images = tf.cast(resized_images,tf.uint8)
    # rgb_image_float = tf.image.convert_image_dtype(resized_images, tf.float32)
    # gray_images = (tf.image.rgb_to_grayscale(resized_images)+2)* 5
    # gray_images = gray_images/ tf.reduce_mean(gray_images)
    # reshaped_images = tf.reshape(gray_images, [-1,_IMAGE_HEIGHT_*_IMAGE_WIDTH_])
    return resized_images


def main():
    with tf.device('/cpu:0'):
        dummy_reader = Dataset_reader_classification(filename=_DATASET_, num_classes=_CLASSES_)
        dummy_reader.pre_process_image(writer_pre_proc)
    # dummy_reader.pre_process_mask(writer_pre_proc_mask)
    # dummy_reader.pre_process_weights(writer_pre_proc_weight)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # Construct model

    Simple_DNN = Model(Model_name=_MODEL_NAME_, Summary=_SUMMARY_, \
                           Batch_size=_BATCH_SIZE_, Image_width=_IMAGE_WIDTH_, Image_height=_IMAGE_HEIGHT_, \
                           Image_cspace=_IMAGE_CSPACE_, Classes=_CLASSES_, Save_dir=_SAVE_DIR_, \
                           State=_STATE_, Dropout=_DROPOUT_, Grad_norm=_GRAD_NORM_, Renorm=_RENORM_,Gen_Dropout=0.5, Dis_Dropout=0.5)

    # Set optimizer
    if not _TEST_:
            Optimizer_params = {'decay': 0.9, 'momentum': 0.01, 'epsilon': 1e-10}

            Optimizer_params_adam = {'beta1': 0.9, 'beta2': 0.99, 'epsilon': 0.1}
            Simple_DNN.Set_optimizer(starter_learning_rate=_LEARNING_RATE_, Optimizer='ADAM',
                                         Optimizer_params=Optimizer_params_adam, decay_steps=5000)

    # Construct op to check accuracy
    Simple_DNN.Construct_Accuracy_op()

    # Training block
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if _TEST_:
            test_path = _TEST_PATH_
            test_images = os.listdir(test_path)
            test_image_path = tf.placeholder(tf.string)
            test_image = tf.image.decode_image(
                tf.read_file(test_image_path))  # So, i've read a test image, i need to pre-process it now
            test_image.set_shape([900, 900, 3])
            # test_image = tf.expand_dims(test_image, 0)
            test_image = writer_pre_proc_seg_test(test_image)
    # lab=['2w','3wc', '3wp', 'bus', 'car', 'jeep-suv', 'lcvmbus', 'lcvmtruck', 'mav3', 'mav4', 'mav5', 'mav6', 'mlcv', 'osv', 'tow' , 'tractor', 'truck', 'van' ]
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        '''
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        while 1:
            batch = dummy_reader.next_batch(1)[0]
            cv2.imshow('image', batch[0])
            cv2.waitKey(0)
        '''
        print('Global Vars initialized')
        if _TEST_:

            Simple_DNN.Construct_Writers()
            Simple_DNN.Try_restore()
            total_images = len(test_images)
            with open('frrucsubmission3.csv', 'w') as submitfile:
                submitfile.write('img,rle_mask\n')
                for index, imag in enumerate(test_images):
                    printProgressBar(index + 1, total_images)
                    test_imag = session.run([test_image], feed_dict={test_image_path: os.path.join(test_path, imag)})[0]
                    gt_name, _ = os.path.splitext(imag)
                    gt_name = gt_name + '_ODsegSoftmap.png'
                    print(imag)
                    print(gt_name)
                    # test_imag = session.run([test_image], feed_dict={test_image_path:'G:/Datasets/MD/Test1/1i.png'})[0]
                    temp_mag = np.reshape(test_imag, (_IMAGE_HEIGHT_, _IMAGE_WIDTH_, 3))
                    cv2.imshow('Test_imag', temp_mag)
                    cv2.waitKey(1)
                    # test_image = np.squeeze(test_image, axis=0)
                    prediction = Simple_DNN.Predict(test_imag, 'G:/Pri/' + gt_name)
                    seg_map = construct_segmap(np.squeeze(prediction), gt_name)
                # submitfile.write(imag+','+rle_to_string(rle_encode(seg_map))+'\n')
                print("DCs:", Dice_coeffs)
                print("Mean Dice coeff:", np.mean(Dice_coeffs))
                print("Mean Jacard coeff:", np.mean(Jacard_coeff))
            # print(seg_map)
            '''
                for ind,image in enumerate(images):
                    #print('image number',ind)
                    with open('output.txt','a+') as fileout:
                        filepath = os.path.join(path,image)
                        #test_image = cv2.imread(filepath,0)
                        #test_image = np.expand_dims(test_image, axis=2)
                        #print(test_image.dtype)
                        #print(test_image.shape)
                        final_image = session.run([ipp], feed_dict={ti : filepath})
                        final_image = np.asarray(final_image)
                        final_image = np.squeeze(final_image,axis=0)
                        #cv2.imshow('f',final_image)
                        #cv2.waitKey(0)
                        prediction = Simple_DNN.Predict(final_image)[0][0]
                        #print(prediction)
                        print('Prediction:',image,'Class:',lab[prediction])
                        fileout.write(image+','+ lab[prediction] +'\n')
            '''
        if not _TEST_:
            print('Constructing Writers')
            Simple_DNN.Construct_Writers()
            print('Writers Constructed')
            Simple_DNN.Train_Iter(iterations=_ITERATIONS_, save_iterations=_SAVE_INTERVAL_, data=dummy_reader,
                                  restore=_RESTORE_, log_iteration=5)
    # Predictions = Simple_DNN.Predict(session,mnist.train.next_batch(10)[0])


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
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

if __name__ == "__main__":
    main()