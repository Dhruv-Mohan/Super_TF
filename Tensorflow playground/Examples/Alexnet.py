from Models.Model_class import Model_class
import tensorflow as tf
from Dataset_IO.Dataset_reader_classification import Dataset_reader_classification
_SUMMARY_           = True
_BATCH_SIZE_        = 48
_IMAGE_WIDTH_       = 299
_IMAGE_HEIGHT_      = 299
_IMAGE_CSPACE_      = 3
_CLASSES_           = 18
_MODEL_NAME_        = 'Alexnet'
_ITERATIONS_        = 500000
_LEARNING_RATE_     = 0.0001
_SAVE_DIR_          = 'E:/Ttest'
_TEST_INTERVAL_     = 1000
_RESTORE_           = True
_DATASET_PATH_      = 'E:\\Alexnet_train.tfrecords'


def writer_pre_proc(images):
    print('adding to graph')
    resized_images = tf.image.resize_images(images, size=[_IMAGE_HEIGHT_,_IMAGE_WIDTH_])
    rgb_image_float = tf.image.convert_image_dtype(resized_images, tf.float32) 
    reshaped_images = tf.reshape(rgb_image_float,[-1,_IMAGE_HEIGHT_*_IMAGE_WIDTH_*_IMAGE_CSPACE_])
    return reshaped_images



def main():
    dummy_reader = Dataset_reader_classification(filename=_DATASET_PATH_, image_shape=[_IMAGE_HEIGHT_, _IMAGE_WIDTH_, _IMAGE_CSPACE_], num_classes=_CLASSES_)
    dummy_reader.pre_process_image(writer_pre_proc)

    with tf.name_scope('AlexNet'):
        Alexnet = Model_class(Model_name=_MODEL_NAME_, Summary=_SUMMARY_, \
            Batch_size=_BATCH_SIZE_, Image_width=_IMAGE_WIDTH_, Image_height=_IMAGE_HEIGHT_, Image_cspace=_IMAGE_CSPACE_, Classes=_CLASSES_, Save_dir=_SAVE_DIR_)

        Alexnet.Construct_Model()


    #Set loss
        with tf.name_scope('Cross_entropy'):
            Alexnet.Set_loss(tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=Alexnet.model_dict['Output_ph'], logits=Alexnet.model_dict['Output'])))



    #Set optimizer
        with tf.name_scope('Train'):
            Alexnet.Set_optimizer(tf.train.AdamOptimizer(_LEARNING_RATE_))

        #Construct op to check accuracy
        Alexnet.Construct_Accuracy_op()
        Alexnet.Construct_Predict_op()

    #Setting test and train placeholders 
    with tf.name_scope('Control Params'):
        Simple_DNN.Set_test_control({'Dropout_prob_ph': 1, 'Train_state': False})
        Simple_DNN.Set_train_control({'Dropout_prob_ph': 0.8, 'Train_state': True})


    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    #Training block
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        Alexnet.Construct_Writers()
        Alexnet.Train_Iter(iterations=_ITERATIONS_, test_iterations=_TEST_INTERVAL_, data=dummy_reader, restore=_RESTORE_)
        Predictions = Alexnet.Predict(session,mnist.train.next_batch(10)[0])


if __name__ == "__main__":
    main()