
from Model_interface.Model import Model
import tensorflow as tf
from Dataset_IO.Classification.Dataset_reader_classification import Dataset_reader_classification

_SUMMARY_           =True
_BATCH_SIZE_        = 5
_IMAGE_WIDTH_       = 224
_IMAGE_HEIGHT_      = 224
_IMAGE_CSPACE_      = 1
_CLASSES_           = 10
_MODEL_NAME_        = 'Alexnet'
_ITERATIONS_        = 2000
_TEST_INTERVAL_     = 10
_LEARNING_RATE_     = 1e-4
_SAVE_DIR_          = 'Save dir path'
_RESTORE_           = False
_DATASET_PATH_      = 'Dataset path'
_DROPOUT_ = 0.7
_STATE_         ='Train'


def writer_pre_proc(images):
    print('adding to graph')
    resized_images = tf.image.resize_images(images, size=[_IMAGE_HEIGHT_,_IMAGE_WIDTH_])
    rgb_image_float = tf.image.convert_image_dtype(resized_images, tf.float32) 
    return rgb_image_float

def main():
    ''' Main function'''

    #Load minst data
    dummy_reader = Dataset_reader_classification(filename=_DATASET_PATH_, num_classes=_CLASSES_)
    dummy_reader.pre_process_image(writer_pre_proc)


    #Construct model
    with tf.name_scope('Alexnet'):
        Simple_DNN = Model(Model_name=_MODEL_NAME_, Summary=_SUMMARY_, \
            Batch_size=_BATCH_SIZE_, Image_width=_IMAGE_WIDTH_, Image_height=_IMAGE_HEIGHT_,\
           Image_cspace=_IMAGE_CSPACE_, Classes=_CLASSES_, Save_dir=_SAVE_DIR_, State=_STATE_, Dropout=_DROPOUT_)
        Simple_DNN.Set_optimizer()
        Simple_DNN.Construct_Accuracy_op()

    #Training block
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        Simple_DNN.Construct_Writers()
        Simple_DNN.Train_Iter(iterations=_ITERATIONS_, data=dummy_reader, restore=_RESTORE_)
        #Predictions = Simple_DNN.Predict(input_data = mnist.train.next_batch(10)[0])


if __name__ == "__main__":
    main()
