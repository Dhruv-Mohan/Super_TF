import tensorflow as tf
from Dataset_IO.Dataset_reader_classification import Dataset_reader_classification

_IMAGE_WIDTH_       = 299
_IMAGE_HEIGHT_      = 299
_IMAGE_CSPACE_      = 1
_DATASET_PATH_      = 'E:\\Dummyset.tfrecords'
_BATCH_SIZE_        = 5
_SHOW_IMAGES        = True


if _SHOW_IMAGES :
    import cv2

def main():
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    dummy_reader = Dataset_reader_classification(filename=_DATASET_PATH_, image_shape=[299,299,1])
    with tf.Session() as sess:
        init_op.run()
        images, labels = dummy_reader.get_next_batch(_BATCH_SIZE_)
        if _SHOW_IMAGES :
            for image in images:
                cv2.imshow('Image', image)
                cv2.waitKey(0)


if __name__ == "__main__":
    main()