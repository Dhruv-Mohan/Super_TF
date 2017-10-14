import tensorflow as tf
from Dataset_IO.Classification.Dataset_reader_classification import Dataset_reader_classification

_IMAGE_WIDTH_       = 299
_IMAGE_HEIGHT_      = 299
_IMAGE_CSPACE_      = 1
_DATASET_PATH_      = 'E:\\Avctet.tfrecords'
_BATCH_SIZE_        = 5
_SHOW_IMAGES_       = True
_CLASSES_           = 18

if _SHOW_IMAGES_ :
    import cv2

def writer_pre_proc(images):
    print('Attaching pre-processing operations to reader graph')
    resized_images = tf.image.resize_images(images, size=[_IMAGE_HEIGHT_, _IMAGE_WIDTH_])
    #rgb_image_float = tf.image.convert_image_dtype(resized_images, tf.float32) 
    return rgb_image_float

def main():
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    dummy_reader = Dataset_reader_classification(filename=_DATASET_PATH_, num_classes=_CLASSES_)
    #dummy_reader.pre_process_image(writer_pre_proc)

    with tf.Session() as sess:
        init_op.run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        images, labels = dummy_reader.next_batch(_BATCH_SIZE_)
        meanimage = sess.run([dummy_reader.mean_image])[0]
        print(meanimage)
        print(images[0])
        if _SHOW_IMAGES_ :
            for image in images:
                cv2.imshow('Image', image)
                cv2.imshow('Meanimage',meanimage)
                cv2.waitKey(0)

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    main()