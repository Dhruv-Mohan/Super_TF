import tensorflow as tf
from Dataset_IO.Classification.Dataset_writer_classification import Dataset_writer_classification


_IMAGE_WIDTH_       = 28
_IMAGE_HEIGHT_      = 28
_IMAGE_CSPACE_      = 1
_DATA_PATH_         = 'G:/Datasets/mnist/testing' #REPLACE WITH PATH TO FOLDER CONTAINING IMAGES
_DATASET_NAME_      = 'Mnist_test.tfrecords'

'''
A classification dataset writer must be given the path to a folder containing the dataset. 
Inside the folder, each class should have it's own subfolder which houses the images.
DATA/
    |
    |-->Class1/
    |          |-Class1_Img1.png
    |          |-Class1_Img2.png
    |    
    |-->Class2/
    |          |-Class2_Img1.png
    |          |-Class2_Img2.png
    |
    |-->Class3/
    |          |-Class3_Img1.png
    |          |-Class3_Img2.png
'''


def main():
    dummy_writer = Dataset_writer_classification(Dataset_filename=_DATASET_NAME_,\
       image_shape=[_IMAGE_HEIGHT_, _IMAGE_WIDTH_, _IMAGE_CSPACE_])
    
    #Writer accepts path of the data as described above or a dictionary with labels as keys and a list of file names of the corresponding lable as values
    dummy_writer.filename_constructor(filename_path=_DATA_PATH_)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        dummy_writer.write_record()



if __name__ == "__main__":
    main()