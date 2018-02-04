from utils.builder import Builder
import tensorflow as tf
from utils.Base_Archs.Base_Classifier import Base_Classifier


class Lenet(Base_Classifier):
    """
    Convolutional neural network introduced in 'Gradient-Based Learning Applied to Document
    Recognition by LeCun et al. http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
    Consists of 2 convolutional layers and 2 fully connected layers. Each conv. layer
    is succeeded by a pooling operation
    """
    def __init__(self, kwargs):
        super().__init__(kwargs)

    def build_net(self):
        with tf.name_scope('LeNet_Model'):
            with Builder(**self.build_params) as lenet_builder:
                input_reshape = lenet_builder.Reshape_input(self.input_placeholder)
                
                conv1 = lenet_builder.Conv2d_layer(input_reshape, k_size=[5, 5])
                pool1 = lenet_builder.Pool_layer(conv1)

                conv2 = lenet_builder.Conv2d_layer(pool1, k_size=[5, 5], filters=64)
                pool2 = lenet_builder.Pool_layer(conv2)

                fc1 = lenet_builder.FC_layer(pool2);
                output = lenet_builder.FC_layer(fc1, filters=self.build_params['Classes'], readout=True)

                return output
