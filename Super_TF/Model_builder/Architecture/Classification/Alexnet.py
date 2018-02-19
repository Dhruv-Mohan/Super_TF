from utils.builder import Builder
import tensorflow as tf
from utils.Base_Archs.Base_Classifier import Base_Classifier


class Alexnet(Base_Classifier):
    '''Alexnet as defined in https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    Consists of 5 pooling layer with increasing tensor channels and 3 fully connected layers
    Conv layer 2, 3 and 5 are followed by a max-pool operation'''

    def __init__(self, kwargs):
        super().__init__(kwargs)

    def build_net(self):
        with tf.name_scope('Alexnet_model'):
            with Builder(**self.build_params) as alexnet_builder:
                #Setting control params
                alexnet_builder.control_params(Dropout_control=self.dropout_placeholder, State=self.state_placeholder)

                #Feature Extraction
                conv1 = alexnet_builder.Conv2d_layer(self.input_placeholder, stride=[1, 4, 4, 1], k_size=[11, 11], filters=96, padding='VALID', Batch_norm=True)
                
                pool1 = alexnet_builder.Pool_layer(conv1, k_size=[1, 3, 3, 1], padding='VALID')

                pad1 = alexnet_builder.Pad_layer(pool1, p_type='SYMMETRIC')
                conv2 = alexnet_builder.Conv2d_layer(pad1, k_size=[5, 5], filters=256, padding='VALID', Batch_norm=True)

                pool2 = alexnet_builder.Pool_layer(conv2, k_size=[1, 3, 3, 1], padding='VALID')

                conv3 = alexnet_builder.Conv2d_layer(pool2, filters=384, Batch_norm=True)
                conv4 = alexnet_builder.Conv2d_layer(conv3, filters=384, Batch_norm=True)
                conv5 = alexnet_builder.Conv2d_layer(conv4, filters=256, Batch_norm=True)

                pool5 = alexnet_builder.Pool_layer(conv5, k_size=[1, 3, 3, 1])

                #Densely Connected
                fc1 = alexnet_builder.FC_layer(pool5, filters=4096)
                drop1 = alexnet_builder.Dropout_layer(fc1)

                fc2 = alexnet_builder.FC_layer(drop1, filters=4096)
                drop2 = alexnet_builder.Dropout_layer(fc2)

                output = alexnet_builder.FC_layer(drop2, filters=self.build_params['Classes'], readout=True)
                return output