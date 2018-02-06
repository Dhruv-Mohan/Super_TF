from utils.builder import Builder
import tensorflow as tf
from utils.Base_Archs.Base_Classifier import Base_Classifier

class Vgg19(Base_Classifier):
    """Vgg19 from the visual geometry group at Oxford as described in
    https://arxiv.org/pdf/1409.1556.pdf """
    def __init__(self, kwargs):
        super().__init__(self, kwargs)

    def build_net(self):
        with tf.name_scope('Vgg19_model'):
            with Builder(**self.build_params) as vgg19_builder:
                #Setting control params
                vgg19_builder.control_params(Dropout_control=self.dropout_placeholder)

                #Feature Extraction
                conv1a = vgg19_builder.Conv2d_layer(self.input_placeholder, filters=64)
                conv1b = vgg19_builder.Conv2d_layer(conv1a, filters=64)

                pool1 = vgg19_builder.Pool_layer(conv1b)

                conv2a = vgg19_builder.Conv2d_layer(pool1, filters=128)
                conv2b = vgg19_builder.Conv2d_layer(conv2a, filters=128)

                pool2 = vgg19_builder.Pool_layer(conv2b)

                conv3a = vgg19_builder.Conv2d_layer(pool2, filters=256)
                conv3b = vgg19_builder.Conv2d_layer(conv3a, filters=256)
                conv3c = vgg19_builder.Conv2d_layer(conv3b, filters=256)
                conv3d = vgg19_builder.Conv2d_layer(conv3c, filters=256)

                pool3 = vgg19_builder.Pool_layer(conv3d)

                conv4a = vgg19_builder.Conv2d_layer(pool3, filters=512)
                conv4b = vgg19_builder.Conv2d_layer(conv4a, filters=512)
                conv4c = vgg19_builder.Conv2d_layer(conv4b, filters=512)
                conv4d = vgg19_builder.Conv2d_layer(conv4c, filters=512)

                pool4 = vgg19_builder.Pool_layer(conv4d)

                conv5a = vgg19_builder.Conv2d_layer(pool4, filters=512)
                conv5b = vgg19_builder.Conv2d_layer(conv5a, filters=512)
                conv5c = vgg19_builder.Conv2d_layer(conv5b, filters=512)
                conv5d = vgg19_builder.Conv2d_layer(conv5c, filters=512)

                pool5 = vgg19_builder.Pool_layer(conv5d)

                #Densely Connected
                fc1 = vgg19_builder.FC_layer(pool5, filters=4096)
                drop1 = vgg19_builder.Dropout_layer(fc1)

                fc2 = vgg19_builder.FC_layer(drop1, filters=4096)
                drop2 = vgg19_builder.Dropout_layer(fc2)

                output = vgg19_builder.FC_layer(drop2, filters=self.build_params['Classes'], readout=True)
                return output




