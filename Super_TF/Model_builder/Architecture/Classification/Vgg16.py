from utils.builder import Builder
import tensorflow as tf


def Build_vgg16(kwargs):
        '''Add paper and brief description'''
        with tf.name_scope('Vgg_model'):
            with Builder(**kwargs) as vgg16_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, kwargs['Image_width']*kwargs['Image_height']*kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, shape=[None, kwargs['Classes']], name='Output')
                dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                input_reshape = vgg16_builder.Reshape_input(input_placeholder, width=kwargs['Image_width'], height=kwargs['Image_height'], colorspace= kwargs['Image_cspace'])

                #Setting control params
                vgg16_builder.control_params(Dropout_control=dropout_prob_placeholder)

                #Feature Extraction
                conv1a = vgg16_builder.Conv2d_layer(input_reshape, filters=64)
                conv1b = vgg16_builder.Conv2d_layer(conv1a, filters=64)

                pool1 = vgg16_builder.Pool_layer(conv1b)

                conv2a = vgg16_builder.Conv2d_layer(pool1, filters=128)
                conv2b = vgg16_builder.Conv2d_layer(conv2a, filters=128)

                pool2 = vgg16_builder.Pool_layer(conv2b)

                conv3a = vgg16_builder.Conv2d_layer(pool2, filters=256)
                conv3b = vgg16_builder.Conv2d_layer(conv3a, filters=256)
                conv3c = vgg16_builder.Conv2d_layer(conv3b, filters=256)

                pool3 = vgg16_builder.Pool_layer(conv3c)

                conv4a = vgg16_builder.Conv2d_layer(pool3, filters=512)
                conv4b = vgg16_builder.Conv2d_layer(conv4a, filters=512)
                conv4c = vgg16_builder.Conv2d_layer(conv4b, filters=512)

                pool4 = vgg16_builder.Pool_layer(conv4c)

                conv5a = vgg16_builder.Conv2d_layer(pool4, filters=512)
                conv5b = vgg16_builder.Conv2d_layer(conv5a, filters=512)
                conv5c = vgg16_builder.Conv2d_layer(conv5b, filters=512)

                pool5 = vgg16_builder.Pool_layer(conv5c)

                #Densely Connected
                fc1 = vgg16_builder.FC_layer(pool5, filters=4096)
                drop1 = vgg16_builder.Dropout_layer(fc1)

                fc2 = vgg16_builder.FC_layer(drop1, filters=4096)
                drop2 = vgg16_builder.Dropout_layer(fc2)

                output = vgg16_builder.FC_layer(drop2, filters=kwargs['Classes'], readout=True)

                #Logit Loss
                with tf.name_scope('Cross_entropy_loss'):
                    softmax_logit_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=output_placeholder, logits=output))

                #Adding collections to graph
                tf.add_to_collection(kwargs['Model_name'] + '_Input_ph', input_placeholder)
                tf.add_to_collection(kwargs['Model_name'] + '_Input_reshape', input_reshape)
                tf.add_to_collection(kwargs['Model_name'] + '_Output_ph', output_placeholder)
                tf.add_to_collection(kwargs['Model_name'] + '_Output', output)
                tf.add_to_collection(kwargs['Model_name'] + '_Dropout_prob_ph', dropout_prob_placeholder)
                tf.add_to_collection(kwargs['Model_name'] + '_State', state_placeholder)
                tf.add_to_collection(kwargs['Model_name'] + '_Loss', softmax_logit_loss)
                
                return 'Classification'


