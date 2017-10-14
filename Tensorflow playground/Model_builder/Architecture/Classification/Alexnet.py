from utils.builder import Builder
import tensorflow as tf



def Build_Alexnet(kwargs):
        '''Add paper and brief description'''
        with tf.name_scope('Alexnet_model'):
            with Builder(**kwargs) as alexnet_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, kwargs['Image_width']*kwargs['Image_height']*kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, shape=[None, kwargs['Classes']], name='Output')
                dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                state_placeholder = tf.placeholder(tf.bool, name="State")

                input_reshape = alexnet_builder.Reshape_input(input_placeholder, width=kwargs['Image_width'], height=kwargs['Image_height'], colorspace= kwargs['Image_cspace'])

                #Setting control params
                alexnet_builder.control_params(Dropout_control=dropout_prob_placeholder, State=state_placeholder)

                #Feature Extraction
                conv1 = alexnet_builder.Conv2d_layer(input_reshape, stride=[1, 4, 4, 1], k_size=[11, 11], filters=96, padding='VALID', Batch_norm=True)
                
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

                output = alexnet_builder.FC_layer(drop2, filters=kwargs['Classes'], readout=True)

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


