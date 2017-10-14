from utils.builder import Builder
import tensorflow as tf


def Build_Inception_Resnet_v2(kwargs):
        ''' Inception-resnet-v2 as described in the paper'''
        with tf.name_scope('Inception_Resnet_v2_model'):
            with Builder(**kwargs) as inceprv2_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, kwargs['Image_width']*kwargs['Image_height']*kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, shape=[None, kwargs['Classes']], name='Output')
                dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                state_placeholder = tf.placeholder(tf.string, name="State")
                input_reshape = inceprv2_builder.Reshape_input(input_placeholder, width=kwargs['Image_width'], height=kwargs['Image_height'], colorspace= kwargs['Image_cspace'])

                #Setting control params
                inceprv2_builder.control_params(Dropout_control=dropout_prob_placeholder, State=state_placeholder)
                
                #Construct functional building blocks
                def stem(input):
                    with tf.name_scope('Stem') as scope:
                        conv1 = inceprv2_builder.Conv2d_layer(input, stride=[1,2,2,1], filters=32, padding='VALID', Batch_norm=True)
                        conv2 = inceprv2_builder.Conv2d_layer(conv1, stride=[1,1,1,1], filters=32, padding='VALID', Batch_norm=True)
                        conv3 = inceprv2_builder.Conv2d_layer(conv2, stride=[1,1,1,1], filters=64, Batch_norm=True)
                        
                        conv1a_split1 = inceprv2_builder.Conv2d_layer(conv3, stride=[1,2,2,1], filters=96, padding='VALID', Batch_norm=True)
                        pool1b_split1 = inceprv2_builder.Pool_layer(conv3, padding='VALID')

                        concat1 = inceprv2_builder.Concat([conv1a_split1, pool1b_split1])

                        conv1a_split2 = inceprv2_builder.Conv2d_layer(concat1, stride=[1, 1, 1, 1], k_size=[1, 1], filters=64, Batch_norm=True)
                        conv2a_split2 = inceprv2_builder.Conv2d_layer(conv1a_split2, stride=[1, 1, 1, 1], k_size=[7, 1], filters=64, Batch_norm=True)
                        conv3a_split2 = inceprv2_builder.Conv2d_layer(conv2a_split2, stride=[1, 1, 1, 1], k_size=[1, 7], filters=64, Batch_norm=True)
                        conv4a_split2 = inceprv2_builder.Conv2d_layer(conv3a_split2, stride=[1, 1, 1, 1], filters=96, padding='VALID', Batch_norm=True)

                        conv1b_split2 = inceprv2_builder.Conv2d_layer(concat1, stride=[1, 1, 1, 1], k_size=[1, 1], filters=64, Batch_norm=True)
                        conv2b_split2 = inceprv2_builder.Conv2d_layer(conv1b_split2, stride=[1, 1, 1, 1], filters=96, padding='VALID', Batch_norm=True)

                        concat2 = inceprv2_builder.Concat([conv4a_split2, conv2b_split2])

                        pool1a_split3 = inceprv2_builder.Pool_layer(concat2, padding="VALID")
                        conv1b_split3 = inceprv2_builder.Conv2d_layer(concat2, stride=[1, 2, 2, 1], filters=192, padding='VALID', Batch_norm=True)

                        concat3 = inceprv2_builder.Concat([pool1a_split3, conv1b_split3])
                        return concat3

                def inception_resnet_A(input):
                    with tf.name_scope('Inception_Resnet_A') as scope:
                        conv1a_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=32, Batch_norm=True)
                        conv2a_split1 = inceprv2_builder.Conv2d_layer(conv1a_split1, stride=[1, 1, 1, 1], filters=48, Batch_norm=True)
                        conv3a_split1 = inceprv2_builder.Conv2d_layer(conv2a_split1, stride=[1, 1, 1, 1], filters=64, Batch_norm=True)

                        conv1b_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=32, Batch_norm=True)
                        conv2b_split1 = inceprv2_builder.Conv2d_layer(conv1b_split1, stride=[1, 1, 1, 1], filters=32, Batch_norm=True)

                        conv1c_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=32, Batch_norm=True)
                        
                        concat1 = inceprv2_builder.Concat([conv3a_split1, conv2b_split1, conv1c_split1])

                        conv2 = inceprv2_builder.Conv2d_layer(concat1, stride=[1, 1, 1, 1], k_size=[1, 1], filters=384, Batch_norm=True, Activation=False)

                        conv2_scale = inceprv2_builder.Scale_activations(conv2,scaling_factor = 1)
                        residual_out = inceprv2_builder.Residual_connect([input, conv2_scale])

                        return residual_out

                def reduction_A(input):
                    with tf.name_scope('Reduction_A') as scope:
                        '''
                        k=256, l=256, m=384, n=384
                        '''
                        conv1a_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=256, Batch_norm=True)
                        conv2a_split1 = inceprv2_builder.Conv2d_layer(conv1a_split1, stride=[1, 1, 1, 1], filters=256, Batch_norm=True)
                        conv3a_split1 = inceprv2_builder.Conv2d_layer(conv2a_split1, stride=[1, 2, 2, 1], filters=384, padding='VALID', Batch_norm=True)

                        conv1b_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 2, 2, 1], filters=384, padding='VALID', Batch_norm=True)

                        pool1c_split1 = inceprv2_builder.Pool_layer(input, padding='VALID')

                        concat = inceprv2_builder.Concat([conv3a_split1, conv1b_split1, pool1c_split1])
                        
                        return concat

                def inception_resnet_B(input):
                    with tf.name_scope('Inception_Resnet_B') as scope:
                        conv1a_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=128, Batch_norm=True)
                        conv2a_split1 = inceprv2_builder.Conv2d_layer(conv1a_split1, stride=[1, 1, 1, 1], k_size=[1, 7], filters=160, Batch_norm=True)
                        conv3a_split1 = inceprv2_builder.Conv2d_layer(conv2a_split1, stride=[1, 1, 1, 1], k_size=[7, 1], filters=192, Batch_norm=True)

                        conv1b_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=192, Batch_norm=True)

                        concat1 = inceprv2_builder.Concat([conv3a_split1, conv1b_split1])

                        conv2 = inceprv2_builder.Conv2d_layer(concat1, stride=[1, 1, 1, 1], k_size=[1, 1], filters=1152, Batch_norm=True, Activation=False) #paper discrepancy filter = 1154
                        conv2_scale = inceprv2_builder.Scale_activations(conv2, scaling_factor=0.4)

                        residual_out = inceprv2_builder.Residual_connect([input, conv2_scale])

                        return residual_out

                def reduction_B(input):
                    with tf.name_scope('Reduction_B') as scope: 
                        conv1a_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=256, Batch_norm=True)
                        conv2a_split1 = inceprv2_builder.Conv2d_layer(conv1a_split1, stride=[1, 1, 1, 1], filters=288, Batch_norm=True)
                        conv3a_split1 = inceprv2_builder.Conv2d_layer(conv2a_split1, stride=[1, 2, 2, 1], filters=384, padding='VALID', Batch_norm=True)

                        conv1b_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=256, Batch_norm=True)
                        conv2b_split1 = inceprv2_builder.Conv2d_layer(conv1b_split1, stride=[1, 2, 2, 1], filters=256, padding='VALID', Batch_norm=True)

                        conv1c_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=256, Batch_norm=True)
                        conv2c_split1 = inceprv2_builder.Conv2d_layer(conv1c_split1, stride=[1, 2, 2, 1], filters=256, padding='VALID', Batch_norm=True)

                        pool1d_split1 = inceprv2_builder.Pool_layer(input, padding='VALID')

                        concat = inceprv2_builder.Concat([conv3a_split1, conv2b_split1, conv2c_split1, pool1d_split1])
                        return concat

                def inception_resnet_C(input):
                    with tf.name_scope('Inception_Resnet_C') as scope:
                        conv1a_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=192, Batch_norm=True)
                        conv2a_split1 = inceprv2_builder.Conv2d_layer(conv1a_split1, stride=[1, 1, 1, 1], k_size=[1, 3], filters=224, Batch_norm=True)
                        conv3a_split1 = inceprv2_builder.Conv2d_layer(conv2a_split1, stride=[1, 1, 1, 1], k_size=[3, 1], filters=256, Batch_norm=True)

                        conv1b_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=192, Batch_norm=True)

                        concat1 = inceprv2_builder.Concat([conv3a_split1, conv1b_split1])

                        conv2 = inceprv2_builder.Conv2d_layer(concat1, stride=[1, 1, 1, 1], k_size=[1, 1], filters=2048, Batch_norm=True, Activation=False)
                        conv2_scale = inceprv2_builder.Scale_activations(conv2)

                        residual_out = inceprv2_builder.Residual_connect([input, conv2_scale])
                        
                        return residual_out
                #Model Construction

                #Stem
                model_stem = stem(input_reshape)
                #5x Inception Resnet A
                inception_A1 = inception_resnet_A(model_stem)
                inception_A2 = inception_resnet_A(inception_A1)
                inception_A3 = inception_resnet_A(inception_A2)
                inception_A4 = inception_resnet_A(inception_A3)
                inception_A5 = inception_resnet_A(inception_A4)
                #Reduction A
                model_reduction_A = reduction_A(inception_A5)
                #10X Inception Resnet B
                inception_B1 = inception_resnet_B(model_reduction_A) #Don't know if i'm missing something or now, but reduction A's output for inception resnetv2 is a tensor of depth 1152
                inception_B2 = inception_resnet_B(inception_B1)
                inception_B3 = inception_resnet_B(inception_B2)
                inception_B4 = inception_resnet_B(inception_B3)
                inception_B5 = inception_resnet_B(inception_B4)
                inception_B6 = inception_resnet_B(inception_B5)
                inception_B7 = inception_resnet_B(inception_B6)
                inception_B8 = inception_resnet_B(inception_B7)
                inception_B9 = inception_resnet_B(inception_B8)
                inception_B10 = inception_resnet_B(inception_B9)
                #Reduction B
                model_reduction_B = reduction_B(inception_B10)
                #5X Inception Resnet C
                inception_C1 = inception_resnet_C(model_reduction_B)
                inception_C2 = inception_resnet_C(inception_C1)
                inception_C3 = inception_resnet_C(inception_C2)
                inception_C4 = inception_resnet_C(inception_C3)
                inception_C5 = inception_resnet_C(inception_C4)
                #Average Pooling
                average_pooling = inceprv2_builder.Pool_layer(inception_C5, k_size=[1, 8, 8, 1], stride=[1, 8, 8, 1], padding='SAME', pooling_type='AVG')
                #Dropout
                drop1 = inceprv2_builder.Dropout_layer(average_pooling)
                #Output
                output = inceprv2_builder.FC_layer(drop1, filters=kwargs['Classes'], readout=True)
                #Logit Loss
                with tf.name_scope('Cross_entropy_loss'):
                    softmax_logit_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=output_placeholder, logits=output))

                #Adding collections to graph
                tf.add_to_collection(kwargs['Model_name'] + '_Endpoints', inception_A5)
                tf.add_to_collection(kwargs['Model_name'] + '_Endpoints', inception_B10)
                tf.add_to_collection(kwargs['Model_name'] + '_Endpoints', inception_C5)
                tf.add_to_collection(kwargs['Model_name'] + '_Input_ph', input_placeholder)
                tf.add_to_collection(kwargs['Model_name'] + '_Input_reshape', input_reshape)
                tf.add_to_collection(kwargs['Model_name'] + '_Output_ph', output_placeholder)
                tf.add_to_collection(kwargs['Model_name'] + '_Output', output)
                tf.add_to_collection(kwargs['Model_name'] + '_Dropout_prob_ph', dropout_prob_placeholder)
                tf.add_to_collection(kwargs['Model_name'] + '_State', state_placeholder)
                tf.add_to_collection(kwargs['Model_name'] + '_Loss', softmax_logit_loss)

                return 'Classification'
