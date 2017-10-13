from utils.builder import Builder
import tensorflow as tf

def Build_Inception_Resnet_v2a(self):
    ''' Inception_Resnet_v2 as written in tf.slim attach issue link'''
        with tf.name_scope('Inception_Resnet_v2a_model'):
            with Builder(**self.kwargs) as inceprv2a_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, self.kwargs['Image_width']*self.kwargs['Image_height']*self.kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, shape=[None, self.kwargs['Classes']], name='Output')
                dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                state_placeholder = tf.placeholder(tf.string, name="State")
                input_reshape = inceprv2a_builder.Reshape_input(input_placeholder, width=self.kwargs['Image_width'], height=self.kwargs['Image_height'], colorspace= self.kwargs['Image_cspace'])

                #Setting control params
                inceprv2a_builder.control_params(Dropout_control=dropout_prob_placeholder, State=state_placeholder)

                #Construct functional building blocks
                def stem(input):
                    with tf.name_scope('Stem'):
                        conv1 = inceprv2a_builder.Conv2d_layer(input, stride=[1, 2, 2, 1], filters=32, Batch_norm=True)
                        conv2 = inceprv2a_builder.Conv2d_layer(conv1, stride=[1, 1, 1, 1], k_size=[3, 3], filters=32, Batch_norm=True, padding='VALID')
                        conv3 = inceprv2a_builder.Conv2d_layer(conv2, stride=[1, 1, 1, 1], k_size=[3, 3], filters=64, Batch_norm=True)
                        pool1 = inceprv2a_builder.Pool_layer(conv3, stride=[1, 2, 2, 1], k_size=[1, 3 ,3, 1], padding='VALID')

                        conv4 = inceprv2a_builder.Conv2d_layer(pool1, stride=[1, 1, 1, 1], filters=80, Batch_norm=True)
                        conv5 = inceprv2a_builder.Conv2d_layer(conv4, stride=[1, 1, 1, 1], k_size=[3, 3], filters=192, Batch_norm=True, padding='VALID')

                        pool2 = inceprv2a_builder.Pool_layer(conv5, stride=[1, 2, 2, 1], k_size=[1, 3 ,3, 1], padding='VALID')

                        conv1a_split1 = inceprv2a_builder.Conv2d_layer(pool2, stride=[1, 1, 1, 1], filters=96, Batch_norm=True)

                        conv1b_split1 = inceprv2a_builder.Conv2d_layer(pool2, stride=[1, 1, 1, 1], filters=48, Batch_norm=True)
                        conv2b_split1 = inceprv2a_builder.Conv2d_layer(conv1b_split1, stride=[1, 1, 1, 1], k_size=[5, 5], filters=64, Batch_norm=True)

                        conv1c_split1 = inceprv2a_builder.Conv2d_layer(pool2, stride=[1, 1, 1, 1], filters=64, Batch_norm=True)
                        conv2c_split1 = inceprv2a_builder.Conv2d_layer(conv1c_split1, stride=[1, 1, 1, 1], k_size=[3, 3], filters=96, Batch_norm=True)
                        conv3c_split1 = inceprv2a_builder.Conv2d_layer(conv2c_split1, stride=[1, 1, 1, 1], k_size=[3, 3], filters=96, Batch_norm=True)

                        avgpool1d_split1 = inceprv2a_builder.Pool_layer(pool2, k_size=[1, 3, 3, 1], stride=[1, 1, 1, 1], pooling_type='AVG')
                        conv1d_split1 = inceprv2a_builder.Conv2d_layer(avgpool1d_split1, k_size=[1, 1], filters=64, Batch_norm=True)

                        concat = inceprv2a_builder.Concat([conv1a_split1, conv2b_split1, conv3c_split1, conv1d_split1])

                        return concat

                def incep_block35(input, Activation=True, scale=1.0):
                    with tf.name_scope('Block35'):
                        conv1a_split1 = inceprv2a_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=32, Batch_norm=True)

                        conv1b_split1 = inceprv2a_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=32, Batch_norm=True)
                        conv2b_split1 = inceprv2a_builder.Conv2d_layer(conv1b_split1, stride=[1, 1, 1, 1], filters=32, Batch_norm=True)

                        conv1c_split1 = inceprv2a_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=32, Batch_norm=True)
                        conv2c_split1 = inceprv2a_builder.Conv2d_layer(conv1c_split1, stride=[1, 1, 1, 1], filters=48, Batch_norm=True)
                        conv3c_split1 = inceprv2a_builder.Conv2d_layer(conv2c_split1, stride=[1, 1, 1, 1], filters=64, Batch_norm=True)

                        concat = inceprv2a_builder.Concat([conv1a_split1, conv2b_split1, conv3c_split1])

                        conv2 = inceprv2a_builder.Conv2d_layer(concat, stride=[1, 1, 1, 1], k_size=[1, 1], filters=input.get_shape()[3], Batch_norm=False, Activation=False)
                        conv2_scale = inceprv2a_builder.Scale_activations(conv2,scaling_factor=scale)
                        residual_out = inceprv2a_builder.Residual_connect([input, conv2_scale], Activation=Activation)

                        return residual_out

                def incep_block17(input, Activation=True, scale=1.0):
                    with tf.name_scope('Block17'):
                        conv1a_split1 = inceprv2a_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=192, Batch_norm=True)

                        conv1b_split1 = inceprv2a_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=128, Batch_norm=True)
                        conv2b_split1 = inceprv2a_builder.Conv2d_layer(conv1b_split1, stride=[1, 1, 1, 1], k_size=[1, 7], filters=160, Batch_norm=True)
                        conv3b_split1 = inceprv2a_builder.Conv2d_layer(conv2b_split1, stride=[1, 1, 1, 1], k_size=[7, 1], filters=192, Batch_norm=True)

                        concat = inceprv2a_builder.Concat([conv1a_split1, conv3b_split1])

                        conv2 = inceprv2a_builder.Conv2d_layer(concat, stride=[1, 1, 1, 1], k_size=[1, 1], filters=input.get_shape()[3], Batch_norm=False, Activation=False)
                        conv2_scale = inceprv2a_builder.Scale_activations(conv2,scaling_factor=scale)
                        residual_out = inceprv2a_builder.Residual_connect([input, conv2_scale], Activation=Activation)

                        return residual_out

                def incep_block8(input, Activation=True, scale=1.0):
                    with tf.name_scope('Block8'):
                        conv1a_split1 = inceprv2a_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=192, Batch_norm=True)

                        conv1b_split1 = inceprv2a_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=192, Batch_norm=True)
                        conv2b_split1 = inceprv2a_builder.Conv2d_layer(conv1b_split1, stride=[1, 1, 1, 1], k_size=[1, 3], filters=224, Batch_norm=True)
                        conv3b_split1 = inceprv2a_builder.Conv2d_layer(conv2b_split1, stride=[1, 1, 1, 1], k_size=[3, 1], filters=256, Batch_norm=True)

                        concat = inceprv2a_builder.Concat([conv1a_split1, conv3b_split1])

                        conv2 = inceprv2a_builder.Conv2d_layer(concat, stride=[1, 1, 1, 1], k_size=[1, 1], filters=input.get_shape()[3], Batch_norm=False, Activation=False)
                        conv2_scale = inceprv2a_builder.Scale_activations(conv2,scaling_factor=scale) #Last layer has no activations, recheck with implementation
                        residual_out = inceprv2a_builder.Residual_connect([input, conv2_scale], Activation=Activation)

                        return residual_out

                def ReductionA(input):
                    with tf.name_scope('Reduction_35x17'):
                        conv1a_split1 = inceprv2a_builder.Conv2d_layer(input, stride=[1, 2, 2, 1], k_size=[3, 3], filters=384, Batch_norm=True, padding='VALID')

                        conv1b_split1 = inceprv2a_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=256, Batch_norm=True)
                        conv2b_split1 = inceprv2a_builder.Conv2d_layer(conv1b_split1, stride=[1, 1, 1, 1], k_size=[3, 3], filters=256, Batch_norm=True)
                        conv3b_split1 = inceprv2a_builder.Conv2d_layer(conv2b_split1, stride=[1, 2, 2, 1], k_size=[3, 3], filters=384, Batch_norm=True, padding='VALID')

                        pool1c_split1 = inceprv2a_builder.Pool_layer(input, stride=[1, 2, 2, 1], k_size=[1, 3, 3, 1], padding='VALID')

                        concat = inceprv2a_builder.Concat([conv1a_split1, conv3b_split1, pool1c_split1])
                        
                        return concat
                def ReductionB(input):
                    with tf.name_scope('Reduction_17x8'):
                        conv1a_split1 = inceprv2a_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=256, Batch_norm=True)
                        conv2a_split1 = inceprv2a_builder.Conv2d_layer(conv1a_split1, stride=[1, 2, 2, 1], k_size=[3, 3], filters=384, Batch_norm=True, padding='VALID')

                        conv1b_split1 = inceprv2a_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=256, Batch_norm=True)
                        conv2b_split1 = inceprv2a_builder.Conv2d_layer(conv1b_split1, stride=[1, 2, 2, 1], k_size=[3, 3], filters=288, Batch_norm=True, padding='VALID')

                        conv1c_split1 = inceprv2a_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=256, Batch_norm=True)
                        conv2c_split1 = inceprv2a_builder.Conv2d_layer(conv1c_split1, stride=[1, 1, 1, 1], k_size=[3, 3], filters=288, Batch_norm=True)
                        conv3c_split1 = inceprv2a_builder.Conv2d_layer(conv2c_split1, stride=[1, 2, 2, 1], k_size=[3, 3], filters=320, Batch_norm=True, padding='VALID')

                        pool1d_split1 = inceprv2a_builder.Pool_layer(input, stride=[1, 2, 2, 1], k_size=[1, 3, 3, 1], padding='VALID')

                        concat = inceprv2a_builder.Concat([conv2a_split1, conv2b_split1, conv3c_split1, pool1d_split1])
                        return concat

                #MODEL CONSTRUCTION

                #STEM
                Block_35 = stem(input_reshape)
                #INCEPTION 35x35
                for index in range(10):
                    Block_35 = incep_block35(Block_35, scale=0.17)
                #Reduction 35->17
                Block_17 = ReductionA(Block_35)
                #INCEPTION 17x17
                for index in range(20):
                    Block_17 = incep_block17(Block_17, scale=0.1)
                #Reduction 17->8
                Block_8 = ReductionB(Block_17)
                for index in range(9):
                    Block_8 = incep_block8(Block_8, scale=0.2)
                Block_8 = incep_block8(Block_8, False)
                #Normal Logits
                with tf.name_scope('Logits'):
                    model_conv = inceprv2a_builder.Conv2d_layer(Block_8, stride=[1, 1, 1, 1], k_size=[1, 1], filters=1536, Batch_norm=True)
                    model_avg_pool = inceprv2a_builder.Pool_layer(model_conv, k_size=[1, 8, 8, 1], stride=[1, 8, 8, 1], padding='SAME', pooling_type='AVG')
                    drop1 = inceprv2a_builder.Dropout_layer(model_avg_pool)
                    output = inceprv2a_builder.FC_layer(drop1, filters=self.kwargs['Classes'], readout=True)
                #AuxLogits
                with tf.name_scope('Auxlogits'):
                    model_aux_avg_pool = inceprv2a_builder.Pool_layer(Block_17, k_size=[1, 5, 5, 1], stride=[1, 3, 3, 1], padding='VALID', pooling_type='AVG')
                    model_aux_conv1 = inceprv2a_builder.Conv2d_layer(model_aux_avg_pool, k_size=[1, 1], stride=[1, 1, 1, 1], filters=128, Batch_norm=True)
                    model_aux_conv2 = inceprv2a_builder.Conv2d_layer(model_aux_conv1, k_size=[5, 5], stride=[1, 1, 1, 1], padding='VALID', filters=768, Batch_norm=True)
                    model_aux_logits = inceprv2a_builder.FC_layer(model_aux_conv2, filters=self.kwargs['Classes'], readout=True)

                #Logit Loss
                with tf.name_scope('Cross_entropy_loss'):
                    softmax_logit_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=output_placeholder, logits=output))

                #AuxLogit Loss
                with tf.name_scope('Cross_entropy_loss'):
                    softmax_auxlogit_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=output_placeholder, logits=model_aux_logits)) * 0.6

                #Adding collections to graph
                tf.add_to_collection(self.model_name + '_Endpoints', Block_35)
                tf.add_to_collection(self.model_name + '_Endpoints', Block_17)
                tf.add_to_collection(self.model_name + '_Endpoints', Block_8)
                tf.add_to_collection(self.model_name + '_Input_ph', input_placeholder)
                tf.add_to_collection(self.model_name + '_Input_reshape', input_reshape)
                tf.add_to_collection(self.model_name + '_Output_ph', output_placeholder)
                tf.add_to_collection(self.model_name + '_Output', output)
                tf.add_to_collection(self.model_name + '_Dropout_prob_ph', dropout_prob_placeholder)
                tf.add_to_collection(self.model_name + '_State', state_placeholder)
                tf.add_to_collection(self.model_name + '_Loss', softmax_logit_loss)
                tf.add_to_collection(self.model_name + '_Loss', softmax_auxlogit_loss)



