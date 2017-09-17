from utils.builder import Builder
import tensorflow as tf


class Factory(object):
    """Factory class to build DNN Architectures"""
    #Look into adding a datastructure to keep track of last layer added to the graph


    def get_model(self):
        #return self.Build_Lenet()
        print('Build_'+self.model_name+'()')
        return (eval('self.Build_'+self.model_name+'()'))
    
    def Build_Unet_resent(self):
        with tf.name_scope('Unet_resnet'):
            with Builder(**self.kwargs) as unet_res_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, self.kwargs['Image_width']*self.kwargs['Image_height']*self.kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, shape=[None, self.kwargs['Classes']], name='Output')
                weight_placeholder = tf.placeholder(tf.float32, shape=[None, self.kwargs['Classes']], name='Weights')
                dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                state_placeholder = tf.placeholder(tf.string, name="State")
                input_reshape = unet_res_builder.Reshape_input(input_placeholder, width=self.kwargs['Image_width'], height=self.kwargs['Image_height'], colorspace= self.kwargs['Image_cspace'])

                #Setting control params
                unet_res_builder.control_params(Dropout_control=dropout_prob_placeholder, State=state_placeholder)

                def stack_encoder(input, out_filters):
                    input = unet_res_builder.Relu(input)

                    conv1a_split1 = unet_res_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=out_filters, Activation=False, Batch_norm=True)

                    conv1b_split1 = unet_res_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[3, 3], filters=out_filters, Batch_norm=True)
                    conv2b_split1 = unet_res_builder.Conv2d_layer(conv1b_split1, stride=[1, 1, 1, 1], k_size=[3, 3], filters=out_filters, Activation=False, Batch_norm=True)

                    res_connect = unet_res_builder.Residual_connect([conv1a_split1, conv2b_split1])

                    return res_connect

                def stack_decoder(input, encoder_connect, out_filters):
                    encoder_connect_shape = encoder_connect.get_shape().as_list()
                    del encoder_connect_shape[0]
                    res_filters = encoder_connect_shape.pop(2)

                    upscale_input = unet_res_builder.Upconv_layer(input, stride=[1, 3, 3, 1], filters=res_filters, Batch_norm=True) #change_filters to match encoder_connect filters

                    u_connect = unet_res_builder.Concat([encoder_connect, upscale_input])
                    conv1 = unet_res_builder.Conv2d_layer(u_connect, stride=[1, 1, 1, 1], k_size=[1, 1], filters=out_filters, Batch_norm=True)


                    conv1a_split1 = unet_res_builder.Conv2d_layer(conv1, stride=[1, 1, 1, 1], k_size=[1, 1], filters=out_filters, Activation=False, Batch_norm=True)

                    conv1b_split1 = unet_res_builder.Conv2d_layer(conv1, stride=[1, 1, 1, 1], k_size=[3, 3], filters=out_filters, Batch_norm=True)
                    conv2b_split1 = unet_res_builder.Conv2d_layer(conv1b_split1, stride=[1, 1, 1, 1], k_size=[3, 3], filters=out_filters, Activation=False, Batch_norm=True)

                    res_connect = unet_res_builder.Residual_connect([conv1a_split1, conv2b_split1])

                    return res_connect


                #Build Encoder

                Encoder1 = stack_encoder(input_reshape, 24)
                Pool1 = unet_res_builder.Pool_layer(Encoder1) #512

                Encoder2 = stack_encoder(Pool1, 64)
                Pool2 = unet_res_builder.Pool_layer(Encoder2) #256

                Encoder3 = stack_encoder(Pool2, 128)
                Pool3 = unet_res_builder.Pool_layer(Encoder3) #128

                Encoder4 = stack_encoder(Pool3, 256)
                Pool4 = unet_res_builder.Pool_layer(Encoder4) #64

                Encoder5 = stack_encoder(Pool4, 512)
                Pool5 = unet_res_builder.Pool_layer(Encoder5) #32

                Encoder6 = stack_encoder(Pool5, 768)
                Pool6 = unet_res_builder.Pool_layer(Encoder6) #16

                #Center
                Conv_center = unet_res_builder.Conv2d_layer(Pool6, stride=[1, 1, 1, 1], filters=768, Batch_norm=True, padding='SAME')
                Pool_center = unet_res_builder.Pool_layer(Conv_center) #8
                #Build Decoder

                Decode1 = stack_decoder(Pool_center, Encoder6, out_filters=512)
                Decode2 = stack_decoder(Decode1, Encoder5, out_filters=256)
                Decode3 = stack_decoder(Decode2, Encoder4, out_filters=128)
                Decode4 = stack_decoder(Decode3, Encoder3, out_filters=64)
                Decode5 = stack_decoder(Decode4, Encoder2, out_filters=24)
                Decode6 = stack_decoder(Decode5, Encoder1, out_filters=24)

                output = unet_res_builder.Conv2d_layer(Decode6, stride=[1, 1, 1, 1], filters=1, Batch_norm=True, k_size=[1, 1]) #output
                #Add loss and debug
                logits = tf.reshape(output, (-1, self.kwargs['Classes']))
                eps = tf.constant(value=1e-5)
                sigmoid = tf.nn.sigmoid(logits) + eps
                BCE_loss = output_placeholder * tf.log(sigmoid) #Fix output and weight shape
                Weighted_BCE_loss = tf.multiply(BCE_loss, weight_placeholder) + tf.multiply(tf.clip_by_value(logits, 0, 1e4), weight_placeholder)
                Weighted_BCE_loss = tf.reduce_mean(Weighted_BCE_loss)

                #Dice Loss
                intersection = tf.reduce_sum(sigmoid * output_placeholder)
                union = eps + tf.reduce_sum(sigmoid) + tf.reduce_sum(output_placeholder)
                Dice_loss = -(2* intersection/(union))

                #Graph Exports
                tf.add_to_collection(self.model_name + '_Input_ph', input_placeholder)
                tf.add_to_collection(self.model_name + '_Input_reshape', input_reshape)
                tf.add_to_collection(self.model_name + '_Weight_placeholder', weight_placeholder)
                tf.add_to_collection(self.model_name + '_Output_ph', output_placeholder)
                tf.add_to_collection(self.model_name + '_Output', output)
                tf.add_to_collection(self.model_name + '_Dropout_prob_ph', dropout_prob_placeholder)
                tf.add_to_collection(self.model_name + '_State', state_placeholder)
                tf.add_to_collection(self.model_name + '_Loss', Weighted_BCE_loss)
                tf.add_to_collection(self.model_name + '_Loss', Dice_loss)


    def Build_Inception_Resnet_v2a(self):
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


    def Build_Inception_Resnet_v2(self):
        with tf.name_scope('Inception_Resnet_v2_model'):
            with Builder(**self.kwargs) as inceprv2_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, self.kwargs['Image_width']*self.kwargs['Image_height']*self.kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, shape=[None, self.kwargs['Classes']], name='Output')
                dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                state_placeholder = tf.placeholder(tf.string, name="State")
                input_reshape = inceprv2_builder.Reshape_input(input_placeholder, width=self.kwargs['Image_width'], height=self.kwargs['Image_height'], colorspace= self.kwargs['Image_cspace'])

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
                #MODEL CONSTRUCTION

                #STEM
                model_stem = stem(input_reshape)
                #5x INCEPTION RESNET A
                inception_A1 = inception_resnet_A(model_stem)
                inception_A2 = inception_resnet_A(inception_A1)
                inception_A3 = inception_resnet_A(inception_A2)
                inception_A4 = inception_resnet_A(inception_A3)
                inception_A5 = inception_resnet_A(inception_A4)
                #REUCTION A
                model_reduction_A = reduction_A(inception_A5)
                #10X INCEPTION RESNET B
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
                #REDUCTION B
                model_reduction_B = reduction_B(inception_B10)
                #5X INCEPTION RESNET C
                inception_C1 = inception_resnet_C(model_reduction_B)
                inception_C2 = inception_resnet_C(inception_C1)
                inception_C3 = inception_resnet_C(inception_C2)
                inception_C4 = inception_resnet_C(inception_C3)
                inception_C5 = inception_resnet_C(inception_C4)
                #AVERAGE POOLING
                average_pooling = inceprv2_builder.Pool_layer(inception_C5, k_size=[1, 8, 8, 1], stride=[1, 8, 8, 1], padding='SAME', pooling_type='AVG')
                #DROPOUT 
                drop1 = inceprv2_builder.Dropout_layer(average_pooling)
                #OUTPUT
                output = inceprv2_builder.FC_layer(drop1, filters=self.kwargs['Classes'], readout=True)
                #LOGIT LOSS
                with tf.name_scope('Cross_entropy_loss'):
                    softmax_logit_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=output_placeholder, logits=output))

                #Adding collections to graph
                tf.add_to_collection(self.model_name + '_Endpoints', inception_A5)
                tf.add_to_collection(self.model_name + '_Endpoints', inception_B10)
                tf.add_to_collection(self.model_name + '_Endpoints', inception_C5)
                tf.add_to_collection(self.model_name + '_Input_ph', input_placeholder)
                tf.add_to_collection(self.model_name + '_Input_reshape', input_reshape)
                tf.add_to_collection(self.model_name + '_Output_ph', output_placeholder)
                tf.add_to_collection(self.model_name + '_Output', output)
                tf.add_to_collection(self.model_name + '_Dropout_prob_ph', dropout_prob_placeholder)
                tf.add_to_collection(self.model_name + '_State', state_placeholder)
                tf.add_to_collection(self.model_name + '_Loss', softmax_logit_loss)

                #Inception_resnetv2_dict = {'Input_ph': input_placeholder, 'Output_ph': output_placeholder, 'Output': output, 'Dropout_prob_ph': dropout_prob_placeholder, 'State' : state_placeholder}
                return dropout_prob_placeholder
                #return Inception_resnetv2_dict


    def Build_vgg19(self):
        with tf.name_scope('Vgg_model'):
            with Builder(**self.kwargs) as vgg19_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, self.kwargs['Image_width']*self.kwargs['Image_height']*self.kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, shape=[None, self.kwargs['Classes']], name='Output')
                dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                input_reshape = vgg19_builder.Reshape_input(input_placeholder, width=self.kwargs['Image_width'], height=self.kwargs['Image_height'], colorspace= self.kwargs['Image_cspace'])

                #Setting control params
                vgg19_builder.control_params(Dropout_control=dropout_prob_placeholder)

                #FEATURE EXTRACTION
                conv1a = vgg19_builder.Conv2d_layer(input_reshape, filters=64)
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

                #DENSELY CONNECTED
                fc1 = vgg19_builder.FC_layer(pool5, filters=4096)
                drop1 = vgg19_builder.Dropout_layer(fc1)

                fc2 = vgg19_builder.FC_layer(drop1, filters=4096)
                drop2 = vgg19_builder.Dropout_layer(fc2)

                output = vgg19_builder.FC_layer(drop2, filters=self.kwargs['Classes'], readout=True)

                VGG19_dict = {'Input_ph': input_placeholder, 'Output_ph': output_placeholder, 'Output': output, 'Dropout_prob_ph': dropout_prob_placeholder }
                return(VGG19_dict)



    def Build_vgg16(self):
        with tf.name_scope('Vgg_model'):
            with Builder(**self.kwargs) as vgg16_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, self.kwargs['Image_width']*self.kwargs['Image_height']*self.kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, shape=[None, self.kwargs['Classes']], name='Output')
                dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                input_reshape = vgg16_builder.Reshape_input(input_placeholder, width=self.kwargs['Image_width'], height=self.kwargs['Image_height'], colorspace= self.kwargs['Image_cspace'])

                #Setting control params
                vgg16_builder.control_params(Dropout_control=dropout_prob_placeholder)

                #FEATURE EXTRACTION
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

                #DENSELY CONNECTED
                fc1 = vgg16_builder.FC_layer(pool5, filters=4096)
                drop1 = vgg16_builder.Dropout_layer(fc1)

                fc2 = vgg16_builder.FC_layer(drop1, filters=4096)
                drop2 = vgg16_builder.Dropout_layer(fc2)

                output = vgg16_builder.FC_layer(drop2, filters=self.kwargs['Classes'], readout=True)

                VGG16_dict = {'Input_ph': input_placeholder, 'Output_ph': output_placeholder, 'Output': output, 'Dropout_prob_ph': dropout_prob_placeholder }
                return(VGG16_dict)



    def Build_Alexnet(self):
        with tf.name_scope('Alexnet_model'):
            with Builder(**self.kwargs) as alexnet_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, self.kwargs['Image_width']*self.kwargs['Image_height']*self.kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, shape=[None, self.kwargs['Classes']], name='Output')
                dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                state_placeholder = tf.placeholder(tf.bool, name="State")

                input_reshape = alexnet_builder.Reshape_input(input_placeholder, width=self.kwargs['Image_width'], height=self.kwargs['Image_height'], colorspace= self.kwargs['Image_cspace'])

                #Setting control params
                alexnet_builder.control_params(Dropout_control=dropout_prob_placeholder, State=state_placeholder)

                #FEATURE EXTRACTION
                conv1 = alexnet_builder.Conv2d_layer(input_reshape, stride=[1, 4, 4, 1], k_size=[11, 11], filters=96, padding='VALID', Batch_norm=True)
                
                pool1 = alexnet_builder.Pool_layer(conv1, k_size=[1, 3, 3, 1], padding='VALID')

                pad1 = alexnet_builder.Pad_layer(pool1, p_type='SYMMETRIC')
                conv2 = alexnet_builder.Conv2d_layer(pad1, k_size=[5, 5], filters=256, padding='VALID', Batch_norm=True)

                pool2 = alexnet_builder.Pool_layer(conv2, k_size=[1, 3, 3, 1], padding='VALID')

                conv3 = alexnet_builder.Conv2d_layer(pool2, filters=384, Batch_norm=True)
                conv4 = alexnet_builder.Conv2d_layer(conv3, filters=384, Batch_norm=True)
                conv5 = alexnet_builder.Conv2d_layer(conv4, filters=256, Batch_norm=True)

                pool5 = alexnet_builder.Pool_layer(conv5, k_size=[1, 3, 3, 1])

                #DENSELY CONNECTED
                fc1 = alexnet_builder.FC_layer(pool5, filters=4096)
                drop1 = alexnet_builder.Dropout_layer(fc1)

                fc2 = alexnet_builder.FC_layer(drop1, filters=4096)
                drop2 = alexnet_builder.Dropout_layer(fc2)

                output = alexnet_builder.FC_layer(drop2, filters=self.kwargs['Classes'], readout=True)
                #tf.summary.image('Inputs', input_reshape)
                #tf.summary.tensor_summary('Outputs', output)
                Alexnet_dict = {'Input_ph': input_placeholder, 'Output_ph': output_placeholder, 'Output': output, 'Dropout_prob_ph': dropout_prob_placeholder, 'State' : state_placeholder}
                return(Alexnet_dict)



    def Build_Lenet(self):
        with tf.name_scope('LeNeT_Model'):
            #with Builder(Summary=True,Batch_size=50,Image_width=28,Image_height=28,Image_cspace=1) as lenet_builder:
            with Builder(**self.kwargs) as lenet_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, self.kwargs['Image_width']*self.kwargs['Image_height']*self.kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, shape=[None, self.kwargs['Classes']], name='Output')
                input_reshape = lenet_builder.Reshape_input(input_placeholder)
                
                conv1 = lenet_builder.Conv2d_layer(input_reshape, k_size=[5, 5])
                pool1 = lenet_builder.Pool_layer(conv1)

                conv2 = lenet_builder.Conv2d_layer(pool1, k_size=[5, 5], filters=64)
                pool2 = lenet_builder.Pool_layer(conv2)

                fc1 = lenet_builder.FC_layer(pool2);
                output = lenet_builder.FC_layer(fc1, filters=self.kwargs['Classes'], readout=True)

                Lenet_dict = {'Input_ph': input_placeholder, 'Output_ph': output_placeholder, 'Output': output}
                return(Lenet_dict)



    def __init__(self, **kwargs):
        #TODO: WRITE ERROR HANDLER AND PARSER 
        self.model_name = kwargs['Model_name']
        self.summary = kwargs['Summary']
        self.kwargs = kwargs
        #Add more params as required