from utils.builder import Builder
import tensorflow as tf

def Build_Unet1024(kwargs):
        '''Add paper and brief description'''
        with tf.name_scope('Unet1024'):
            with Builder(**kwargs) as unet_res_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, kwargs['Image_width']*kwargs['Image_height']*kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, kwargs['Image_width']*kwargs['Image_height']], name='Mask')
                weight_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, kwargs['Image_width']*kwargs['Image_height']], name='Weight')
                dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                state_placeholder = tf.placeholder(tf.string, name="State")
                input_reshape = unet_res_builder.Reshape_input(input_placeholder, \
                    width=kwargs['Image_width'], height=kwargs['Image_height'], colorspace= kwargs['Image_cspace'])
                #batch_size = tf.slice(tf.shape(input_placeholder),[0],[1])
                #Setting control params
                unet_res_builder.control_params(Dropout_control=dropout_prob_placeholder, State=state_placeholder)

                def stack_encoder(input, out_filters):
                    with tf.name_scope('Encoder'):
                        input = unet_res_builder.Relu(input)

                        #conv1a_split1 = unet_res_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=out_filters, Activation=False, Batch_norm=True)

                        conv1 = unet_res_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[3, 3], filters=out_filters, Batch_norm=True)
                        conv2 = unet_res_builder.Conv2d_layer(conv1, stride=[1, 1, 1, 1], k_size=[3, 3], filters=out_filters, Batch_norm=True)

                        #res_connect = unet_res_builder.Residual_connect([conv1a_split1, conv2b_split1])

                        return conv2
                def stack_decoder(input, encoder_connect, out_filters, output_shape, infilter=None):
                    with tf.name_scope('Decoder'):
                        encoder_connect_shape = encoder_connect.get_shape().as_list()
                        del encoder_connect_shape[0]
                        res_filters = encoder_connect_shape.pop(2)

                        if infilter is not None:
                            res_filters=infilter
                        #upscale_input = unet_res_builder.Upconv_layer(input, stride=[1, 2, 2, 1], filters=res_filters, Batch_norm=True, output_shape=output_shape) #change_filters to match encoder_connect filters
                        upscale_input = unet_res_builder.Conv_Resize_layer(input, stride=[1,2,2,1], filters=res_filters, Batch_norm=True)
                        uconnect = unet_res_builder.Concat([encoder_connect, upscale_input])
                        conv1 = unet_res_builder.Conv2d_layer(uconnect, stride=[1, 1, 1, 1], k_size=[3, 3], filters=out_filters, Batch_norm=True)
                        conv2 = unet_res_builder.Conv2d_layer(conv1, stride=[1, 1, 1, 1], k_size=[3, 3], filters=out_filters, Batch_norm=True)
                        conv3 = unet_res_builder.Conv2d_layer(conv2, stride=[1, 1, 1, 1], k_size=[3, 3], filters=out_filters, Batch_norm=True)
                        return conv3
                        '''
                        u_connect = unet_res_builder.Concat([encoder_connect, upscale_input])
                        conv1 = unet_res_builder.Conv2d_layer(u_connect, stride=[1, 1, 1, 1], k_size=[1, 1], filters=out_filters, Batch_norm=True)


                        conv1a_split1 = unet_res_builder.Conv2d_layer(conv1, stride=[1, 1, 1, 1], k_size=[1, 1], filters=out_filters, Activation=False, Batch_norm=True)

                        conv1b_split1 = unet_res_builder.Conv2d_layer(conv1, stride=[1, 1, 1, 1], k_size=[3, 3], filters=out_filters, Batch_norm=True)
                        conv2b_split1 = unet_res_builder.Conv2d_layer(conv1b_split1, stride=[1, 1, 1, 1], k_size=[3, 3], filters=out_filters, Activation=False, Batch_norm=True)

                        res_connect = unet_res_builder.Residual_connect([conv1a_split1, conv2b_split1])

                        return res_connect
                        '''
                def Center_pool(input, filters=768):
                    ''' Dense dialations '''
                    with tf.name_scope('Dense_Dialated_Center'):
                        Dconv1 = unet_res_builder.DConv_layer(input, filters=filters, Batch_norm=True, D_rate=1, Activation=False)
                        Dense_connect1 = unet_res_builder.Residual_connect([input, Dconv1])

                        Dconv2 = unet_res_builder.DConv_layer(Dense_connect1, filters=filters, Batch_norm=True, D_rate=2, Activation=False)
                        Dense_connect2 = unet_res_builder.Residual_connect([input, Dconv1, Dconv2])

                        Dconv4 = unet_res_builder.DConv_layer(Dense_connect2, filters=filters, Batch_norm=True, D_rate=4, Activation=False)
                        Dense_connect3 = unet_res_builder.Residual_connect([input, Dconv1, Dconv2, Dconv4 ])

                        Dconv8 = unet_res_builder.DConv_layer(Dense_connect3, filters=filters, Batch_norm=True, D_rate=8, Activation=False)
                        Dense_connect4 = unet_res_builder.Residual_connect([input, Dconv1, Dconv2, Dconv4, Dconv8])

                        Dconv16 = unet_res_builder.DConv_layer(Dense_connect4, filters=filters, Batch_norm=True, D_rate=16, Activation=False)
                        Dense_connect5 = unet_res_builder.Residual_connect([input, Dconv1, Dconv2, Dconv4, Dconv8, Dconv16])

                        Dconv32 = unet_res_builder.DConv_layer(Dense_connect5, filters=filters, Batch_norm=True, D_rate=32, Activation=False)
                        Dense_connect6 = unet_res_builder.Residual_connect([input, Dconv1, Dconv2, Dconv4, Dconv8, Dconv16, Dconv32])

                        Scale_output = unet_res_builder.Scale_activations(Dense_connect6,scaling_factor=0.9)

                        return Scale_output
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

                Encoder7 = stack_encoder(Pool6, 768)
                Pool7 = unet_res_builder.Pool_layer(Encoder7) #8

                #Center
                #Conv_center = unet_res_builder.Conv2d_layer(Pool7, stride=[1, 1, 1, 1], filters=768, Batch_norm=True, padding='SAME')
                Conv_center = Center_pool(Pool7)
                #Pool_center = unet_res_builder.Pool_layer(Conv_center) #8
                #Build Decoder
                Decode1 = stack_decoder(Conv_center, Encoder7, out_filters=768, output_shape=[16, 16])
                Decode2 = stack_decoder(Decode1, Encoder6, out_filters=768, output_shape=[32, 32])
                Decode3 = stack_decoder(Decode2, Encoder5, out_filters=512, output_shape=[64, 64], infilter=768)
                Decode4 = stack_decoder(Decode3, Encoder4, out_filters=256, output_shape=[128, 128], infilter=512)
                Decode5 = stack_decoder(Decode4, Encoder3, out_filters=128, output_shape=[256, 256], infilter=256)
                Decode6 = stack_decoder(Decode5, Encoder2, out_filters=64, output_shape=[512,512],  infilter=128)
                Decode7 = stack_decoder(Decode6, Encoder1, out_filters=24, output_shape=[1024,1024], infilter=64)
                output = unet_res_builder.Conv2d_layer(Decode7, stride=[1, 1, 1, 1], filters=1, Batch_norm=True, k_size=[1, 1], Activation=False) #output
                logits = tf.reshape(output, shape= [-1, kwargs['Image_width']*kwargs['Image_height']])
                '''
                Encoder1 = stack_encoder(input_reshape, 128)
                Pool1 = unet_res_builder.Pool_layer(Encoder1) #64

                Encoder2 = stack_encoder(Pool1, 256)
                Pool2 = unet_res_builder.Pool_layer(Encoder2) #32

                Encoder3 = stack_encoder(Pool2, 512)
                Pool3 = unet_res_builder.Pool_layer(Encoder3) #16

                Encoder4 = stack_encoder(Pool3, 1024)
                Pool4 = unet_res_builder.Pool_layer(Encoder4) #8

                Conv_center = unet_res_builder.Conv2d_layer(Pool4, stride=[1, 1, 1, 1], filters=1024, Batch_norm=True, padding='SAME')

                Decode1 = stack_decoder(Conv_center, Encoder4, out_filters=512, output_shape=[16, 16])
                Decode2 = stack_decoder(Decode1, Encoder3, out_filters=256, output_shape=[32, 32])
                Decode3 = stack_decoder(Decode2, Encoder2, out_filters=128, output_shape=[64, 64])
                Decode4 = stack_decoder(Decode3, Encoder1, out_filters=64, output_shape=[128, 128])

                output = unet_res_builder.Conv2d_layer(Decode4, stride=[1, 1, 1, 1], filters=1, Batch_norm=True, k_size=[1, 1]) #output
                unet_res_builder.variable_summaries(output, name='output')
                unet_res_builder.variable_summaries(input_placeholder, name='input')
                '''
                #Add loss and debug
                with tf.name_scope('BCE_Loss'):
                    offset = 1e-5
                    Threshold = 0.5
                    Probs = tf.nn.sigmoid(logits)
                    Probs_processed = tf.clip_by_value(Probs, offset, 1.0)
                    Con_Probs_processed = tf.clip_by_value(1-Probs, offset, 1.0)
                    W_I = (-output_placeholder * tf.log(Probs_processed) - (1-output_placeholder)*tf.log(Con_Probs_processed))
                    Weighted_BCE_loss = tf.reduce_sum(W_I) / tf.cast(tf.maximum(tf.count_nonzero(W_I -0.01),0), tf.float32)

                #Dice Loss
                
                with tf.name_scope('Dice_Loss'):
                    eps = tf.constant(value=1e-5, name='eps')
                    sigmoid = tf.nn.sigmoid(logits,name='sigmoid') + eps
                    intersection =tf.reduce_sum(sigmoid * output_placeholder,axis=1,name='intersection')
                    union = eps + tf.reduce_sum(sigmoid,1,name='reduce_sigmoid') + (tf.reduce_sum(output_placeholder,1,name='reduce_mask'))
                    Dice_loss = 2 * intersection / (union)
                    Dice_loss = 1 - tf.reduce_mean(Dice_loss, name='diceloss')
                    unet_res_builder.variable_summaries(sigmoid, name='logits')
                    Jacard_loss = intersection / (eps + tf.reduce_sum(sigmoid,1,name='reduce_sigmoid') + (tf.reduce_sum(output_placeholder,1,name='reduce_mask')) - intersection)
                    Jacard_loss = 1- tf.reduce_mean(Jacard_loss, name='jclosss')
                #Graph Exports
                tf.add_to_collection(kwargs['Model_name'] + '_Input_ph', input_placeholder)
                tf.add_to_collection(kwargs['Model_name'] + '_Input_reshape', input_reshape)
                tf.add_to_collection(kwargs['Model_name'] + '_Weight_ph', weight_placeholder)
                tf.add_to_collection(kwargs['Model_name'] + '_Output_ph', output_placeholder)
                tf.add_to_collection(kwargs['Model_name'] + '_Output', output)
                tf.add_to_collection(kwargs['Model_name'] + '_Dropout_prob_ph', dropout_prob_placeholder)
                tf.add_to_collection(kwargs['Model_name'] + '_State', state_placeholder)
                tf.add_to_collection(kwargs['Model_name'] + '_Loss', Weighted_BCE_loss)
                tf.add_to_collection(kwargs['Model_name'] + '_Loss', Dice_loss)
                #tf.add_to_collection(kwargs['Model_name'] + '_Loss', Jacard_loss)

                if kwargs['Summary']:
                    unet_res_builder.variable_summaries(sigmoid, name='logits')
                    #tf.add_to_collection(kwargs['Model_name'] + '_Loss', final_focal_loss)
                    tf.summary.scalar('WBCE loss', Weighted_BCE_loss)
                    tf.summary.image('WCBE', tf.reshape(W_I, [-1,kwargs['Image_width'], kwargs['Image_height'], 1]))
                    #tf.summary.scalar('Count WCBE loss', W_I_count)
                    #tf.summary.scalar('WCBE losssum ', tf.reduce_sum(W_I))
                    tf.summary.scalar('Dice loss', Dice_loss)
                    tf.summary.scalar('JC loss', Jacard_loss)
                    #tf.summary.scalar('Focal loss', final_focal_loss)
                return 'Segmentation'
                return 'Segmentation'

