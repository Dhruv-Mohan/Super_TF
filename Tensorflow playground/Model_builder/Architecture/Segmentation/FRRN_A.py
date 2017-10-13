from utils.builder import Builder
import tensorflow as tf

def Build_FRRN_A():
        with tf.name_scope('FRRN_A'):
            with Builder(**self.kwargs) as frnn_a_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, self.kwargs['Image_width']*self.kwargs['Image_height']*self.kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, self.kwargs['Image_width']*self.kwargs['Image_height']], name='Mask')
                weight_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, self.kwargs['Image_width']*self.kwargs['Image_height']], name='Weight')
                dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                state_placeholder = tf.placeholder(tf.string, name="State")
                input_reshape = frnn_a_builder.Reshape_input(input_placeholder, \
                    width=self.kwargs['Image_width'], height=self.kwargs['Image_height'], colorspace= self.kwargs['Image_cspace'])

                #Setting control params
                frnn_a_builder.control_params(Dropout_control=dropout_prob_placeholder, State=state_placeholder)

                #Construct functional building blocks
                def RU(input, filters):
                    with tf.name_scope('Residual_Unit'):
                        Conv1 = frnn_a_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], filters=filters, Batch_norm=True)
                        Conv2 = frnn_a_builder.Conv2d_layer(Conv1, stride=[1, 1, 1, 1], filters=filters, Batch_norm=True)
                        Conv3 = frnn_a_builder.Conv2d_layer(Conv2, k_size=[1, 1], stride=[1, 1, 1, 1], filters=filters, Activation=False)

                        return frnn_a_builder.Residual_connect([input, Conv3])

                def FRRU(Residual_stream, Pooling_stream, scale_factor, filters, res_filters=32):
                    with tf.name_scope('Full_Resolution_Unit'):
                        scale_dims = [1, scale_factor, scale_factor, 1]
                        Pool, Ind = frnn_a_builder.Pool_layer(Residual_stream, k_size=scale_dims, stride=scale_dims, pooling_type='MAXIND')

                        Concat = frnn_a_builder.Concat([Pool, Pooling_stream])

                        #Conv0 = frnn_a_builder.Conv2d_layer(Concat, stride=[1,1,1,1], k_size=[1,1], filters=filters, Batch_norm=True)
                        Conv1 = frnn_a_builder.Conv2d_layer(Concat, stride=[1, 1, 1, 1], filters=filters, Batch_norm=True)
                        Conv2 = frnn_a_builder.Conv2d_layer(Conv1, stride=[1, 1, 1, 1], filters=filters, Batch_norm=True)

                        #Res_connect = frnn_a_builder.Residual_connect([Conv0, Conv2])
                        Conv3 = frnn_a_builder.Conv2d_layer(Conv2, k_size=[1, 1], stride=[1, 1, 1, 1], filters=res_filters, Activation=False)

                        Unpool = frnn_a_builder.Unpool_layer(Conv3, Ind, k_size = scale_dims)
                    Residual_stream_out = frnn_a_builder.Residual_connect([Residual_stream, Unpool])
                    Pooling_stream_out = Conv2

                    return Residual_stream_out, Pooling_stream_out
                    #return Conv2

                #Model Construction
                Stem = frnn_a_builder.Conv2d_layer(input_reshape, stride=[1, 1, 1, 1], k_size=[5, 5], filters=48, Batch_norm=True)
                Stem = RU(Stem, 48)
                Stem_pool = frnn_a_builder.Pool_layer(Stem)
                
                Stem_pool = RU(Stem_pool, 48)
                Stem_pool = RU(Stem_pool, 48)

                Residual_stream = frnn_a_builder.Conv2d_layer(Stem_pool, stride=[1, 1, 1, 1], k_size=[1, 1], filters=32, Batch_norm=True)
                Pooling_stream, ind1 = frnn_a_builder.Pool_layer(Stem_pool, pooling_type='MAXIND')

                #Encoder
                scale_factor = 2
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=96)
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=96)
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=96)

                Pooling_stream, ind2 = frnn_a_builder.Pool_layer(Pooling_stream, pooling_type='MAXIND')

                scale_factor = 4
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=192)
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=192)
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=192)
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=192)

                Pooling_stream, ind3 = frnn_a_builder.Pool_layer(Pooling_stream, pooling_type='MAXIND')

                scale_factor=8
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=192)
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=192)

                Pooling_stream, ind4 = frnn_a_builder.Pool_layer(Pooling_stream, pooling_type='MAXIND')

                scale_factor=16
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=384)
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=384)

                Pooling_stream, ind5 = frnn_a_builder.Pool_layer(Pooling_stream, pooling_type='MAXIND')

                scale_factor=32
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=192)
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=384)
                Pooling_stream, ind6 = frnn_a_builder.Pool_layer(Pooling_stream, pooling_type='MAXIND')

                scale_factor=64
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=192)
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=384)
                #Decoder
                Pooling_stream = frnn_a_builder.Unpool_layer(Pooling_stream, ind6)
                scale_factor = 32
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=192)
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=384)
                Pooling_stream = frnn_a_builder.Unpool_layer(Pooling_stream, ind5)
                scale_factor = 16
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=192)
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=192)


                Pooling_stream = frnn_a_builder.Unpool_layer(Pooling_stream, ind4)

                scale_factor = 8
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=192)
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=192)

                
                Pooling_stream = frnn_a_builder.Unpool_layer(Pooling_stream, ind3)

                scale_factor = 4
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=192)
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=192)

                Pooling_stream = frnn_a_builder.Conv2d_layer(Pooling_stream, stride=[1, 1, 1, 1], k_size=[1, 1], filters=96, Batch_norm=True)
                Pooling_stream = frnn_a_builder.Unpool_layer(Pooling_stream, ind2)

                scale_factor = 2
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=96)
                Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=96)

                Pooling_stream = frnn_a_builder.Conv2d_layer(Pooling_stream, stride=[1, 1, 1, 1], k_size=[1, 1], filters=48, Batch_norm=True)
                Pooling_stream = frnn_a_builder.Unpool_layer(Pooling_stream, ind1)

                RP_stream_merge = frnn_a_builder.Concat([Pooling_stream, Residual_stream])
                Conv3 = frnn_a_builder.Conv2d_layer(RP_stream_merge, stride=[1, 1, 1, 1], k_size=[1, 1], filters=48, Batch_norm=True)
                
                Conv3 = RU(Conv3, 48)
                Conv3 = RU(Conv3, 48)


                
                Upconv = frnn_a_builder.Upconv_layer(Conv3, stride=[1, 2, 2, 1], filters=48, Batch_norm=True, output_shape=[self.kwargs['Image_width'], self.kwargs['Image_height']])
                Res_connect = frnn_a_builder.Residual_connect([Stem, Upconv])
                Res_connect = RU(Res_connect, 48)
                output = frnn_a_builder.Conv2d_layer(Res_connect, filters=1, stride=[1, 1, 1, 1], k_size=[1, 1], Batch_norm=True, Activation=False)

                #Add loss and debug
                with tf.name_scope('BCE_Loss'):
                    weights = tf.reshape(weight_placeholder, shape=[-1, self.kwargs['Image_width']*self.kwargs['Image_height']])
                    w2 = weights
                    print(self.kwargs['Image_width']*self.kwargs['Image_height'])
                    logits = tf.reshape(output, shape= [-1, self.kwargs['Image_width']*self.kwargs['Image_height']])
                    P = tf.minimum(tf.nn.sigmoid(logits)+1e-4,1.0) #safe for log sigmoid
                    F1= -output_placeholder*tf.pow(1-P,2)*tf.log(P) -(1-output_placeholder)*tf.pow(P,2)*tf.log(1-P+1e-4)
                    tf.summary.image('FOCAL Loss', tf.reshape(F1,[1, 1024, 1024, 1]))
                    F1_count = tf.count_nonzero(tf.maximum(F1-0.1,0))
                    final_focal_loss = tf.multiply(tf.reduce_sum(F1)/ tf.to_float(F1_count), 0.1)
                    tf.summary.scalar('Count focal loss', F1_count)
                    tf.summary.scalar('Focal losssum ', tf.reduce_sum(F1))
                    #focal_loss = tf.multiply(tf.multiply(Y, tf.square(1 - P)),L) + tf.multiply(tf.multiply(1-Y, tf.square(P)),max_x+L)
                    #final_focal_loss = tf.reduce_mean(focal_loss)
                    #eps = tf.constant(value=1e-5)
                    #sigmoid = tf.nn.sigmoid(logits) + eps
                    W_I = tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=output_placeholder),w2)
                    tf.summary.image('WCBE', tf.reshape(W_I, [1, 1024, 1024, 1]))
                    W_I_count = tf.count_nonzero(tf.maximum(W_I-0.1,0))
                    W_Is = tf.reduce_sum(W_I) / tf.to_float(W_I_count)
                    Weighted_BCE_loss = tf.multiply(W_Is,0.1) #0.8
                    tf.summary.scalar('Count WCBE loss', W_I_count)
                    tf.summary.scalar('WCBE losssum ', tf.reduce_sum(W_I))
                    #Weighted_BCE_loss = tf.reduce_mean(output_placeholder * tf.log(sigmoid)) #Fix output and weight shape
                    #Weighted_BCE_loss = tf.multiply(BCE_loss, weight_placeholder) + tf.multiply(tf.clip_by_value(logits, 0, 1e4), weight_placeholder)
                    #Weighted_BCE_loss = tf.reduce_mean(Weighted_BCE_loss)

                #Dice Loss
                
                with tf.name_scope('Dice_Loss'):

                    eps = tf.constant(value=1e-5, name='eps')
                    sigmoid = tf.nn.sigmoid(logits,name='sigmoid') + eps
                    intersection =tf.reduce_sum(sigmoid * output_placeholder,axis=1,name='intersection')
                    union = tf.reduce_sum(sigmoid,1,name='reduce_sigmoid') + tf.reduce_sum(output_placeholder,1,name='reduce_mask') + 1e-5
                    Dice_loss = 2 * intersection / (union)
                    Dice_loss = 1 - tf.reduce_mean(Dice_loss,name='diceloss')
                    frnn_a_builder.variable_summaries(sigmoid, name='logits')
                
                #Graph Exports
                tf.add_to_collection(self.model_name + '_Input_ph', input_placeholder)
                tf.add_to_collection(self.model_name + '_Input_reshape', input_reshape)
                tf.add_to_collection(self.model_name + '_Weight_ph', weight_placeholder)
                tf.add_to_collection(self.model_name + '_Output_ph', output_placeholder)
                tf.add_to_collection(self.model_name + '_Output', output)
                tf.add_to_collection(self.model_name + '_Dropout_prob_ph', dropout_prob_placeholder)
                tf.add_to_collection(self.model_name + '_State', state_placeholder)
                tf.add_to_collection(self.model_name + '_Loss', Weighted_BCE_loss)
                tf.add_to_collection(self.model_name + '_Loss', Dice_loss)
                tf.add_to_collection(self.model_name + '_Loss', final_focal_loss)
                tf.summary.scalar('WBCE loss', Weighted_BCE_loss)
                tf.summary.scalar('Dice loss', Dice_loss)
                tf.summary.scalar('Focal loss', final_focal_loss)
                return 'Segmentation'

