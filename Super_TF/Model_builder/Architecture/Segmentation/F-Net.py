from utils.builder import Builder
import tensorflow as tf

def Build_FRRN_D(kwargs):
        with tf.name_scope('F-Net'):
            with Builder(**kwargs) as frnn_c_builder:
                frnn_c_builder.BNscope=50
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, kwargs['Image_width']*kwargs['Image_height']*kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, kwargs['Image_width']*kwargs['Image_height']], name='Mask')
                weight_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, kwargs['Image_width']*kwargs['Image_height']], name='Weight')
                dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                state_placeholder = tf.placeholder(tf.string, name="State")
                input_reshape = frnn_c_builder.Reshape_input(input_placeholder, \
                    width=kwargs['Image_width'], height=kwargs['Image_height'], colorspace= kwargs['Image_cspace'])
                #Build P-Net output + Input
                prior_image_path = tf.placeholder(tf.string)
                prior_image = tf.image.decode_image(tf.read_file(prior_image_path)) 
                prior_image.set_shape([900, 900, 1])
                prior_image =  tf.image.convert_image_dtype( tf.image.resize_images(prior_image, size=[kwargs['Image_height'],kwargs['Image_width']]), tf.float32) 
                prior_image = tf.stack([prior_image,prior_image,prior_image], axis=2)
                prior_image= tf.multiply(prior_image,0.0001)
                prior_image=tf.expand_dims(tf.squeeze( prior_image),0)
                input_reshape = input_reshape+prior_image

                frnn_c_builder.control_params(Dropout_control=dropout_prob_placeholder, State=state_placeholder)

                #Construct functional building blocks
                def RU(input, filters):
                    with tf.name_scope('Residual_Unit'):
                        Conv1 = frnn_c_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], filters=filters, Batch_norm=True)
                        Conv2 = frnn_c_builder.Conv2d_layer(Conv1, stride=[1, 1, 1, 1], filters=filters, Batch_norm=True,Activation=False )


                        return frnn_c_builder.Residual_connect([input, Conv2])

                def FRRU(Residual_stream, Pooling_stream, scale_factor, filters, res_filters=32, D_rate=1, k_size=[3,3]):
                    with tf.name_scope('Full_Resolution_Unit'):
                        scale_dims = [1, scale_factor, scale_factor, 1]
                        Pool = frnn_c_builder.Pool_layer(Residual_stream, k_size=scale_dims, stride=scale_dims)

                        Concat = frnn_c_builder.Concat([Pool, Pooling_stream])
                        Conv1 = frnn_c_builder.DConv_layer(Concat,  filters=filters, Batch_norm=True, D_rate=D_rate)
                        Conv2 = frnn_c_builder.DConv_layer(Conv1, filters=filters, Batch_norm=True, D_rate=D_rate)
                        Conv3 = frnn_c_builder.Conv2d_layer(Conv2, k_size=[1, 1], stride=[1, 1, 1, 1], filters=res_filters, Activation=False, Batch_norm=True)

                        Unpool = frnn_c_builder.Conv_Resize_layer(Conv3, k_size=k_size, output_scale=scale_factor, Batch_norm=False )

                    Residual_stream_out = frnn_c_builder.Residual_connect([Residual_stream, Unpool], Activation=False)
                    Pooling_stream_out = Conv2
                    return Residual_stream_out, Pooling_stream_out


                #Model Construction
                with tf.name_scope('F-Net'):
                    Stem = frnn_c_builder.Conv2d_layer(input_reshape, stride=[1, 1, 1, 1], k_size=[3, 3], filters=64, Batch_norm=True)
                    Stem = RU(Stem, 64)
                    
                    Residual_stream = frnn_c_builder.Conv2d_layer(Stem, stride=[1, 1, 1, 1], k_size=[1, 1], filters=32, Batch_norm=True)
                    Stem_pool = frnn_c_builder.Pool_layer(Stem)
                    Stem_pool = RU(Stem_pool, 64)
                    Pooling_stream = frnn_c_builder.Pool_layer(Stem_pool)

                    #Encoder
                    scale_factor = 4
                    Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=64)
                    Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=64, D_rate=2)
                    Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=64, D_rate=6)
                    Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=128, D_rate=12)
                    Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=128, D_rate=18)
                    Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=256, D_rate=24)
                    Residual_stream, Pooling_stream = FRRU(Residual_stream=Residual_stream, Pooling_stream=Pooling_stream, scale_factor=scale_factor, filters=256, D_rate=32)
                    Pooling_stream = frnn_c_builder.Conv_Resize_layer(Pooling_stream, k_size=[3, 3], Batch_norm=True, filters=64)
                    Pooling_stream = RU(Pooling_stream, 64)
                    Pooling_stream = frnn_c_builder.Conv_Resize_layer(Pooling_stream, k_size=[3, 3], Batch_norm=True)
                    RP_stream_merge = frnn_c_builder.Concat([Pooling_stream, Residual_stream])
                    Conv3 = frnn_c_builder.Conv2d_layer(RP_stream_merge, stride=[1, 1, 1, 1], k_size=[1, 1], filters=64, Batch_norm=True)
                    Res_connect = frnn_c_builder.Residual_connect([Stem, Conv3])
                    Res_connect = RU(Res_connect, 64)
                    
                output = frnn_c_builder.Conv2d_layer(Res_connect, filters=1, stride=[1, 1, 1, 1], k_size=[1, 1], Batch_norm=False, Activation=False)
                weights = tf.reshape(weight_placeholder, shape=[-1, kwargs['Image_width']*kwargs['Image_height']])
                logits = tf.reshape(output, shape= [-1, kwargs['Image_width']*kwargs['Image_height']])

                #Add loss and debug
                '''
                with tf.name_scope('Focal_Loss'):
                    
                    P = tf.minimum(tf.nn.sigmoid(logits)+1e-4,1.0) #safe for log sigmoid
                    F1= -output_placeholder*tf.pow(1-P,5)*tf.log(P) -(1-output_placeholder)*tf.pow(P,5)*tf.log(1-P+1e-4)
                    tf.summary.image('FOCAL Loss', tf.reshape(F1,[1, 1024, 1024, 1]))
                    F1_count = tf.count_nonzero(tf.maximum(F1-0.05,0))
                    #final_focal_loss = tf.multiply(tf.reduce_mean(F1),1)
                    final_focal_loss = tf.multiply(tf.reduce_sum(F1)/ tf.to_float(tf.maximum(F1_count, 1024*5)), 1)
                    tf.summary.scalar('Count focal loss', F1_count)
                    tf.summary.scalar('Focal losssum ', tf.reduce_sum(F1))
                    #focal_loss = tf.multiply(tf.multiply(Y, tf.square(1 - P)),L) + tf.multiply(tf.multiply(1-Y, tf.square(P)),max_x+L)
                    #final_focal_loss = tf.reduce_mean(focal_loss)
                    #eps = tf.constant(value=1e-5)
                    #sigmoid = tf.nn.sigmoid(logits) + eps
                '''
                
                with tf.name_scope('BCE_Loss'):
                    offset = 1e-5
                    Threshold = 0.8
                    Probs = tf.nn.sigmoid(logits)
                    Wmask = 1-Probs+Threshold
                    Wmask_con =tf.floor( 1-Wmask) *(1- output_placeholder)
                    Wmask = tf.floor(Wmask) *output_placeholder
                    Probs_processed = tf.clip_by_value(Probs, offset, 1.0)
                    Con_Probs_processed = tf.clip_by_value(1-Probs, offset, 1.0)
                    W_I = (-Wmask * tf.log(Probs_processed) - (1-output_placeholder)*tf.log(Con_Probs_processed))
                    Weighted_BCE_loss = tf.reduce_sum(W_I) /( tf.reduce_sum(Wmask)+100)
                    EU_loss =tf.losses.huber_loss(output_placeholder, Probs)

                #Dice Loss
                with tf.name_scope('Dice_Loss'):

                    eps = tf.constant(value=1e-5, name='eps')
                    sigmoid = tf.nn.sigmoid(output,name='sigmoid') + eps
                    sigmoid =tf.reshape(sigmoid, shape= [-1, kwargs['Image_width']*kwargs['Image_height']])
                    intersection =sigmoid * output_placeholder 
                    union = tf.reduce_sum(intersection,1) / ( tf.reduce_sum(sigmoid , 1, name='reduce_sigmoid') + tf.reduce_sum(output_placeholder ,1,name='reduce_mask') + 1e-5)
                    Dice_loss = 2. *  (union)
                    Dice_loss = 1 - tf.reduce_mean(Dice_loss,name='diceloss')
                    
                
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
                tf.add_to_collection(kwargs['Model_name'] + '_Loss', EU_loss)
                tf.add_to_collection(kwargs['Model_name'] +'_Prior_path', prior_image_path)

                
                #Graph Summaries
                if kwargs['Summary']:
                    tf.summary.image('PI', prior_image)
                    #frnn_c_builder.variable_summaries(sigmoid, name='logits')
                    #tf.add_to_collection(kwargs['Model_name'] + '_Loss', final_focal_loss)
                    tf.summary.scalar('WBCE loss', Weighted_BCE_loss)
                    tf.summary.image('WCBE', tf.reshape(W_I, [1,kwargs['Image_width'], kwargs['Image_height'], 1]))
                    tf.summary.image('mask', tf.reshape(Wmask, [1,kwargs['Image_width'], kwargs['Image_height'], 1]))
                    #tf.summary.scalar('Count WCBE loss', W_I_count)
                    #tf.summary.scalar('WCBE losssum ', tf.reduce_sum(W_I))
                    #tf.summary.scalar('Dice loss', Dice_loss)
                    #tf.summary.scalar('Focal loss', final_focal_loss)
                return 'Segmentation'




