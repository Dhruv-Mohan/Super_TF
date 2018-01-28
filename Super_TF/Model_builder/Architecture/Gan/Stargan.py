from utils.builder import Builder
import tensorflow as tf


def Build_Stargan(kwargs):
    '''
    Builds stargan, generator and decoder
    '''
    with tf.name_scope('Stargan_generator'):
        with Builder(**kwargs) as stargen_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, kwargs['Image_width']*kwargs['Image_height']*kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, kwargs['Image_width']*kwargs['Image_height']], name='Mask')
                weight_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, kwargs['Image_width']*kwargs['Image_height']], name='Weight')
                dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                state_placeholder = tf.placeholder(tf.string, name="State")
                input_reshape = stargen_builder.Reshape_input(input_placeholder, \
                    width=kwargs['Image_width'], height=kwargs['Image_height'], colorspace= kwargs['Image_cspace'])
                stargen_builder.control_params(Dropout_control=dropout_prob_placeholder, State=state_placeholder)
                
                def residual_unit(input, filters=256):
                    Conv_rl1 = stargen_builder.Conv2d_layer(input, k_size=[3, 3], stride=[1, 2, 2, 1], Batch_norm=True, filters=filters)
                    Conv_rl2 = stargen_builder.Conv2d_layer(Conv_rl1, k_size=[3, 3], stride=[1, 2, 2, 1], Batch_norm=True, filters=filters, Activation=False)
                    Res_connect = stargen_builder.Residual_connect([input, Conv_rl2])
                    return Res_connect

                with tf.name_scope('Downsample'):
                    Conv1 = stargen_builder.Conv2d_layer(input_reshape, k_size=[7, 7], Batch_norm=True, filters=64)
                    Conv2 = stargen_builder.Conv2d_layer(Conv1, k_size=[4, 4], stride=[1, 2, 2, 1], Batch_norm=True, filters=128, padding='VALID')
                    Conv3 = stargen_builder.Conv2d_layer(Conv2, k_size=[4, 4], stride=[1, 2, 2, 1], Batch_norm=True, filters=256, padding='VALID')


                with tf.name_scope('Bottleneck'):
                    Res_bl1 = residual_unit(Conv3)
                    Res_bl2 = residual_unit(Res_bl1)
                    Res_bl3 = residual_unit(Res_bl2)

                with tf.name_scope('Upsample'):
                    Upconv1 = stargen_builder.Conv_Resize_layer(Res_bl3, filters=128, Batch_norm=True, Activation=True)
                    Upconv2 = stargen_builder.Conv_Resize_layer(Upconv2, filters=64, Batch_norm=True, Activation=True)
                    Final_conv = stargen_builder.Conv2d_layer(Conv2, k_size=[7, 7], filters=3, Activation=False)
                    Output = stargen_builder.Activation(Final_conv, Type='TANH')


        with Builder(**kwargs) as stardis_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, kwargs['Image_width']*kwargs['Image_height']*kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, kwargs['Image_width']*kwargs['Image_height']], name='Mask')
                weight_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, kwargs['Image_width']*kwargs['Image_height']], name='Weight')
                dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                state_placeholder = tf.placeholder(tf.string, name="State")
                input_reshape = stardis_builder.Reshape_input(input_placeholder, \
                    width=kwargs['Image_width'], height=kwargs['Image_height'], colorspace= kwargs['Image_cspace'])
                stardis_builder.control_params(Dropout_control=dropout_prob_placeholder, State=state_placeholder)

                Dis_conv1 = stardis_builder.Conv2d_layer(input_reshape, k_size=[4, 4], stride=[1, 2, 2, 1], filters=64, padding='VALID')
                Dis_conv2 = stardis_builder.Conv2d_layer(Dis_conv1, k_size=[4, 4], stride=[1, 2, 2, 1], filters=128, padding='VALID')
                Dis_conv3 = stardis_builder.Conv2d_layer(Dis_conv2, k_size=[4, 4], stride=[1, 2, 2, 1], filters=256, padding='VALID')
                Dis_conv4 = stardis_builder.Conv2d_layer(Dis_conv3, k_size=[4, 4], stride=[1, 2, 2, 1], filters=512, padding='VALID')
                Dis_conv5 = stardis_builder.Conv2d_layer(Dis_conv4, k_size=[4, 4], stride=[1, 2, 2, 1], filters=1024, padding='VALID')
                Dis_conv6 = stardis_builder.Conv2d_layer(Dis_conv5, k_size=[4, 4], stride=[1, 2, 2, 1], filters=2048, padding='VALID')

                Output_Dsrc = stardis_builder.Conv2d_layer(Dis_conv6, k_size=[3, 3], stride=[1, 2, 2, 1], filters=1, Activation=False)
                Output_Dcls = stardis_builder.Conv2d_layer(Dis_conv6, k_size=[kwargs['Image_height']/64, kwargs['Image_width']/64], Activation=False)


