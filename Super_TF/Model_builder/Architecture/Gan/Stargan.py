from utils.builder import Builder
import tensorflow as tf
from utils.Architect import Architect

class Stargan(Architect):
    def __init__(self, kwargs):
        self.gen_input_placeholder = tf.placeholder(tf.float32, \
            shape=[None, kwargs['Image_width'], kwargs['Image_height'], kwargs['Image_cspace']], name='Input')
        self.gen_class_placeholder = tf.placeholder(tf.float32, \
            shape=[None, kwargs['Image_width'], kwargs['Image_height']], name='Mask') #TODO: Epand class to input size and concat, fix class size
        self.dis_class_placeholder = tf.placeholder(tf.float32, \
            shape=[None, kwargs['Image_width'], kwargs['Image_height']], name='Mask') #TODO: Epand class to input size and concat, fix class size
        self.build_params = kwargs
        self.gen_dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
        self.gen_state_placeholder = tf.placeholder(tf.string, name="State")
        self.dis_dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
        self.dis_state_placeholder = tf.placeholder(tf.string, name="State")

    def generator(self, gen_input, gen_class):
        with tf.variable_scope('Stargan_generator'):
            with Builder(**self.build_params) as stargen_builder:
                    
                    stargen_builder.control_params(Dropout_control = self.gen_dropout_prob_placeholder, State = self.gen_state_placeholder, Share_var=True)
                
                    def residual_unit(input, filters=256):
                        Conv_rl1 = stargen_builder.Conv2d_layer(input, k_size=[3, 3], Batch_norm=True, filters=filters)
                        Conv_rl2 = stargen_builder.Conv2d_layer(Conv_rl1, k_size=[3, 3], Batch_norm=True, filters=filters, Activation=False)
                        Res_connect = stargen_builder.Residual_connect([input, Conv_rl2])
                        return Res_connect

                    with tf.name_scope('Downsample'):
                        Conv1 = stargen_builder.Conv2d_layer(gen_input, k_size=[7, 7], Batch_norm=True, filters=64)
                        Conv2 = stargen_builder.Conv2d_layer(Conv1, k_size=[4, 4], stride=[1, 2, 2, 1], Batch_norm=True, filters=128, padding=[[0, 0], [1, 1], [1, 1], [0, 0]])
                        Conv3 = stargen_builder.Conv2d_layer(Conv2, k_size=[4, 4], stride=[1, 2, 2, 1], Batch_norm=True, filters=256, padding=[[0, 0], [1, 1], [1, 1], [0, 0]])


                    with tf.name_scope('Bottleneck'):
                        Res_bl1 = residual_unit(Conv3)
                        Res_bl2 = residual_unit(Res_bl1)
                        Res_bl3 = residual_unit(Res_bl2)

                    with tf.name_scope('Upsample'):
                        Upconv1 = stargen_builder.Conv_Resize_layer(Res_bl3, filters=128, Batch_norm=True, Activation=True)
                        Upconv2 = stargen_builder.Conv_Resize_layer(Upconv1, filters=64, Batch_norm=True, Activation=True)
                        Final_conv = stargen_builder.Conv2d_layer(Upconv2, k_size=[7, 7], filters=3, Activation=False)
                        Output_gen = stargen_builder.Activation(Final_conv, Type='TANH')
                    return (Output_gen)

    def discriminator(self, dis_input):
        with tf.variable_scope('Stargan_discriminator'):
            with Builder(**self.build_params) as stardis_builder:
                    stardis_builder.control_params(Dropout_control=self.dis_dropout_prob_placeholder, State=self.dis_state_placeholder, Share_var=True)

                    Dis_conv1 = stardis_builder.Conv2d_layer(dis_input, k_size=[4, 4], stride=[1, 2, 2, 1], filters=64, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], Activation=False)
                    Dis_conv1 = stardis_builder.Activation(Dis_conv1, Type='LRELU')
                    Dis_conv2 = stardis_builder.Conv2d_layer(Dis_conv1, k_size=[4, 4], stride=[1, 2, 2, 1], filters=128, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], Activation=False)
                    Dis_conv2 = stardis_builder.Activation(Dis_conv2, Type='LRELU')
                    Dis_conv3 = stardis_builder.Conv2d_layer(Dis_conv2, k_size=[4, 4], stride=[1, 2, 2, 1], filters=256, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], Activation=False)
                    Dis_conv3 = stardis_builder.Activation(Dis_conv3, Type='LRELU')
                    Dis_conv4 = stardis_builder.Conv2d_layer(Dis_conv3, k_size=[4, 4], stride=[1, 2, 2, 1], filters=512, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], Activation=False)
                    Dis_conv4 = stardis_builder.Activation(Dis_conv4, Type='LRELU')
                    Dis_conv5 = stardis_builder.Conv2d_layer(Dis_conv4, k_size=[4, 4], stride=[1, 2, 2, 1], filters=1024, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], Activation=False)
                    Dis_conv5 = stardis_builder.Activation(Dis_conv5, Type='LRELU')
                    Dis_conv6 = stardis_builder.Conv2d_layer(Dis_conv5, k_size=[4, 4], stride=[1, 2, 2, 1], filters=2048, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], Activation=False)
                    Dis_conv6 = stardis_builder.Activation(Dis_conv6, Type='LRELU')

                    Output_Dsrc = stardis_builder.Conv2d_layer(Dis_conv6, k_size=[3, 3], stride=[1, 1, 1, 1], filters=1, Activation=False)
                    Output_Dcls = stardis_builder.Conv2d_layer(Dis_conv6, k_size=[ int(self.build_params['Image_height']/64), int(self.build_params['Image_width']/64)], Activation=False, padding='VALID')
                    return (Output_Dsrc, Output_Dcls)

    def set_train_dict(self):
        return super().set_train_dict()

    def set_test_dict(self):
        return super().set_test_dict()

    def set_accuracy_op(self):
        return super().set_accuracy_op()

    def construct_loss(self):
        with tf.variable_scope('Output', reuse=tf.AUTO_REUSE):
            fake_gen_out = self.generator(self.gen_input_placeholder, self.gen_class_placeholder)
            reconst_gen_out = self.generator(fake_gen_out, self.dis_class_placeholder) #Reconstruct given image from generated image
            real_outsrc, real_outcls = self.discriminator(self.gen_input_placeholder)
            fake_outsrc, fake_outcls = self.discriminator(fake_gen_out)

        with tf.variable_scope('Losses', reuse=tf.AUTO_REUSE):
            Dis_loss_real = -tf.reduce_mean(real_outsrc)
            Dis_loss_fake =  tf.reduce_mean(fake_outsrc)
            Dis_loss_cls =   tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=real_outcls, labels=self.dis_class_placeholder))

            #Dis loss stage1
            Dis_stage1_loss = Dis_loss_cls + Dis_loss_fake + Dis_loss_real

            #Gen loss
            Gen_loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fake_outcls, labels=self.dis_class_placeholder))
            Gen_loss_rec = tf.reduce_mean(tf.abs(reconst_gen_out - fake_gen_out))
            Gen_loss = Gen_loss_rec - Dis_loss_fake + Gen_loss_cls

    def construct_predict_op(self):
        return self.generator(self.gen_input_placeholder, self.gen_class_placeholder)

    def set_train_ops(self):
        return super().set_train_ops()

    def train(self):
        return super().train()
