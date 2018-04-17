from utils.builder import Builder
import tensorflow as tf
from utils.Base_Archs.Base_Gan import Base_Gan
import random
import numpy as np

class Stargan(Base_Gan):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.gen_class_placeholder = tf.placeholder(tf.float32, shape=[None, self.build_params['Classes']], name='Gen_class')
        self.dis_class_placeholder = tf.placeholder(tf.float32, shape=[None, self.build_params['Classes']], name='Dis_class')

        self.gen_name = 'Stargan_generator'
        self.dis_name = 'Stargan_discriminator'

        self.test_dict = self.construct_control_dict('TEST')
        self.train_dict = self.construct_control_dict('TRAIN')

        # Losses
        self.Dis_stage1_loss = None
        self.Dis_stage2_loss = None
        self.Gen_loss = None

        # Train ops
        self.dis_stage1_op = None
        self.dis_stage2_op = None
        self.gen_loss_op = None

        # Output
        self.fake_gen_out = None

        # Alpha ops
        self.alpha = tf.random_uniform([], 0, 1)

        # Hyper_params
        self.dis_cycles = 10
        self.lambda_gp = 10
        self.lambda_cls = 0



    def generator(self, gen_input, gen_class):
        with tf.variable_scope(self.gen_name):
            with Builder(**self.build_params) as stargen_builder:
                    
                    stargen_builder.control_params(Dropout_control = self.gen_dropout_prob_placeholder, State = self.gan_state_placeholder, Share_var=True)
                
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
        with tf.variable_scope(self.dis_name):
            with Builder(**self.build_params) as stardis_builder:
                    stardis_builder.control_params(Dropout_control=self.dis_dropout_prob_placeholder, State=self.gan_state_placeholder, Share_var=True)

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

                    Output_Dsrc = stardis_builder.Conv2d_layer(Dis_conv6, k_size=[3, 3], stride=[1, 1, 1, 1], filters=1,
                                                               Activation=False)
                    Output_Dcls = stardis_builder.Conv2d_layer(Dis_conv6,
                                                               k_size=[int(self.build_params['Image_height']/64),
                                                                       int(self.build_params['Image_width']/64)],
                                                               filters=self.build_params['Classes'],
                                                               Activation=False,
                                                               padding='VALID')
                    Output_Dcls = tf.squeeze(Output_Dcls)
                    return (Output_Dsrc, Output_Dcls)

    def set_accuracy_op(self):
        return super().set_accuracy_op()

    def construct_loss(self):
        with tf.variable_scope('Output', reuse=tf.AUTO_REUSE):
            self.fake_gen_out = self.generator(self.gen_input_placeholder, self.gen_class_placeholder)
            reconst_gen_out = self.generator(self.fake_gen_out, self.dis_class_placeholder)  # Reconstruct given image from generated image
            real_outsrc, real_outcls = self.discriminator(self.gen_input_placeholder)
            fake_outsrc, fake_outcls = self.discriminator(self.fake_gen_out)

        with tf.variable_scope('Losses', reuse=tf.AUTO_REUSE):
            Dis_loss_real = -tf.reduce_mean(real_outsrc)
            Dis_loss_fake =  tf.reduce_mean(fake_outsrc)
            Dis_loss_cls =   tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=real_outcls, labels=self.dis_class_placeholder))

            # Dis loss stage1
            self.Dis_stage1_loss = Dis_loss_cls + Dis_loss_fake + Dis_loss_real

            # Dis loss stage2
            #with tf.control_dependencies(self.alpha.initializer.run()):
            print(self.alpha.shape)
            print(self.fake_gen_out.shape)
            interp = tf.scalar_mul(self.alpha, self.gen_input_placeholder) + tf.scalar_mul(1-self.alpha, self.fake_gen_out)

            Dis_loss_intrp, _ = self.discriminator(interp)
            grads = tf.gradients(Dis_loss_intrp, interp)
            slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[1]))
            self.Dis_stage2_loss = tf.reduce_mean((slopes - 1.) ** 2) * self.lambda_gp

            # Gen loss
            Gen_loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fake_outcls, labels=self.dis_class_placeholder))
            Gen_loss_rec = tf.reduce_mean(tf.abs(reconst_gen_out - self.fake_gen_out))
            self.Gen_loss = Gen_loss_rec - Dis_loss_fake + Gen_loss_cls

            tf.summary.scalar('Dis_loss_stage1', self.Dis_stage1_loss)
            tf.summary.scalar('Dis_loss_stage2', self.Dis_stage2_loss)
            tf.summary.scalar('Gen_loss', self.Gen_loss)
            tf.summary.image('Generator_reconst', reconst_gen_out)
            tf.summary.image('Generator_fake_out', self.fake_gen_out)
            tf.summary.scalar('Gen_loss_cls', Gen_loss_cls)
            tf.summary.scalar('Dis_loss_cls', Dis_loss_cls)

    def predict(self, kwargs):
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']

        predict_io_dict = self.construct_IO_dict([kwargs['Input_Im'], kwargs['Out_Class']])
        predict_feed_dict = {**predict_io_dict, **self.test_dict}
        return session.run([self.fake_gen_out], feed_dict=predict_feed_dict)


    def set_train_ops(self, optimizer):
        gen_train_vars = [v for v in tf.trainable_variables() if self.gen_name in v.name]
        dis_train_vars = [v for v in tf.trainable_variables() if self.dis_name in v.name]
        self.dis_stage1_op = optimizer.minimize(loss=self.Dis_stage1_loss, var_list=dis_train_vars,
                                                global_step=self.global_step)
        self.dis_stage2_op = optimizer.minimize(loss=self.Dis_stage2_loss, var_list=dis_train_vars,
                                                global_step=None)
        self.gen_loss_op = optimizer.minimize(loss=self.Gen_loss, var_list=gen_train_vars, global_step=None)

    def gen_random_lab(self):
        cls_idx = np.array([random.randrange(self.build_params['Classes'])])
        one_hot_vec = np.zeros((self.build_params['Batch_size'], self.build_params['Classes']))
        one_hot_vec[np.arange(self.build_params['Batch_size']), cls_idx] = 1
        return one_hot_vec

    def Construct_IO_dict(self, batch):  # need to write predict io dict too
        return {self.gen_input_placeholder: batch[0], self.dis_class_placeholder: batch[1],
                self.gen_class_placeholder: self.gen_random_lab()}

    def train(self, **kwargs):
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']

        # Dis cycles
        for i in range(self.dis_cycles):
            batch = kwargs['data'].next_batch(self.build_params['Batch_size'])
            IO_feed_dict = self.Construct_IO_dict(batch)
            train_feed_dict = {**IO_feed_dict, **self.train_dict}
            #Dis stage 1
            session.run([self.dis_stage1_op], feed_dict=train_feed_dict)

            #Dis stage 2
            session.run([self.dis_stage2_op], feed_dict=train_feed_dict)

        #Gen
        session.run([self.gen_loss_op], feed_dict=train_feed_dict)

    def test(self, **kwargs):
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']

        batch = kwargs['data'].next_batch(self.build_params['Batch_size'])
        IO_feed_dict = self.Construct_IO_dict(batch)
        test_feed_dict = {**IO_feed_dict, **self.test_dict}

        return session.run([kwargs['merged']], \
            feed_dict=test_feed_dict)[0]
