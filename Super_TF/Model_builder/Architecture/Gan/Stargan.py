from utils.builder import Builder
import tensorflow as tf
from utils.Base_Archs.Base_Gan import Base_Gan
import random
import numpy as np

class Stargan(Base_Gan):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        print(self.build_params['Classes'])
        self.target_lab = tf.placeholder(tf.float32, shape=[None, self.build_params['Classes']], name='Gen_class')
        self.real_im_lab = tf.placeholder(tf.float32, shape=[None, self.build_params['Classes']], name='Dis_class')

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

    def wgan_gp_loss(self, real_img, fake_img):  # gradient penalty
        alpha = tf.random_uniform(
            shape=[self.build_params['Batch_size'], 1, 1, 1],
            minval=0.,
            maxval=1.
        )

        hat_img = alpha * real_img + (1. - alpha) * fake_img
        gradients = tf.gradients(self.discriminator(hat_img, reuse=True)[0], xs=[hat_img])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.))

        return gradient_penalty

    def generator(self, gen_input, gen_class, reuse=False):
        print('Building generator')
        with tf.variable_scope(self.gen_name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            with Builder(**self.build_params) as stargen_builder:
                    inp_shape = gen_input.get_shape().as_list()
                    proto_class = tf.cast(tf.reshape(gen_class, [-1, 1, 1, self.build_params['Classes']]), tf.float32)
                    print('proto')
                    print(proto_class.get_shape().as_list())
                    print(inp_shape[1])
                    gen_tensor = tf.tile(proto_class, (1, inp_shape[1], inp_shape[2], 1))
                    stargen_builder.control_params(Dropout_control = self.gen_dropout_prob_placeholder, State = self.gan_state_placeholder, Share_var=True)
                
                    def residual_unit(input, filters=256):
                        Conv_rl1 = stargen_builder.Conv2d_layer(input, k_size=[3, 3], Batch_norm=True, filters=filters)
                        Conv_rl2 = stargen_builder.Conv2d_layer(Conv_rl1, k_size=[3, 3], Batch_norm=True, filters=filters, Activation=False)
                        Res_connect = stargen_builder.Residual_connect([input, Conv_rl2])
                        return Res_connect

                    #with tf.name_scope('Downsample'):
                    gen_input = tf.concat([gen_input, gen_tensor], axis=-1)
                    Conv1 = stargen_builder.Conv2d_layer(gen_input, k_size=[7, 7], Batch_norm=True, filters=64)
                    Conv2 = stargen_builder.Conv2d_layer(Conv1, k_size=[4, 4], stride=[1, 2, 2, 1], Batch_norm=True, filters=128, padding=[[0, 0], [1, 1], [1, 1], [0, 0]])
                    Conv3 = stargen_builder.Conv2d_layer(Conv2, k_size=[4, 4], stride=[1, 2, 2, 1], Batch_norm=True, filters=256, padding=[[0, 0], [1, 1], [1, 1], [0, 0]])


                    #with tf.name_scope('Bottleneck'):
                    Res_bl1 = residual_unit(Conv3)
                    Res_bl2 = residual_unit(Res_bl1)
                    Res_bl3 = residual_unit(Res_bl2)


                    Upconv1 = stargen_builder.Conv_Resize_layer(Res_bl3, filters=128, Batch_norm=True, Activation=True)
                    Upconv2 = stargen_builder.Conv_Resize_layer(Upconv1, filters=64, Batch_norm=True, Activation=True)
                    Final_conv = (stargen_builder.Conv2d_layer(Upconv2, k_size=[7, 7], filters=3, Activation=False))
                    Output_gen = stargen_builder.Activation(Final_conv, Type='TANH')
                    #Output_gen = tf.nn.sigmoid(Final_conv/50)
                    return (Output_gen)

    def discriminator(self, dis_input, reuse=False):
        print('Building Discriminator')
        with tf.variable_scope(self.dis_name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
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
                    Output_Dcls =  tf.reshape(Output_Dcls, [-1, self.build_params['Classes']])
                    #Output_Dcls = tf.squeeze(Output_Dcls)
                    return (Output_Dsrc, Output_Dcls)

    def set_accuracy_op(self):
        return super().set_accuracy_op()

    def construct_loss(self):

            self.fake_im = self.generator(self.real_im, self.target_lab, reuse=False)
            self.recon_im = self.generator(self.fake_im, self.real_im_lab, reuse=True)
            self.src_real_im, self.cls_real_im = self.discriminator(self.real_im, reuse=False)
            self.src_fake_im, self.cls_fake_im = self.discriminator(self.fake_im, reuse=True)

            #disc_adv_loss
            self.gp_loss = 10 * self.wgan_gp_loss(self.real_im, self.fake_im, )
            self.d_loss_fake = tf.reduce_mean(self.src_fake_im)
            self.d_loss_real = -tf.reduce_mean(self.src_real_im)
            self.d_adv_loss = self.d_loss_fake + self.d_loss_real + self.gp_loss


            #disc class loss
            self.d_real_cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.real_im_lab, logits=self.cls_real_im))
            self.d_loss = self.d_adv_loss + 1* self.d_real_cls_loss

            #Gen adv loss
            self.g_adv_loss = -tf.reduce_mean(self.src_fake_im)

            #Gen class loss
            self.g_fake_cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target_lab, logits=self.cls_fake_im))

            #reconst loss
            self.g_recon_loss = tf.reduce_mean(tf.abs(self.real_im - self.recon_im))

            self.g_loss = self.g_adv_loss + 1*self.g_fake_cls_loss + 10*self.g_recon_loss
            tf.summary.scalar('G_loss', self.g_loss)
            tf.summary.scalar('D_loss', self.d_loss)
            tf.summary.image('real_im', self.real_im)
            tf.summary.image('reconst_im', self.recon_im)
            tf.summary.image('fake_im', self.fake_im)
            tf.summary.scalar('G_adv_loss', self.g_adv_loss)
            tf.summary.scalar('G_cls_loss', self.g_fake_cls_loss)
            tf.summary.scalar('G_reconst_loss', self.g_recon_loss)
            tf.summary.scalar('D_adv_loss', self.d_adv_loss)
            tf.summary.scalar('D_cls_loss', self.d_real_cls_loss)
            tf.summary.scalar('D_loss_fake', self.d_loss_fake)
            tf.summary.scalar('D_loss_real', self.d_loss_real)
            '''
            real_outsrc, real_outcls = self.discriminator(self.gen_input_placeholder, reuse=False)
            Dis_loss_real = - tf.reduce_mean(real_outsrc)
            Dis_loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=real_outcls, labels=self.dis_class_placeholder))
            self.fake_gen_out = self.generator(self.gen_input_placeholder, self.gen_class_placeholder, reuse=False)
            fake_outsrc, fake_outcls = self.discriminator(self.fake_gen_out, reuse=True)
            Dis_loss_fake = tf.reduce_mean(fake_outsrc)


            #self.fake_gen_out = self.generator(self.gen_input_placeholder, self.gen_class_placeholder)
            #fake_outsrc, fake_outcls = self.discriminator(self.fake_gen_out)
            reconst_gen_out = self.generator(self.gen_input_placeholder,
                                             self.dis_class_placeholder, reuse=True)  # Reconstruct given image from generated image
            #Dis_loss_real = -tf.reduce_mean(real_outsrc)
            #Dis_loss_fake =  tf.reduce_mean(fake_outsrc)
            #Dis_loss_cls =   tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=real_outcls, labels=self.dis_class_placeholder))

            # Dis loss stage1
            #self.Dis_stage1_loss = Dis_loss_cls + Dis_loss_fake + Dis_loss_real

            # Dis loss stage2
            #with tf.control_dependencies(self.alpha.initializer.run()):
            #print(self.fake_gen_out.shape)
            #interp = tf.scalar_mul(self.alpha, self.gen_input_placeholder) + tf.scalar_mul(1-self.alpha, self.fake_gen_out)

            #Dis_loss_intrp, _ = self.discriminator(interp, reuse=True)
            #grads = tf.gradients(Dis_loss_intrp, interp)
            #slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[1]))
            #self.Dis_stage2_loss = tf.reduce_mean((slopes - 1.) ** 2) * self.lambda_gp
            # Gen loss

            #Gen_loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fake_outcls, labels=self.dis_class_placeholder))
            Gen_loss_rec = tf.reduce_mean(tf.abs(reconst_gen_out - self.gen_input_placeholder))
            self.Gen_loss = Gen_loss_rec
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            #tf.summary.scalar('Dis_loss_stage1', self.Dis_stage1_loss)
            tf.summary.scalar('Gen_loss', self.Gen_loss)
            tf.summary.image('Generator_reconst', reconst_gen_out)
            tf.summary.image('Generator_Input', self.gen_input_placeholder)
            tf.summary.image('Generator_fake_out', self.fake_gen_out)
            #tf.summary.scalar('Gen_loss_cls', Gen_loss_cls)
            tf.summary.scalar('Dis_loss_cls', Dis_loss_cls)
            tf.summary.scalar('Dis_loss_fake', Dis_loss_fake)
            tf.summary.scalar('Dis_loss_real', Dis_loss_real)
            '''

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
        for var in gen_train_vars: print(var.name)
        dis_train_vars = [v for v in tf.trainable_variables() if self.dis_name in v.name]
        for var in dis_train_vars: print(var.name)
        print(dis_train_vars)

        #self.dis_stage1_op = optimizer.minimize(loss=self.Dis_stage1_loss, var_list=dis_train_vars,
                                                #global_step=self.global_step)
        #self.dis_stage2_op = optimizer.minimize(loss=self.Dis_stage2_loss, var_list=dis_train_vars,
                                                #global_step=None)
        self.opti_dis = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9).minimize(self.d_loss, var_list=dis_train_vars, global_step=self.global_step)
        self.opti_gen = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9).minimize(self.g_loss, var_list=gen_train_vars)


    def gen_random_lab(self, real_idx):
        cls_idx = np.abs(real_idx-1)
        cls_idx = cls_idx.astype(np.uint8)
        #cls_idx = np.array([random.randrange(self.build_params['Classes'])])
        one_hot_vec = np.zeros((self.build_params['Batch_size'], self.build_params['Classes']))
        one_hot_vec[np.arange(self.build_params['Batch_size']), cls_idx] = 1
        return one_hot_vec

    def Construct_IO_dict(self, batch):  # need to write predict io dict too
        return {self.real_im: batch[0], self.real_im_lab: batch[1],
                self.target_lab: self.gen_random_lab(np.argmax(batch[1]))}

    def train(self, **kwargs):
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']

        # Dis cycles
        #for i in range(self.dis_cycles):
        #batch = kwargs['data']
        batch = kwargs['data'].next_batch(self.build_params['Batch_size'])
        print(np.argmax(batch[1]))
        IO_feed_dict = self.Construct_IO_dict(batch)
        train_feed_dict = {**IO_feed_dict, **self.train_dict}
        session.run([self.opti_dis, self.d_loss, self.gp_loss], feed_dict=train_feed_dict)
        #Dis stage 1
        #session.run([self.dis_stage1_op], feed_dict=train_feed_dict)

        #Dis stage 2
        #session.run([self.dis_stage2_op], feed_dict=train_feed_dict)

        #Gen

        if self.global_step.eval() % 2 == 0:
            print('updating gen')
            session.run([self.opti_gen, self.g_loss], feed_dict=train_feed_dict)
            #session.run([self.gen_loss_op, self.update_ops], feed_dict=train_feed_dict)

    def test(self, **kwargs):
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']
        #batch = kwargs['data']
        batch = kwargs['data'].next_batch(self.build_params['Batch_size'])
        IO_feed_dict = self.Construct_IO_dict(batch)
        test_feed_dict = {**IO_feed_dict, **self.test_dict}

        return session.run([kwargs['merged']], \
            feed_dict=test_feed_dict)[0]
