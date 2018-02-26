from utils.builder import Builder
import tensorflow as tf
from utils.Base_Archs.Base_Classifier import Base_Classifier

class Inception_resnet_v2a(Base_Classifier):
    """Inception_Resnet_v2 as written in tf.slim"""

    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.builder = None
        self.Endpoints = {}

    def build_net(self):
        with tf.name_scope('Inception_Resnet_v2a_model'):
            with Builder(**self.build_params) as inceprv2a_builder:
                self.builder = inceprv2a_builder
                #Setting control params
                inceprv2a_builder.control_params(Dropout_control=self.dropout_placeholder, State=self.state_placeholder, Renorm=self.build_params['Renorm'], Share_var=True)


                #Construct functional building blocks
                def stem(input):
                    with tf.variable_scope('Stem'):
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
                    with tf.variable_scope('Block35'):
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
                    with tf.variable_scope('Block17'):
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
                    with tf.variable_scope('Block8'):
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
                    with tf.variable_scope('Reduction_35x17'):
                        conv1a_split1 = inceprv2a_builder.Conv2d_layer(input, stride=[1, 2, 2, 1], k_size=[3, 3], filters=384, Batch_norm=True, padding='VALID')

                        conv1b_split1 = inceprv2a_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=256, Batch_norm=True)
                        conv2b_split1 = inceprv2a_builder.Conv2d_layer(conv1b_split1, stride=[1, 1, 1, 1], k_size=[3, 3], filters=256, Batch_norm=True)
                        conv3b_split1 = inceprv2a_builder.Conv2d_layer(conv2b_split1, stride=[1, 2, 2, 1], k_size=[3, 3], filters=384, Batch_norm=True, padding='VALID')

                        pool1c_split1 = inceprv2a_builder.Pool_layer(input, stride=[1, 2, 2, 1], k_size=[1, 3, 3, 1], padding='VALID')

                        concat = inceprv2a_builder.Concat([conv1a_split1, conv3b_split1, pool1c_split1])
                        
                        return concat
                def ReductionB(input):
                    with tf.variable_scope('Reduction_17x8'):
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

                #Model Construction

                #Stem
                Block_35 = stem(self.input_placeholder)
                #Inception 35x35
                for index in range(10):
                    Block_35 = incep_block35(Block_35, scale=0.17)
                #Reduction 35->17
                self.Endpoints['Block_35'] = Block_35

                Block_17 = ReductionA(Block_35)
                #Inception 17x17
                for index in range(20):
                    Block_17 = incep_block17(Block_17, scale=0.1)
                self.Endpoints['Block_17'] = Block_17

                #Reduction 17->8
                Block_8 = ReductionB(Block_17)
                for index in range(9):
                    Block_8 = incep_block8(Block_8, scale=0.2)
                Block_8 = incep_block8(Block_8, False)
                self.Endpoints['Block_8'] = Block_8

                #Normal Logits
                with tf.variable_scope('Logits'):
                    model_conv = inceprv2a_builder.Conv2d_layer(Block_8, stride=[1, 1, 1, 1], k_size=[1, 1], filters=1024, Batch_norm=True) #1536
                    model_conv_shape = model_conv.get_shape().as_list()
                    model_avg_pool = inceprv2a_builder.Pool_layer(model_conv, k_size=[1, model_conv_shape[1], model_conv_shape[2], 1], stride=[1, model_conv_shape[1], model_conv_shape[2], 1], padding='SAME', pooling_type='AVG')
                    #model_conv = inceprv2a_builder.Conv2d_layer(Block_8, stride=[1, 1, 1, 1], k_size=[1, 1], filters=512, Batch_norm=True) #1536
                    #model_conv = tf.reshape(model_conv, shape=[-1,  model_conv_shape[1] * model_conv_shape[2], model_conv_shape[3]])   #stacking heightwise for attention module
                    self.Endpoints['Model_conv'] = model_conv
                    drop1 = inceprv2a_builder.Dropout_layer(model_avg_pool)
                    output = inceprv2a_builder.FC_layer(drop1, filters=self.build_params['Classes'], readout=True)
                    return output

    def set_train_ops(self, optimizer):
        loss = tf.add_n(self.loss, 'Loss_accu')
        #self.train_step = optimizer.minimize(loss, global_step=self.global_step)
        grads = tf.gradients(loss, tf.trainable_variables())
        nomed_grads, _ = tf.clip_by_global_norm(grads, self.build_params['Grad_norm'])
        self.train_step = optimizer.apply_gradients(zip(nomed_grads,  tf.trainable_variables()), global_step=self.global_step)

    def construct_loss(self):
        if self.output is None:
            self.set_output()
        logit_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.output_placeholder,
                                                                               logits=self.output))
        with tf.variable_scope('Aux_logits'):
            aux_hook = self.Endpoints['Block_17']
            model_aux_avg_pool = self.builder.Pool_layer(aux_hook, k_size=[1, 5, 5, 1], stride=[1, 3, 3, 1], padding='VALID', pooling_type='AVG')
            model_aux_conv1 = self.builder.Conv2d_layer(model_aux_avg_pool, k_size=[1, 1], stride=[1, 1, 1, 1], filters=128, Batch_norm=True)
            model_aux_conv2 = self.builder.Conv2d_layer(model_aux_conv1, k_size=[5, 5], stride=[1, 1, 1, 1], padding='VALID', filters=768, Batch_norm=True)
            model_aux_logits = self.builder.FC_layer(model_aux_conv2, filters=self.build_params['Classes'], readout=True)
            aux_logit_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.output_placeholder,
                                                                                       logits=model_aux_logits)) * 0.6

        self.loss.append(logit_loss)
        self.loss.append(aux_logit_loss)