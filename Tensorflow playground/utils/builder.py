import tensorflow as tf


#TODO:ADD WEIGHT DECAY PARAMS TO CONV AND FC, ADD CONTECT MANAGER FUNC DEFAULTS
class Builder(object):
    def __init__(self, **kwargs):
        self.Summary = kwargs['Summary']
        self.Batch_size = kwargs['Batch_size']
        self.Image_width = kwargs['Image_width']
        self.Image_height = kwargs['Image_height']
        self.Image_cspace = kwargs['Image_cspace']
        self.Dropout_control = None
        self.State = None

    def __enter__(self):
        return self 


    def __exit__(self, exc_type, exc_value, traceback):
        print('Building complete')

    def control_params(self, Dropout_control= None, State= None):
        self.Dropout_control = Dropout_control
        self.State = State

    def Weight_variable(self, shape, weight_decay=0.0004):
        with tf.name_scope('Weight') as scope:
            ##with tf.variable_scope("Weight") as var_scope:
            #weights = tf.get_variable(name='Weight', initializer=tf.truncated_normal(shape, stddev=0.1), trainable=True, regularizer=self.Regloss_l2)
            initi = tf.contrib.layers.xavier_initializer()
            weights = tf.Variable(initi(shape))
            #weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.Regloss_l2(weights, weight_decay))
            #if self.Summary:
                ##self.variable_summaries(weights)
            return weights


    def Bias_variable(self, shape, weight_decay=0.04):
        with tf.name_scope('Bias') as scope:
            #with tf.variable_scope("Bias") as var_scope:
            biases = tf.Variable(tf.constant(0.01, shape=[int(shape)]))
            #tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.Regloss_l2(biases, weight_decay))
            #biases = tf.get_variable(name='Bias', shape=shape, initializer=tf.constant_initializer(0.1) , trainable=True, regularizer=self.Regloss_l2)

                #if self.Summary:
                ##self.variable_summaries(biases)
            return biases

    def Relu(self, input):
        return tf.nn.relu(input, name='Relu')

    def Conv2d_layer(self, input, *, batch_type=None, stride=[1, 1, 1, 1], k_size=[3, 3], filters=32, padding='SAME', Batch_norm=False, Activation=True, weight_decay=0.00004):
        with tf.name_scope('Conv') as scope:

            if batch_type is None:
                batch_type=self.State

            bias = self.Bias_variable(filters, weight_decay)
            input_shape = input.get_shape().as_list()[3]
            weight_shape = k_size + [input_shape, int(filters)]
            weights = self.Weight_variable(weight_shape, weight_decay)
            final_conv = tf.nn.conv2d(input, weights, strides=stride, padding=padding, name="CONV") + bias

            if Activation: #Prepare for Resnet
                final_conv = self.Relu(final_conv)

            if Batch_norm:
                final_conv = self.Batch_norm(final_conv, batch_type=batch_type)

            return final_conv

    def Upconv_layer(self, input, *, batch_type=None, stride=[1, 1, 1, 1], filters=32, output_shape=None, padding='SAME', Batch_norm=False, Activation=True, weight_decay=0.00001, k_size=[3, 3]):
        with tf.name_scope('Deconv') as scope:

            if batch_type is None:
                batch_type=self.State

            bias = self.Bias_variable(filters, weight_decay)
            input_shape = input.get_shape().as_list()[3]
            weight_shape = k_size + [input_shape, int(filters)]
            weights = self.Weight_variable(weight_shape, weight_decay)
            #final_shape  = [batch_size, output_shape[0], output_shape[1], int(filters)]
            #final_shape= tf.convert_to_tensor([input.get_shape().as_list()[0], output_shape[0], output_shape[1], int(filters)])
            batch_size = tf.shape(input)[0]
            final_shape = tf.stack([batch_size, output_shape[0], output_shape[1], int(filters)])
            final_deconv = tf.nn.conv2d_transpose(input, weights, output_shape=final_shape, strides=stride, padding=padding)
            final_deconv = tf.reshape(final_deconv,[-1, output_shape[0], output_shape[1], int(filters)])
            if Activation:
                final_deconv = self.Relu(final_deconv)

            if Batch_norm:
                final_deconv = self.Batch_norm(final_deconv, batch_type=batch_type)

            return final_deconv

    def Pool_layer(self, input, k_size=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='SAME', pooling_type='MAX'):
        with tf.name_scope('Pool') as scope:
            if pooling_type == 'MAX':
                Pool = tf.nn.max_pool(input, ksize=k_size, \
                    strides=stride, padding=padding, name="MAX_POOL")
            elif pooling_type == 'AVG':
                Pool = tf.nn.avg_pool(input, ksize=k_size, \
                    strides=stride, padding=padding, name="AVG_POOL")
            elif pooling_type == 'MAXIND':
                Pool, ind = tf.nn.max_pool_with_argmax(input, ksize=k_size,\
                    strides=stride, padding=padding, name='MAX_POOL_WITH_ARGMAX')
                return Pool, ind
            #if self.Summary:
                #tf.summary.histogram('Pool_activations', Pool)
            return Pool

    def Unpool_layer(self, pool, ind, k_size=[1, 2, 2, 1]):
        # https://github.com/tensorflow/tensorflow/issues/2169
        """
           Unpooling layer after max_pool_with_argmax.
           Args:
               pool:   max pooled output tensor
               ind:      argmax indices
               k_size:     k_size is the same as for the pool
           Return:
               unpool:    unpooling tensor
        """
        with tf.variable_scope('Unpool'):
            input_shape =  tf.shape(pool)
            input_shape_aslist = pool.get_shape().as_list()
            output_shape = tf.stack([input_shape[0], input_shape[1] * k_size[1], input_shape[2] * k_size[2], input_shape[3]])
            output_shapeaslist = [-1,  input_shape_aslist[1]* k_size[1] ,  input_shape_aslist[2]  * k_size[2], input_shape_aslist[3]]

            pool_ = tf.reshape(pool, [input_shape_aslist[1] * input_shape_aslist[2] * input_shape_aslist[3]])
            batch_range = tf.reshape(tf.range(tf.cast(input_shape[0], tf.int64), dtype=ind.dtype), 
                                              shape=tf.stack([input_shape[0], 1, 1, 1]))
            b = tf.ones_like(ind) * batch_range
            b = tf.reshape(b, tf.stack([ input_shape_aslist[1] * input_shape_aslist[2] * input_shape_aslist[3], 1]))
            ind_ = tf.reshape(ind, tf.stack( [input_shape_aslist[1] * input_shape_aslist[2] * input_shape_aslist[3], 1]))
            ind_ = tf.concat([b, ind_], 1)
            ret = tf.scatter_nd(ind_, pool_, shape=tf.cast([-1, output_shapeaslist[1] * output_shapeaslist[2] * output_shapeaslist[3] ], tf.int64))
            ret = tf.reshape(ret, [-1, output_shapeaslist[1], output_shapeaslist[2], output_shapeaslist[3]])
            return ret
    def FC_layer(self, input, filters=1024, readout=False, weight_decay=0.00004): #Expects flattened layer
        with tf.name_scope('FC') as scope:

            input_shape = input.get_shape().as_list()
            if len(input_shape) > 2:
                input = tf.reshape(input, [-1, input_shape[1] * input_shape[2] * input_shape[3]])

            bias = self.Bias_variable(filters, weight_decay)

            weight = self.Weight_variable([input.get_shape().as_list()[1], int(filters)], weight_decay)

            proto_output = tf.matmul(input, weight) + bias;
            #if self.Summary:
            #tf.summary.histogram('Pre_activations', proto_output)
            if readout:
                return(proto_output)

            final_output = tf.nn.relu(proto_output)
            #if self.Summary:
                #tf.summary.histogram('Final_activations', final_output)
            return(final_output)


    def Reshape_input(self, input, width=28, height=28, colorspace=1):
        with tf.name_scope('Pre-proc') as scope:
            return (tf.reshape(input, [-1, int(width), int(height), int(colorspace)]))


    def Pad_layer(self, input, p_size=[2, 2], p_type="CONSTANT"):
        with tf.name_scope('Pad') as scope:
            return(tf.pad(input, [[0, 0], [p_size[0], p_size[0]], [p_size[1], p_size[1]], [0, 0]], mode=p_type, name='Pad'))


    def Dropout_layer(self, input, keep_prob=None, seed=None):
        if keep_prob is None:
            keep_prob=self.Dropout_control
        return(tf.nn.dropout(input, keep_prob=keep_prob, seed=seed, name="Dropout"))


    def _BN_TRAIN(self, input, pop_mean, pop_var, scale, beta, epsilon, decay):
        with tf.name_scope('BN_TRAIN') as scope:
            batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2])
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1- decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1- decay))

            with tf.control_dependencies([train_mean, train_var]): #Further study into control_D required 
                return tf.nn.batch_normalization(input, batch_mean, batch_var, beta, scale, epsilon, name="BN_TRAIN")


    def _BN_TEST(self, input, pop_mean, pop_var, scale, beta, epsilon):
        with tf.name_scope('BN_TEST') as scope:
            return tf.nn.batch_normalization(input, pop_mean, pop_var, beta, scale, epsilon, name="BN_TEST")


    def Batch_norm(self, input, *, batch_type, decay=0.9997, epsilon=1e-3):
        ''' https://r2rt.com/implementing-batch-normalization-in-tensorflow.html for an explanation of the code'''
        with tf.name_scope('Batch_norm') as scope:
            #with tf.variable_scope("Batch_norm") as var_scope:
            pop_mean = tf.Variable(tf.zeros([input.get_shape()[-1]]), trainable=False)
            pop_var = tf.Variable(tf.ones([input.get_shape()[-1]]), trainable=False)
                #pop_mean = tf.get_variable(name="pop_mean", shape=[input.get_shape()[-1]], initializer=tf.zeros_initializer(), trainable=False)
                #pop_var = tf.get_variable(name="pop_var", shape=[input.get_shape()[-1]], initializer=tf.ones_initializer(), trainable=False)
                
                #scale = tf.get_variable(name="Bn_scale", shape=[input.get_shape()[-1]], initializer=tf.ones_initializer())
                #beta = tf.get_variable(name="Bn_beta", shape=[input.get_shape()[-1]], initializer=tf.zeros_initializer())
                #var_scope.reuse_variables()
            scale = tf.Variable(tf.ones([input.get_shape()[-1]]))
            beta = tf.Variable(tf.zeros([input.get_shape()[-1]]))

            return tf.cond(tf.equal(batch_type, 'TRAIN'), lambda: self._BN_TRAIN(input, pop_mean, pop_var, scale, beta, epsilon, decay), lambda: self._BN_TEST(input, pop_mean, pop_var, scale, beta, epsilon ))

    def Concat(self, inputs, axis=3):
        with tf.name_scope('Concat') as scope:
            return tf.concat(inputs, axis=axis)

    def Scale_activations(self, input, scaling_factor=0.2):
        with tf.name_scope('Scale_Activations') as scope:
            return tf.multiply(input, scaling_factor)

    def Residual_connect(self, input, Activation=True):
        with tf.name_scope('Residual_Connection') as scope:
            layer_sum = tf.add(input[0], input[1])
            if Activation:
                layer_sum = tf.nn.relu(layer_sum, name='Relu')
            return layer_sum

    def Regloss_l2(self, input, weight=1.0):
        with tf.name_scope('L2Regularizer'):
            l2_weight = tf.convert_to_tensor(weight, dtype=input.dtype.base_dtype, name='weight')
            return tf.multiply(l2_weight, tf.nn.l2_loss(input), name='value')
      

    
    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name + 'mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar(name + 'stddev', stddev)
            tf.summary.scalar(name + 'max', tf.reduce_max(var))
            tf.summary.scalar(name + 'min', tf.reduce_min(var))
            tf.summary.histogram(name + 'histogram', var)
    
