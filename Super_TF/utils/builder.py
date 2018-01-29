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
        self.global_step = tf.get_collection('Global_Step')[0]
        self.BNscope = 0
        self.debug = True
        self.renorm = None
    def __enter__(self):
        return self 


    def __exit__(self, exc_type, exc_value, traceback):
        print('Building complete')

    def control_params(self, Dropout_control=None, State=None, Renorm=None, Rmax=3, Dmax=5, R_Iter=200*5*50, D_Iter=550*5*50):
        self.Dropout_control = Dropout_control
        self.State = State
        self.renorm=Renorm
        self.construct_renorm_dict(Rmax=Rmax, Dmax=Dmax, R_Iter=R_Iter, D_Iter=D_Iter)

    def construct_renorm_dict(self, Rmax, Dmax, R_Iter, D_Iter):
        rmax = tf.Variable(1.0, trainable=False, name='Rmax', dtype=tf.float32)
        rmin = tf.Variable(0.99, trainable=False, name='Rmin', dtype=tf.float32)
        dmax = tf.Variable(0.0, trainable=False, name='Dmax', dtype=tf.float32)
        update_rmax = tf.cond(self.global_step<R_Iter, self.assign_add(rmax, 1, Rmax, R_Iter), self.make_noop).op
        update_dmax = tf.cond(self.global_step<D_Iter, self.assign_add(dmax, 0, Dmax, D_Iter), self.make_noop).op
        update_rmin = tf.cond(self.global_step<R_Iter, self.assign_inv(rmin, rmax), self.make_noop).op

        tf.summary.scalar('rmax', rmax)
        tf.summary.scalar('rmin', rmin)
        tf.summary.scalar('dmax', dmax)
        
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_rmax)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_dmax)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_rmin)
        
        self.renorm_dict = {'rmax':rmax, 'rmin':0.0, 'dmax':dmax}

    def Weight_variable(self, shape, weight_decay=0.000004):
        with tf.name_scope('Weight') as scope:
            #weights = tf.get_variable(name='Weight', initializer=tf.truncated_normal(shape, stddev=0.1), trainable=True, regularizer=self.Regloss_l2)
            #initi = tf.contrib.layers.xavier_initializer()
            #initi = tf.orthogonal_initializer()
            initi = tf.random_uniform_initializer(minval=-0.08, maxval=0.08)
            weights = tf.Variable(initi(shape))
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights)* weight_decay)
            return weights


    def Bias_variable(self, shape, weight_decay=0.04):
        with tf.name_scope('Bias') as scope:
            biases = tf.Variable(tf.constant(0.01, shape=[int(shape)]))
            return biases

    def Relu(self, input):
        return tf.nn.relu(input, name='Relu')

    def Tanh(self, input):
        return tf.nn.tanh(input, name='Tanh')

    def Leaky_relu(self, input, alpha=0.01):
        return tf.nn.leaky_relu(input, alpha=alpha)

    def Activation(self, input, Type='RELU', alpha=0.01):
        if Type is 'RELU':
            return self.Relu(input)

        elif Type is 'TANH':
            return self.Tanh(input)

        elif Type is 'LRELU':
            return self.Leaky_relu(input, alpha)

    def Conv2d_layer(self, input, *, batch_type=None, stride=[1, 1, 1, 1], k_size=[3, 3], filters=32, padding='SAME', Batch_norm=False, Activation=True, weight_decay=0.00001, name=None):
        with tf.name_scope('Conv') as scope:
            #weight_decay=0.00002 0.000001
            if batch_type is None:
                batch_type=self.State

            bias = self.Bias_variable(filters, weight_decay)
            input_shape = input.get_shape().as_list()[3]
            weight_shape = k_size + [input_shape, int(filters)]
            weights = self.Weight_variable(weight_shape, weight_decay)
            if name is None:
                final_conv = tf.nn.conv2d(input, weights, strides=stride, padding=padding, name="CONV") + bias
            else:
                final_conv = tf.nn.conv2d(input, weights, strides=stride, padding=padding, name=name) + bias

            if Activation: #Prepare for Resnet
                final_conv = self.Relu(final_conv)

            if Batch_norm:
                final_conv = self.Batch_norm(final_conv, batch_type=batch_type)

            return final_conv

    def DConv_layer(self, input, *, batch_type=None, stride=[1, 1], k_size=[3, 3], filters=32, padding='SAME', Batch_norm=False, Activation=True, weight_decay=0.00001, D_rate=1, name=None):
        with tf.name_scope('DConv') as scope:
            if name is None:
                dia_conv = tf.layers.conv2d(input, dilation_rate=[D_rate,D_rate], kernel_size=[3,3], filters=filters, strides=stride,padding=padding,\
                    kernel_initializer= tf.contrib.layers.xavier_initializer())
            else:
                dia_conv = tf.layers.conv2d(input, dilation_rate=[D_rate,D_rate], kernel_size=[3,3], filters=filters, strides=stride,padding=padding,\
                    kernel_initializer= tf.contrib.layers.xavier_initializer(), name=name)
            return dia_conv

    def Upconv_layer(self, input, *, batch_type=None, stride=[1, 1, 1, 1], filters=32, output_shape=None, padding='SAME', Batch_norm=False, Activation=True, weight_decay=0.00001, k_size=[3, 3]):
        with tf.name_scope('Deconv') as scope:

            if batch_type is None:
                batch_type=self.State

            bias = self.Bias_variable(filters, weight_decay)
            input_shape = input.get_shape().as_list()[3]
            weight_shape = k_size + [input_shape, int(filters)]
            weights = self.Weight_variable(weight_shape, weight_decay)
            batch_size = tf.shape(input)[0]
            final_shape = tf.stack([batch_size, output_shape[0], output_shape[1], int(filters)])
            final_deconv = tf.nn.conv2d_transpose(input, weights, output_shape=final_shape, strides=stride, padding=padding)
            final_deconv = tf.reshape(final_deconv,[-1, output_shape[0], output_shape[1], int(filters)])
            if Activation:
                final_deconv = self.Relu(final_deconv)

            if Batch_norm:
                final_deconv = self.Batch_norm(final_deconv, batch_type=batch_type)

            return final_deconv

    def Conv_Resize_layer(self, input, *, batch_type=None, stride=[1, 1, 1, 1], filters=None, output_scale=2, padding='SAME', Batch_norm=False, Activation=False, weight_decay=0.0001, k_size=[3, 3]):
        '''Resize + conv layer mentioned in
        https://distill.pub/2016/deconv-checkerboard/ '''

        with tf.name_scope('Conv_Resize') as scope:
            if batch_type is None:
                batch_type=self.State
            input_shape = input.get_shape().as_list()
            output_shape = [input_shape[1] * output_scale , input_shape[2] * output_scale]
            if filters is None:
                filters= input_shape[3]
            upscaled_input = tf.image.resize_nearest_neighbor(input, output_shape)
            final_reconv = self.Conv2d_layer(input=upscaled_input, batch_type=batch_type, stride=stride, filters=filters, padding=padding, Batch_norm=Batch_norm,\
                Activation=Activation, weight_decay=weight_decay, k_size=k_size)

            return final_reconv

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
        with tf.name_scope('Unpool'):
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
            ret = tf.scatter_nd(ind_, pool_, shape=tf.cast([input_shape[0], output_shapeaslist[1] * output_shapeaslist[2] * output_shapeaslist[3] ], tf.int64))
            ret = tf.reshape(ret, [-1, output_shapeaslist[1], output_shapeaslist[2], output_shapeaslist[3]])
            return ret

    def FC_layer(self, input, filters=1024, readout=False, flatten=True, weight_decay=0.00004): #Expects flattened layer
        with tf.name_scope('FC') as scope:

            input_shape = input.get_shape().as_list()
            if flatten:
                if len(input_shape) > 2:
                    input = tf.reshape(input, [-1, input_shape[1] * input_shape[2] * input_shape[3]])

            bias = self.Bias_variable(filters, weight_decay)

            weight = self.Weight_variable([input.get_shape().as_list()[1], int(filters)], weight_decay)

            proto_output = tf.matmul(input, weight) + bias;

            if readout:
                return(proto_output)

            final_output = tf.nn.relu(proto_output)
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



    
    def _BN_CUS(self, input, decay=0.945, epsilon=0.001, is_training=True):
        shape = input.get_shape().as_list()
        scale = tf.Variable(tf.ones([shape[-1]]))
        beta = tf.Variable(tf.zeros([shape[-1]]))
        pop_mean = tf.Variable(tf.zeros([shape[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([shape[-1]]), trainable=False)

        if is_training:
            mean=None
            variance=None
        else:
            mean =pop_mean
            variance = pop_var

        output, batch_m, batch_var = tf.nn.fused_batch_norm(x=input, offset=beta, scale=scale, epsilon=epsilon, is_training=is_training, mean=mean, variance=variance)

        update_pop_avg=tf.assign(pop_mean, pop_mean*decay+batch_m*(1-decay))
        update_pop_var=tf.assign(pop_var, pop_var*decay+batch_var*(1-decay))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_pop_avg)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_pop_var)
        return output
    
    def BatchNorm_layer(self, x, scope, train, epsilon=0.001, decay=.99):
        # Perform a batch normalization after a conv layer or a fc layer
        # gamma: a scale factor
        # beta: an offset
        # epsilon: the variance epsilon - a small float number to avoid dividing by 0
        with tf.variable_scope(scope, reuse=True):
            with tf.variable_scope('BatchNorm', reuse=True) as bnscope:
                gamma, beta = tf.get_variable("gamma"), tf.get_variable("beta")
                moving_avg, moving_var = tf.get_variable("moving_avg"), tf.get_variable("moving_var")
                shape = x.get_shape().as_list()
                control_inputs = []
                if train:
                    avg, var = tf.nn.moments(x, range(len(shape)-1))
                    update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
                    update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
                    control_inputs = [update_moving_avg, update_moving_var]
                else:
                    avg = moving_avg
                    var = moving_var
                with tf.control_dependencies(control_inputs):
                    output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)
        return output

    def initialize_batch_norm(self, shape):
        self.BNscope = self.BNscope+1
        with tf.variable_scope("BATCHNM_" + str(self.BNscope)) as bnscope:
                 gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0))
                 beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0))
                 moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
                 moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
                 bnscope.reuse_variables()

    def Batch_norm(self, input, *, batch_type, decay=0.92, epsilon=0.001, train=True, scope=None):
        
        if self.renorm is not None:
            output = tf.layers.batch_normalization(input, momentum=decay, epsilon=epsilon, training=train, renorm=True, renorm_clipping=self.renorm_dict)
            return output
        else:
            shape = input.get_shape().as_list()
            self.initialize_batch_norm(shape)
            with tf.variable_scope("BATCHNM_" + str(self.BNscope), reuse=True) as bnscope:
                    gamma, beta = tf.get_variable("gamma"), tf.get_variable("beta")
                    avg, var = tf.nn.moments(input, [0, 1, 2])
                    moving_avg, moving_var = tf.get_variable("moving_avg"), tf.get_variable("moving_var")
                
                    if self.debug:
                        self.variable_summaries(moving_avg, 'MOVING_AVG')
                        self.variable_summaries(moving_var, 'MOVING_VAR')
                    control_inputs = []
                    if train:
                        if self.debug:
                            self.debug = False
                            self.variable_summaries(avg, 'Instant_AVG')
                            self.variable_summaries(var, 'Instant_VAR')
                        avg, var = tf.nn.moments(input, [0, 1, 2])
                        update_moving_avg = tf.assign(moving_avg, moving_avg * decay + avg * (1- decay))
                        update_moving_var = tf.assign(moving_var, moving_var * decay + var * (1- decay))
                        control_inputs = [update_moving_avg, update_moving_var]
                        with tf.control_dependencies(control_inputs):
                            output = tf.nn.batch_normalization(input, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)
                            return output
                    else:
                        output = tf.nn.batch_normalization(input, moving_avg, moving_var, offset=beta, scale=gamma, variance_epsilon=epsilon)
                        return output

    def Concat(self, inputs, axis=3):
        with tf.name_scope('Concat') as scope:
            return tf.concat(inputs, axis=axis)

    def Scale_activations(self, input, scaling_factor=0.2):
        with tf.name_scope('Scale_Activations') as scope:
            return tf.multiply(input, scaling_factor)

    def Residual_connect(self, input, Activation=True):
        with tf.name_scope('Residual_Connection') as scope:
            layer_sum = tf.add_n(input)
            if Activation:
                layer_sum = tf.nn.relu(layer_sum, name='Relu')
            return layer_sum

    def Regloss_l2(self, input, weight=1.0):
        with tf.name_scope('L2Regularizer'):
            l2_weight = tf.convert_to_tensor(weight, dtype=input.dtype.base_dtype, name='weight')
            return tf.multiply(l2_weight, tf.nn.l2_loss(input), name='value')


    def Lstm_cell(self, num_units=512, state_is_tuple=True):
        with tf.name_scope('Basic_LSTM_Cell') as scope:
            return(tf.contrib.rnn.BasicLSTMCell(num_units=num_units, state_is_tuple=state_is_tuple))


    def Lstm_cell_LayerNorm(self, num_units=512, state_is_tuple=True, keep_prob=None):
        with tf.name_scope('Basic_LSTM_Cell_LN') as scope:
            if keep_prob is None:
                keep_prob=self.Dropout_control
            return(tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=num_units, dropout_keep_prob=keep_prob))
            

    def Rnn_dropout(self, input, keep_prob=None, seed=None):
        with tf.name_scope('Basic_RNN_Dropout') as scope:
            if keep_prob is None:
                keep_prob=self.Dropout_control
            return(tf.contrib.rnn.DropoutWrapper(input, input_keep_prob=keep_prob, output_keep_prob=keep_prob))

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

    #https://gist.github.com/yaroslavvb/d592394c0cedd32513f8fbb87ca05938#file-smart_initialize-py-L22
    def make_noop(self) : return tf.no_op()

    def assign_add(self, var, varinit, varmax, iterations):
        def f():
            return tf.assign_add(var, tf.cast( (varmax-varinit)/iterations, dtype=tf.float32)).op
        return f

    def assign_inv(self, var, var2):
        def f():
            return tf.assign(var, tf.cast(1/var2, dtype=tf.float32)).op
        return f
    #TODO: Add losses 
