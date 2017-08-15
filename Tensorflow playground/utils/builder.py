import tensorflow as tf



class Builder(object):
    def __init__(self, **kwargs):
        self.Summary = kwargs['Summary']
        self.Batch_size = kwargs['Batch_size']
        self.Image_width = kwargs['Image_width']
        self.Image_height = kwargs['Image_height']
        self.Image_cspace = kwargs['Image_cspace']


    def __enter__(self):
        return self 


    def __exit__(self, exc_type, exc_value, traceback):
        print('Building complete')


    def Weight_variable(self, shape):
        with tf.name_scope('Weight') as scope:
            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
            if self.Summary:
                self.variable_summaries(weights)
            return weights


    def Bias_variable(self, shape):
        with tf.name_scope('Bias') as scope:
            biases = tf.Variable(tf.constant(0.1, shape=[int(shape)]))
            if self.Summary:
                self.variable_summaries(biases)
            return biases


    def Conv2d_layer(self, input, *, batch_type, stride=[1, 1, 1, 1], k_size=[3, 3], filters=32, padding='SAME', Batch_norm=False, Activation=True):
        with tf.name_scope('Conv') as scope:
            bias = self.Bias_variable(filters)
            input_shape = input.get_shape().as_list()[3]
            weight_shape = k_size + [input_shape, int(filters)]
            weights = self.Weight_variable(weight_shape)
            proto_conv = tf.nn.conv2d(input, weights, strides=stride, padding=padding, name="CONV") + bias
            if Activation: #Prepare for Resnet
                final_conv = tf.nn.relu(proto_conv)

            ''' Add batch norm
            if Batch_norm:
                #Append bathc norm block
            '''

            if self.Summary:
                tf.summary.histogram('Pre_activations', proto_conv)
                tf.summary.histogram('Final_activations', final_conv)
            return final_conv


    def Pool_layer(self, input, k_size=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='SAME'):
        with tf.name_scope('Pool') as scope:
            Pool = tf.nn.max_pool(input, ksize=k_size, \
                strides=stride, padding=padding, name="POOL")
            if self.Summary:
                tf.summary.histogram('Pool_activations', Pool)
            return Pool


    def FC_layer(self, input, filters=1024, readout=False): #Expects flattened layer
        with tf.name_scope('FC') as scope:

            input_shape = input.get_shape().as_list()
            if len(input_shape) > 2:
                input = tf.reshape(input, [-1, input_shape[1] * input_shape[2] * input_shape[3]])

            bias = self.Bias_variable(filters)

            weight = self.Weight_variable([input.get_shape().as_list()[1], int(filters)])

            proto_output = tf.matmul(input, weight) + bias;
            if self.Summary:
                tf.summary.histogram('Pre_activations', proto_output)
            if readout:
                return(proto_output)

            final_output = tf.nn.relu(proto_output)
            if self.Summary:
                tf.summary.histogram('Final_activations', final_output)
            return(final_output)


    def Reshape_input(self, input, width=28, height=28, colorspace=1):
        with tf.name_scope('Pre-proc') as scope:
            return (tf.reshape(input, [-1, int(width), int(height), int(colorspace)]))


    def Pad_layer(self, input, p_size=[2, 2], p_type="CONSTANT"):
        with tf.name_scope('Pad') as scope:
            return(tf.pad(input, [[0, 0], [p_size[0], p_size[0]], [p_size[1], p_size[1]], [0, 0]], mode=p_type, name='Pad'))


    def Dropout_layer(self, input, keep_prob, seed=None):
        return(tf.nn.dropout(input, keep_prob=keep_prob, seed=seed, name="Dropout"))


    def _BN_TRAIN(input, pop_mean, pop_var, scale, beta, decay):
        with tf.name_scope('BN_TRAIN') as scope:
            batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2])
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1- decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1- decay))

            with tf.control_dependencies([train_mean, train_var]): #Further study into control_D required 
                return tf.nn.batch_normalization(input, batch_mean, batch_var, beta, scale, epsilon, name="BN_TRAIN")


    def _BN_TEST(input, pop_mean, pop_var, scale, beta):
        with tf.name_scope('BN_TEST') as scope:
            return tf.nn.batch_normalization(input, pop_mean, pop_var, beta, scale, epsilon, name="BN_TEST")


    def Batch_norm(self, input, batch_type, decay=0.99, epsilon=1e-3):
        ''' https://r2rt.com/implementing-batch-normalization-in-tensorflow.html for an explanation of the code'''
        with tf.name_scope('Batch_norm') as scope:
            pop_mean = tf.Variable(tf.zeros([input.get_shape()[-1]]), trainable=False)
            pop_var = tf.Variable(tf.ones([input.get_shape()[-1]]), trainable=False)

            scale = tf.Variable(tf.ones([input.get_shape()[-1]]))
            beta = tf.Variable(tf.zeros([input.get_shape()[-1]]))

            return tf.cond(tf.equal(batch_type, True), lambda: self._BN_TRAIN(input, pop_mean, pop_var, scale, beta, epsilon, decay), lambda: self._BN_TEST(input, pop_mean, pop_var, scale, beta, epsilon ))



    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
