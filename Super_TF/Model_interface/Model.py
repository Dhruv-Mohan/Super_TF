from Model_builder.Build_factory import Factory
import tensorflow as tf
import numpy as np
import cv2
import os


class Model(object):
    """This class contains the model architecture, optimizer and loss function"""
    def __init__(self, **kwargs):
        self.Model_name = kwargs['Model_name']
        print(self.Model_name)
        self.kwargs = kwargs
        self.NN_arch = self.Construct_Model()
        self.global_step = tf.get_collection('Global_Step')[0]
        self.optimizer = None
        self.merged = None

    def Set_optimizer(self, starter_learning_rate=0.0001, decay_steps=100000, decay_rate=None, Optimizer='RMS', Optimizer_params=None, Gradient_norm=None): #0.0001
        #TODO: CHANGE TO OPTIMIZER FACTORY
        def replace_none_with_zero(i, shape):
            return np.zeros(shape=shape) if i==None else i

        if decay_rate is not None:
            learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
        else:
            learning_rate = starter_learning_rate
        if self.kwargs['Summary'] is True:
            tf.summary.scalar('Learning_rate', learning_rate)
        
        clip_min = - 1.0
        clip_max = 1.0
        
        #Select Optimizer
        
        if Optimizer is 'ADAM':
            if Optimizer_params is None:
                self.optimizer = tf.train.AdamOptimizer(learning_rate)
            else:
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=Optimizer_params['beta1'], beta2=Optimizer_params['beta2'], epsilon=Optimizer_params['epsilon'])
        elif Optimizer is 'RMS':
            if Optimizer_params is None:
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate)
            else:
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=Optimizer_params['decay'], momentum=Optimizer_params['momentum'], epsilon=Optimizer_params['epsilon'])
        elif Optimizer is 'SGD':
                self.optimizer= tf.train.GradientDescentOptimizer(learning_rate)
        elif Optimizer is 'MOM':
            if Optimizer_params is None:
                self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.8)
            else:
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=Optimizer_params['momentum'], use_nesterov=Optimizer_params['Nestrov'])

        self.NN_arch.construct_loss()
        self.NN_arch.set_train_ops(self.optimizer)
        '''#Failed gpu memory saving experiment
        vars = tf.trainable_variables()
        total_length = len(vars)/2
        first_half = []
        second_half = []
        for index,grad in enumerate(vars):
            if index < total_length:
                first_half.append(grad)
            else:
                second_half.append(grad)
        #grads1 = self.optimizer.compute_gradients(self.loss, var_list=first_half)
        #grads2 = self.optimizer.compute_gradients(self.loss, var_list=second_half)
                self.train_step = self.optimizer.minimize(self.loss,global_step=self.global_step, var_list=first_half)
        self.train_step2 = self.optimizer.minimize(self.loss,global_step=self.global_step, var_list=second_half)
        '''
        
        """
        tvs =all_trainable = [v for v in tf.trainable_variables() if 'Pnet' not in v.name]
        acc_grads = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        self.reset_accumulated_grads = [tv.assign(tf.zeros_like(tv)) for tv in acc_grads]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        gvs, tvars = zip(*self.optimizer.compute_gradients(self.loss, tvs))
        #with tf.control_dependencies(update_ops):
        with tf.control_dependencies(update_ops):
            self.accumulate_gradients = [acc_grads[i].assign_add(replace_none_with_zero(gv, acc_grads[i].get_shape().as_list())) for i, gv in enumerate(gvs)]

        if Gradient_norm is not None:
            gradients, _ = tf.clip_by_global_norm(acc_grads, Gradient_norm)
        else:
            gradients = acc_grads

        
        self.train_step = self.optimizer.apply_gradients(zip(gradients,tvars), global_step=self.global_step)
        """

    def Construct_Model(self):
        return Factory(**self.kwargs).get_model()

    def Construct_Accuracy_op(self):
        with tf.name_scope('accuracy'):
            self.NN_arch.set_accuracy_op()
            """
            if self.model_dict['Model_Type'] is 'Classification' :
                correct_prediction = tf.equal(tf.argmax(self.model_dict['Output'], 1), tf.argmax(self.model_dict['Output_ph'], 1))
                false_images = tf.boolean_mask(self.model_dict['Reshaped_input'], tf.logical_not(correct_prediction))
                tf.summary.image(name='False images', tensor=false_images)
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)
                self.accuracy_op = True

            elif self.model_dict['Model_Type'] is 'Segmentation' :
                probs = tf.reshape((tf.sigmoid(self.model_dict['Output'])), shape=[ self.kwargs['Batch_size'], -1])
                lab = tf.reshape(self.model_dict['Output_ph'], shape=[self.kwargs['Batch_size'], -1])
                probs = tf.ceil(probs - 0.5 + 1e-10)
                intersection = tf.reduce_sum(probs * lab, axis=1) 
                union =  tf.reduce_sum(probs, 1) + tf.reduce_sum(lab, 1) 
                tf.summary.image(name='Input images',tensor = self.model_dict['Reshaped_input'])
                tf.summary.image(name='Mask',tensor = tf.reshape(self.model_dict['Output_ph'], [-1, self.kwargs['Image_width'], self.kwargs['Image_height'], 1]))
                tf.summary.image(name='Weight',tensor = tf.reshape(self.model_dict['Weight_ph'], [-1, self.kwargs['Image_width'], self.kwargs['Image_height'], 1]))
                tf.summary.image(name='Output',tensor = (tf.sigmoid(self.model_dict['Output'])))
                self.accuracy = tf.reduce_mean(2 * intersection / (union))
                tf.summary.scalar('accuracy', self.accuracy)
                self.accuracy_op = True

            elif self.model_dict['Model_Type'] is 'Sequence' :
                
                correct_prediction = tf.equal(tf.argmax(self.model_dict['Output'], 1), tf.reshape(tf.cast(tf.reshape(self.model_dict['Output_ph'], shape=[-1]), tf.int64), [-1]))
                pre_acc = tf.to_float(correct_prediction) * tf.to_float(tf.reshape(self.model_dict['Mask'], [-1]))
                pre_acc = tf.reduce_sum(pre_acc)
                self.accuracy = tf.div(pre_acc,  tf.maximum(1.0,tf.reduce_sum(tf.to_float(tf.reshape(self.model_dict['Mask'], [-1])))))
                tf.reduce_sum(tf.to_float(tf.reshape(self.model_dict['Mask'], [-1])))
                self.accuracy_op = True
                tf.summary.scalar('accuracy', self.accuracy)
                self.out_op = tf.argmax(self.model_dict['Output'], 1)
            #tf.cond(self.accuracy > 0.92, lambda: tf.summary.image(name='False images', tensor=false_images), lambda: tf.summary.tensor_summary(name='correct_predictions', tensor=correct_prediction))
            """

    def Construct_Writers(self, session=None):
        #Get default session

        self.saver = tf.train.Saver()
        if session is None:
            session = tf.get_default_session()

        self.log_writer = tf.summary.FileWriter(self.kwargs['Save_dir'] + '/logs/', session.graph)

    def Set_initial_state(self, image, session=None):
        
        if session is None:
            session = tf.get_default_session()
        lstm_init_dict = {self.model_dict['Input_ph']: image}
        init_feed_dict = {**lstm_init_dict, **self.test_dict}
        output = session.run(self.Predict_op, feed_dict=init_feed_dict)
        return(output)
        
    def Get_LSTM_prediction(self, input_feed, session=None):
        if session is None:
            session = tf.get_default_session()
        lstm_predict_dict = {self.model_dict['Input_seq']: input_feed, self.model_dict['Lstm_state_feed']: self.initial_state_lstm}
        predict_feed_dict = {**lstm_predict_dict, **self.test_dict}
        output, self.initial_state_lstm = session.run([self.Predict_op, self.model_dict['Lstm_state']], feed_dict=predict_feed_dict)
        print('output', output)
        return output

    def Try_restore(self,session=None):
        #Get default session
        if session is None:
            session = tf.get_default_session()
        latest_ckpt = tf.train.latest_checkpoint(self.kwargs['Save_dir'] + '/mdl/')
        if latest_ckpt:
            print('Ckpt_found')
            print(latest_ckpt)
            self.saver.restore(session,latest_ckpt)
        else:
            print('Ckpt_not_found')

    def Predict(self, kwargs):
        return self.NN_arch.predict(kwargs)

    def Train_Iter(self, iterations, save_iterations=100, data=None, log_iteration=2, restore=True, session=None, micro_batch=2):
        #Get default session
        if session is None:
            session = tf.get_default_session()

        #Try restore
        if restore:
            print('Restoring Default session')
            self.Try_restore(session)
            print('Default session restored')

        merged = tf.summary.merge_all()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        self.global_step.initializer.run()

        for step in range(iterations):
            self.NN_arch.train(session=session, data=data, Batch_size=self.kwargs['Batch_size'])

            if (step + 1) % save_iterations == 0:
                print('Saving Checkpoint')
                self.saver.save(session, self.kwargs['Save_dir'] + '/mdl/' + self.Model_name + '.ckpt',
                                global_step=self.global_step)

            if (step + 1) % log_iteration == 0:
                print('Logging')
                test_out = self.NN_arch.test(session=session, data=data, Batch_size=self.kwargs['Batch_size'], merged=merged)
                glo_step = session.run([self.global_step])[0]
                self.log_writer.add_summary(test_out, glo_step)

        """
        for step in range(iterations):
            step = session.run([self.global_step])[0]
            session.run(self.reset_accumulated_grads)
            print('Getting batch')
            
            for i in range(micro_batch):
                batch = data.next_batch(self.kwargs['Batch_size'])            #IO feed dict
                print('Getting micro batch')
                #print(batch[2])
                IO_feed_dict = self.Construct_IO_dict(batch)            #Construct train dict
                if self.prior_path is not None:                                 #Stop gap solution to accomodate F-Net
                    prior_path=batch[3]
                    abs_path = os.path.splitext(prior_path)[0]
                    complete_path = 'G:/Pri/'+abs_path+'_ODsegSoftmap.png'
                    prior_path_dict = {self.prior_path: complete_path}

                    print('Constructing IO feed dict')
                    train_feed_dict = {**IO_feed_dict, **self.train_dict, **prior_path_dict}
                else:
                    print('Constructing IO feed dict')
                    train_feed_dict = {**IO_feed_dict, **self.train_dict}
                _,  loss,output = session.run([self.accumulate_gradients,  self.loss, self.Predict_op], feed_dict=train_feed_dict)
                print(output)
            #Train Step
            session.run(self.train_step)
            print ('Step: ',step+1,'Loss: ',loss)
            if(step + 1) % save_iterations == 0:
                print('Saving Checkpoint')
                self.saver.save(session, self.kwargs['Save_dir'] + '/mdl/' + self.Model_name + '.ckpt', global_step=self.global_step)

            #logger block
            if(step + 1) % log_iteration == 0:
                if self.prior_path is not None:
                    test_feed_dict = {**IO_feed_dict, **self.test_dict, **prior_path_dict}               #Construst Test dict
                else:
                    test_feed_dict = {**IO_feed_dict, **self.test_dict}               #Construst Test dict
                if self.accuracy_op:
                    summary, train_accuracy, glo_step = session.run([self.merged, self.accuracy, self.global_step], \
                                feed_dict=test_feed_dict)
                else:
                    summary, glo_step = session.run([self.merged, self.global_step], \
                                feed_dict=test_feed_dict)
                self.log_writer.add_summary(summary, glo_step)

        """
        coord.request_stop()
        coord.join(threads)



