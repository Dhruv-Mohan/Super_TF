from Model_builder.Build_factory import Factory
import tensorflow as tf
class Model_class(object):
    """This class contains the model architecture, optimizer and loss function"""


    def __init__(self, **kwargs):
        self.Model_name = kwargs['Model_name']
        self.kwargs = kwargs
        self.global_step=tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

        #Init class dicts
        self.test_dict = {}
        self.train_dict = {}
        self.model_dict = {}


    def Set_test_control(self, control_placeholders_dict):
        for key, value in control_placeholders_dict.items():
            self.test_dict[self.model_dict[key]] = value


    def Set_train_control(self, control_placeholders_dict):
        for key, value in control_placeholders_dict.items():
            self.train_dict[self.model_dict[key]] = value


    def Set_loss(self, params):
        self.loss = params  #Const loss func
        if self.kwargs['Summary']:
            tf.summary.scalar('cross_entropy', self.loss)


    def Set_optimizer(self, params):
        self.optimizer = params #Const optimizer params 
        #gradients,_ = self.optimizer.compute_gradients(loss) Add max-norm
        self.train_step = self.optimizer.minimize(self.loss,global_step=self.global_step)


    def Construct_Model(self):
        self.model_dict = Factory(**self.kwargs).get_model()
        print('Model Placeholders')
        for key, value in self.model_dict.items():
            print(key)


    def Construct_Accuracy_op(self):
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.model_dict['Output'], 1), tf.argmax(self.model_dict['Output_ph'], 1))
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)

    def Construct_Writers(self, session=None):
        #Get default session

        self.saver = tf.train.Saver()
        if session is None:
            session = tf.get_default_session()
        #if self.kwargs['Summary']:
        self.log_writer = tf.summary.FileWriter(self.kwargs['Save_dir'] + '/logs/', session.graph)
            #self.test_writer = tf.summary.FileWriter(self.kwargs['Save_dir'] + '/test/')
        #else:
            #graph_writer = tf.summary.FileWriter(self.kwargs['Save_dir'] + '/train/')
            #graph_writer.add_graph(session.graph)
            #graph_writer.close();


    def Construct_Predict_op(self):
        with tf.name_scope('Predict'):
            self.Predict_op = tf.argmax(self.model_dict['Output'], 1)


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


    def Predict(self, input_data, session=None):
        #Get default session
        if session is None:
            session = tf.get_default_session()
        #Construct Predict dict
        IO_predict_dict = {self.model_dict['Input_ph']: input_data}

        #Construct Predict feed_dict
        predict_feed_dict = {**IO_predict_dict, **self.test_dict}

        out = session.run([self.Predict_op], feed_dict=predict_feed_dict)
        return(out)


    def Train_Iter(self, iterations, save_iterations, data, log_iteration=10, restore=True, session=None):
        #Get default session
        if session is None:
            session = tf.get_default_session()
        #Try restore
        if restore:
            self.Try_restore(session)


        self.merged = tf.summary.merge_all()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for step in range(iterations):
            print('step %d' %(step + 1))
            batch = data.next_batch(self.kwargs['Batch_size'])            #IO feed dict
            IO_feed_dict = {self.model_dict['Input_ph']: batch[0], self.model_dict['Output_ph']: batch[1]}            #Construct train dict
            train_feed_dict = {**IO_feed_dict, **self.train_dict}


            #saver block
            if(step + 1) % save_iterations == 0:

                self.saver.save(session, self.kwargs['Save_dir'] + '/mdl/' + self.Model_name + '.ckpt', global_step=self.global_step)

            #logger block
            if(step + 1) % log_iteration == 0:
                test_feed_dict = {**IO_feed_dict, **self.test_dict}               #Construst Test dict
                summary, train_accuracy, glo_step = session.run([self.merged, self.accuracy, self.global_step], \
                        feed_dict=test_feed_dict)
                self.log_writer.add_summary(summary, glo_step)


                #print(train_accuracy)




            #Train Step

            _, loss = session.run([self.train_step ,self.loss], feed_dict=train_feed_dict)
            print (loss)
        coord.request_stop()
        coord.join(threads)



