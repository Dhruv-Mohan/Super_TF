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


    def Construct_Writers(self,session):
        self.saver = tf.train.Saver()
        if self.kwargs['Summary']:
            self.train_writer = tf.summary.FileWriter(self.kwargs['Save_dir'] + '/train/', session.graph)
            self.test_writer = tf.summary.FileWriter(self.kwargs['Save_dir'] + '/test/')
        else:
            graph_writer = tf.summary.FileWriter(self.kwargs['Save_dir'] + '/train/')
            graph_writer.add_graph(session.graph)
            graph_writer.close();


    def Construct_Predict_op(self):
        with tf.name_scope('Predict'):
            self.Predict_op = tf.argmax(self.model_dict['Output'], 1)


    def Try_restore(self,session):
        latest_ckpt = tf.train.latest_checkpoint(self.kwargs['Save_dir'] + '/mdl/')
        if latest_ckpt:
            print('Ckpt_found')
            print(latest_ckpt)
            self.saver.restore(session,latest_ckpt)
        else:
            print('Ckpt_not_found')


    def Predict(self, session, input_data):
        #Construct Predict dict
        IO_predict_dict = {self.model_dict['Input_ph']: input_data}

        #Construct Predict feed_dict
        predict_feed_dict = {**IO_predict_dict, **self.test_dict}

        out = session.run([self.Predict_op], feed_dict=predict_feed_dict)
        return(out)


    def Train_Iter(self, session, iterations, data, restore=True):
        #Try restore
        if restore:
            self.Try_restore(session)

        if self.kwargs['Summary']:
            self.merged = tf.summary.merge_all()

        for step in range(iterations):
            batch = data.next_batch(self.kwargs['Batch_size'])
            #IO feed dict
            IO_feed_dict = {self.model_dict['Input_ph']: batch[0], self.model_dict['Output_ph']: batch[1]}
            #Construct train dict
            train_feed_dict = {**IO_feed_dict, **self.train_dict}
            
            #Test block TODO: Add param for variable test iteration
            if(step + 1) % 100 == 0:
                #Construst Test dict
                test_feed_dict = {**IO_feed_dict, **self.test_dict}
                self.saver.save(session, self.kwargs['Save_dir'] + '/mdl/lenet.ckpt', global_step=self.global_step)
                if self.kwargs['Summary']:
                    summary, train_accuracy = session.run([self.merged, self.accuracy], \
                        feed_dict=test_feed_dict)
                    self.test_writer.add_summary(summary, step)
                else:
                    train_accuracy = session.run([self.accuracy], \
                        feed_dict=test_feed_dict)

                print(train_accuracy)
            print('step %d' %(step))

            #Train Step
            if self.kwargs['Summary']:
                summary,  _ = session.run([self.merged, self.train_step], \
                    feed_dict=train_feed_dict)
                self.train_writer.add_summary(summary, step)
            else:
                _ = session.run([self.train_step], \
                    feed_dict=train_feed_dict)



