from Model_builder.Build_factory import Factory
import tensorflow as tf
class Model_class(object):
    """This class contains the model architecture, optimizer and loss function"""


    def __init__(self, **kwargs):
        self.Model_name = kwargs['Model_name']
        self.kwargs = kwargs
        self.global_step=tf.Variable(0,trainable=False,dtype=tf.int32,name='global_step')




    def Set_loss(self, params):
        self.loss = params  #Const loss func
        if self.kwargs['Summary']:
            tf.summary.scalar('cross_entropy', self.loss)


    def Set_optimizer(self, params):
        self.optimizer = params #Const optimizer params 
        self.train_step = self.optimizer.minimize(self.loss,global_step=self.global_step)


    def Construct_Model(self):
        [self.input_placeholder, self.output_placeholder, self.output] = Factory(**self.kwargs).get_model()


    def Construct_Accuracy_op(self):
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.output_placeholder, 1))
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


    def Try_restore(self,session):
        latest_ckpt = tf.train.latest_checkpoint(self.kwargs['Save_dir'] + '/mdl/')
        if latest_ckpt:
            print('Ckpt_found')
            print(latest_ckpt)
            self.saver.restore(session,latest_ckpt)
        else:
            print('Ckpt_not_found')


    def Train_Iter(self, session, iterations, data , restore=True):

        #Try restore
        if restore:
            self.Try_restore(session)

        if self.kwargs['Summary']:
            self.merged = tf.summary.merge_all()

        for step in range(iterations):
            batch = data.next_batch(self.kwargs['Batch_size'])
            if(step + 1) % 100 == 0:
                #enter test and saver block
                #add saving op
                self.saver.save(session, self.kwargs['Save_dir'] + '/mdl/lenet.ckpt',global_step=self.global_step)
                if self.kwargs['Summary']:
                    summary, train_accuracy = session.run([self.merged, self.accuracy], \
                        feed_dict={self.input_placeholder: batch[0], self.output_placeholder: batch[1]})
                    self.test_writer.add_summary(summary, step)
                else:
                    train_accuracy = session.run([self.accuracy], \
                        feed_dict={self.input_placeholder: batch[0], self.output_placeholder: batch[1]})

                print(train_accuracy)
            print('step %d' %(step))

            #Train Step
            if self.kwargs['Summary']:
                summary,  _ = session.run([self.merged, self.train_step], \
                    feed_dict={self.input_placeholder: batch[0], self.output_placeholder: batch[1]})
                self.train_writer.add_summary(summary, step)
            else:
                _ = session.run([self.train_step], \
                    feed_dict={self.input_placeholder: batch[0], self.output_placeholder: batch[1]})

