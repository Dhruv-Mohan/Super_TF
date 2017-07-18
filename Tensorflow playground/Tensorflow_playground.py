from Models.Model_class import Model_class
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



_SUMMARY_           = False
_BATCH_SIZE_        = 50
_IMAGE_WIDTH_       = 28
_IMAGE_HEIGHT_      = 28
_IMAGE_CSPACE_      = 1
_CLASSES_           = 10
_MODEL_NAME_        = 'Lenet'
_ITERATIONS_        = 200

def main():
    ''' Main function'''

    #Load minst data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


    #Construct model
    Simple_DNN = Model_class(Model_name=_MODEL_NAME_, Summary=_SUMMARY_, \
        Batch_size=_BATCH_SIZE_, Image_width=_IMAGE_WIDTH_, Image_height=_IMAGE_HEIGHT_, Image_cspace=_IMAGE_CSPACE_, Classes=_CLASSES_)

    Simple_DNN.Construct_Model()


    #Set loss
    with tf.name_scope('Cross_entropy'):
        Simple_DNN.Set_loss(tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Simple_DNN.output_placeholder, logits=Simple_DNN.output)))



    #Set optimizer
    with tf.name_scope('Train'):
        Simple_DNN.Set_optimizer(tf.train.AdamOptimizer(1e-4))



    #Op to check accuracy
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(Simple_DNN.output, 1), tf.argmax(Simple_DNN.output_placeholder, 1))

        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    #Train and save
    if _SUMMARY_:
        merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    with tf.Session() as sess:

    #Construct writers if summary
        if _SUMMARY_:
            train_writer = tf.summary.FileWriter('E:/Ttest/train', sess.graph)
            test_writer = tf.summary.FileWriter('E:/Ttest/test')
        else:
            graph_writer = tf.summary.FileWriter('E:/Ttest/train')
            graph_writer.add_graph(sess.graph)
            graph_writer.close();
    #Init global vars
    sess.run(tf.global_variables_initializer())
    for i in range(_ITERATIONS_):
        batch = mnist.train.next_batch(_BATCH_SIZE_)
        if i % 100 == 0:

            #Save ckpt
            saver.save(sess, "E:/Ttest/mdl/lenet.ckpt", i)

            #Calculate accuracy
            if _SUMMARY_:
                summary, train_accuracy = sess.run([merged, accuracy], \
                    feed_dict={Simple_DNN.input_placeholder: batch[0], Simple_DNN.output_placeholder: batch[1]})
                test_writer.add_summary(summary, i)
            else:
                train_accuracy = sess.run([accuracy], \
                    feed_dict={Simple_DNN.input_placeholder: batch[0], Simple_DNN.output_placeholder: batch[1]})

            print(train_accuracy)
        print('step %d' %(i))

        #Train Step
        if _SUMMARY_:
            summary, _ = sess.run([merged, Simple_DNN.train_step], \
                feed_dict={Simple_DNN.input_placeholder: batch[0], Simple_DNN.output_placeholder: batch[1]})
            train_writer.add_summary(summary, i)
        else:
            _ = sess.run([Simple_DNN.train_step], \
                feed_dict={Simple_DNN.input_placeholder: batch[0], Simple_DNN.output_placeholder: batch[1]})





if __name__ == "__main__":
    main()
