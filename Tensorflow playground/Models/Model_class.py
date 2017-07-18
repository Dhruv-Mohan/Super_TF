from Model_builder.Build_factory import Factory
import tensorflow as tf
class Model_class(object):
    """This class contains the model architecture, optimizer and loss function"""


    def __init__(self, **kwargs):
        self.Model_name = kwargs['Model_name']
        self.kwargs = kwargs


    def Set_loss(self, params):
        self.loss = params  #Const loss func
        if self.kwargs['Summary']:
            tf.summary.scalar('cross_entropy', self.loss)


    def Set_optimizer(self, params):
        self.optimizer = params #Const optimizer params 
        self.train_step = self.optimizer.minimize(self.loss)

    def Construct_Model(self):
        [self.input_placeholder, self.output_placeholder, self.output] = Factory(**self.kwargs).get_model()
