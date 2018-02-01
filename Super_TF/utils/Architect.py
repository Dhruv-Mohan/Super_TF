from abc import ABC, abstractmethod
import tensorflow as tf


class Architect(ABC):
    """Inherited by network architectures
        provides interface for functions used my model class
    """
    def __init__(self):
        self.global_step=tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

    @abstractmethod
    def construct_control_dict(self):
        pass

    @abstractmethod
    def set_accuracy_op(self):
        pass

    @abstractmethod
    def set_train_ops(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def construct_loss(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

