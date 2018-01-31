from abc import ABC, abstractmethod

class Architect(ABC):
    """Inherited by network architectures
        provides interface for functions used my model class
    """
    @abstractmethod
    def set_train_dict(self):
        pass

    @abstractmethod
    def set_test_dict(self):
        pass

    @abstractmethod
    def set_accuracy_op(self):
        pass

    @abstractmethod
    def set_train_ops(self):
        pass

    @abstractmethod
    def construct_predict_op(self):
        pass

    @abstractmethod
    def construct_loss(self):
        pass

    @abstractmethod
    def train(self):
        pass

