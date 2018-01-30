class Architect(object):
    """Inherited by network architectures
        provides interface for functions used my model class
    """

    def set_train_dict(self): {}

    def set_test_dict(self): {}

    def set_accuracy_op(self): {}

    def set_train_ops(self): {}

    def predict(self): {}

    def build(self): {}

    def set_loss(self): {}

    def train(self): {}

