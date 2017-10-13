from utils.builder import Builder
import tensorflow as tf
from Model_builder.Architecture.Classification import *
from Model_builder.Architecture.Segmentation import * 
class Factory(object):
    """Factory class to build DNN Architectures"""
    #Look into adding a datastructure to keep track of last layer added to the graph

    def get_model(self):
        print('Building ' + self.model_name+'()')
        return (eval(self.model_name + '.Build_' + self.model_name+'()'))

    def __init__(self, **kwargs):
        #TODO: WRITE ERROR HANDLER AND PARSER 
        self.model_name = kwargs['Model_name']
        self.summary = kwargs['Summary']
        self.kwargs = kwargs
        #Add more params as required