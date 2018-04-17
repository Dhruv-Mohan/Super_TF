from utils.builder import Builder
import tensorflow as tf
import os
from Model_builder.Architecture.Gan.Stargan import Stargan

#from Model_builder.Architecture.Classification.Lenet import Lenet

Segnets = os.path.dirname(os.path.realpath(__file__)) +'/Architecture/Segmentation'
Classnets = os.path.dirname(os.path.realpath(__file__)) +'/Architecture/Classification'
Seqnets = os.path.dirname(os.path.realpath(__file__)) +'/Architecture/Sequencegen'
Gans = os.path.dirname(os.path.realpath(__file__)) +'/Architecture/Gan'

segnet_archs = os.listdir(Segnets)
classnet_archs = os.listdir(Classnets)
seqnet_archs = os.listdir(Seqnets)
gan_archs = os.listdir(Gans)

for classnet in classnet_archs:
    #if ".pyc" not in segnet and "__init__" not in segnet and ".py" in segnet:
        #exec("from Model_builder.Architecture.Segmentation." + segnet[:-3] + " import " + segnet[:-3] )

    if ".pyc" not in  classnet and "__init__" not in  classnet and ".py" in  classnet:
        exec("from Model_builder.Architecture.Classification." + classnet[:-3] + " import " + classnet[:-3] )

    #if ".pyc" not in  seqnet and "__init__" not in  seqnet and ".py" in  seqnet:
        #exec("from Model_builder.Architecture.Sequencegen." + seqnet[:-3] + " import " + seqnet[:-3] )

    #if ".pyc" not in  gan and "__init__" not in  gan and ".py" in  gan:
        #exec("from Model_builder.Architecture.Gan." + gan[:-3] + " import " + gan[:-3] )

for segnet in segnet_archs:
    if ".pyc" not in segnet and "__init__" not in segnet and ".py" in segnet:
        exec("from Model_builder.Architecture.Segmentation." + segnet[:-3] + " import " + segnet[:-3] )

for sequnet in seqnet_archs:
    if ".pyc" not in sequnet and "__init__" not in sequnet and ".py" in sequnet:
        exec("from Model_builder.Architecture.Sequencegen." + sequnet[:-3] + " import " + sequnet[:-3] )

for gannet in gan_archs:
    if ".pyc" not in gannet and "__init__" not in gannet and ".py" in gannet:
        exec("from Model_builder.Architecture.Gan." + gannet[:-3] + " import " + gannet[:-3] )


class Factory(object):
    """Factory class to build DNN Architectures"""
    #Look into adding a datastructure to keep track of last layer added to the graph

    def get_model(self):

        print('Building ' + self.model_name+'()')
        #return (eval('Build_' + self.model_name+'(self.kwargs)'))
        return (eval(self.model_name+'(self.kwargs)'))

    def __init__(self, **kwargs):
        #TODO: WRITE ERROR HANDLER AND PARSER 
        self.model_name = kwargs['Model_name']
        self.kwargs = kwargs
        #Add more params as required