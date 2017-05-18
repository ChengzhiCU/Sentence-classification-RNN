import theano
import theano.tensor as T
from utils import LOG_INFO
import numpy as np


class Network(object):
    def __init__(self):
        self.layer_list = []
        self.params = []
        self.num_layers = 0

    def add(self, layer):
        self.layer_list.append(layer)
        self.num_layers += 1
        if layer.trainable:
            self.params += layer.params()

    def set_lr(self, learning_rate, optimizer):
        optimizer.set_lr(learning_rate)

    def get_lr(self, optimizer):
        return optimizer.get_lr()

    def compile(self, input_placeholder, label_placeholder, trainable, loss, optimizer):
        x = input_placeholder        
        for k in range(self.num_layers):
            if 'dropout' in self.layer_list[k].name:
                print 'find drop'
                x = self.layer_list[k].forward(x,trainable)
            else:
                x = self.layer_list[k].forward(x)
            
        self.loss = loss.forward(x, label_placeholder)
        self.updates = optimizer.get_updates(self.loss, self.params)
        self.accuracy = T.mean(T.eq(T.argmax(x, axis=-1),
                               T.argmax(label_placeholder, axis=-1)))

        LOG_INFO('start compiling model...')
        self.train = theano.function(
            inputs=[input_placeholder, label_placeholder],
            outputs=[self.loss, self.accuracy],
            updates=self.updates,
            givens ={
            trainable: np.cast['int32'](1)
            },
            allow_input_downcast=True)
        '''
        self.debuger = theano.function(
            inputs=[input_placeholder],
            outputs=[x,z],
            #updates=self.updates,
            allow_input_downcast=True)
        '''
        self.test = theano.function(
            inputs=[input_placeholder, label_placeholder],
            outputs=[self.accuracy, self.loss],
            givens ={
            trainable: np.cast['int32'](0)
            },
            allow_input_downcast=True)
        '''
        self.predict = theano.function(
            inputs=[input_placeholder],
            outputs=[x],
            allow_input_downcast=True)
        '''
        LOG_INFO('model compilation done!')
