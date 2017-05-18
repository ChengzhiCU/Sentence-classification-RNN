from network import Network
from layers import Relu, Softmax, Linear, Convolution, Pooling, Batchnorm,Dropout, PoolingSquare
from loss import CrossEntropyLoss
from optimizer import SGDOptimizerHalf, AdagradOptimizer,RMSpropOptimizer
from solve_net import solve_net
from load_ag_news import load_ag_onehotembed
from utils import shared_dataset

import theano.tensor as T

#train_data, test_data, train_label, test_label = load_ag_news()
#datasets = load_ag_news()
datasets = load_ag_onehotembed()
print 'loading finished'
train_data, train_label = datasets[0]
test_data, test_label = datasets[1]


model = Network()
'''
model.add(Convolution('conv1', 7, 1, 256, 0.05))   
model.add(Relu('relu1'))
model.add(Pooling('pool1', 3))                  # output size: N x 4 x 12 x 12
model.add(Convolution('conv2', 7, 256, 256, 0.05))   # output size: N x 8 x 8 x 8
#model.add(Batchnorm('bn2'))
model.add(Relu('relu2'))
model.add(Pooling('pool2', 3))                  # output size: N x 8 x 4 x 4

model.add(Convolution('conv3', 3, 256, 256, 0.05))   

model.add(Relu('relu3'))
model.add(Convolution('conv4', 3, 256, 256, 0.05))   
model.add(Relu('relu4'))
model.add(Convolution('conv5', 3, 256, 256, 0.05))   
model.add(Relu('relu5'))
model.add(Convolution('conv6', 3, 256, 256, 0.05))   
#model.add(Batchnorm('bn6'))
model.add(Relu('relu6'))
model.add(Pooling('pool6', 3))   

model.add(Linear('fc7', 34*70*256, 1024, 0.005))     
#model.add(Batchnorm('bn7'))
model.add(Relu('relu7'))  

model.add(Linear('fc8', 1024, 4, 0.005))
model.add(Softmax('softmax'))
'''

model.add(Convolution('conv1', 7, 1, 25, 0.05))   
model.add(Relu('relu1'))
model.add(PoolingSquare('pool1', 3))

model.add(Convolution('conv2', 7, 25, 20, 0.05))   
model.add(Relu('relu4'))
model.add(PoolingSquare('pool5', 3))

model.add(Convolution('conv4', 3, 20, 256, 0.05))   
model.add(Relu('relu4'))
model.add(Convolution('conv5', 3, 256, 20, 0.05))   
model.add(Relu('relu5'))

model.add(Dropout('Dropout2',0.6))
model.add(Linear('fc7', 380, 30, 0.005))     
#model.add(Batchnorm('bn7'))
model.add(Relu('relu7'))  
model.add(Dropout('Dropout1',0.6))
model.add(Linear('fc8', 30, 4, 0.005))
model.add(Softmax('softmax'))

loss = CrossEntropyLoss(name='xent')

optim = SGDOptimizerHalf(learning_rate=0.001, weight_decay=0, momentum=0.9)
#optim = AdagradOptimizer(learning_rate = 0.02)
#optim=RMSpropOptimizer(learning_rate=0.01, rho=0.9, eps=1e-8)

input_placeholder = T.ftensor4('input')
label_placeholder = T.fmatrix('label')
trainable = T.iscalar('trainable')
model.compile(input_placeholder, label_placeholder, trainable, loss, optim)

solve_net(model, optim, train_data, train_label, test_data, test_label,
          batch_size=4, max_epoch=10000, disp_freq=2500, test_freq=8000)
#batch <30
