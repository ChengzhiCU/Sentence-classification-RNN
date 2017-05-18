from network import Network
from layers import Relu, Softmax, Linear, Convolution, Pooling, Batchnorm,Dropout
from loss import CrossEntropyLoss
from optimizer import SGDOptimizer, AdagradOptimizer
from solve_net import solve_net
from load_ag_news import load_ag_news,load_ag_onehotembed
from utils import shared_dataset

import theano.tensor as T

#train_data, test_data, train_label, test_label = load_ag_news()
#datasets = load_ag_news()
datasets = load_ag_onehotembed()
print 'loading finished'
train_data, train_label = datasets[0]
test_data, test_label = datasets[1]


model = Network()
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

loss = CrossEntropyLoss(name='xent')

optim = SGDOptimizer(learning_rate=0.001, weight_decay=0.00001, momentum=0.9)
#optim = AdagradOptimizer(learning_rate = 0.02)

input_placeholder = T.ftensor4('input')
label_placeholder = T.fmatrix('label')
model.compile(input_placeholder, label_placeholder, loss, optim)

solve_net(model, train_data, train_label, test_data, test_label,
          batch_size=10, max_epoch=10, disp_freq=50, test_freq=200)
#batch <30
