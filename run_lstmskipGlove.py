from network import Network
from layers_1 import Relu, Softmax, Linear, Convolution,  ConvolutionSquare, Pooling, PoolingSquare, Batchnorm,Dropout,RNN4D,LSTM4D
from loss import CrossEntropyLoss
from optimizer import SGDOptimizer, AdagradOptimizer, SGDOptimizerHalf
from solve_net import solve_net
from load_ag_news import load_ag_onehotembed,load_ag_Glove_embed
from utils import shared_dataset

import theano.tensor as T

#train_data, test_data, train_label, test_label = load_ag_news()
#datasets = load_ag_news()
#datasets = load_ag_onehotembed()

model = Network()
'''
model.add(Convolution('conv1', 7, 1, 256, 0.05))
model.add(ConvolutionT('conv1T', 7, 1, 256, 0.05))   #74*44
model.add(Relu('relu1'))
model.add(PoolingSquare('pool1', 2))                  # output size: 37*22
model.add(Convolution('conv2', 6, 256, 256, 0.05))   # output size: 32*22
model.add(ConvolutionT('conv2T', 5, 1, 256, 0.05)) 	# 32*18
#model.add(Batchnorm('bn2'))
model.add(Relu('relu2'))
model.add(PoolingSquare('pool2', 2))                  # output size: 16*9

model.add(ConvolutionSquare('conv3', 3, 256, 256, 0.05))   

model.add(Relu('relu3'))
model.add(Convolution('conv4', 3, 256, 256, 0.05))   #12*7
model.add(Relu('relu4'))
model.add(ConvolutionT('conv5', 2, 256, 256, 0.05))   #12*6
model.add(PoolingSquare('pool6', 2))   # 
'''
batch_size = 20
model.add(ConvolutionSquare('conv1',5,1,64,0.05))  #76 * 46
model.add(Batchnorm('bn1'))
model.add(Relu('relu1'))
model.add(PoolingSquare('pool1', 2)) #38*23

model.add(ConvolutionSquare('conv1',5,64,128,0.05))  #34 * 19
model.add(Batchnorm('bn2'))
model.add(Relu('relu2'))
model.add(PoolingSquare('pool2', 2, False)) #17 * 10
#model.add(RNN4D('rnn1', batch_size, 128, 16, 6,0.1))		#	HIDDEN_DIM, INPUT_DIM, 0.1
model.add(LSTM4D('rnn1', batch_size, 128, 32, 10 ,0.1, True))		#	HIDDEN_DIM, INPUT_DIM, 0.1


model.add(Linear('fc7',128*32*17 + 128*10*17 , 2048, 0.005))
model.add(Batchnorm('bn3'))
model.add(Relu('relu7'))  
model.add(Dropout('dropout1'))
model.add(Linear('fc7',2048 , 2048, 0.005))
model.add(Batchnorm('bn4'))
model.add(Relu('relu7'))
model.add(Dropout('dropout2'))
model.add(Linear('fc8', 2048, 4, 0.005))
model.add(Softmax('softmax'))

loss = CrossEntropyLoss(name='xent')

optim = SGDOptimizerHalf(learning_rate=0.00001, weight_decay=0, momentum=0.9)
#optim = AdagradOptimizer(learning_rate = 0.02)

input_placeholder = T.ftensor4('input')
label_placeholder = T.fmatrix('label')
trainable = T.iscalar('trainable')
model.compile(input_placeholder, label_placeholder, trainable, loss, optim)

datasets = load_ag_Glove_embed(group=1)
print 'loading finished'
train_data, train_label = datasets[0]
test_data, test_label = datasets[1]

solve_net(model, optim, train_data, train_label, test_data, test_label,
          batch_size=batch_size, max_epoch=50, disp_freq=250, test_freq=1000)
#batch <30
