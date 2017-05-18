from network import Network
from layers_1 import Relu, Softmax, Linear, Convolution,  ConvolutionSquare, Pooling, PoolingSquare, Batchnorm,Dropout,RNN4D,RNN4D_noshare,LSTMnoshare4D,LSTMnoshare4D_back
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
name = 'lstmnosharebi'

batch_size = 20
model.add(ConvolutionSquare('conv1',5,1,64,0.05))  #76 * 46
model.add(Batchnorm('bn1'))
model.add(Relu('relu1'))
model.add(PoolingSquare('pool1', 2)) #38*23

model.add(ConvolutionSquare('conv1',5,64,128,0.05))  #34 * 19
model.add(Batchnorm('bn2'))
model.add(Relu('relu2'))
model.add(PoolingSquare('pool2', 2, False)) #17 * 10
model.add(LSTMnoshare4D('rnn1', batch_size, 128, 32, 10, 0.1))		#	HIDDEN_DIM, INPUT_DIM, 0.1
model.add(LSTMnoshare4D_back('rnnb', batch_size, 128, 32, 32, 0.1))	


model.add(Linear('fc7',128*17*32 , 2048, 0.005))
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

optim = SGDOptimizerHalf(learning_rate=0.000001, weight_decay=0, momentum=0.9)
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
          batch_size=batch_size, max_epoch=50, disp_freq=250, test_freq=250, half_step = 2, network_name = name)
#batch <30
