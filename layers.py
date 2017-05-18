import theano.tensor as T
import numpy as np
import theano
from utils import sharedX
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import bn


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable

    def forward(self, inputs):
        pass

    def params(self):
        pass


rng = np.random.RandomState(1234)
srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))

def drop(input, p, rng=rng):            
    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask

class Dropout(Layer):
    def __init__(self, name, p=0.5):
        super(Dropout, self).__init__(name)
        self.p = p

    def forward(self, inputs, is_train):
        train_output = drop(input = np.cast[theano.config.floatX](1./self.p) * inputs, p=self.p,rng=rng)
        return T.switch(T.neq(is_train, 0), train_output, inputs)


class Batchnorm(Layer):
    def __init__(self,name):
        super(Batchnorm,self).__init__(name)
        gamma=1
        beta=0
        self.gamma = sharedX(gamma)
        self.beta = sharedX(beta)
    def forward(self,inputs):

        return bn.batch_normalization(
            inputs = inputs,
            gamma = self.gamma,
            beta = self.beta,
            mean = T.mean(inputs),
            std = T.std(inputs)
            )
    def params(self):
        return [self.gamma, self.beta]



class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, inputs):
        return T.nnet.relu(inputs)


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, inputs):
        return T.nnet.sigmoid(inputs)

class Softmax(Layer):
    def __init__(self, name):
        super(Softmax, self).__init__(name)

    def forward(self, inputs):
        return T.nnet.softmax(inputs)

class Linear(Layer):
    def __init__(self, name, inputs_dim, num_output, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.W = sharedX(np.random.randn(inputs_dim, num_output) * init_std, name=name + '/W')
        self.b = sharedX(np.zeros((num_output)), name=name + '/b')

    def forward(self, inputs): 
        inputs_dim=inputs.shape[0]
        inputs = inputs.reshape((-1, inputs_dim))
        return T.dot(inputs, self.W) + self.b

    def params(self):
        return [self.W, self.b]

class Convolution(Layer):
    def __init__(self, name, kernel_size, num_input, num_output, init_std):
        super(Convolution, self).__init__(name, trainable=True)
        W_shape = (num_output, num_input, kernel_size,1)
        self.W = sharedX(np.random.randn(*W_shape) * init_std, name=name + '/W')
        self.b = sharedX(np.zeros((num_output)), name=name + '/b')

    def forward(self, inputs):
        return conv2d(inputs, self.W) + self.b.dimshuffle('x', 0, 'x', 'x')

    def params(self):
        return [self.W, self.b]

class ConvolutionSquare(Layer):
    def __init__(self, name, kernel_size, num_input, num_output, init_std):
        super(ConvolutionSquare, self).__init__(name, trainable=True)
        W_shape = (num_output, num_input, kernel_size,kernel_size)
        self.W = sharedX(np.random.randn(*W_shape) * init_std, name=name + '/W')
        self.b = sharedX(np.zeros((num_output)), name=name + '/b')

    def forward(self, inputs):
        return conv2d(inputs, self.W) + self.b.dimshuffle('x', 0, 'x', 'x')

    def params(self):
        return [self.W, self.b]


class Pooling(Layer):
    def __init__(self, name, kernel_size):
        super(Pooling, self).__init__(name)
        self.kernel_size = kernel_size

    def forward(self, inputs): 
        return pool_2d(inputs, (self.kernel_size, 1), ignore_border=True)

class PoolingSquare(Layer):
    def __init__(self, name, kernel_size):
        super(PoolingSquare, self).__init__(name)
        self.kernel_size = kernel_size

    def forward(self, inputs): 
        return pool_2d(inputs, (self.kernel_size, self.kernel_size), ignore_border=True)
        
class Linear(Layer):
    def __init__(self, name, inputs_dim, num_output, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.W = sharedX(np.random.randn(inputs_dim, num_output) * init_std, name=name + '/W')
        self.b = sharedX(np.zeros((num_output)), name=name + '/b')

    def forward(self, inputs):
        # Your codes here
        inputs = inputs.reshape((inputs.shape[0],-1))
        return T.dot(inputs, self.W) + self.b.dimshuffle('x', 0)


    def params(self):
        return [self.W, self.b]

class RNN4D(Layer):
    def __init__(self, name, batch_size, channel, hidden_dim, input_dim, init_std):
        super(RNN4D, self).__init__(name, trainable=True)
        self.hidden_dim = hidden_dim
        self.Wx = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name=name + '/Wx')
        self.Wh = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Wh')
        self.b = sharedX(np.zeros((batch_size, channel, hidden_dim)), name=name + '/b')
        self.batch_size = batch_size
        self.channel = channel
    def forward(self, inputs):
        insz = inputs.shape
        inputs_sh = inputs.dimshuffle(2,0,1,3)
        hs, _ = theano.scan(
            self.step,         
            sequences=inputs_sh,
            outputs_info=[None, T.zeros((self.batch_size, self.channel, self.hidden_dim))]) 

        out = hs[0].reshape((-1, self.batch_size, self.channel, self.hidden_dim))
        return out.dimshuffle(1,2,0,3) # out shape : batch,channel,hidden_dim

    def step(self, x_t, h_t_prev):
        x_tsz=x_t.shape
        print 'x_t shape',x_tsz
        # batch_size * 128 * 6
        h_t = T.tanh(T.dot(x_t, self.Wx) + T.dot(h_t_prev, self.Wh) + self.b)
        return [h_t, h_t]

    def params(self):
        return [self.Wx, self.Wh, self.b]

    
class LSTM(Layer):
    def __init__(self, name, hidden_dim, input_dim, init_std):
        super(LSTM, self).__init__(name, trainable=True)
        self.hidden_dim = hidden_dim

        # input gate
        self.Wi = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name=name + '/Wi')
        self.Ui = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Ui')
        self.Vi = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Vi')
        self.bi = sharedX(np.zeros((hidden_dim)), name=name + '/bi')        
        # forget gate
        self.Wf = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name=name + '/Wf')
        self.Uf = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Uf')
        self.Vf = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Vf')
        self.bf = sharedX(np.zeros((hidden_dim)), name=name + '/bf')
        # output gate
        self.Wo = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name=name + '/Wo')
        self.Uo = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Uo')
        self.Vo = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Vo')
        self.bo = sharedX(np.zeros((hidden_dim)), name=name + '/bo')
        # cell
        self.Wc = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name=name + '/Wc')
        self.Uc = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Uc')
        self.bc = sharedX(np.zeros((hidden_dim)), name=name + '/bc')

    def forward(self, inputs):
        results, _ = theano.scan(
            self.step,
            sequences=inputs,
            outputs_info=[T.zeros(self.hidden_dim), T.zeros(self.hidden_dim)])        
        return results[0]

    def step(self, x_t, h_t_prev, c_t_prev):
        i_t= T.nnet.sigmoid(T.dot(x_t,self.Wi)+T.dot(h_t_prev,self.Ui)+T.dot(c_t_prev,self.Vi)+self.bi)
        f_t= T.nnet.sigmoid(T.dot(x_t,self.Wf)+T.dot(h_t_prev,self.Uf)+T.dot(c_t_prev,self.Vf)+self.bf)
        o_t= T.nnet.sigmoid(T.dot(x_t,self.Wo)+T.dot(h_t_prev,self.Uo)+T.dot(c_t_prev,self.Vo)+self.bo)
        cb=  T.tanh(T.dot(x_t,self.Wc)+T.dot(h_t_prev,self.Uc)+self.bc)
        c_t=f_t*c_t_prev+i_t*cb
        h_t=o_t*T.tanh(c_t)
        return [h_t,c_t]


    def params(self):

        return [self.Wi, self.Ui, self.Vi, self.bi, self.Wf, self.Uf, self.Vf, self.bf, self.Wo, self.Uo, self.Vo, self.bo, self.Wc, self.Uc, self.bc]   