import theano.tensor as T
import numpy as np
import numpy
import theano
from utils import sharedX
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import bn
import theano.tensor.nnet.nnet as N

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

class ConvolutionSquareConst(Layer):
    def __init__(self, name, kernel_size, num_input, num_output, init_value):
        super(ConvolutionSquareConst, self).__init__(name, trainable=True)
        W_shape = (num_output, num_input, kernel_size,kernel_size)
        self.W = sharedX(np.ones((1,1,1,1)) * init_value, name=name + '/W')
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
    def __init__(self, name, kernel_size, ignore_border= True):
        super(PoolingSquare, self).__init__(name)
        self.kernel_size = kernel_size
        self.ignore_border = ignore_border

    def forward(self, inputs): 
        return pool_2d(inputs, (self.kernel_size, self.kernel_size), ignore_border=self.ignore_border)
        
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

class LSTM4D(Layer):
    def __init__(self, name, batch_size, channel, hidden_dim, input_dim, init_std, is_skip=False):
        super(LSTM4D, self).__init__(name, trainable=True)
        self.hidden_dim = hidden_dim
        # Your codes here, do weights initilization
        self.Wi = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name = name + '/Wi')
        self.Wf = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name = name + '/Wf')
        self.Wo = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name = name + '/Wo')
        self.Wc = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name = name + '/Wc')
        self.Ui = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Ui')
        self.Uf = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Uf')
        self.Uo = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Uo')
        self.Uc = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Uc')
        self.Vi = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Vi')
        self.Vf = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Vf')
        self.Vo = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Vo')
        self.bi = sharedX(np.zeros((hidden_dim)), name=name + '/bi')
        self.bf = sharedX(np.zeros((hidden_dim)), name=name + '/bf')
        self.bo = sharedX(np.zeros((hidden_dim)), name=name + '/bo')
        self.bc = sharedX(np.zeros((hidden_dim)), name=name + '/bc')
        self.batch_size = batch_size
        self.channel = channel
        self.is_skip = is_skip

    def forward(self, inputs):
        insz = inputs.shape
        inputs_sh = inputs.dimshuffle(2,0,1,3)
        hs, _ = theano.scan(
            self.step,
            sequences=inputs_sh,
            outputs_info=[None,  T.zeros((self.batch_size, self.channel, self.hidden_dim)), T.zeros((self.batch_size, self.channel, self.hidden_dim))])
        # outputs_info : initialize here
        out = hs[0].reshape((-1, self.batch_size, self.channel, self.hidden_dim))
        if self.is_skip == False:
            return out.dimshuffle(1,2,0,3)
        elif self.is_skip == True:
            return T.concatenate([out.dimshuffle(1,2,0,3).flatten(2), inputs.flatten(2)], axis=1)

    def step(self, x_t, h_t_prev ,c_t_prev):

        i_t = N.sigmoid(T.dot(x_t,self.Wi) + T.dot(h_t_prev,self.Ui) + T.dot(h_t_prev,self.Vi) + self.bi)
        f_t = N.sigmoid(T.dot(x_t,self.Wf) + T.dot(h_t_prev,self.Uf) + T.dot(h_t_prev,self.Vf) + self.bf)
        o_t = N.sigmoid(T.dot(x_t,self.Wo) + T.dot(h_t_prev,self.Uo) + T.dot(h_t_prev,self.Vo) + self.bo)
        c_hat_t = T.tanh(T.dot(x_t,self.Wc) + T.dot(h_t_prev,self.Uc) + self.bc)
        c_t = f_t * c_t_prev + i_t * c_hat_t
        h_t = o_t * T.tanh(c_t)
        return [h_t, h_t, c_t] # the real output, hidden state, cell state

    def params(self):
        # Your codes here
        return [self.Wi, self.Wf, self.Wo, self.Wc, self.Ui, self.Uf, self.Uo, self.Uc, self.Vi, self.Vf, self.Vo, \
        self.bi, self.bf, self.bo, self.bc]



def get_Wres(connective,width):
    rng2 = np.random.RandomState(123555)

    z = np.asarray(rng2.uniform(low=0,high=1,size=(width,width)))
    mask = [z<connective]
    mask = numpy.asarray(mask)
    # mask is a bool matrix, indicate which element is set to zero

    rng3 = numpy.random.RandomState(666)
    raw = np.asarray(rng3.uniform(low=-0.5,high=0.5,size=(width,width)))
    Wres_raw = np.multiply(raw,mask)      
    # here we get a sparsely connected Matrix

    #print "Wres_raw[0][i]",Wres_raw[0][i].shape
    eig,other = np.linalg.eig(Wres_raw)
    max_eig_i = np.max(eig)
    max_eig_norm_i = np.sqrt(np.real(max_eig_i)**2 + np.imag(max_eig_i)**2)
    Wres_raw = Wres_raw / max_eig_norm_i*0.999
    
    #Wres = theano.shared(Wres / max_eig_norm,borrow=True)
    Wres = np.reshape(Wres_raw,(width,width))
    Wres =np.asarray(Wres,dtype=theano.config.floatX)
    return Wres

class ESN_4D(Layer):
    def __init__(self, name, batch_size, channel, hidden_dim, input_dim, init_std):
        super(ESN_4D, self).__init__(name, trainable=True)
        self.hidden_dim = hidden_dim
        self.Wx = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name=name + '/Wx')
        #self.Wh = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Wh')

        self.connective = 0.02
        width = hidden_dim 
        Wres_g = get_Wres(connective = self.connective, width = width)#wrong
        print 'Wres_g.shape', Wres_g.shape
        self.Wh = sharedX(Wres_g , name=name + '/Wh')


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
        t = out.dimshuffle(1,2,0,3) # out shape : batch,channel,hidden_dim
        return T.concatenate([t.flatten(2), inputs.flatten(2)], axis=1)

    def step(self, x_t, h_t_prev):
        x_tsz=x_t.shape
        print 'x_t shape',x_tsz
        # batch_size * 128 * 6
        h_t = T.tanh(T.dot(x_t, self.Wx) + T.dot(h_t_prev, self.Wh) + self.b)
        return [h_t, h_t]

    def params(self):
        return [] #[self.Wx, self.Wh, self.b]
'''
class LSTM4D_back(Layer):
    def __init__(self, name, batch_size, channel, hidden_dim, input_dim, init_std, iterate_length, is_skip=False):
        super(LSTM4D_back, self).__init__(name, trainable=True)
        self.hidden_dim = hidden_dim
        # Your codes here, do weights initilization
        self.Wi = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name = name + '/Wi')
        self.Wf = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name = name + '/Wf')
        self.Wo = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name = name + '/Wo')
        self.Wc = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name = name + '/Wc')
        self.Ui = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Ui')
        self.Uf = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Uf')
        self.Uo = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Uo')
        self.Uc = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Uc')
        self.Vi = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Vi')
        self.Vf = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Vf')
        self.Vo = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Vo')
        self.bi = sharedX(np.zeros((hidden_dim)), name=name + '/bi')
        self.bf = sharedX(np.zeros((hidden_dim)), name=name + '/bf')
        self.bo = sharedX(np.zeros((hidden_dim)), name=name + '/bo')
        self.bc = sharedX(np.zeros((hidden_dim)), name=name + '/bc')
        self.batch_size = batch_size
        self.channel = channel
        self.is_skip = is_skip
        self.i = theano.tensor.arange(iterate_length)
        self.iterate_length = iterate_length
        #t = theano.tensor.arange(9)
        # t = theano.tensor.arange(9).reshape((3,3))
        #t[t > 4].eval()  # an array with shape (3, 3, 3)

    def forward(self, inputs):
        insz = inputs.shape
        inputs_sh = inputs.dimshuffle(2,0,1,3) # iter_axis, batch, channel, in_dim        
        hs, _ = theano.scan(
            self.step,
            sequences=self.i,
            non_sequences = inputs_sh,
            outputs_info=[None,  T.zeros((self.batch_size, self.channel, self.hidden_dim)), T.zeros((self.batch_size, self.channel, self.hidden_dim))])

        out = hs[0].reshape((-1, self.batch_size, self.channel, self.hidden_dim))
        if self.is_skip == False:
            return out.dimshuffle(1,2,0,3)
        elif self.is_skip == True:
            return T.concatenate([out.dimshuffle(1,2,0,3).flatten(2), inputs.flatten(2)], axis=1)

    def step(self, i_, h_t_prev ,c_t_prev, inputs_sh):
        i_int = theano.tensor.cast(i_, dtype = 'int32')
        x_t = inputs_sh[self.iterate_length - 1 - i_][:]
        i_t = N.sigmoid(T.dot(x_t,self.Wi) + T.dot(h_t_prev,self.Ui) + T.dot(h_t_prev,self.Vi) + self.bi)
        f_t = N.sigmoid(T.dot(x_t,self.Wf) + T.dot(h_t_prev,self.Uf) + T.dot(h_t_prev,self.Vf) + self.bf)
        o_t = N.sigmoid(T.dot(x_t,self.Wo) + T.dot(h_t_prev,self.Uo) + T.dot(h_t_prev,self.Vo) + self.bo)
        c_hat_t = T.tanh(T.dot(x_t,self.Wc) + T.dot(h_t_prev,self.Uc) + self.bc)
        c_t = f_t * c_t_prev + i_t * c_hat_t
        h_t = o_t * T.tanh(c_t)
        return [h_t, h_t, c_t] # the real output, hidden state, cell state

    def params(self):
        # Your codes here
        return [self.Wi, self.Wf, self.Wo, self.Wc, self.Ui, self.Uf, self.Uo, self.Uc, self.Vi, self.Vf, self.Vo, \
        self.bi, self.bf, self.bo, self.bc]
'''

class LSTM4D_back(Layer):
    def __init__(self, name, batch_size, channel, hidden_dim, input_dim, init_std, is_skip=False):
        super(LSTM4D_back, self).__init__(name, trainable=True)
        self.hidden_dim = hidden_dim
        # Your codes here, do weights initilization
        self.Wi = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name = name + '/Wi')
        self.Wf = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name = name + '/Wf')
        self.Wo = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name = name + '/Wo')
        self.Wc = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name = name + '/Wc')
        self.Ui = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Ui')
        self.Uf = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Uf')
        self.Uo = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Uo')
        self.Uc = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Uc')
        self.Vi = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Vi')
        self.Vf = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Vf')
        self.Vo = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Vo')
        self.bi = sharedX(np.zeros((hidden_dim)), name=name + '/bi')
        self.bf = sharedX(np.zeros((hidden_dim)), name=name + '/bf')
        self.bo = sharedX(np.zeros((hidden_dim)), name=name + '/bo')
        self.bc = sharedX(np.zeros((hidden_dim)), name=name + '/bc')
        self.batch_size = batch_size
        self.channel = channel
        self.is_skip = is_skip

    def forward(self, inputs):
        insz = inputs.shape # 20 128 17 10
        inputs_sh = inputs.dimshuffle(2,0,1,3) # 17 20 128 10

        in_sh_sz = inputs_sh.shape #
        inputs_sh_flip = T.zeros(in_sh_sz)
        inputs_sh_flip = T.set_subtensor(inputs_sh_flip[::-1,::1,::1,::1], inputs_sh)
        # theano.tensor.set_subtensor(inputs_sh_flip,inputs_sh[::-1][:])
        hs, _ = theano.scan(
            self.step,
            sequences=inputs_sh_flip,
            outputs_info=[None,  T.zeros((self.batch_size, self.channel, self.hidden_dim)), T.zeros((self.batch_size, self.channel, self.hidden_dim))])
        # outputs_info : initialize here

        outSz = hs[0].shape
        out = T.zeros(outSz)
        out = T.set_subtensor(out[::-1,::1,::1,::1], hs[0])
        out = out.reshape((-1, self.batch_size, self.channel, self.hidden_dim))
        if self.is_skip == False:
            return out.dimshuffle(1,2,0,3)
        elif self.is_skip == True:
            return T.concatenate([out.dimshuffle(1,2,0,3).flatten(2), inputs.flatten(2)], axis=1)

    def step(self, x_t, h_t_prev ,c_t_prev):

        i_t = N.sigmoid(T.dot(x_t,self.Wi) + T.dot(h_t_prev,self.Ui) + T.dot(h_t_prev,self.Vi) + self.bi)
        f_t = N.sigmoid(T.dot(x_t,self.Wf) + T.dot(h_t_prev,self.Uf) + T.dot(h_t_prev,self.Vf) + self.bf)
        o_t = N.sigmoid(T.dot(x_t,self.Wo) + T.dot(h_t_prev,self.Uo) + T.dot(h_t_prev,self.Vo) + self.bo)
        c_hat_t = T.tanh(T.dot(x_t,self.Wc) + T.dot(h_t_prev,self.Uc) + self.bc)
        c_t = f_t * c_t_prev + i_t * c_hat_t
        h_t = o_t * T.tanh(c_t)
        return [h_t, h_t, c_t] # the real output, hidden state, cell state

    def params(self):
        # Your codes here
        return [self.Wi, self.Wf, self.Wo, self.Wc, self.Ui, self.Uf, self.Uo, self.Uc, self.Vi, self.Vf, self.Vo, \
        self.bi, self.bf, self.bo, self.bc]


class LSTMnoshare4D_back(Layer):
    def __init__(self, name, batch_size, channel, hidden_dim, input_dim, init_std, is_skip=False):
        super(LSTMnoshare4D_back, self).__init__(name, trainable=True)
        self.hidden_dim = hidden_dim
        # Your codes here, do weights initilization
        self.Wi = sharedX(np.random.randn(channel,input_dim, hidden_dim) * init_std, name = name + '/Wi')
        self.Wf = sharedX(np.random.randn(channel,input_dim, hidden_dim) * init_std, name = name + '/Wf')
        self.Wo = sharedX(np.random.randn(channel,input_dim, hidden_dim) * init_std, name = name + '/Wo')
        self.Wc = sharedX(np.random.randn(channel,input_dim, hidden_dim) * init_std, name = name + '/Wc')
        self.Ui = sharedX(np.random.randn(channel,hidden_dim, hidden_dim) * init_std, name=name + '/Ui')
        self.Uf = sharedX(np.random.randn(channel,hidden_dim, hidden_dim) * init_std, name=name + '/Uf')
        self.Uo = sharedX(np.random.randn(channel,hidden_dim, hidden_dim) * init_std, name=name + '/Uo')
        self.Uc = sharedX(np.random.randn(channel,hidden_dim, hidden_dim) * init_std, name=name + '/Uc')
        self.Vi = sharedX(np.random.randn(channel,hidden_dim, hidden_dim) * init_std, name=name + '/Vi')
        self.Vf = sharedX(np.random.randn(channel,hidden_dim, hidden_dim) * init_std, name=name + '/Vf')
        self.Vo = sharedX(np.random.randn(channel,hidden_dim, hidden_dim) * init_std, name=name + '/Vo')
        self.bi = sharedX(np.zeros((batch_size, channel, hidden_dim)), name=name + '/bi')
        self.bf = sharedX(np.zeros((batch_size, channel, hidden_dim)), name=name + '/bf')
        self.bo = sharedX(np.zeros((batch_size, channel, hidden_dim)), name=name + '/bo')
        self.bc = sharedX(np.zeros((batch_size, channel, hidden_dim)), name=name + '/bc')
        self.batch_size = batch_size
        self.channel = channel
        self.is_skip = is_skip

    def forward(self, inputs):
        #insz = inputs.shape
        #inputs_sh = inputs.dimshuffle(2,0,1,3)
        insz = inputs.shape # 20 128 17 10
        inputs_sh = inputs.dimshuffle(2,0,1,3) # 17 20 128 10

        in_sh_sz = inputs_sh.shape #
        inputs_sh_flip = T.zeros(in_sh_sz)
        inputs_sh_flip = T.set_subtensor(inputs_sh_flip[::-1,::1,::1,::1], inputs_sh)

        hs, _ = theano.scan(
            self.step,
            sequences=inputs_sh_flip,
            outputs_info=[None,  T.zeros((self.batch_size, self.channel, self.hidden_dim)), T.zeros((self.batch_size, self.channel, self.hidden_dim))])
        # outputs_info : initialize here
        #out = hs[0].reshape((-1, self.batch_size, self.channel, self.hidden_dim))
        outSz = hs[0].shape
        out = T.zeros(outSz)
        out = T.set_subtensor(out[::-1,::1,::1,::1], hs[0])
        out = out.reshape((-1, self.batch_size, self.channel, self.hidden_dim))
        
        if self.is_skip == False:
            return out.dimshuffle(1,2,0,3)
        elif self.is_skip == True:
            return T.concatenate([out.dimshuffle(1,2,0,3).flatten(2), inputs.flatten(2)], axis=1)

    def step(self, x_t, h_t_prev ,c_t_prev):

        tempi=N.sigmoid(T.batched_dot(x_t.dimshuffle(1,0,2),self.Wi) +T.batched_dot(h_t_prev.dimshuffle(1,0,2), self.Ui) + T.batched_dot(h_t_prev.dimshuffle(1,0,2),self.Vi) )
        tempf=N.sigmoid(T.batched_dot(x_t.dimshuffle(1,0,2),self.Wf) +T.batched_dot(h_t_prev.dimshuffle(1,0,2), self.Uf) + T.batched_dot(h_t_prev.dimshuffle(1,0,2),self.Vf) )
        tempo=N.sigmoid(T.batched_dot(x_t.dimshuffle(1,0,2),self.Wo) +T.batched_dot(h_t_prev.dimshuffle(1,0,2), self.Uo) + T.batched_dot(h_t_prev.dimshuffle(1,0,2),self.Vo) )
        temch=T.batched_dot(x_t.dimshuffle(1,0,2), self.Wc) + T.batched_dot(h_t_prev.dimshuffle(1,0,2),self.Uc) 

        #i_t = N.sigmoid(T.dot(x_t,self.Wi) + T.dot(h_t_prev,self.Ui) + T.dot(h_t_prev,self.Vi) + self.bi)

        i_t =  N.sigmoid(tempi.dimshuffle(1,0,2)+ self.bi)
        f_t =  N.sigmoid(tempf.dimshuffle(1,0,2)+ self.bf)
        o_t =  N.sigmoid(tempo.dimshuffle(1,0,2)+ self.bo)
        c_hat_t = T.tanh(temch.dimshuffle(1,0,2)+ self.bc)

        c_t = f_t * c_t_prev + i_t * c_hat_t
        h_t = o_t * T.tanh(c_t)
        return [h_t, h_t, c_t] # the real output, hidden state, cell state

    def params(self):
        # Your codes here
        return [self.Wi, self.Wf, self.Wo, self.Wc, self.Ui, self.Uf, self.Uo, self.Uc, self.Vi, self.Vf, self.Vo, \
        self.bi, self.bf, self.bo, self.bc]


class RNN4D_noshare(Layer):
    def __init__(self, name, batch_size, channel, hidden_dim, input_dim, init_std):
        super(RNN4D_noshare, self).__init__(name, trainable=True)
        self.hidden_dim = hidden_dim

        self.Wx = sharedX(np.random.randn(channel,input_dim, hidden_dim) * init_std, name=name + '/Wx')
        self.Wh = sharedX(np.random.randn(channel,hidden_dim, hidden_dim) * init_std, name=name + '/Wh')
        # channel dimension added
        # so different channel have different connection, ie. noshare connection
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
        #print 'x_t shape',x_tsz
        # batch_size * 128 * 6
        x_t_sh = x_t.dimshuffle(1,0,2)
        h_t_prev_sh = h_t_prev.dimshuffle(1,0,2)
        temp = T.batched_dot(x_t_sh, self.Wx) + T.batched_dot(h_t_prev_sh, self.Wh)
        h_t = T.tanh(temp.dimshuffle(1,0,2) + self.b)
        return [h_t, h_t]

    def params(self):
        return [self.Wx, self.Wh, self.b]

def get_Wres_3D(connective,width,column):
    rng2 = np.random.RandomState(123555)
    z=np.asarray(rng2.uniform(low=0,high=1,size=(column,width,width)))
    mask = [z<connective]
    mask = numpy.asarray(mask)

    rng3=numpy.random.RandomState(666)
    raw = np.asarray(rng3.uniform(low=-0.5,high=0.5,size=(column,width,width)))
    Wres_raw = np.multiply(raw,mask)
    for i in xrange(column):    
        #print "Wres_raw[0][i]",Wres_raw[0][i].shape
        eig,other = np.linalg.eig(Wres_raw[0][i])
        max_eig_i = np.max(eig)
        max_eig_norm_i = np.sqrt(np.real(max_eig_i)**2 + np.imag(max_eig_i)**2)
        Wres_raw[0][i] = Wres_raw[0][i] / max_eig_norm_i*0.99
        if i%10 == 0:
            print "finished",i
    #Wres = theano.shared(Wres / max_eig_norm,borrow=True)
    Wres = np.reshape(Wres_raw,(column,width,width))
    Wres =np.asarray(Wres,dtype=theano.config.floatX)
    return Wres
class ESN_4D_noshare(Layer):
    def __init__(self, name, batch_size, channel, hidden_dim, input_dim, init_std):
        super(ESN_4D_noshare, self).__init__(name, trainable=True)
        self.hidden_dim = hidden_dim
        self.Wx = sharedX(np.random.randn(channel, input_dim, hidden_dim) * init_std, name=name + '/Wx')
        #self.Wh = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Wh')

        self.connective = 0.05
        width = hidden_dim 
        Wres_g = get_Wres_3D(connective = self.connective, width = width, column = channel)
        print 'Wres_g.shape', Wres_g.shape
        self.Wh = sharedX(Wres_g , name=name + '/Wh')


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
        t = out.dimshuffle(1,2,0,3) # out shape : batch,channel,hidden_dim
        return T.concatenate([t.flatten(2), inputs.flatten(2)], axis=1)

    def step(self, x_t, h_t_prev):
        x_tsz=x_t.shape
        print 'x_t shape',x_tsz
        # batch_size * 128 * 6
        temph_t = T.batched_dot(x_t.dimshuffle(1,0,2), self.Wx) + T.batched_dot(h_t_prev.dimshuffle(1,0,2), self.Wh)
        h_t = T.tanh(temph_t.dimshuffle(1,0,2) + self.b)
        return [h_t, h_t]

    def params(self):
        return [] #[self.Wx, self.Wh, self.b]
class LSTMnoshare4D(Layer):
    def __init__(self, name, batch_size, channel, hidden_dim, input_dim, init_std, is_skip=False):
        super(LSTMnoshare4D, self).__init__(name, trainable=True)
        self.hidden_dim = hidden_dim
        # Your codes here, do weights initilization
        self.Wi = sharedX(np.random.randn(channel,input_dim, hidden_dim) * init_std, name = name + '/Wi')
        self.Wf = sharedX(np.random.randn(channel,input_dim, hidden_dim) * init_std, name = name + '/Wf')
        self.Wo = sharedX(np.random.randn(channel,input_dim, hidden_dim) * init_std, name = name + '/Wo')
        self.Wc = sharedX(np.random.randn(channel,input_dim, hidden_dim) * init_std, name = name + '/Wc')
        self.Ui = sharedX(np.random.randn(channel,hidden_dim, hidden_dim) * init_std, name=name + '/Ui')
        self.Uf = sharedX(np.random.randn(channel,hidden_dim, hidden_dim) * init_std, name=name + '/Uf')
        self.Uo = sharedX(np.random.randn(channel,hidden_dim, hidden_dim) * init_std, name=name + '/Uo')
        self.Uc = sharedX(np.random.randn(channel,hidden_dim, hidden_dim) * init_std, name=name + '/Uc')
        self.Vi = sharedX(np.random.randn(channel,hidden_dim, hidden_dim) * init_std, name=name + '/Vi')
        self.Vf = sharedX(np.random.randn(channel,hidden_dim, hidden_dim) * init_std, name=name + '/Vf')
        self.Vo = sharedX(np.random.randn(channel,hidden_dim, hidden_dim) * init_std, name=name + '/Vo')
        self.bi = sharedX(np.zeros((batch_size, channel, hidden_dim)), name=name + '/bi')
        self.bf = sharedX(np.zeros((batch_size, channel, hidden_dim)), name=name + '/bf')
        self.bo = sharedX(np.zeros((batch_size, channel, hidden_dim)), name=name + '/bo')
        self.bc = sharedX(np.zeros((batch_size, channel, hidden_dim)), name=name + '/bc')
        self.batch_size = batch_size
        self.channel = channel
        self.is_skip = is_skip

    def forward(self, inputs):
        insz = inputs.shape
        inputs_sh = inputs.dimshuffle(2,0,1,3)
        hs, _ = theano.scan(
            self.step,
            sequences=inputs_sh,
            outputs_info=[None,  T.zeros((self.batch_size, self.channel, self.hidden_dim)), T.zeros((self.batch_size, self.channel, self.hidden_dim))])
        # outputs_info : initialize here
        out = hs[0].reshape((-1, self.batch_size, self.channel, self.hidden_dim))
        if self.is_skip == False:
            return out.dimshuffle(1,2,0,3)
        elif self.is_skip == True:
            return T.concatenate([out.dimshuffle(1,2,0,3).flatten(2), inputs.flatten(2)], axis=1)

    def step(self, x_t, h_t_prev ,c_t_prev):

        tempi=N.sigmoid(T.batched_dot(x_t.dimshuffle(1,0,2),self.Wi) +T.batched_dot(h_t_prev.dimshuffle(1,0,2), self.Ui) + T.batched_dot(h_t_prev.dimshuffle(1,0,2),self.Vi) )
        tempf=N.sigmoid(T.batched_dot(x_t.dimshuffle(1,0,2),self.Wf) +T.batched_dot(h_t_prev.dimshuffle(1,0,2), self.Uf) + T.batched_dot(h_t_prev.dimshuffle(1,0,2),self.Vf) )
        tempo=N.sigmoid(T.batched_dot(x_t.dimshuffle(1,0,2),self.Wo) +T.batched_dot(h_t_prev.dimshuffle(1,0,2), self.Uo) + T.batched_dot(h_t_prev.dimshuffle(1,0,2),self.Vo) )
        temch=T.batched_dot(x_t.dimshuffle(1,0,2), self.Wc) + T.batched_dot(h_t_prev.dimshuffle(1,0,2),self.Uc) 

        #i_t = N.sigmoid(T.dot(x_t,self.Wi) + T.dot(h_t_prev,self.Ui) + T.dot(h_t_prev,self.Vi) + self.bi)

        i_t =  N.sigmoid(tempi.dimshuffle(1,0,2)+ self.bi)
        f_t =  N.sigmoid(tempf.dimshuffle(1,0,2)+ self.bf)
        o_t =  N.sigmoid(tempo.dimshuffle(1,0,2)+ self.bo)
        c_hat_t = T.tanh(temch.dimshuffle(1,0,2)+ self.bc)

        c_t = f_t * c_t_prev + i_t * c_hat_t
        h_t = o_t * T.tanh(c_t)
        return [h_t, h_t, c_t] # the real output, hidden state, cell state

    def params(self):
        # Your codes here
        return [self.Wi, self.Wf, self.Wo, self.Wc, self.Ui, self.Uf, self.Uo, self.Uc, self.Vi, self.Vf, self.Vo, \
        self.bi, self.bf, self.bo, self.bc]


class LSTM2D(Layer):
    def __init__(self, name, batch_size, hidden_dim, input_dim, init_std, is_skip=False):
        super(LSTM2D, self).__init__(name, trainable=True)
        self.hidden_dim = hidden_dim
        # Your codes here, do weights initilization
        self.Wi = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name = name + '/Wi')
        self.Wf = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name = name + '/Wf')
        self.Wo = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name = name + '/Wo')
        self.Wc = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name = name + '/Wc')
        self.Ui = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Ui')
        self.Uf = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Uf')
        self.Uo = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Uo')
        self.Uc = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Uc')
        self.Vi = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Vi')
        self.Vf = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Vf')
        self.Vo = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Vo')
        self.bi = sharedX(np.zeros((hidden_dim)), name=name + '/bi')
        self.bf = sharedX(np.zeros((hidden_dim)), name=name + '/bf')
        self.bo = sharedX(np.zeros((hidden_dim)), name=name + '/bo')
        self.bc = sharedX(np.zeros((hidden_dim)), name=name + '/bc')
        self.batch_size = batch_size        
        self.is_skip = is_skip
        self.input_dim = input_dim

    def forward(self, inputs):
        # in batch * 50 * 80
        #inputs_reshape = inputs.reshape((self.batch_size, -1, self.input_dim))
        print inputs.eval().shape
        inputs_sh = inputs.dimshuffle(1,0,2) # 80*20*50
        hs, _ = theano.scan(
            self.step,
            sequences=inputs_sh,
            outputs_info=[None,  T.zeros((self.batch_size, self.hidden_dim)), T.zeros((self.batch_size, self.hidden_dim))])
        # outputs_info : initialize here
        out = hs[0].reshape((-1, self.batch_size, self.hidden_dim))
        if self.is_skip == False:
            return out.dimshuffle(1,0,2)
        elif self.is_skip == True:
            return T.concatenate([out.dimshuffle(1,0,2).flatten(2), inputs.flatten(2)], axis=1)

    def step(self, x_t, h_t_prev ,c_t_prev):

        i_t = N.sigmoid(T.dot(x_t,self.Wi) + T.dot(h_t_prev,self.Ui) + T.dot(h_t_prev,self.Vi) + self.bi)
        f_t = N.sigmoid(T.dot(x_t,self.Wf) + T.dot(h_t_prev,self.Uf) + T.dot(h_t_prev,self.Vf) + self.bf)
        o_t = N.sigmoid(T.dot(x_t,self.Wo) + T.dot(h_t_prev,self.Uo) + T.dot(h_t_prev,self.Vo) + self.bo)
        c_hat_t = T.tanh(T.dot(x_t,self.Wc) + T.dot(h_t_prev,self.Uc) + self.bc)
        c_t = f_t * c_t_prev + i_t * c_hat_t
        h_t = o_t * T.tanh(c_t)
        return [h_t, h_t, c_t] # the real output, hidden state, cell state

    def params(self):
        # Your codes here
        return [self.Wi, self.Wf, self.Wo, self.Wc, self.Ui, self.Uf, self.Uo, self.Uc, self.Vi, self.Vf, self.Vo, \
        self.bi, self.bf, self.bo, self.bc]

