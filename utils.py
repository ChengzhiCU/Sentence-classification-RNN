import theano
import numpy as np
from datetime import datetime
import theano.tensor as T

def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=True)
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=True)
    return shared_x, T.cast(shared_y, 'int32')
    
def sharedX(X, name=None):
    return theano.shared(
        np.asarray(X, dtype=theano.config.floatX),
        name=name,
        borrow=True)


def LOG_INFO(msg):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-3]
    print display_now + ' ' + msg
