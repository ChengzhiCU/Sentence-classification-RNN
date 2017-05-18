import theano.tensor as T
import theano
import numpy as np
from utils import sharedX

class SGDOptimizerHalf(object):
    def __init__(self, learning_rate, weight_decay=0.005, momentum=0.9):
        self.lr = sharedX(learning_rate)
        self.wd = weight_decay
        self.mm = momentum        
        self.cnt=0

    def get_updates(self, cost, params):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        self.cnt += 1
        print 'cnt add one',self.cnt       
        for p, g in zip(params, grads):
            d = sharedX(p.get_value() * 0.0)            
            new_d = self.mm * d - self.lr * (g + self.wd * p)            
            updates.append((d, new_d))
            updates.append((p, p + new_d))
        return updates

    def get_lr(self):
        return self.lr.get_value()

    def set_lr(self, learning_rate):
        self.lr.set_value(np.cast[theano.config.floatX](learning_rate))

class SGDOptimizer(object):
    def __init__(self, learning_rate, weight_decay=0.005, momentum=0.9):
        self.lr = learning_rate
        self.wd = weight_decay
        self.mm = momentum

    def get_updates(self, cost, params):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            d = sharedX(p.get_value() * 0.0)
            new_d = self.mm * d - self.lr * (g + self.wd * p)
            updates.append((d, new_d))
            updates.append((p, p + new_d))

        return updates


class AdagradOptimizer(object):
    def __init__(self, learning_rate, eps=1e-8):
        self.lr = learning_rate
        self.eps = eps

    def get_updates(self, cost, params):
        # Your codes here
        # hint: implementation idea
        #       cache += dx ** 2
        #       p = p - self.lr * dx / (sqrt(cache) + self.eps)
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        
        for p, g in zip(params, grads):
            cache = sharedX(p.get_value() * 0.0)
            new_cache = cache + g ** 2
            updates.append((cache, new_cache))
            updates.append((p, p - self.lr * g / (T.sqrt(new_cache) + self.eps)))

        return updates

class RMSpropOptimizer(object):
    def __init__(self, learning_rate, rho=0.9, eps=1e-8):
        # rho: decay_rate
        self.lr = learning_rate
        self.rho = rho #decay
        self.eps = eps

    def get_updates(self, cost, params):
        # Your codes here
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            cache = sharedX(p.get_value() * 0.0)
            new_cache = self.rho * cache + (1 - self.rho) * g**2

            updates.append((cache, new_cache))
            updates.append((p, p-self.lr * g / (np.sqrt(new_cache) + self.eps)))

        return updates
