import theano.tensor as T


class CrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, inputs, labels):
        
        #  labels are already in one-hot form
        return T.nnet.categorical_crossentropy(inputs, labels).sum()