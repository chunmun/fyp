import theano as th
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
from utils import init_mat, ts
import os

class NN:
    def __init__(self):
        self.params = []


    def get_names(self):
        return [p.name for p in self.params]

    def save(self, folder, varlist):
        for param, name in zip(self.params, self.get_names()):
            np.save(os.path.join(folder, '_' + name + '_'.join(varlist)), param.get_value())

    def load(self, folder, varlist):
        for param, name in zip(self.params, self.get_names()):
            param.set_value(np.load(os.path.join(folder, '_' + name + '_'.join(varlist) + '.npy')))

class HiddenLayer(NN):
    def __init__(self, input, n_in, n_out, activation=T.nnet.sigmoid):
        rf = 1.0
        self.W = init_mat('w', (n_out, n_in), rf)
        self.b = ts('b', np.zeros(n_out, dtype=th.config.floatX))

        self.params = [self.W, self.b]
        self.output = activation(T.dot(self.W, input) + self.b)

class Embedding(NN):
    """
    :params de Dimension of embedding
    :params n  Number of embeddings
    """
    def __init__(self, input, de, n):
        self.emb = init_mat('emb', 1, (n, de))
        self.params = [self.emb]

        self.normalize = th.function( inputs = [],
                updates = { self.emb:\
                        self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

        self.output = self.emb[input]
        self.create_helpers(de, n)

    def create_helpers(self, de, n):
        emb = init_mat('emb', 0, (n, de))
        self.params_helper = [emb]

class Dropout(NN):
    def __init__(self, input, de, rate):
        srng = RandomStreams(seed=1234)
        self.output = T.switch(srng.binomial(size=input.shape,p=rate), input, np.float32(0))
