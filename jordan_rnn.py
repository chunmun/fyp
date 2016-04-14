import theano
import numpy
import os

from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class JordanRnn(object):

    def __init__(self, nh, nc, ne, de, cs):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size
        '''
        #assert st in ['proba', 'argmax']

        self.emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (ne, de)).astype(theano.config.floatX))
        self.Wx  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de * cs, nh)).astype(theano.config.floatX))
        self.Wh  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nc)).astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
        self.h0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))

        # bundle
        self.params = [ self.emb, self.Wx, self.Wh, self.W, self.bh, self.b, self.h0 ]

        # For adagrad
        self.emb_helper = theano.shared(numpy.zeros((ne, de)).astype(theano.config.floatX))
        self.Wx_helper  = theano.shared(numpy.zeros((de * cs, nh)).astype(theano.config.floatX))
        self.Wh_helper  = theano.shared(numpy.zeros((nh, nh)).astype(theano.config.floatX))
        self.W_helper   = theano.shared(numpy.zeros((nh, nc)).astype(theano.config.floatX))
        self.bh_helper  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.b_helper   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
        self.h0_helper  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.params_helper = [ self.emb_helper, self.Wx_helper, self.Wh_helper, \
                self.W_helper, self.bh_helper, self.b_helper, self.h0_helper ]

        self.names  = ['embeddings', 'Wx', 'Wh', 'W', 'bh', 'b', 'h0']

        srng = RandomStreams(seed=0)

        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence

        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y_sentence = T.ivector('y_sentence') # labels

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + \
                                 T.dot(h_tm1, self.Wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)[0]
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence, \
            sequences=x, outputs_info=[self.h0, None], \
            n_steps=x.shape[0])

        y_pred = T.argmax(s, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        rho = 0.1

        sentence_nll = -T.mean(T.log(s)[T.arange(x.shape[0]), y_sentence])
        errors = T.sum(T.neq(y_pred, y_sentence)) # 0-1 loss

        loss_with_reg = sentence_nll + T.sum(self.Wx**2) + T.sum(self.Wh**2) + T.sum(self.b**2) \
                + T.sum(self.bh**2)

        sentence_gradients = T.grad( sentence_nll, self.params )

        sentence_updates = [(h, h + g ** 2) for p,h,g in
                zip( self.params, self.params_helper, sentence_gradients)]

        sentence_updates.extend([(p, p-lr*g*rho / (rho+T.sqrt(h + g ** 2))) for p,h,g in
                zip( self.params, self.params_helper, sentence_gradients)])

        # theano functions
        self.test = theano.function(inputs=[idxs, y_sentence], outputs=[sentence_nll, errors])

        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

        self.train = theano.function( inputs  = [idxs, y_sentence, lr],
                                      outputs = sentence_nll,
                                      updates = sentence_updates )

        self.normalize = theano.function( inputs = [],
                         updates = {self.emb:\
                         self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())

    def load(self, folder):
        for param, name in zip(self.params, self.names):
            param.set_value(numpy.load(os.path.join(folder, name+'.npy')))
