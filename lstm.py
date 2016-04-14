import theano as th
import numpy as np
import theano.tensor as T
import theano.d3viz as d3v
from math import exp
from utils import init_mat, ts, init_mat_sig
from nn import NN

class LSTM(NN):
    """
    :params l2 coeff of l2 reg
    :params dh dimension of hidden layer
    :params ne number of words
    :params nc number of classes (tags)
    :params de dimension of word embedding
    """
    def __init__(self, input, l2reg, dh, ne, nc, de):
        self.input = input
        self.l2reg = l2reg

        self.wf = init_mat_sig('wf', (dh, de))
        self.wi = init_mat_sig('wi', (dh, de))
        self.wc = init_mat_sig('wc', (dh, de))
        self.wo = init_mat_sig('wo', (dh, de))

        self.uf = init_mat_sig('uf', (dh, dh))
        self.ui = init_mat_sig('ui', (dh, dh))
        self.uc = init_mat_sig('uc', (dh, dh))
        self.uo = init_mat_sig('uo', (dh, dh))

        self.bf = ts('bf', 2*np.ones(dh, dtype=th.config.floatX))
        self.bi = ts('bi', np.zeros(dh, dtype=th.config.floatX))
        self.bc = ts('bc', np.zeros(dh, dtype=th.config.floatX))
        self.bo = ts('bo', np.zeros(dh, dtype=th.config.floatX))

        self.whl = init_mat_sig('whl', (dh, de))
        self.bhl= ts('bhl', np.zeros(de, dtype=th.config.floatX))

        self.ws = init_mat_sig('ws', (dh, nc))
        self.bs = ts('bs', np.zeros(nc, dtype=th.config.floatX))

        self.h_0 = ts('h_0', np.zeros(dh, dtype=th.config.floatX))
        self.c_0 = ts('c_0', np.zeros(dh, dtype=th.config.floatX))

        self.params = [self.h_0, self.c_0, \
                self.wf, self.wi, self.wc, self.wo,\
                self.uf, self.ui, self.uc, self.uo,\
                self.bf, self.bi, self.bc, self.bo, \
                self.ws, self.bs, self.whl, self.bhl]

        self.create_helpers(dh, ne, nc, de)

        def recurrence(x_t, h_tm1, c_tm1):
            ft = T.nnet.sigmoid(T.dot(self.wf, x_t) + T.dot(self.uf, h_tm1) + self.bf)
            it = T.nnet.sigmoid(T.dot(self.wi, x_t) + T.dot(self.ui, h_tm1) + self.bi)
            ctp = T.tanh(T.dot(self.wc, x_t) + T.dot(self.uc, h_tm1) + self.bc)
            ot = T.nnet.sigmoid(T.dot(self.wo, x_t) + T.dot(self.uo, h_tm1) + self.bo)

            c_t = c_tm1 * ft + it * ctp
            h_t = ot * T.tanh(c_t)
            return [h_t, c_t]

        [h_t,c_t], _ = th.scan(fn=recurrence, \
                sequences=self.input,\
                outputs_info=[self.h_0, self.c_0], \
                n_steps=self.input.shape[0])

        [h_t2, c_t2], _ = th.scan(fn=recurrence,
                sequences=self.input,
                outputs_info=[h_t[-1], c_t[-1]],
                n_steps=self.input.shape[0])

        self.p_y_given_x = T.nnet.softmax(T.dot(h_t2, self.ws) + self.bs)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.output = self.p_y_given_x

    def errors_with_rare_dropout(self, y, dropout_mask):
        e = T.sum(T.neq(self.y_pred, y) * dropout_mask)
        sentence_nll = -T.mean(T.log(self.p_y_given_x)[T.arange(self.input.shape[0]), y] * dropout_mask)\
                + self.l2reg * sum([T.mean(T.sqr(p)) for p in self.params])
        return [e, sentence_nll]

    def errors(self, y):
        e = T.sum(T.neq(self.y_pred, y)) # 0-1 loss
        sentence_nll = -T.mean(T.log(self.p_y_given_x)[T.arange(self.input.shape[0]), y])\
                + self.l2reg * sum([T.mean(T.sqr(p)) for p in self.params])
        return [e, sentence_nll]

    def create_helpers(self, dh, ne, nc, de):
        wf = init_mat('wf', 0, (dh, de))
        wi = init_mat('wi', 0, (dh, de))
        wc = init_mat('wc', 0, (dh, de))
        wo = init_mat('wo', 0, (dh, de))

        uf = init_mat('uf', 0, (dh, dh))
        ui = init_mat('ui', 0, (dh, dh))
        uc = init_mat('uc', 0, (dh, dh))
        uo = init_mat('uo', 0, (dh, dh))

        bf = ts('bf', 2*np.ones(dh, dtype=th.config.floatX))
        bi = ts('bi', np.zeros(dh, dtype=th.config.floatX))
        bc = ts('bc', np.zeros(dh, dtype=th.config.floatX))
        bo = ts('bo', np.zeros(dh, dtype=th.config.floatX))

        whl = init_mat('whl', 0, (dh, de))
        bhl= ts('bhl', np.zeros(de, dtype=th.config.floatX))

        ws = init_mat('ws', 0, (dh, nc))
        bs = ts('bs', np.zeros(nc, dtype=th.config.floatX))

        h_0 = ts('h_0', np.zeros(dh, dtype=th.config.floatX))
        c_0 = ts('c_0', np.zeros(dh, dtype=th.config.floatX))

        self.params_helper = [h_0, c_0, \
                wf, wi, wc, wo,\
                uf, ui, uc, uo,\
                bf, bi, bc, bo, \
                ws, bs, whl, bhl]


