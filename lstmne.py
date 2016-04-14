import theano as th
import numpy as np
import os
import theano.tensor as T
import theano.d3viz as d3v
from math import exp

class LSTM_NE:
    """
    :params l2 coeff of l2 reg
    :params dh dimension of hidden layer
    :params ne number of words EXPLOSION here
    :params nc number of classes (tags)
    :params de dimension of word embedding
    :params cs constext window size
    """
    def __init__(self, l2reg, dh, ne, nc, de, cs):
        ts = lambda n, v: th.shared(name=n, value=v, borrow=True)

        def init_mat(name, rf, shape):
            mat = np.random.uniform(-rf, rf, shape).astype(th.config.floatX)
            _,e,_ = np.linalg.svd(mat)
            mat /= e[0]
            return ts(name, mat)

        dx = de * cs # dimension of the input vectors
        ns = dh + dx

        idxs = T.fmatrix('idxs')

        rf = np.sqrt(6)/np.sqrt(dh + ns)

        self.wf = init_mat('wf', rf, (dh, ns))
        self.wi = init_mat('wi', rf, (dh, ns))
        self.wc = init_mat('wc', rf, (dh, ns))
        self.wo = init_mat('wo', rf, (dh, ns))

        self.bf = ts('bf', 2*np.ones(dh, dtype=th.config.floatX))
        self.bi = ts('bi', np.zeros(dh, dtype=th.config.floatX))
        self.bc = ts('bc', np.zeros(dh, dtype=th.config.floatX))
        self.bo = ts('bo', np.zeros(dh, dtype=th.config.floatX))

        rf =  np.sqrt(6)/np.sqrt(dh + nc)
        self.ws = init_mat('ws', rf, (dh, nc))
        self.bs = ts('bs', np.zeros(nc, dtype=th.config.floatX))

        self.wf_helper = ts('wf_helper', np.zeros((dh, ns)).astype(th.config.floatX))
        self.wi_helper = ts('wi_helper', np.zeros((dh, ns)).astype(th.config.floatX))
        self.wc_helper = ts('wc_helper', np.zeros((dh, ns)).astype(th.config.floatX))
        self.wo_helper = ts('wo_helper', np.zeros((dh, ns)).astype(th.config.floatX))

        self.ws_helper = ts('ws_helper', np.zeros((dh, nc)).astype(th.config.floatX))

        self.bf_helper = ts('bf_helper', np.zeros(dh, dtype=th.config.floatX))
        self.bi_helper = ts('bi_helper', np.zeros(dh, dtype=th.config.floatX))
        self.bc_helper = ts('bc_helper', np.zeros(dh, dtype=th.config.floatX))
        self.bo_helper = ts('bo_helper', np.zeros(dh, dtype=th.config.floatX))

        self.bs_helper = ts('bs_helper', np.zeros(nc, dtype=th.config.floatX))

        x = idxs.reshape((idxs.shape[0], de*cs))
        y_sentence = T.ivector('y_sentence') # labels

        self.h0 = ts('h0', np.zeros(dh, dtype=th.config.floatX))
        self.c0 = ts('c0', np.zeros(dh, dtype=th.config.floatX))

        self.h0_helper = ts('h0_helper', np.zeros(dh, dtype=th.config.floatX))
        self.c0_helper = ts('c0_helper', np.zeros(dh, dtype=th.config.floatX))

        self.params = [self.h0, self.c0, self.wf, self.wi, self.wc, self.wo,\
                self.bf, self.bi, self.bc, self.bo, self.ws, self.bs]

        self.names = [p.name for p in self.params]

        self.params_helper = [self.h0_helper, self.c0_helper,\
                self.wf_helper, self.wi_helper, self.wc_helper, self.wo_helper,\
                self.bf_helper, self.bi_helper, self.bc_helper, self.bo_helper,\
                self.ws_helper, self.bs_helper]

        def recurrence(x_t, h_tm1, c_tm1):
            st = T.concatenate([h_tm1, x_t])

            ft = T.nnet.sigmoid(T.dot(self.wf, st) + self.bf)
            it = T.nnet.sigmoid(T.dot(self.wi, st) + self.bi)
            ctp = T.tanh(T.dot(self.wc, st) + self.bc)
            ot = T.nnet.sigmoid(T.dot(self.wo, st) + self.bo)

            c_t = c_tm1 * ft + it * ctp
            h_t = ot * T.tanh(c_t)
            return [h_t, c_t]

        [h_t,c_t], _ = th.scan(fn=recurrence, \
                sequences=x,\
                outputs_info=[self.h0, self.c0],
                n_steps=x.shape[0])

        p_y_given_x = T.nnet.softmax(T.dot(h_t, self.ws) + self.bs)
        y_pred = T.argmax(p_y_given_x, axis=1)

        errors = T.sum(T.neq(y_pred, y_sentence)) # 0-1 loss
        sentence_nll = -T.mean(T.log(p_y_given_x)[T.arange(x.shape[0]), y_sentence])\
                + l2reg * sum([T.mean(T.sqr(p)) for p in self.params])

        lr = T.scalar('lr')
        rho = 10
        sentence_gradients = T.grad( sentence_nll, self.params )

        sgd_updates = [(p, p-lr*g) for p, g in zip(self.params, sentence_gradients)]

        sentence_updates = [(h, (h + g ** 2).clip(-5, 5)) for p,h,g in
                zip( self.params, self.params_helper, sentence_gradients)]

        sentence_updates.extend([(p, (p-lr*g*rho / (rho+T.sqrt(h + g ** 2))).clip(-5,5)) for p,h,g in
                zip( self.params, self.params_helper, sentence_gradients)])

        self.test = th.function(inputs=[idxs, y_sentence], outputs=[sentence_nll, errors])

        self.classify = th.function(inputs=[idxs], outputs=[y_pred, p_y_given_x])

        self.train = th.function( inputs  = [idxs, y_sentence, lr],
                                  outputs = [sentence_nll],
                                  updates = sentence_updates,
                                  allow_input_downcast = True)

        zero_updates = [(h, h*0) for h in self.params_helper]
        self.zero_helpers = th.function( inputs=[], outputs=[], updates = zero_updates)


    def save(self, folder, varlist):
        for param, name in zip(self.params, self.names):
            np.save(os.path.join(folder, '_' + name + '_'.join(varlist)), param.get_value())

    def load(self, folder, varlist):
        for param, name in zip(self.params, self.names):
            param.set_value(np.load(os.path.join(folder, '_' + name + '_'.join(varlist) + '.npy')))

if __name__ == "__main__":
    """
    :params dh dimension of hidden layer
    :params ne number of words
    :params nc number of classes (tags)
    :params de dimension of word embedding
    :params cs constext window size
    """
    model = LSTM_NE(10, 2, 3, 4, 2)
    #print(model.classify([[1,1,1,1, 0,0,0,0],[1,1,1,1, 0,0,0,0], [0,0,0,0,1,1,1,1]]))
    for i in range(100):
        print(model.train([[1,1,1,1, 0,0,0,0],[1,1,1,1, 0,0,0,0], [0,0,0,0,1,1,1,1]],
            [1,1,0], 0.1))
