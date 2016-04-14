import theano as th
import numpy as np
import theano.tensor as T
import theano.d3viz as d3v
from math import exp

th.config.compute_test_value = 'raise'

class LSTM:
    """
    :params dh dimension of hidden layer
    :params ne number of words
    :params nc number of classes (tags)
    :params de dimension of word embedding
    :params cs constext window size
    """
    def __init__(self, dh, ne, nc, de, cs):
        ts = lambda n, v: th.shared(name=n, value=v, borrow=True)

        #model = LSTM(10, 10, 2, 1, 1)
        """
        dh = 2 # hidden size
        ne = 3 # number of words
        nc = 4  # number of tags
        de = 5  # dim of emb
        cs = 6  # window size
        """

        dx = de * cs # 5*4 = 20
        ns = nc + dx # 20+4=24
        ds = dh + dx # 2 + 20 = 22

        idxs = T.imatrix('idxs')

        emb = ts('emb', 0.2*np.random.uniform(-1.0, 1, (ne, de)).astype(th.config.floatX))

        rf = 1/np.sqrt(nc)
        wf = ts('wf', 0.2*np.random.uniform(-rf, rf, (nc, ns)).astype(th.config.floatX))
        wi = ts('wi', 0.2*np.random.uniform(-rf, rf, (nc, ns)).astype(th.config.floatX))
        wc = ts('wc', 0.2*np.random.uniform(-rf, rf, (nc, ns)).astype(th.config.floatX))
        wo = ts('wo', 0.2*np.random.uniform(-rf, rf, (nc, ns)).astype(th.config.floatX))

        bf = ts('bf', np.zeros(nc, dtype=th.config.floatX))
        bi = ts('bi', np.zeros(nc, dtype=th.config.floatX))
        bc = ts('bc', np.zeros(nc, dtype=th.config.floatX))
        bo = ts('bo', np.zeros(nc, dtype=th.config.floatX))

        #c_tm1 = ts('c_tm1', np.zeros(nc, dtype=th.config.floatX))
        #h_tm1 = ts('h_tm1', np.zeros(nc, dtype=th.config.floatX))

        c_tm1 = T.zeros(nc)
        h_tm1 = T.zeros(nc)

        idxs.tag.test_value = np.asarray([[0, 1, 2, 1, 1, 2], [1, 2, 1, 0, 0, 0]], dtype=np.int32)

        x = emb[idxs].reshape((idxs.shape[0], de*cs))
        x_t = x[0]

        st = T.concatenate([h_tm1, x_t])

        print('h_tm1', h_tm1.tag.test_value.shape)
        print('c_tm1', c_tm1.tag.test_value.shape)
        print('st', st.tag.test_value.shape)

        ft = T.nnet.sigmoid(T.dot(wf, st) + bf)
        print('ft', ft.tag.test_value.shape)
        it = T.nnet.sigmoid(T.dot(wi, st) + bi)
        print('it', it.tag.test_value.shape)
        ctp = T.tanh(T.dot(wc, st) + bc)
        print('ctp', ctp.tag.test_value.shape)
        ot = T.nnet.sigmoid(T.dot(wo, st) + bo)
        print('ot', ot.tag.test_value.shape)

        c_t = c_tm1 * ft + it * ctp
        print('c_t', c_t.tag.test_value.shape)
        h_t = ot * T.tanh(c_t)
        print('h_t', h_t.tag.test_value.shape)

        for i in range(10):
            print('-'*20)
            # =============
            x_t = x[1]
            h_tm1 = h_t
            c_tm1 = c_t
            st = T.concatenate([h_tm1, x_t])

            print('h_tm1', h_tm1.tag.test_value.shape)
            print('c_tm1', c_tm1.tag.test_value.shape)
            print('st', st.tag.test_value.shape)

            ft = T.nnet.sigmoid(T.dot(wf, st) + bf)
            it = T.nnet.sigmoid(T.dot(wi, st) + bi)
            ctp = T.tanh(T.dot(wc, st) + bc)
            ot = T.nnet.sigmoid(T.dot(wo, st) + bo)

            c_t = c_tm1 * ft + it * ctp
            #print('c_t', c_t.tag.test_value.shape)
            h_t = T.tanh(c_t) * ot
            #print('h_t', h_t.tag.test_value.shape)

        self.classify = th.function(inputs=[idxs], outputs=[h_t])


window_size = 1
model = LSTM(2, 3, 4, 5, 6)
print(model.classify([[0, 1, 2, 1, 1, 2], [1, 2, 1, 0, 0, 0]]))
