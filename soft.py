import theano, numpy
import theano.tensor as T

x = T.fvector('x')

f = theano.function(inputs=[x], outputs=[T.nnet.softmax(x)])
g = numpy.asarray([1,2,3,4], dtype=theano.config.floatX)
print(g)
print(f(g)[0])
