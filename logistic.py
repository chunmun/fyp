import theano as th
import theano.tensor as T
from pprint import pprint

# Logistic function
x = T.matrix('x')
s = 1/(1+T.exp(-x))
logistic = th.function([x], s)
print(logistic([[0,1], [-1,-2]]))

# Multiple functions
a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff ** 2
f = th.function([a, b], [diff, abs_diff, diff_squared])
pprint(f([[1, 1], [1, 1,]], [[0, 1], [2, 3]]))

# Default values
from theano import In
from theano import function
x, y = T.dscalars('x', 'y')
z = x + y
f = function([x, In(y, value=1)], z)
print(f(33))
print(f(33, 2))

from theano import shared
state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])
print(state.get_value())
print(accumulator(1))
print(state.get_value())
print(accumulator(100))
print(state.get_value())

# Random stream
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
srng = RandomStreams(seed=234)
rv_u = srng.uniform((2, 2))
rv_n = srng.normal((2, 2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True) # Not updating rv_n.rng
nearly_zeros = function([], rv_u + rv_u - 2 * rv_n)
print(f())
print(f())
print(g())
print(g())

# Actual Logistic Regression!
import numpy as np
rng = np.random

N = 400
feats = 784

# generate a dataset: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 10000

# Declare Theano symbolic variables
x = T.matrix('x')
y = T.vector('y')

# Initialize the weight vector w randomly
#
# this and the following bias variable b
# are shared so they keep their values
# between the training iterations (updates)
w = th.shared(rng.randn(feats), name='w')

# initialize the bias term
b = th.shared(0., name='b')

print("Initial model:")
print(w.get_value())
print(b.get_value())

# Construct the Theano expression graph
p_1 = 1/(1+T.exp(-T.dot(x, w) - b))                # Prob that target = 1
prediction = p_1 > 0.5                             # The prediction threshold
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)      # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()         # The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # w.r.t weight vector w and bias b

# Compile
train = th.function(
        inputs=[x, y],
        outputs=[prediction, xent],
        updates=((w, w-0.1*gw), (b, b-0.1*gb)),
        allow_input_downcast=True)
predict = th.function(inputs=[x], outputs=prediction, allow_input_downcast=True)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print('Final model:')
print(w.get_value())
print(b.get_value())
print('Target values for D:')
print(D[1])
print('Prediction on D:')
print(predict(D[0]))
print('Error')
print(D[1] - predict(D[0]))
