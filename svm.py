import theano, numpy
import theano.tensor as T

"""
Model is linear and as such
y is the label and x is the input

y = sign(wx)
"""
ftype = theano.config.floatX

x = T.fscalar()
y = T.fscalar()

w = theano.shared(numpy.asarray([0], dtype=ftype), borrow=True)
b = theano.shared(numpy.asarray([0], dtype=ftype), borrow=True)
test_set_x = theano.shared(numpy.asarray([-1.0, 1.0], dtype=ftype), borrow=True)
test_set_y = theano.shared(numpy.asarray([0.0, 1.0], dtype=ftype), borrow=True)

idx = T.iscalar()

def classify(x):
    return w * x + b

p_1 = T.nnet.sigmoid(w*x+b) > 0.5
errors = T.abs_(y - p_1)
cost = errors.mean() + w.sum() + b.sum()

g_w, g_b = T.grad(cost, [w, b])
lr = 1.0
updates = [(w, w+g_w*lr*errors), (b, b+g_b*lr*errors)]

predict = theano.function(inputs=[idx], \
        outputs=p_1,\
        givens={x: test_set_x[idx] })

test_model = theano.function(inputs=[idx], \
        outputs=errors,\
        givens={x: test_set_x[idx], y: test_set_y[idx] })

check_updates = theano.function(inputs=[idx], \
        outputs=[g_w, g_b],\
        givens={x: test_set_x[idx], y: test_set_y[idx] })

train_model = theano.function(inputs=[idx], \
        outputs=cost, \
        givens={x: test_set_x[idx], y: test_set_y[idx] }, \
        updates=updates)


print(predict(0))
print(predict(1))
print("after doing 1 useless training")
print(train_model(0))

print(predict(0))
print(predict(1))

print("after doing 1 useful training")
train_model(1)

print(predict(0))
print(predict(1))

"""
print([predict(j) for j in range(2)])
print(w.get_value(), b.get_value())
for i in range(1000):

    train_model(0)
    train_model(1)
    if i % 100 == 0:
        for j in range(2):
            print('prediction for ', j,  predict(j))
        print(w.get_value(), b.get_value())
        print('\n')
"""
