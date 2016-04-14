import theano.tensor as T
import theano, numpy, timeit, sys, pickle, os, gzip

class LogisticRegression: # Multi-class Logistic Regression
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
                value=numpy.zeros((n_in, n_out),
                    dtype=theano.config.floatX),
                name='W', borrow=True)
        self.b = theano.shared(
                value=numpy.zeros((n_out,),
                    dtype=theano.config.floatX),
                name='n', borrow=True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                    'y should have the same shape as self.y_pred',
                    ('y', y_type, 'y_pred', self.y_pred.type))

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
        dataset='mnist.pkl.gz', batch_size=600):
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print('... building the model')

    index = T.lscalar() # index to minibatch

    x = T.matrix('x')
    y = T.ivector('y')

    classifier = LogisticRegression(input=x, n_in=28*28, n_out=10)
    cost = classifier.negative_log_likelihood(y)

    test_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index+1) * batch_size],
                y: test_set_y[index * batch_size: (index+1) * batch_size]})

    validate_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index+1) * batch_size],
                y: valid_set_y[index * batch_size: (index+1) * batch_size]})

    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
            (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index+1) * batch_size],
                y: train_set_y[index * batch_size: (index+1) * batch_size]})


    print('... training the model')
    # Early stopping parameters
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = numpy.inf
    test_score = 0
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)

            # iteration number
            iter = (epoch + 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validate error %f %%' %
                        (epoch, minibatch_index + 1,
                        n_train_batches, this_validation_loss * 100))

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(' epoch %i, minibatch %i/%i, test error of best model %f %%' %
                            (epoch, minibatch_index + 1,
                            n_test_batches, test_score * 100))

                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

        if patience <= iter:
            done_looping = True
            break
    end_time = timeit.default_timer()
    print('Optimization complete with best validation score of %f %%, with test performance of %f %%' %
            (best_validation_loss * 100, test_score * 100))

    print('The code run for %d epochs, with %f epochs/sec' % (epoch, 1.0 * epoch / (end_time - start_time)))
    print(('The code for file ' + os.path.split(__file__)[1]
        + ' ran for %.1fs' % (end_time -start_time)), file=sys.stderr)

def predict():
    classifier = pickle.load(open('best_model.pkl'))

    predict_model = theano.function(
            inputs=[classifier.input],
            outputs=classifier.y_pred)

    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print('Predicted values for the first 10 examples in the test set:')
    print(predicted_values)

if __name__ == "__main__":
    sgd_optimization_mnist()


