from Reader import Reader, Metadata, Token
import utils, argparse, os, theano, numpy, timeit, pickle, sys
import theano.tensor as T
from MLP import MLP
from jordan_rnn import JordanRnn

"""
Compute the log of the sum of exponentials of input elements
"""
def logsumexp(a, axis=None):
    if axis is None:
        a = a.ravel()
    else:
        a = numpy.rollaxis(a, axis)
    a_max = a.max(axis=0)
    return numpy.log(numpy.sum(np.exp(a-a_max), axis=0)) + a_max

def theano_logsumexp(x, axis=None):
    """
    Compute log(sum(exp(x), axis=axis) in a numerically stable
    fashion.
    Parameters
    ----------
    x : tensor_like
        A Theano tensor (any dimension will do).
    axis : int or symbolic integer scalar, or None
        Axis over which to perform the summation. `None`, the
        default, performs over all axes.
    Returns
    -------
    result : ndarray or scalar
        The result of the log(sum(exp(...))) operation.
    """
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + tensor.log(tensor.exp(x - xmax).sum(axis=axis))


parser = argparse.ArgumentParser(usage="usage: train.py [options] filename")
parser.add_argument('filename')
parser.add_argument('--num-features',\
        type=int,\
        default=50,\
        help='number of features for word vectors')

parser.add_argument('--window',\
        type=int,\
        default=5,\
        help='Size of word window (default: 5)')

parser.add_argument('--load-reader',\
        help='Loads the reader instead of initialising another one')

parser.add_argument('--learning-rate',\
        default=0.01,\
        help='Learning rate of the model (default: 0.01)')

parser.add_argument('--num-tag-features',\
        type=int,\
        default=10,\
        help='Number of features for tag vectors')

parser.add_argument('--hidden',\
        type=int,\
        default=50,\
        help='Size of hidden layer (default: 50)')

parser.add_argument('--iterations',\
        type=int,\
        default=10,\
        help='number of iterations of training (default: 10)')

if __name__=="__main__":
    args = parser.parse_args()
    #md = Metadata('/home/chunmun/fyp/variable.txt.proc')
    #md = Metadata('/home/chunmun/fyp/all.vardec')
    md = Metadata(args.filename)
    directory_model = 'bestModel'

    if args.load_reader:
        with open(os.path.join(directory_model, 'reader.pkl'), 'rb') as f:
            reader = pickle.load(f)
    else:
        reader = Reader(md)
        reader.save(directory_model)

    # Generate the training set
    num_sentences = len(reader.sentences)
    num_words = len(reader.word_dict)
    codified_sentences = [numpy.asarray(\
            utils.contextwin([t.codified_word for t in s], args.window,\
            reader.get_padding_left(), reader.get_padding_right()\
            ), dtype=numpy.int32)\
            for s in reader.sentences]

    #print('codified_sentences', codified_sentences)
    #sentences_shared = theano.shared(codified_sentences)

    num_tags = len(reader.tag_dict)
    codified_tags = [numpy.asarray([t.codified_tag for t in s], dtype=numpy.int32) for s in reader.sentences]

    #print('codified_tags', codified_tags)
    #tags_shared = theano.shared(codified_tags)

    model = JordanRnn(args.hidden, num_tags, num_words, args.num_features, args.window)
    print('... loading models')
    model.load(directory_model)

    print('... training the model')
    print('#sentences : {}, #tags : {}, learning rate : {}, #hidden : {}, embedding size: {} '.format(\
        num_sentences, num_tags, args.learning_rate, args.hidden, args.num_features))
    print('window size: {}'.format(args.window))

    # Early stopping parameters
    patience = 5000
    patience_increase = 2

    best_mean_nll = numpy.inf
    start_time = timeit.default_timer()
    validation_frequency = num_sentences

    done_looping = False
    epoch = 0
    while (epoch < args.iterations) and (not done_looping):
        epoch = epoch + 1
        nll = 0
        total_errors = 0
        parsed = 0

        """
        # shuffle
        r = numpy.random.random()
        utils.shuffle(codified_sentences, r)
        utils.shuffle(codified_tags, r)
        """
        for minibatch_index in range(num_sentences):
            print('batch', minibatch_index)
            model.train(codified_sentences[minibatch_index],\
                    codified_tags[minibatch_index],\
                    numpy.float32(args.learning_rate))
            model.normalize()

            n, e = model.test(codified_sentences[minibatch_index],\
                    codified_tags[minibatch_index])

            nll += n
            total_errors += e
            parsed += 1

            # iteration number
            iter = (epoch + 1) + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                mean_nll = nll/parsed
                print('epoch', epoch, 'mean nll', mean_nll, 'total errors', total_errors)

                if mean_nll < best_mean_nll:
                    if mean_nll < best_mean_nll * 0.9:
                        patience = max(patience, iter * patience_increase)

                    best_mean_nll = mean_nll

                    # Save the model
                    print('Saved')
                    model.save(directory_model)

                nll = 0
                total_errors = 0
                parsed = 0

        if patience <= iter:
            done_looping = True
            break

    end_time = timeit.default_timer()
    print('Optimization complete with best sentence negative log likelihood of %f %%, with training error of %f %%' %
            (best_mean_nll * 100, total_errors / num_sentences * 100))

    print('The code run for %d epochs, with %f epochs/sec' % (epoch, 1.0 * epoch / (end_time - start_time)))
    print(('The code for file ' + os.path.split(__file__)[1]
        + ' ran for %.1fs' % (end_time -start_time)), file=sys.stderr)
