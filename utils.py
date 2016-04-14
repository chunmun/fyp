import os, sys, gzip, theano, pickle, numpy, random
import theano.tensor as T
import theano as th
import numpy as np

def load_data(dataset):
    data_dir, data_file = os.path.split(dataset)

    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(
                os.path.split(__file__)[0],
                '..',
                'data',
                dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
            dtype=theano.config.floatX),
            borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
            dtype=theano.config.floatX),
            borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def contextwin(l, win, left, right):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)

    lpadded = win//2 * [left] + l + win//2 * [right]
    out = [ lpadded[i:i+win] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out

def create_directory(directory_model):
    if not os.path.exists(directory_model):
        os.mkdir(directory_model)

def shuffle(lol, seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)

def generate_feature_vectors(num_vectors, num_features, min_val=-0.1, max_val=0.1):
    return (max_val - min_val) * numpy.random.random((num_vectors, num_features)) + min_val

def create_feature_tables(args, metadata, reader):
    feature_tables = []

    # Create random word vectors for each word in the dictionary
    table_size = len(reader.word_dict)
    types_table = generate_feature_vectors(table_size, args.num_features)
    types_table_shared = theano.shared(types_table, dtype=theano.config.floatX,
        borrow=True)
    feature_tables.append(types_table)

    table_size = len(reader.tag_dict)
    pos_table = generate_feature_vectors(table_size, args.num_tag_features)
    # TODO POS table is not appended here

    return feature_tables

def ts(n, v):
    return th.shared(name=n, value=v, borrow=True)

def init_mat(name, rf, shape):
    mat = np.random.uniform(-rf, rf, shape).astype(th.config.floatX)
    #_,e,_ = np.linalg.svd(mat)
    #mat /= e[0]
    return ts(name, mat)

def init_mat_sig(name, shape):
    rf = np.sqrt(6)/np.sqrt(sum(shape))
    return init_mat(name, rf, shape)

def dropout_rare(sentence, reader, rate):
    new = [i for i in sentence]
    dropout_mask = [1 for i in sentence]
    for i,k in enumerate(new):
        if np.random.random() < rate:
            new[i] = reader.codify_string(reader.word_dict.rare)[0]
            dropout_mask[i] = 0
    n = lambda x: np.asarray(x, dtype=np.int32)
    return [n(new), n(dropout_mask)]


def viterbi(probabilities):
    tags = len(probilities[0])
    probs = [1 for i in range(tags)]
    sequences = [() for i in range(tags)]

    for word in probilities:
        new_prob = probs[:]
        for i,p in enumerate(probs):
            highest = 0
            highest_idx = 0
            for j, pr in enumerate(word):
                if pr * p > highest:
                    highest = pr * p
                    highest_idx = j
            new_prob[i] = highest
            sequences[i] += (highest_idx,)
        probs = new_prob

    return sequences
