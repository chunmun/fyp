from Reader import Reader, Metadata, Token
import utils, argparse, os, theano, numpy, timeit, pickle, sys
import theano.tensor as T
from MLP import MLP
from jordan_rnn import JordanRnn

"""
Compute the log of the sum of exponentials of input elements
"""

parser = argparse.ArgumentParser(usage="usage: tag.py [options]")
parser.add_argument('--num-features',\
        type=int,\
        default=50,\
        help='number of features for word vectors')

parser.add_argument('--window',\
        type=int,\
        default=5,\
        help='Size of word window (default: 5)')

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
    directory_model = 'bestModel'

    #reader = Reader(md)

    with open(os.path.join(directory_model, 'reader.pkl'), 'rb') as f:
        reader = pickle.load(f)

    num_tags = len(reader.tag_dict)
    num_words = len(reader.word_dict)
    model = JordanRnn(args.hidden, num_tags, num_words, args.num_features, args.window)
    print('... loading models')
    model.load(directory_model)

    print('>>> READY')
    while True:
        sent = input()
        coded = reader.codify_string(sent)
        framed = numpy.asarray(\
                    utils.contextwin(coded, args.window,\
                    reader.get_padding_left(), reader.get_padding_right()\
                    ), dtype=numpy.int32)
        coded_tags = model.classify(framed)
        tags = [reader.reverse_tag_dict[t] for t in coded_tags]

        print('[INPUT] ' + str(sent))
        print('[CODED] ' + str(coded))
        print('[ TAG ] ' + str(coded_tags))
        print('[UNTAG] ' + str(tags))
        print()


