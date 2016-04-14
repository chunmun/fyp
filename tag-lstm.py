from Reader import Reader, Metadata, Token
import utils, argparse, os, timeit, pickle, sys
import theano.tensor as T
import theano as th
import numpy as np
from lstm import LSTM
from nn import Dropout, Embedding
from pprint import pprint
from termcolor import colored

parser = argparse.ArgumentParser(usage="usage: tag.py [options]")
parser.add_argument('filename')

parser.add_argument('--validation-filename',\
        help='Loads another file with the validation test set')

parser.add_argument('--num-features',\
        type=int,\
        default=50,\
        help='number of features for word vectors')

parser.add_argument('--l2',\
        type=float,\
        default=0,
        help='Coefficient of L2 regularization factors')

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

parser.add_argument('--dropout-rare',\
        type=float,\
        default=0,
        help='Proabability of a word turning into a rare word')

parser.add_argument('--dropout',\
        type=float,\
        default=0,
        help='Proabability of zeroing embedded word vectors')

parser.add_argument('--fixed-embeddings',\
        help='Loads the corresponding embeddings from the given word embedding file')

parser.add_argument('--learn-embeddings',\
        help='Loads the corresponding embeddings from that only exists in the test sentence')



if __name__=="__main__":
    args = parser.parse_args()

    varlist = list(map(str, [os.path.basename(args.filename), os.path.basename(args.validation_filename), \
        args.iterations, args.hidden, args.l2, args.dropout_rare, args.dropout,\
        args.fixed_embeddings is not None, args.learn_embeddings is not None]))
    #reader = Reader(md)
    directory_model = 'Model_' + '_'.join(varlist)

    try:
        with open(os.path.join(directory_model, 'reader.pkl'), 'rb') as f:
            reader = pickle.load(f)
    except:
        md = Metadata(args, args.filename, args.fixed_embeddings or args.learn_embeddings)
        reader = Reader(md, minimum_occurrence=2)

    num_tags = len(reader.tag_dict)
    num_words = len(reader.word_dict)
    print('... loading models')

    x = T.ivector('x')

    emb = Embedding(x, args.num_features, num_words+1)
    lstm = LSTM(emb.output, args.l2, args.hidden, num_words + 1, num_tags, args.num_features)

    emb.load(directory_model, varlist)
    lstm.load(directory_model, varlist)

    classify = th.function(
            inputs = [x],
            outputs = [lstm.y_pred, lstm.p_y_given_x])

    print('#words: {}, #tags : {}, #hidden : {}, embedding size: {} '.format(\
            len(reader.word_dict), len(reader.tag_dict), args.hidden, args.num_features))

    print('>>> READY')
    while True:
        sent = input()
        coded = reader.codify_string(sent)
        coded_tags, probilities = classify(coded)
        tags = [reader.reverse_tag_dict[t] for t in coded_tags]
        sent = sent.split(' ')

        p = lambda s: ' '.join(["{:>14}".format(x) for x in s])
        c = lambda x: x if ' O' in x else colored(x, 'green')
        cp = lambda s: ' '.join([c("{:>14}".format(x)) for x in s])

        print('[INPUT] ' + p(sent))
        print('[CODED] ' + p(coded))
        print('[ TAG ] ' + p(coded_tags))
        print('[UNTAG] ' + cp(tags))
        print('[PROBS]')
        probs = [sorted([(p, reader.reverse_tag_dict[i]) for i,p in enumerate(w)])[-5:][::-1] for w in probilities]
        for w, p in zip(sent, probs):
            print('\t<{}>'.format(w))
            for t in p:
                print('\t\t{:>20}:\t{:7.4f}'.format(t[1], t[0]))
        print()


