from Reader import Reader, Metadata, Token
import utils, argparse, os, theano, numpy, timeit, pickle, sys
import theano.tensor as T
from lstm import LSTM
from lstmne import LSTM_NE
from lstmle import LSTM_LE

parser = argparse.ArgumentParser(usage="usage: lstm-test.py [options] filename")
parser.add_argument('filename')
parser.add_argument('--num-features',\
        type=int,\
        default=50,\
        help='number of features for word vectors')

parser.add_argument('--window',\
        type=int,\
        default=5,\
        help='Size of word window (default: 5)')

parser.add_argument('--l2',\
        type=float,\
        default=0,
        help='Coefficient of L2 regularization factors')

parser.add_argument('--load-reader',\
        help='Loads the reader instead of initialising another one',
        action='store_true')

parser.add_argument('--load-models',\
        help='Loads the models instead of initialising another one',
        action='store_true')

parser.add_argument('--validation-filename',\
        help='Loads another file with the validation test set')

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

parser.add_argument('--print-errors',\
        type=int,\
        default=-1,\
        help='Only print if the number of errors that sentence is lower than or equal x (default: -1)')

parser.add_argument('--fixed-embeddings',\
        help='Loads the corresponding embeddings from the given word embedding file')

parser.add_argument('--learn-embeddings',\
        help='Loads the corresponding embeddings from that only exists in the test sentence')

if __name__=="__main__":
    args = parser.parse_args()
    md = Metadata(args, args.filename, args.fixed_embeddings or args.learn_embeddings)

    varlist = list(map(str, [os.path.basename(args.filename), os.path.basename(args.validation_filename), \
        args.iterations, args.hidden, args.window, args.l2, args.fixed_embeddings is not None,
        args.learn_embeddings is not None]))

    directory_model = 'Model_' + '_'.join(varlist)
    utils.create_directory(directory_model)

    if args.load_reader:
        print('... loading reader')
        with open(os.path.join(directory_model, 'reader.pkl'), 'rb') as f:
            reader = pickle.load(f)
    else:
        print('... Generating new reader')
        reader = Reader(md, minimum_occurrence=2)
        #reader.save(directory_model)

    """
    Special options
    """
    #reader.load_files(directory_model)
    #reader.codify_sentences()

    # Generate the training set
    num_sentences = len(reader.sentences)
    num_words = len(reader.word_dict)
    num_tags = len(reader.tag_dict)

    if args.validation_filename:
        valid_md = Metadata(args, args.validation_filename, args.fixed_embeddings or args.learn_embeddings)
        valid_reader = Reader(valid_md)
        valid_reader.word_dict = reader.word_dict
        valid_reader.tag_dict = reader.tag_dict
        valid_reader.codify_sentences()

    if args.fixed_embeddings:
        codified_sentences = [numpy.concatenate(numpy.asarray(\
                utils.contextwin([reader.get_embedding(t.codified_word) for t in s],
                    args.window,\
                reader.get_padding_left(), reader.get_padding_right()\
                ), dtype=theano.config.floatX), axis=0)\
                for s in reader.sentences]

        if args.validation_filename:
            codified_sentences_valid = [numpy.concatenate(numpy.asarray(\
                    utils.contextwin([reader.get_embedding(t.codified_word) for t in s],
                        args.window,\
                    reader.get_padding_left(), reader.get_padding_right()\
                    ), dtype=theano.config.floatX), axis=0)\
                    for s in valid_reader.sentences]

    else:

        codified_sentences = [numpy.asarray(\
                utils.contextwin([t.codified_word for t in s], args.window,\
                reader.get_padding_left(), reader.get_padding_right()\
                ), dtype=numpy.int32)\
                for s in reader.sentences]

        if args.validation_filename:
            codified_sentences_valid = [numpy.asarray(\
                    utils.contextwin([t.codified_word for t in s], args.window,\
                    reader.get_padding_left(), reader.get_padding_right()\
                    ), dtype=numpy.int32)\
                    for s in valid_reader.sentences]

    codified_tags = [numpy.asarray([t.codified_tag for t in s], dtype=numpy.int32) for s in reader.sentences]
    if args.validation_filename:
        codified_tags_valid = [numpy.asarray([t.codified_tag for t in s], dtype=numpy.int32)
            for s in valid_reader.sentences]
    print('#sentences : {}, #words: {}, #tags : {}, learning rate : {}, #hidden : {}, embedding size: {} '.format(\
        num_sentences, num_words, num_tags, args.learning_rate, args.hidden, args.num_features))
    print('window size: {}'.format(args.window))

    print('#valid_sentences : {},'.format(len(valid_reader.sentences)))

    """
    :params dh dimension of hidden layer
    :params ne number of words
    :params nc number of classes (tags)
    :params de dimension of word embedding
    :params cs constext window size
    """
    if args.fixed_embeddings:
        model = LSTM_NE(args.l2, args.hidden, num_words, num_tags, args.num_features, args.window)
    elif args.learn_embeddings:
        model = LSTM_LE(args.l2, args.hidden, num_words + 1, num_tags, \
                args.num_features, args.window, reader.embedding_matrix)
    else:
        model = LSTM(args.l2, args.hidden, num_words + 1, num_tags, args.num_features, args.window)

    if args.load_models:
        print('... loading models')
        model.load(directory_model, varlist)

    print('... training the model')

    # Early stopping parameters
    best_mean_nll = numpy.inf
    start_time = timeit.default_timer()
    validation_frequency = num_sentences
    lowest_nll = 10000
    lowest_error = 10000000
    eps = 1e-3
    ignored = set()

    model.save(directory_model, varlist)

    epoch = 0
    while (epoch < args.iterations):
        epoch = epoch + 1
        nll = 0
        total_errors = 0
        parsed = 0
        max_e = 0
        epoch_start_time = timeit.default_timer()

        for minibatch_index in range(num_sentences):
            if minibatch_index in ignored:
                continue

            """
            model.train(codified_sentences[minibatch_index],\
                    codified_tags[minibatch_index],\
                    numpy.float32(args.learning_rate))

            if not args.fixed_embeddings:
                model.normalize()
            """

            n, e = model.test(codified_sentences[minibatch_index],\
                    codified_tags[minibatch_index])

            """
            if e == 0 and n < eps:
                ignored.add(minibatch_index)
            """
            nll += n
            total_errors += e
            parsed += 1
            max_e = max(e, max_e)

            if e >= 1 and False:
                current_sentence = [reader.uncodify_word(t[args.window//2]) for t in codified_sentences[minibatch_index]]
                current_tags = codified_tags[minibatch_index]
                y_pred, probs = model.classify(codified_sentences[minibatch_index])
                print('---------------------------')
                print('sent', current_sentence)
                print('gold', [reader.reverse_tag_dict[t] for t in current_tags])
                print('gues', [reader.reverse_tag_dict[t] for t in y_pred])

        if args.validation_filename:
            nllv = 0
            total_errorsv = 0
            parsedv = 0
            misclass = {}
            for minibatch_index in range(len(valid_reader.sentences)):
                nv, ev = model.test(codified_sentences_valid[minibatch_index],\
                        codified_tags_valid[minibatch_index])
                nllv += nv
                total_errorsv += ev
                parsedv += 1

                if ev >= 1:
                    current_sentence = [reader.uncodify_word(t[args.window//2]) for t in codified_sentences_valid[minibatch_index]]
                    current_tags = codified_tags_valid[minibatch_index]
                    y_pred, probs = model.classify(codified_sentences_valid[minibatch_index])
                    gold = [reader.reverse_tag_dict[t] for t in current_tags]
                    gues = [reader.reverse_tag_dict[t] for t in y_pred]
                    print('---------------------------')
                    print('sent', current_sentence)
                    print('gold', gold)
                    print('gues', gues)
                    for i in zip(gues, gold):
                        if i[0] != i[1]:
                            misclass[i] = misclass.get(i, 0) + 1


            from pprint import pprint
            pprint(misclass)
            print(sum(misclass.values()))
            exit()

            mean_nllv = nllv / parsedv
            print('epoch', epoch, 'mean nll [valid]', mean_nllv, 'total errors [valid]', total_errorsv, 'parsed [valid]', parsedv)

        epoch_time = timeit.default_timer() - epoch_start_time
        mean_nll = nll/parsed
        print('epoch', epoch, 'mean nll', mean_nll, 'total errors', total_errors, 'ignored', len(ignored), '/', parsed, 'time taken(s)', epoch_time)

        if mean_nll < best_mean_nll or total_errors < lowest_error:
            best_mean_nll = min(mean_nll, best_mean_nll)
            lowest_error = min(lowest_error, total_errors)

            # Save the model
            print('Saved')
            model.save(directory_model, varlist)
        else:
            #model.zero_helpers()
            ignored.clear()

        nll = 0
        total_errors = 0
        parsed = 0

    end_time = timeit.default_timer()
    print('The code run for %d epochs, with %f epochs/sec' % (epoch, 1.0 * epoch / (end_time - start_time)))
    print(('The code for file ' + os.path.split(__file__)[1]
        + ' ran for %.1fs' % (end_time -start_time)), file=sys.stderr)
