from Reader import Reader, Metadata, Token
import utils, argparse, os, timeit, pickle, sys
import theano as th
import numpy as np
import theano.tensor as T
from nn import Embedding, Dropout
from lstm import LSTM
from lstmne import LSTM_NE
from lstmle import LSTM_LE

parser = argparse.ArgumentParser(usage="usage: train.py [options] filename")
parser.add_argument('filename')
parser.add_argument('--num-features',\
        type=int,\
        default=50,\
        help='number of features for word vectors')

parser.add_argument('--l2',\
        type=float,\
        default=0,
        help='Coefficient of L2 regularization factors')

parser.add_argument('--dropout-rare',\
        type=float,\
        default=0,
        help='Proabability of a word turning into a rare word')

parser.add_argument('--dropout',\
        type=float,\
        default=0,
        help='Proabability of zeroing embedded word vectors')

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
        args.iterations, args.hidden, args.l2, args.dropout_rare, args.dropout,\
        args.fixed_embeddings is not None, args.learn_embeddings is not None]))

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

    # Generate the training set
    num_sentences = len(reader.sentences)
    num_words = len(reader.word_dict)
    num_tags = len(reader.tag_dict)

    n = lambda x: np.asarray(x, dtype=np.int32)

    codified_sentences = [n([t.codified_word for t in s]) for s in reader.sentences]
    codified_tags = [n([t.codified_tag for t in s]) for s in reader.sentences]

    print('#sentences : {}, #words: {}, #tags : {}, learning rate : {}, #hidden : {}, embedding size: {} '.format(\
        num_sentences, num_words, num_tags, args.learning_rate, args.hidden, args.num_features))

    if args.validation_filename != None:
        valid_md = Metadata(args, args.validation_filename)
        reader_valid = Reader(valid_md)
        reader_valid.word_dict = reader.word_dict
        reader_valid.tag_dict = reader.tag_dict
        reader_valid.codify_sentences()

        codified_sentences_valid = [n([t.codified_word for t in s]) for s in reader_valid.sentences]
        codified_tags_valid = [n([t.codified_tag for t in s]) for s in reader_valid.sentences]

    x = T.ivector('x')
    y = T.ivector('y')
    mask  = T.ivector('mask')

    emb = Embedding(x, args.num_features, num_words+1)
    if args.dropout:
        dropout = Dropout(emb.output, args.num_features, args.dropout)
        lstm = LSTM(dropout.output, args.l2, args.hidden, num_words + 1, num_tags, args.num_features)
    else:
        lstm = LSTM(emb.output, args.l2, args.hidden, num_words + 1, num_tags, args.num_features)

    if args.load_models:
        print('... Loaded Models')
        emb.load(directory_model, varlist)
        lstm.load(directory_model, varlist)

    te, nll = lstm.errors(y)
    params = emb.params + lstm.params
    params_helper = emb.params_helper + lstm.params_helper

    rho = 10
    lr = np.float32(float(args.learning_rate))

    sentence_gradients = T.grad( nll, params )

    sentence_updates = [(h, (h + g ** 2).clip(-5, 5)) for p,h,g in
                zip( params, params_helper, sentence_gradients)]

    sentence_updates.extend([(p, (p-lr*g*rho / (rho+T.sqrt(h + g ** 2))).clip(-5,5)) for p,h,g in
                zip( params, params_helper, sentence_gradients)])

    #sentence_updates.extend([(p, p - (lr*g).clip(-5,5)) for p,g in zip( params, sentence_gradients)])

    train = th.function(
            inputs  = [x, y],
            outputs = lstm.errors(y),
            updates = sentence_updates)

    te_with_rare_dropout, nll_with_rare_dropout = lstm.errors_with_rare_dropout(y, mask)

    sentence_gradients_with_rare_dropout = T.grad( nll_with_rare_dropout, params )
    sentence_updates_with_rare_dropout = [(p, p - (lr*g).clip(-5,5)) \
            for p,g in zip( params, sentence_gradients_with_rare_dropout)]

    train_with_rare_dropout = th.function(
            inputs  = [x, y, mask],
            outputs = lstm.errors_with_rare_dropout(y, mask),
            updates = sentence_updates_with_rare_dropout)

    test = th.function(
            inputs = [x, y],
            outputs = lstm.errors(y))

    classify = th.function(
            inputs = [x],
            outputs = lstm.y_pred)


    print('... training the model')

    # Early stopping parameters
    best_mean_nll = np.inf
    start_time = timeit.default_timer()
    lowest_nll = 10000
    lowest_error = 10000000
    eps = 1e-3

    #model.save(directory_model, varlist)

    epoch = 0
    while (epoch < args.iterations):
        epoch = epoch + 1
        total_nll = np.float32(0)
        total_errors = 0
        parsed = 0
        max_e = 0
        epoch_start_time = timeit.default_timer()

        for minibatch_index in range(num_sentences):
            if args.dropout_rare:
                [s, mask] = utils.dropout_rare(codified_sentences[minibatch_index], reader, args.dropout_rare)
                e, nll = train_with_rare_dropout(s, codified_tags[minibatch_index], mask)
            else:
                s = codified_sentences[minibatch_index]
                e, nll = train(s, codified_tags[minibatch_index])
            #print(nll)
            emb.normalize()

            total_errors += e
            total_nll += nll
            parsed += 1

            if args.print_errors and e>0 and e >= args.print_errors and False:
                w = [t.word for t in reader.sentences[minibatch_index]]
                print(w)
                print([reader.reverse_tag_dict[t] for t in codified_tags[minibatch_index]])
                print([reader.reverse_tag_dict[t] for t in classify(codified_sentences[minibatch_index])])

        # Compute validation errors and mean nll
        total_errors_valid = 0
        total_nll_valid = np.float32(0)
        parsed_valid = 0
        for minibatch_index in range(len(codified_sentences_valid)):
            e, nll = test(codified_sentences_valid[minibatch_index], codified_tags_valid[minibatch_index])
            total_errors_valid += e
            total_nll_valid += nll
            parsed_valid += 1

        mean_nll = total_nll / parsed
        mean_nll_valid = total_nll_valid / parsed_valid

        if lowest_error > total_errors_valid or best_mean_nll > mean_nll_valid:
            best_mean_nll = min(best_mean_nll, mean_nll_valid)
            lowest_error = min(lowest_error, total_errors_valid)
            print('Saved Model')
            emb.save(directory_model, varlist)
            lstm.save(directory_model, varlist)

        time_taken = timeit.default_timer() - epoch_start_time
        print('Epoch {:3} - Total Error: {:4}, Mean nll: {:6.4f}, Valid E : {:4}, Valid mean nll: {:6.4f}, Elapsed : {:6.4f}'.format(epoch, total_errors, mean_nll, total_errors_valid, mean_nll_valid, time_taken))

    end_time = timeit.default_timer()
    print('The code run for %d epochs, with %f epochs/sec' % (epoch, 1.0 * epoch / (end_time - start_time)))
    print(('The code for file ' + os.path.split(__file__)[1]
        + ' ran for %.1fs' % (end_time -start_time)), file=sys.stderr)
