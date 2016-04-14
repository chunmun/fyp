import os, utils, pickle
import numpy as np
import theano as th
from collections import Counter

class Metadata:
    def __init__(self, args, input=None, filename_embedding=None):
        directory, filename = os.path.split(input)
        self.directory = directory
        self.filename_input = input
        self.filename_embedding = filename_embedding
        self.num_features = args.num_features
        self.args = args

def is_number(s):
    return len(s) > 0 and (s.isdigit() or (s[0] == '-' and s[1:].isdigit()))

class ConllConfig(object):
    """
    Dummy class for storing the position of each field in a
    CoNLL data file.
    """
    id = 0
    word = 1
    lemma = 2
    pos = 3
    morph = 4
    parse = 7
    pred = 8
    srl = 9

    rare_tag = 'O'

variables = ['double', 'float', 'string', 'integer', 'int', 'short', 'unsigned', 'char']

class Token:
    def __init__(self, line):
        self.fields = line.strip().split() # for one token
        self.is_pred = self.fields[ConllConfig.pred] != '-'
        self.word = self.fields[ConllConfig.word].lower()
        self.lemma = self.fields[ConllConfig.lemma].lower()
        self.pos = self.fields[ConllConfig.pos]
        # TODO : Assuming tags of the form (A0*), (V*), ..
        # TODO : Only using 1 tag, fuck da rest
        tag = self.fields[ConllConfig.srl]
        self.srl_tag = tag[1:-2] if tag !='*' else ConllConfig.rare_tag

        # Special treatment for numbers
        self.is_number = is_number(self.word)
        if self.is_number:
            self.word = WordDictionary.number

        # Special treatment for types
        """
        if self.word in variables:
            self.word = WordDictionary.variable
        """

        self.codified_word = None
        self.codified_tag = None

    @classmethod
    def initFromWord(cls, word, tag='*'):
        line = ' '.join(['0', word, word, '*SPECIAL*', '*', '*', '*', '*', '-', tag])
        return cls(line)

    def codify_token(self, word_dict, tag_dict):
        self.codified_word = word_dict[self.word]
        self.codified_tag = tag_dict[self.srl_tag]

    def __str__(self):
        return self.word
    def __eq__(self, o):
        return (self.is_number and o.is_number) or self.word == o.word
    def __hash__(self):
        return hash(self.word)

class WordDictionary(dict):
    padding_left = '*left*'
    padding_right = '*right*'
    rare = '*rare*'
    number = '*number*'
    variable = '*variable*'

    def __init__(self, tokens=None, minimum_occurrence=1):
        self.reverse_dict = {}
        if tokens:
            self.insertFromTokenList(tokens, minimum_occurrence)

    def insertFromTokenList(self, tokens, minimum_occurrence):
        counter = Counter(tokens)

        top_tokens = [key for key, number in counter.most_common() \
                    if number >= minimum_occurrence or \
                    key.word in [WordDictionary.padding_left,
                        WordDictionary.padding_right,
                        WordDictionary.rare,
                        WordDictionary.number]]
        top_tokens.sort(key=lambda t: t.word)

        for token in top_tokens:
            idx = len(self)
            self[token.word] = idx
            self.reverse_dict[idx] = token

    def insertFromWordList(self, wordlist):
        for w in wordlist:
            t = Token.initFromWord(w)
            self.insert(t)

    def insert(self, token):
        if type(token) is str:
            token = Token.initFromWord(token)
        idx = len(self)
        self[token.word] = idx
        self.reverse_dict[idx] = token

    def insertSpecials(self):
        if not self.get(WordDictionary.rare):
            specials = [Token.initFromWord(WordDictionary.rare),\
                    Token.initFromWord(WordDictionary.padding_left),
                    Token.initFromWord(WordDictionary.padding_right),
                    Token.initFromWord(WordDictionary.number)]

            for s in specials:
                idx =  len(self)
                self[s.word] = idx
                self.reverse_dict[idx] = s

    def __contains__(self, word):
        if type(word) == Token:
            word = word.word
        if is_number(word):
            return number
        return super().__contains__(key)

    def __getitem__(self, word):
        if type(word) == Token:
            word = word.word
        else:
            t = Token.initFromWord(word)
            word = t.word
        res = super().get(word, False)
        if res:
            return res
        return super().get(WordDictionary.rare)

# This reader is used purely for predicate detection
class Reader:
    def __init__(self, metadata, minimum_occurrence=1):
        self.metadata = metadata
        self.tag_dict = {ConllConfig.rare_tag: 0}
        self.reverse_tag_dict = {0: ConllConfig.rare_tag}

        self.word_dict = WordDictionary()
        self.sentences = None
        self.predicates = None
        self.max_sent_length = 0

        if metadata.filename_input is not None:
            self.read_conll(minimum_occurrence)
            self.word_dict.insertSpecials()
            self.codify_sentences()
        else:
            self.word_dict.insertSpecials()

        if metadata.filename_embedding is not None:
            print('Retrieving embedding from ', metadata.filename_embedding)
            self.embedding_matrix = []
            self.load_embedding_file()


    def load_embedding_file(self):
        with open(self.metadata.filename_embedding, 'r') as f:
            rare_idx = self.word_dict[WordDictionary.rare]
            self.embedding_matrix = [[1e-10] * self.metadata.num_features] * (len(self.word_dict) + 1)
            for line in f:
                line = line.split(' ')
                word_idx = self.word_dict[Token.initFromWord(line[0])]
                if word_idx < rare_idx:
                    self.embedding_matrix[word_idx] = [float(x) for x in line[1:]]
            self.embedding_matrix = np.asarray(self.embedding_matrix, dtype=th.config.floatX)


    def get_embedding(self, wordidx):
        if wordidx >= len(self.embedding_matrix): # This is a rare word
            return np.zeros(self.metadata.num_features, dtype=th.config.floatX)
        return self.embedding_matrix[wordidx]

    """
    Reads in the input from the filename in metadata
    Generates the word_dict and tag_dict mapping
    """
    def read_conll(self, minimum_occurrence=1):
        self.sentences = []
        self.predicates = []
        tokens = []
        all_toks = []

        def collect():
            nonlocal tokens
            if len(tokens) > 0:
                self.sentences.append(tokens)
                self.max_sent_length = max(len(tokens), self.max_sent_length)
                self.predicates.append([i for i,e in enumerate(tokens) if e.is_pred])
                tokens = []

        with open(self.metadata.filename_input, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    collect()
                    continue
                t = Token(line)
                tokens.append(t)
                all_toks.append(t)

                if t.srl_tag not in self.tag_dict:
                    self.tag_dict[t.srl_tag] = len(self.tag_dict)
                    self.reverse_tag_dict[len(self.reverse_tag_dict)] = t.srl_tag

            if len(tokens) > 0: # collect the last line
                collect()

        if self.word_dict is None:
            self.word_dict = WordDictionary(all_toks, minimum_occurrence)
        else:
            self.word_dict.insertFromTokenList(all_toks, minimum_occurrence)

    def get_padding_left(self):
        return self.word_dict[self.word_dict.padding_left]

    def get_padding_right(self):
        return self.word_dict[self.word_dict.padding_right]

    # TODO : Gonna do front padding, and wonder if that'll work
    def pad_sentences(self):
        rare_tok = Token.initFromWord(WordDictionary.rare)
        for s in self.sentences:
            while len(s) < self.max_sent_length:
                s.insert(0, rare_tok)

    def codify_string(self, sent_str):
        return [self.word_dict[w.strip()] for w in sent_str.strip().split()]

    def codify_sentences(self):
        for tokens in self.sentences:
            for token in tokens:
                token.codify_token(self.word_dict, self.tag_dict)

    def uncodify_sentence(self, sentence_idx):
        return [self.uncodify_word(i) for i in sentence_idx]

    def uncodify_word(self, word_idx):
        if word_idx not in self.word_dict.reverse_dict:
            return self.word_dict.rare
        return self.word_dict.reverse_dict[word_idx].word

    def save(self, folder):
        with open(os.path.join(folder, 'reader.pkl'), 'wb') as f:
            pickle.dump(self, f)

    def load(self, folder):
        with open(os.path.join(folder, 'reader.pkl'), 'rb') as f:
            self = pickle.load(f)

    def save_files(self, folder):
        with open(os.path.join(folder, 'word_dict.pkl'), 'wb') as f:
            pickle.dump(self.word_dict, f)

        with open(os.path.join(folder, 'reverse_tag_dict.pkl'), 'wb') as f:
            pickle.dump(self.reverse_tag_dict, f)

        with open(os.path.join(folder, 'tag_dict.pkl'), 'wb') as f:
            pickle.dump(self.tag_dict, f)

    def load_files(self, folder):
        with open(os.path.join(folder, 'word_dict.pkl'), 'rb') as f:
            self.word_dict = pickle.load(f)

        with open(os.path.join(folder, 'reverse_tag_dict.pkl'), 'rb') as f:
            self.reverse_tag_dict = pickle.load(f)

        with open(os.path.join(folder, 'tag_dict.pkl'), 'rb') as f:
            self.tag_dict = pickle.load(f)

