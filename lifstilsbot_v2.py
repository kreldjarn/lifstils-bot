#!/usr/bin/env python3
import sys
import getopt

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

def usage():
    pass

def train_model(corpus_path, epochs=20):
    corpus = open(corpus_path).read().lower()

    alphabet = sorted(list(set(corpus)))
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))

    corpus_length = len(corpus)
    alphabet_length = len(alphabet)
    print('Corpus length: {}'.format(corpus_length))
    print('Alphabet length: {}'.format(alphabet_length))

    seq_length = 4
    dataX = []
    dataY = []
    for i in range(0, corpus_length - seq_length):
        seq_in = corpus[i:i + seq_length]
        seq_out = corpus[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])

    n_patterns = len(dataX)

    print('Number of patterns: {}'.format(n_patterns))
    # [samples, time steps, features]
    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    # Normalise
    X = X / alphabet_length
    # Cast to factors?
    y = np_utils.to_categorical(dataY)

    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    out_file = "models/lifstils-bot-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(out_file, monitor='loss', verbose=1, save_best_only=True, mode='min')

    model.fit(X, y, epochs=20, batch_size=128, callbacks=[checkpoint])

def generate_text(model, length):
    pass

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 't:e:g:l:h', ['train=', 'epochs=', 'generate=', 'length=', 'help'])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit()

        elif opt in ('-t', '--train'):
            try:
                epochs = opts['-e']
            except KeyError:
                try:
                    epochs = opts['--epochs']
                except KeyError:
                    epochs = 20
            train_model(arg, epochs)

        elif opt in ('-g', '--generate'):
            try:
                length = opts['-l']
            except KeyError:
                try:
                    length = opts['--length']
                except KeyError:
                    length = 100

            generate_text(arg, length)


