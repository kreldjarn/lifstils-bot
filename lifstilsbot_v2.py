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

def create_model(corpus_path):
    corpus = open(corpus_path).read().lower()

    alphabet = sorted(list(set(corpus)))
    char2int = dict((c, i) for i, c in enumerate(alphabet))
    int2char = dict((i, c) for i, c in enumerate(alphabet))

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
        dataX.append([char2int[char] for char in seq_in])
        dataY.append(char2int[seq_out])

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
    return model, dataX, alphabet_length, int2char

def train_model(corpus_path, epochs=20):
    model = create_model(corpus_path)[0]
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    out_file = "models/lifstils-bot-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(out_file, monitor='loss', verbose=1, save_best_only=True, mode='min')
    model.fit(X, y, epochs=epochs, batch_size=128, callbacks=[checkpoint])

def generate_text(corpus_path, model, length):
    model, dataX, alphabet_length, int2char = create_model(corpus_path)
    model.load_weights(model)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    seed = dataX[numpy.random.randint(0, len(dataX) - 1)]

    for i in range(length):
        x = numpy.reshape(seed, (1, len(seed), 1)) / alphabet_length
        prediction = model.predict(x, verbose=0)
        idx = numpy.argmax(prediction)
        res = int2char[idx]
        sys.stdout.write(res)
        seed = seed.append(idx)[1:]



if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'tf:e:g:l:h', ['train', 'file=', 'epochs=', 'generate=', 'length=', 'help'])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    filename = ''
    try:
        filename = opts['-f']
    except KeyError:
        pass
    try:
        filename = opts['--file']
    except KeyError:
        pass

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit()

        elif opt in ('-t', '--train'):
            if len(filename) == 0:
                usage()
                sys.exit(2)
            try:
                epochs = opts['-e']
            except KeyError:
                try:
                    epochs = opts['--epochs']
                except KeyError:
                    epochs = 20
            train_model(filename, epochs)

        elif opt in ('-g', '--generate'):
            if len(filename) == 0:
                usage()
                sys.exit(2)
            try:
                length = opts['-l']
            except KeyError:
                try:
                    length = opts['--length']
                except KeyError:
                    length = 100

            generate_text(filename, arg, length)
        else:
            usage()
            sys.exit(2)


