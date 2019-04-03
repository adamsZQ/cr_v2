"""Convert raw GloVe word vector text file to h5."""
import argparse

import h5py
import numpy as np


def glove_h5(file_prefix):
    ite_num = 100000
    i = 0
    vocab = []
    vectors = []
    with open(file_prefix + 'embeddings/glove.840B.300d.txt', 'r') as f:
        for line in f:
            line = line.strip().split()
            vocab.append(line[0])
            vectors.append([float(val) for val in line[1:]])
            i = i + 1
            print(i)
            if i % ite_num == 0:
                try:
                    if i == ite_num:
                        w = h5py.File(file_prefix + 'embeddings/glove.840B.300d.h5', 'w')
                        atw = w.create_dataset(data=vectors, name='embedding', maxshape=(None, None,), chunks=True, )
                        # w.create_dataset(data=vocab, name='words_flatten')
                        shape_atw = atw.shape
                        print(w['embedding'][:])
                        # print(w['words_flatten'][:])
                        w.close()

                        # vocab = []
                        vectors = []
                    else:
                        w = h5py.File('glove.840B.300d.h5', 'a')
                        a = w['embedding']
                        # b = w['words_flatten']
                        shape_0 = a.shape[0]
                        shape_1 = a.shape[1]
                        a.resize(a.shape[0] + len(vectors), axis=0)
                        a[-len(vectors):, :] = vectors
                        ppp = a[:]
                        # b.resize(b.shape[0] + len(vocab), b.shape[1])
                        # b[-len(vocab):, :] = vocab

                        # vocab = []
                        vectors = []
                        w.close()

                except Exception as e:
                    print(e)
                    w.close()

            elif i > 2196016:
                w = h5py.File('glove.840B.300d.h5', 'a')
                a = w['embedding']
                # b = w['words_flatten']
                shape_0 = a.shape[0]
                shape_1 = a.shape[1]
                a.resize(a.shape[0] + len(vectors), axis=0)
                a[-len(vectors):, :] = vectors
                ppp = a[:]
                # b.resize(b.shape[0] + len(vocab), b.shape[1])
                # b[-len(vocab):, :] = vocab
                print('ppp', len(ppp))
                # vocab = []
                vectors = []
                w.close()

    vocab = '\n'.join(vocab)
    print('vocab', len(vocab))

    f = h5py.File(file_prefix + 'embeddings/glove.840B.300d.h5', 'a')
    f.create_dataset(data=vocab, name='words_flatten')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix") # data and model prefix
    args = parser.parse_args()

    HIDDEN_DIM = 4
    FILE_PREFIX = args.prefix

    if FILE_PREFIX is None:
        FILE_PREFIX = '/path/to/models/'
    glove_h5(FILE_PREFIX)
