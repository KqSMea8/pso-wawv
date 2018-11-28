import sys
import pickle
import numpy as np
from deap import base
from deap import creator
from gensim.models import KeyedVectors
import pandas as pd

def doc_vec(word_weights, doc_words, wv_model):
    erase_list = []
    w = None
    i = 0
    for v in wv_model.vocab:
        if v in doc_words:
            if w is None:
                w = wv_model[v]
            else:
                w = np.vstack((w, wv_model[v]))
        else:
            erase_list.append(i)
        i += 1
    prunned_word_weights = np.delete(word_weights, erase_list)
    num_docs = len(doc_words)
    if num_docs == 1:
        w = np.array([w])
    return np.matmul(prunned_word_weights, w) / np.sum(prunned_word_weights)

def load_ds(ds_filename):
    doc_words = []
    ds_file = open(ds_filename, 'r')
    for d in ds_file:
        id_class_text = d.split(';')
        words = set(id_class_text[2].split())
        doc_words.append(words)
    ds_file.close()
    return doc_words

def main(argv = None):
    ds_filename = sys.argv[1]
    wv_filename = sys.argv[2]
    results_filename = sys.argv[3]
    dvs_filename = sys.argv[4]
    weights_filename = sys.argv[5]

    doc_words = load_ds(ds_filename)
    wv_model = KeyedVectors.load_word2vec_format(wv_filename, binary=False)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, best=None)
    results = pickle.load(open(results_filename, 'rb'))

    print "Generating doc vecs...",
    sys.stdout.flush()
    pso_wawv_dvs = None
    for ws in doc_words:
        if pso_wawv_dvs is None:
            pso_wawv_dvs = doc_vec(results['test_best_part'], ws, wv_model)
        else:
            pso_wawv_dvs = np.vstack((pso_wawv_dvs, doc_vec(results['test_best_part'], ws, wv_model)))
    num_docs = len(doc_words)
    if num_docs == 1:
        w = np.array([pso_wawv_dvs])
    print "OK!"

    # Save doc vecs
    print "Saving doc vecs...",
    sys.stdout.flush()
    np.savetxt(dvs_filename, pso_wawv_dvs, fmt='%1.4f', delimiter=' ')
    print "OK!"

    # Save weights
    print "Saving vocab weights...",
    sys.stdout.flush()
    pd.DataFrame({'word' : pd.Categorical(wv_model.vocab.keys()), 'weight' : list(results['test_best_part'])}).to_csv(weights_filename)
    print "OK!"
        

if __name__ == '__main__':
    main()
