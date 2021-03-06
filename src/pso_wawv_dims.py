#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter("ignore")

import sys
import argparse
import operator
import random
from multiprocessing import Pool
import pickle
import math

import numpy as np

from deap import base
from deap import creator
from deap import tools

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB

from gensim.models import KeyedVectors

def load_ds(ds_filename):
    doc_words = []
    doc_classes = []
    ds_file = open(ds_filename, 'r')
    for d in ds_file:
        class_and_text = d.split(';')
        words = set(class_and_text[2].split())
        if len(words) == 0:
            continue
        doc_words.append(words)
        doc_classes.append(class_and_text[1])
    ds_file.close()
    return doc_words, doc_classes


def generate(size, pmin, pmax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
    part.speed = [random.uniform(pmin, pmax) for _ in range(size)]
    return part


def updateParticle(part, best, constriction_factor, c1, c2):
    c1r1 = np.array([c1*random.uniform(0, 1) for _ in range(len(part))])
    c2r2 = np.array([c2*random.uniform(0, 1) for _ in range(len(part))])
    cog_comp = c1r1 * list(map(operator.sub, part.best, part))
    soc_comp = c2r2 * list(map(operator.sub, best, part))
    part.speed = list(constriction_factor * (part.speed + cog_comp + soc_comp))
    part[:] = list(map(operator.add, part, part.speed))
#     ce1 = (c1*random.uniform(0, 1) for _ in range(len(part)))
#     ce2 = (c2*random.uniform(0, 1) for _ in range(len(part)))
#     ce1_p = map(operator.mul, ce1, map(operator.sub, best, part))
#     ce2_g = map(operator.mul, ce2, map(operator.sub, part.best, part))
#     a = map(operator.sub,
#                       map(operator.mul,
#                                     itertools.repeat(constriction_factor),
#                                     map(operator.add, ce1_p, ce2_g)),
#                       map(operator.mul,
#                                      itertools.repeat(1-constriction_factor),
#                                      part.speed))
#     part.speed = list(map(operator.add, part.speed, a))
#     for i, speed in enumerate(part.speed):
#         if speed < part.smin:
#             part.speed[i] = part.smin
#         elif speed > part.smax:
#             part.speed[i] = part.smax
#     part[:] = list(map(operator.add, part, part.speed))


def doc_vec(word_weights, dim_weights, doc_words, wv_model):
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
    part1 = np.matmul(prunned_word_weights, w)
    part2 = np.sum(prunned_word_weights)
    part3 = part1 / part2
    return dim_weights * part3
#     
#     dv = None
#     for w in doc_words:
#         if dv is None:
#             dv = wv_model[w]
#         else:
#             dv = dv + wv_model[w]
#     dv = dim_weights * dv / len(doc_words)
#     return dv


def evalClassif(particle, train_X, train_y, valid_X, valid_y, wv_model, multiclass, alt_classif):
    # Train document vectors
    vocab_size = len(wv_model.wv.vocab)
    train_dvs = []
    for x in train_X:
        train_dvs.append(doc_vec(particle[:vocab_size], particle[vocab_size:], x, wv_model))
    # Valid document vectors
    valid_dvs = []
    for x in valid_X:
        valid_dvs.append(doc_vec(particle[:vocab_size], particle[vocab_size:], x, wv_model))

    classif = None
    if alt_classif == 1:
#         classif = neighbors.KNeighborsClassifier(k, weights='uniform')
#         classif = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
        classif = GaussianNB()
    else:
        # Classifier training and validation
        selected_solver = 'liblinear'
        selected_mc = 'ovr'
        if multiclass:
            #selected_solver = 'lbfgs'
            selected_solver = 'newton-cg'
            selected_mc = 'multinomial'
        classif = LogisticRegression(max_iter=1000000, solver=selected_solver, multi_class=selected_mc, n_jobs=-1)
   
    classif.fit(train_dvs, train_y)

    valid_pred_y = classif.predict(valid_dvs)
    fitness = f1_score(valid_y, valid_pred_y, average = "macro")
    
    return fitness,


# def evalClassif_old(particle, train_doc_words, train_doc_classes, valid_fraction, wv_model):
#     # Train and valid sets
#     train_X, valid_X, train_y, valid_y = train_test_split(train_doc_words, train_doc_classes,
#                                                           test_size = valid_fraction,
#                                                           stratify = train_doc_classes)
#     # Train document vectors
#     train_dvs = []
#     for x in train_X:
#         train_dvs.append(doc_vec(particle, x, wv_model))
#     # Valid document vectors
#     valid_dvs = []
#     for x in valid_X:
#         valid_dvs.append(doc_vec(particle, x, wv_model))
# 
#     # Classifier training and validation                                                                             
#     classif = LogisticRegression(max_iter=1000000)
#     classif.fit(train_dvs, train_y)
# 
#     valid_pred_y = classif.predict(valid_dvs)
#     fitness = f1_score(valid_y, valid_pred_y, average = "macro")
# 
#     return fitness,


def test_classif(particle, train_doc_words, train_doc_classes, test_doc_words, test_doc_classes, wv_model, multiclass):
    # Train document vectors
    vocab_size = len(wv_model.wv.vocab)
    train_dvs = []
    for x in train_doc_words:
        train_dvs.append(doc_vec(particle[:vocab_size], particle[vocab_size:], x, wv_model))
    # Test document vectors
    test_dvs = []
    for x in test_doc_words:
        test_dvs.append(doc_vec(particle[:vocab_size], particle[vocab_size:], x, wv_model))
    # Classifier
    selected_solver = 'liblinear'
    selected_mc = 'ovr'
    if multiclass:
        #selected_solver = 'lbfgs'                                                                             
        selected_solver = 'newton-cg'
        selected_mc = 'multinomial'
    classif = LogisticRegression(max_iter=1000000, solver=selected_solver, multi_class=selected_mc, n_jobs=-1)
    classif.fit(train_dvs, train_doc_classes)
    # Test predictions
    test_pred = classif.predict(test_dvs)
    #Score
    test_score = f1_score(test_doc_classes, test_pred, average = "macro")
    return test_score

#     # Classifier
#     classif = LogisticRegression(max_iter=1000000)
#     classif.fit(train_dvs, train_doc_classes)
#     # Test predictions
#     test_pred = classif.predict(test_dvs)
#     #Score
#     test_score = f1_score(test_doc_classes, test_pred, average = "macro")
#     return test_score

def main(argv = None):

    # Parsing command line arguments
    if argv is None:
        argv = sys.argv
    parser = argparse.ArgumentParser(
        description = "Generates document vectors from word vectors through PSO (word and dims weights).")
    parser.add_argument("-tr", "--train_filename", help = "Train data", required = True)
    parser.add_argument("-te", "--test_filename", help = "Test data", required = True)
    parser.add_argument("-w", "--wv_filename", help = "Word vectors file", required = True)
    parser.add_argument("-o", "--output_filename", help = "Output file", required = True)
    parser.add_argument("-vf", "--valid_fraction", help = "Fraction of train dataset used for validation",
                        type = float, default = 0.5)
    parser.add_argument("-tf", "--test_fraction", help = "Fraction of dataset used for testing",
                        type = float, default = 0.3)
    parser.add_argument("-p", "--pop_size", help = "Number of particles (population size)",
                        type = int, default = 50)
    parser.add_argument("-g", "--num_generations", help = "Number of generations",
                        type = int, default = 1000)
    parser.add_argument("-t", "--num_threads", help = "Number of threads",
                        type = int, default = 24)
    parser.add_argument("-sv", "--stop_value", help = "Number of generations of tolerance without progress",
                        type = int, default = 100)
    parser.add_argument("-ii", "--inertia_init", help = "Initial value of inertia",
                        type = float, default = 0.9)
    parser.add_argument("-if", "--inertia_final", help = "Final value of inertia",
                        type = float, default = 0.4)
    parser.add_argument("-pmin", "--pmin", help = "Minimum initial value of a particle dimension",
                        type = float, default = 0.9)
    parser.add_argument("-pmax", "--pmax", help = "Maximum initial value of a particle dimension",
                        type = float, default = 1.1)
#     parser.add_argument("-smin", "--smin", help = "Minimum speed value of a particle",
#                         type = float, default = -0.1)
#     parser.add_argument("-smax", "--smax", help = "Maximum speed value of a particle dimension",
#                         type = float, default = 0.1)
    parser.add_argument("-c1", "--c1", help = "C1", type = float, default = 2.0)
    parser.add_argument("-c2", "--c2", help = "C2", type = float, default = 2.0)
    parser.add_argument("-ge", "--greater_equal", help = "Consider greater or equal values as progress",
                        type = int, default = 0)
    parser.add_argument("-ac", "--alternative_classifier", help = "Use alternative classifier",
                        type = int, default = 0)
    args = parser.parse_args(argv[1:])

    # Loading datasets
    print("Building train and test datasets...", end = " ")
    train_doc_words, train_doc_classes = load_ds(args.train_filename)
    train_X, valid_X, train_y, valid_y = train_test_split(train_doc_words, train_doc_classes,
                                                          test_size = args.test_fraction)
    test_doc_words, test_doc_classes = load_ds(args.test_filename)
    mc = (len(set(train_y)) > 2) or (len(set(valid_y)) > 2)
    print("OK!")

    # Loading word vectors
    print("Loading word vectors...", end = " ")
    wv_model = KeyedVectors.load_word2vec_format(args.wv_filename, binary=False)
    print("OK!")

    # Fitness and particle functions
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, best=None)
    
    # Constriction factor
    phi = args.c1 + args.c2
    chi = 2.0 / abs(2.0 - phi - math.sqrt(math.pow(phi, 2.0) - 4.0 * phi))

    # Toolbox
    toolbox = base.Toolbox()
    toolbox.register("particle", generate, size=(len(wv_model.wv.vocab) + wv_model.vector_size), pmin=args.pmin, pmax=args.pmax)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, constriction_factor = chi, c1 = args.c1, c2 = args.c2)
    toolbox.register("evaluate", evalClassif, train_X = train_X, train_y = train_y,
                     valid_X = valid_X, valid_y = valid_y, wv_model = wv_model, multiclass=mc,
                     alt_classif=args.alternative_classifier)

    # Initial population
    pop = toolbox.population(n = args.pop_size)

    # Statistics definitions
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Log
    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    # Thread pool
    pool = Pool(args.num_threads)
    toolbox.register("map", pool.map)

    best = None
    test_best = 0.0
    test_best_particle = None
    num_gen_no_progress = 0
    for g in range(args.num_generations):
        has_best = False
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if args.greater_equal == 0:
                if not part.best or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if not best or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values
                    has_best = True
            else:
                if not part.best or part.best.fitness <= part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if not best or best.fitness <= part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values
                    has_best = True
        if has_best:
            num_gen_no_progress = 0
            current_test_best = test_classif(best, train_doc_words, train_doc_classes,
                                             test_doc_words, test_doc_classes, wv_model, mc)
            if not test_best_particle or test_best < current_test_best:
                test_best_particle = creator.Particle(best)
                test_best_particle.fitness.values = best.fitness.values
                test_best = current_test_best
                print("\n*** NEW BEST OVERALL:", test_best, " ***\n")
            else:
                print("\nNEW BEST:", current_test_best, "\n")
            
#             test_best = test_classif(best, train_X, train_y,
#                                      valid_X, valid_y, wv_model)
#             print("\n\tNEW BEST:", test_best, "\n")
        else:
            num_gen_no_progress += 1
        for part in pop:
            if part != best:
                toolbox.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)

        # Save results                                                                                               
        dump_file = open(args.output_filename, 'wb')
        pickle.dump(dict(best = best, population = pop, log = logbook, score = test_best,
                         test_best_part = test_best_particle), dump_file, 2)
        dump_file.close()

        # Stop if no progress
        if num_gen_no_progress >= args.stop_value:
            print("Stop criteria met (no progress)")
            break

    print("\n*** BEST OVERALL:", test_best, " ***\n")
    pool.close()


if __name__ == '__main__':
    main()
