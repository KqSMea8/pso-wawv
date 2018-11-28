#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import random

def chunks(xs, n):
    ys = list(xs)
    random.shuffle(ys)
    ylen = len(ys)
    size = int(ylen / n)
    chunks = [ys[0+size*i : size*(i+1)] for i in range(n)]
    leftover = ylen - size*n
    edge = size*n
    for i in range(leftover):
            chunks[i%n].append(ys[edge+i])
    return chunks

def main(argv = None):
    if argv is None:
        argv = sys.argv
    ds_fn = argv[1]
    num_folds = int(argv[2])
    ds_file = open(ds_fn, 'r')
    file_lines = ds_file.readlines()
    ds_file.close()
    folds = chunks(file_lines, num_folds)
    for i in range(num_folds):
        fold_file = open(str(i), "w")
        for l in folds[i]:
            fold_file.write(l)
        fold_file.close()
        
if __name__ == '__main__':
    main()
