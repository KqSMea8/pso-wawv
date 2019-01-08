#!/bin/bash

# GENERATE DOCUMENT VECTORS

# 20nshort, sanders
ds=$1

# cbow, sg, glove
wvmodel=$2

# 20, 50, 100
vdim=$3

# Greater or equal
ge=$4

# Initial iter
init_iter=$5

# Final iter
final_iter=$6

# Validation fraction
vf=$7

# Number of threads
nt=$8


for ((fold=init_iter; fold<=final_iter; fold++))
do
    train_fn=/tmp/${ds}_train_fold-${fold}.csv
    rm -f $train_fn
    touch $train_fn
    for i in {0..9}
    do
	if [ $i -eq $fold  ]; then
	    continue
	fi
	cat data/${ds}/${i} >> $train_fn
    done
    test_fn=data/${ds}/${fold}
    wv_fn=word_vectors_by_dataset/${ds}/${wvmodel}/${vdim}/wv_${ds}_${wvmodel}-${vdim}_w2v-format.vec
    out_fn=results/pso_wawv_dims_${ds}_${wvmodel}-${vdim}_fold-${fold}.pkl

    echo ""
    echo "======= EXECUCAO FOLD ${fold} ======="
    echo ""

    python3 src/pso_wawv_dims.py -tr ${train_fn} -te ${test_fn} -w ${wv_fn} -o ${out_fn} -ge ${ge} -vf ${vf} -t ${nt} -ac 1

    rm $train_fn
done
