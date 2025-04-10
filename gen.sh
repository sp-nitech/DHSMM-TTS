#!/bin/sh
#. ./venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

set -eu

ver=demo
ltype=test
iter=0
#iter=final
indir=./data/phn/phone_hl

dir=gen/$ver/$ltype/$iter
log=$dir/gen.log
mkdir -p $(dirname $log)

date > $log
list=./list/${ltype}.list



for base in `sed 's| |/|g' $list`; do
    infile=$indir/$base.phn
    python3 -u -B scripts/model/$ver/generate.py \
            --input_file $infile \
            -o $dir \
            -c model/$ver/checkpoint_$iter \
            >> $log 2>&1
done

date >> $log
echo done >> $log

