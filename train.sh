#!/bin/sh
#. ./venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

set -eu

debug=false
#debug=true

ver=demo

log=./model/$ver/logs/train.`date +"%Y-%m-%d_%H:%M:%S"`.log

mkdir -p $(dirname $log)

opt=""
if $debug; then
    opt="$opt --debug"
    log=./model/$ver/logs/debug.log
fi
opt="-c model/$ver/checkpoint_0"

date > $log
python3 -u -B scripts/model/$ver/train.py \
        $opt \
        -o model/$ver \
        >> $log 2>&1

date >> $log
echo done >> $log
