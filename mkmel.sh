#!/bin/sh
#. ./venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

set -eu

ver=01
log=data/mel/$ver/logs/mkmel.log
mkdir -p $(dirname $log)

echo >> $log
date >> $log
hostname >> $log

dbdir=/db/ATR-TTS-JP-CORPUS/F009
outdir=./data/mel/$ver
scpdir=./scripts/mel/$ver

listdir=./list
alllist=$listdir/all.list
trnlist=$listdir/train.list

echo Start extracting spectrograms [$(date)] >> $log
python3 -u -B $scpdir/extract_spectrogram.py \
        $alllist \
        --wavdir $dbdir \
        --outdir $outdir \
        >> $log.extrc 2>&1
echo done [$(date)] >> $log
echo >> $log

echo Start calculating statistics [$(date)] >> $log
python3 -u -B $scpdir/normalize.py \
        $trnlist \
        --specdir $outdir \
        --calcstats \
        >> $log.stats 2>&1
echo done [$(date)] >> $log
echo >> $log

echo Start normalizing spectrograms [$(date)] >> $log
python3 -u -B $scpdir/normalize.py \
        $alllist \
        --specdir $outdir \
        --normalize \
        >> $log.norml 2>&1
echo done [$(date)] >> $log
echo >> $log
