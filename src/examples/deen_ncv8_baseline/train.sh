#!/usr/bin/env bash
#install_name_tool -change libboost_program_options.dylib /Users/roeeaharoni/boost_1_55_0/stage/lib/libboost_program_options.dylib /Users/roeeaharoni/git/dynet/build/dynet/libdynet.dylib
#install_name_tool -change libboost_serialization.dylib /Users/roeeaharoni/boost_1_55_0/stage/lib/libboost_serialization.dylib /Users/roeeaharoni/git/dynet/build/dynet/libdynet.dylib
base_path=/home/nlp/aharonr6
#base_path=/Users/roeeaharoni

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

python $base_path/git/dynet-seq2seq-attn/src/dynmt.py \
--dynet-devices GPU:1 \
--dynet-mem 12000 \
--input-dim=256 \
--hidden-dim=512 \
--epochs=100 \
--eval-after=10000 \
--lstm-layers=1 \
--optimization=ADAM \
--batch-size=64 \
--beam-size=12 \
--max-len=50 \
--plot \
--eval-script=$base_path/git/dynet-seq2seq-attn/src/examples/deen_ncv8_baseline/validate.sh \
$base_path/git/research/string-to-tree-nmt/data/news-de-en/train/news-commentary-v8.de-en.tok.penntrg.clean.true.bpe.de \
$base_path/git/research/string-to-tree-nmt/data/news-de-en/train/news-commentary-v8.de-en.tok.penntrg.clean.true.bpe.en \
$base_path/git/research/string-to-tree-nmt/data/news-de-en/dev/newstest2015-deen.tok.penntrg.clean.true.bpe.de \
$base_path/git/research/string-to-tree-nmt/data/news-de-en/dev/newstest2015-deen.tok.penntrg.clean.true.bpe.en \
$base_path/git/research/string-to-tree-nmt/data/news-de-en/test/newstest2016-deen.tok.penntrg.clean.true.bpe.de \
$base_path/git/research/string-to-tree-nmt/data/news-de-en/test/newstest2016-deen.tok.penntrg.clean.true.bpe.en \
$base_path/git/dynet-seq2seq-attn/results/deen_ncv8_baseline/deen_ncv8_baseline

