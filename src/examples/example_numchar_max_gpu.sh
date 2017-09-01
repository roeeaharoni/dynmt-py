#!/usr/bin/env bash
#install_name_tool -change libboost_program_options.dylib /Users/roeeaharoni/boost_1_55_0/stage/lib/libboost_program_options.dylib /Users/roeeaharoni/git/dynet/build/dynet/libdynet.dylib
#install_name_tool -change libboost_serialization.dylib /Users/roeeaharoni/boost_1_55_0/stage/lib/libboost_serialization.dylib /Users/roeeaharoni/git/dynet/build/dynet/libdynet.dylib
base_path=/home/nlp/aharonr6
#base_path=/Users/roeeaharoni

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
python ../dynmt.py \
--dynet-mem 1000 \
--dynet-devices GPU:0 \
--input-dim=100 \
--hidden-dim=100 \
--epochs=100 \
--lstm-layers=1 \
--optimization=ADADELTA \
--batch-size=1 \
--beam-size=1 \
--plot \
--max \
/Users/roeeaharoni/git/dynet-seq2seq-attn/data/input.txt \
/Users/roeeaharoni/git/dynet-seq2seq-attn/data/output.txt \
/Users/roeeaharoni/git/dynet-seq2seq-attn/data/input.txt \
/Users/roeeaharoni/git/dynet-seq2seq-attn/data/output.txt \
/Users/roeeaharoni/git/dynet-seq2seq-attn/data/input.txt \
/Users/roeeaharoni/git/dynet-seq2seq-attn/data/output.txt \
/Users/roeeaharoni/git/dynet-seq2seq-attn/results/test_numchar_max_gpu

# test files
#$base_path/git/research/nmt/data/news-de-en/test/newstest2016-deen.tok.penntrg.clean.true.bpe.de \
#$base_path/git/research/nmt/data/news-de-en/test/newstest2016-deen.tok.penntrg.clean.true.bpe.en \

# large train files
#$base_path/git/research/nmt/data/news-de-en-50/train/news-commentary-v8.de-en.tok.penntrg.clean.true.bpe.de \
#$base_path/git/research/nmt/data/news-de-en-50/train/news-commentary-v8.de-en.tok.penntrg.clean.true.bpe.en \