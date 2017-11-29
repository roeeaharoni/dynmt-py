import os
import src.common as common

def main():
    # base_path = '/Users/roeeaharoni'
    base_path = '/home/nlp/aharonr6'
    dev_target = base_path + '/git/dynet-seq2seq-attn/results/test_numchar_eval_script.dev.predictions'
    postprocess_command = '{}/git/phrasing/src/nmt_scripts/nematus/postprocess-en.sh < {} > {}.postprocessed 2> /dev/null'.format(
        base_path,
        dev_target,
        dev_target)
    os.system(postprocess_command)
    postprocessed_path = dev_target + '.postprocessed'
    gold_path = base_path + '/git/research/string-to-tree-nmt/data/news-de-en/dev/newstest2015-deen.en'
    bleu = common.evaluate_bleu_from_files(gold_path, postprocessed_path)
    print bleu
    return

if __name__=='__main__':
    main()