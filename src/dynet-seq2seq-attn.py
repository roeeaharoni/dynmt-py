"""

Sequence to sequence learning with an attention mechanism implemented using dynet's python bindings.

Usage:
  dynet-seq2seq-attn.py [--dynet-mem MEM] [--dynet-gpu-ids IDS] [--input-dim=INPUT] [--hidden-dim=HIDDEN]
  [--epochs=EPOCHS] [--lstm-layers=LAYERS] [--optimization=OPTIMIZATION] [--reg=REGULARIZATION] [--batch-size=BATCH]
  [--beam-size=BEAM] [--learning=LEARNING] [--plot] [--override] [--eval] [--ensemble=ENSEMBLE]
  [--vocab-size=VOCAB] [--eval-after=EVALAFTER] TRAIN_INPUTS_PATH TRAIN_OUTPUTS_PATH DEV_INPUTS_PATH DEV_OUTPUTS_PATH TEST_INPUTS_PATH
  TEST_OUTPUTS_PATH RESULTS_PATH...

Arguments:
  TRAIN_INPUTS_PATH    train inputs path
  TRAIN_OUTPUTS_PATH   train outputs path
  DEV_INPUTS_PATH      development inputs path
  DEV_OUTPUTS_PATH     development outputs path
  TEST_INPUTS_PATH     test inputs path
  TEST_OUTPUTS_PATH    test outputs path
  RESULTS_PATH  results file path

Options:
  -h --help                     show this help message and exit
  --dynet-mem MEM               allocates MEM bytes for dynet
  --dynet-gpu-ids IDS           GPU ids to use
  --input-dim=INPUT             input embeddings dimension
  --hidden-dim=HIDDEN           LSTM hidden layer dimension
  --epochs=EPOCHS               amount of training epochs
  --layers=LAYERS               amount of layers in LSTM
  --optimization=OPTIMIZATION   chosen optimization method (ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA)
  --reg=REGULARIZATION          regularization parameter for optimization
  --learning=LEARNING           learning rate parameter for optimization
  --plot                        plot a learning curve while training each model
  --override                    override existing model with the same name, if exists
  --ensemble=ENSEMBLE           ensemble model paths separated by a comma
  --batch-size=BATCH            batch size
  --beam-size=BEAM              beam size in beam search
  --vocab-size=VOCAB            vocabulary size
  --eval-after=EVALAFTER        amount of train batches to wait before evaluation
"""

import numpy as np
import random
import prepare_data
import progressbar
import datetime
import time
import os
import common
import dynet as dn

import matplotlib
# to run on headless server
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from docopt import docopt
from collections import defaultdict

# default values
INPUT_DIM = 300
HIDDEN_DIM = 100
EPOCHS = 1
LAYERS = 2
MAX_PREDICTION_LEN = 50
OPTIMIZATION = 'ADADELTA'
EARLY_STOPPING = True
MAX_PATIENCE = 100
REGULARIZATION = 0.0
LEARNING_RATE = 0.0001  # 0.1
BEAM_WIDTH = 5
MAX_VOCAB_SIZE = 30000
BATCH_SIZE = 1
MAX_SEQ_LEN = 50
EVAL_AFTER = 1000
GRAD_CLIP = 5.0
EPOCH_EVAL = False

# consts
UNK = 'UNK'
BEGIN_SEQ = '<s>'
END_SEQ = '</s>'


# finish data preproc - DONE
# add support for large vocab/UNKs (limit vocab size) - DONE
# debug on cpu - DONE
# moses BLEU evaluation - DONE
# debug on gpu - DONE
# minibatch support - DONE
# sort inputs by length - DONE
# divide into batches of batch size - DONE
# compute loss for batch with attention - DONE
# debug batching with toy example (chars to letters?) - DONE
# add multi-checkpoint support + save/evaluate model by checkpoints (every X batches) - DONE
# TODO: debug batching on GPU - In progress


# evaluate model
# save model
# log
# plot
# all according to current checkpoint (checkpoint id is total batches)

# TODO: OOP refactoring?
# TODO: debug with non english output (reverse translation from en to heb will work)
# TODO: do word lookup once in the training stage and not in each epoch
# TODO: data preproc with better (moses?) scripts
# TODO: add beamsearch support
# TODO: BPE support
# TODO: ensembling support (by interpolating probabilities)
# TODO: find better value for max seq len in the literature (50 is standard in nmt/80 in moses clean)


def main(train_inputs_path, train_outputs_path, dev_inputs_path, dev_outputs_path, test_inputs_path, test_outputs_path,
         results_file_path, input_dim, hidden_dim, epochs, layers, optimization, regularization, learning_rate, plot,
         override, eval_only, ensemble, batch_size, beam_size, vocab_size, eval_after):
    hyper_params = {'INPUT_DIM': input_dim,
                    'HIDDEN_DIM': hidden_dim,
                    'EPOCHS': epochs,
                    'LAYERS': layers,
                    'MAX_PREDICTION_LEN': MAX_PREDICTION_LEN,
                    'OPTIMIZATION': optimization,
                    'PATIENCE': MAX_PATIENCE,
                    'REGULARIZATION': regularization,
                    'LEARNING_RATE': learning_rate,
                    'EVAL_AFTER': eval_after}

    # write model config file (.modelinfo)
    common.write_model_config_file(hyper_params, train_inputs_path, train_outputs_path, dev_inputs_path,
                                   dev_outputs_path, test_inputs_path, test_outputs_path, results_file_path)

    # debug prints
    print 'train input path = {}'.format(str(train_inputs_path))
    print 'train output path = {}'.format(str(train_outputs_path))
    print 'test inputs path = {}'.format(str(test_inputs_path))
    print 'test output path = {}\n'.format(str(test_outputs_path))
    for param in hyper_params:
        print param + '=' + str(hyper_params[param])

    # load train, dev and test data
    train_inputs, input_vocabulary, train_outputs, output_vocabulary = \
        prepare_data.load_parallel_data(train_inputs_path, train_outputs_path, vocab_size, MAX_SEQ_LEN)

    dev_inputs, dev_in_vocab, dev_outputs, dev_out_vocab  = \
        prepare_data.load_parallel_data(dev_inputs_path, dev_outputs_path, vocab_size, MAX_SEQ_LEN)

    test_inputs, test_in_vocab, test_outputs, test_out_vocab = \
        prepare_data.load_parallel_data(test_inputs_path, test_outputs_path, vocab_size, MAX_SEQ_LEN)

    # add unk symbols to vocabularies
    input_vocabulary.append(UNK)
    output_vocabulary.append(UNK)

    # add begin/end sequence symbols to vocabularies
    input_vocabulary.append(BEGIN_SEQ)
    input_vocabulary.append(END_SEQ)
    output_vocabulary.append(BEGIN_SEQ)
    output_vocabulary.append(END_SEQ)

    # symbol 2 int and int 2 symbol
    x2int = dict(zip(input_vocabulary, range(0, len(input_vocabulary))))
    y2int = dict(zip(output_vocabulary, range(0, len(output_vocabulary))))
    int2y = {index: x for x, index in y2int.items()}

    print 'input vocab size: {}'.format(len(x2int))
    print 'output vocab size: {}'.format(len(y2int))

    # try to load existing model
    model_file_name = '{}_bestmodel.txt'.format(results_file_path)
    if os.path.isfile(model_file_name) and not override:
        print 'loading existing model from {}'.format(model_file_name)
        model, params = load_best_model(input_vocabulary, output_vocabulary, results_file_path, input_dim, hidden_dim,
                                        layers)
        print 'loaded existing model successfully'
    else:
        print 'could not find existing model or explicit override was requested. started training from scratch...'
        model, params = build_model(input_vocabulary, output_vocabulary, input_dim, hidden_dim, layers)

    # train the model
    if not eval_only:
        model, params, last_epoch, best_epoch = train_model(model, params, train_inputs, train_outputs, dev_inputs,
                                                            dev_outputs, x2int, y2int, int2y, epochs, optimization,
                                                            results_file_path, plot, batch_size, eval_after)

        print 'last epoch is {}'.format(last_epoch)
        print 'best epoch is {}'.format(best_epoch)
        print 'finished training'
    else:
        print 'skipped training, evaluating on test set...'

    # evaluate using an ensemble
    if ensemble:
        # predict test set using ensemble majority
        predicted_sequences = predict_with_ensemble_majority(input_vocabulary, output_vocabulary, x2int, y2int,
                                                             int2y, ensemble, hidden_dim, input_dim, layers,
                                                             test_inputs, test_outputs)
    else:
        # predict test set using a single model
        predicted_sequences = predict_multiple_sequences(params, x2int, y2int, int2y, test_inputs)
    if len(predicted_sequences) > 0:

        # evaluate the test predictions
        amount, accuracy = evaluate_model(predicted_sequences, test_inputs, test_outputs, print_results=False)
        print 'test bleu: {}% '.format(accuracy)

        final_results = []
        for i in xrange(len(test_outputs)):
            index = ' '.join(test_inputs[i])
            final_output = ' '.join(predicted_sequences[index])
            final_results.append(final_output)

        # write output files
        predictions_path = common.write_results_files(results_file_path, final_results)

        # bleu = common.evaluate_bleu_from_files(test_outputs_path, predictions_path)

    return


def predict_with_ensemble_majority(input_vocabulary, output_vocabulary, x2int, y2int, int2y, ensemble,
                                   hidden_dim, input_dim, layers, test_inputs, test_outputs):
    ensemble_model_names = ensemble.split(',')
    print 'ensemble paths:\n {}'.format('\n'.join(ensemble_model_names))
    ensemble_models = []

    # load ensemble models
    for ens in ensemble_model_names:
        model, params = load_best_model(input_vocabulary, output_vocabulary, ens, input_dim, hidden_dim, layers)
        ensemble_models.append((model, params))

    # predict the entire test set with each model in the ensemble
    ensemble_predictions = []
    for em in ensemble_models:
        model, params = em
        predicted_sequences = predict_multiple_sequences(params, x2int, y2int, int2y, test_inputs)
        ensemble_predictions.append(predicted_sequences)

    # perform voting for each test input
    majority_predicted_sequences = {}
    string_to_template = {}
    test_data = zip(test_inputs, test_outputs)
    for i, (input_seq, output_seq) in enumerate(test_data):
        joint_index = input_seq
        prediction_counter = defaultdict(int)
        for ens in ensemble_predictions:
            prediction_str = ''.join(ens[joint_index])
            prediction_counter[prediction_str] += 1
            string_to_template[prediction_str] = ens[joint_index]
            print 'template: {} prediction: {}'.format(''.join([e.encode('utf-8') for e in ens[joint_index]]),
                                                       prediction_str.encode('utf-8'))

        # return the most predicted output
        majority_prediction_string = max(prediction_counter, key=prediction_counter.get)
        print 'chosen:{} with {} votes\n'.format(majority_prediction_string.encode('utf-8'),
                                                 prediction_counter[majority_prediction_string])
        majority_predicted_sequences[joint_index] = string_to_template[majority_prediction_string]

    return majority_predicted_sequences


def save_best_model(model, results_file_path):
    tmp_model_path = results_file_path + '_bestmodel.txt'
    print 'saving to ' + tmp_model_path
    model.save(tmp_model_path)
    print 'saved to {0}'.format(tmp_model_path)


def load_best_model(input_vocabulary, output_vocabulary, results_file_path, input_dim, hidden_dim, layers):
    tmp_model_path = results_file_path + '_bestmodel.txt'
    model, params = build_model(input_vocabulary, output_vocabulary, input_dim, hidden_dim, layers)

    print 'trying to load model from: {}'.format(tmp_model_path)
    model.load(tmp_model_path)
    return model, params


def build_model(input_vocabulary, output_vocabulary, input_dim, hidden_dim, layers):
    print 'creating model...'

    model = dn.Model()

    params = {}

    # input embeddings
    params['input_lookup'] = model.add_lookup_parameters((len(input_vocabulary), input_dim))

    # init vector for input feeding
    params['init_lookup'] = model.add_lookup_parameters((1, 3 * hidden_dim))

    # output embeddings
    params['output_lookup'] = model.add_lookup_parameters((len(output_vocabulary), input_dim))

    # used in softmax output
    params['readout'] = model.add_parameters((len(input_vocabulary), 3 * hidden_dim))
    params['bias'] = model.add_parameters(len(input_vocabulary))

    # rnn's
    params['encoder_frnn'] = dn.LSTMBuilder(layers, input_dim, hidden_dim, model)
    params['encoder_rrnn'] = dn.LSTMBuilder(layers, input_dim, hidden_dim, model)

    # attention MLPs - Luong-style with extra v_a from Bahdanau

    # concatenation layer for h (hidden dim), c (2 * hidden_dim)
    params['w_c'] = model.add_parameters((3 * hidden_dim, 3 * hidden_dim))

    # concatenation layer for h_input (2*hidden_dim), h_output (hidden_dim)
    params['w_a'] = model.add_parameters((hidden_dim, hidden_dim))

    # concatenation layer for h (hidden dim), c (2 * hidden_dim)
    params['u_a'] = model.add_parameters((hidden_dim, 2 * hidden_dim))

    # concatenation layer for h_input (2*hidden_dim), h_output (hidden_dim)
    params['v_a'] = model.add_parameters((1, hidden_dim))

    # 1 * HIDDEN_DIM - gets only the feedback input
    params['decoder_rnn'] = dn.LSTMBuilder(layers, 3 * hidden_dim + input_dim, hidden_dim, model)

    print 'finished creating model'

    return model, params


def train_model(model, params, train_inputs, train_outputs, dev_inputs, dev_outputs, x2int, y2int, int2y, epochs,
                optimization, results_file_path, plot, batch_size, eval_after):
    print 'training...'

    np.random.seed(17)
    random.seed(17)

    # sort training sentences by length in descending order
    train_data = zip(train_inputs, train_outputs)
    train_data.sort(key=lambda t: - len(t[0]))
    train_order = [x * batch_size for x in range(len(train_data) / batch_size + 1)]

    # sort dev sentences by length in descending order
    dev_data = zip(dev_inputs, dev_outputs)
    dev_data.sort(key=lambda t: - len(t[0]))
    dev_order = [x * batch_size for x in range(len(dev_data) / batch_size + 1)]

    if optimization == 'ADAM':
        trainer = dn.AdamTrainer(model)  # lam=REGULARIZATION, alpha=LEARNING_RATE, beta_1=0.9, beta_2=0.999, eps=1e-8)
    elif optimization == 'MOMENTUM':
        trainer = dn.MomentumSGDTrainer(model)
    elif optimization == 'SGD':
        trainer = dn.SimpleSGDTrainer(model)
    elif optimization == 'ADAGRAD':
        trainer = dn.AdagradTrainer(model)
    elif optimization == 'ADADELTA':
        trainer = dn.AdadeltaTrainer(model)
    else:
        trainer = dn.SimpleSGDTrainer(model)

    trainer.set_clip_threshold(GRAD_CLIP)
    best_avg_train_loss = 99999999
    total_loss = 0
    best_dev_loss = 99999999
    best_dev_bleu = -1
    best_train_bleu = -1
    best_dev_epoch = 0
    best_train_epoch = 0
    patience = 0
    train_len = len(train_outputs)
    dev_len = len(dev_inputs)
    train_bleu = -1
    epochs_x = []
    train_loss_y = []
    dev_loss_y = []
    train_bleu_y = []
    dev_bleu_y = []
    avg_train_loss = -1
    total_batches = 0
    train_loss_patience = 0
    train_loss_patience_threshold = 50
    e = 0
    log_path = results_file_path + '_log.txt'
    # train_sanity_set_size = 100

    # progress bar init
    widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
    train_progress_bar = progressbar.ProgressBar(widgets=widgets, maxval=epochs).start()

    for e in xrange(epochs):

        # shuffle the batch start indices in each epoch
        random.shuffle(train_order)

        # go through batches
        for i, batch_start_index in enumerate(train_order, start=1):

            total_batches += 1

            # get batch examples
            batch_inputs = [x[0] for x in train_data[batch_start_index:batch_start_index + batch_size]]
            batch_outputs = [x[1] for x in train_data[batch_start_index:batch_start_index + batch_size]]

            # skip empty batch
            if len(batch_inputs) == 0 or len(batch_inputs[0]) == 0:
                continue

            # compute batch loss
            loss = compute_batch_loss(params, batch_inputs, batch_outputs, x2int, y2int)

            # update parameters
            total_loss += loss.scalar_value()
            loss.backward()
            trainer.update()

            # avg loss per sample
            avg_train_loss = total_loss / float(i * batch_size + e * train_len)

            if avg_train_loss < best_avg_train_loss:
                best_avg_train_loss = avg_train_loss
                train_loss_patience = 0
            else:
                train_loss_patience += 1
                if train_loss_patience > train_loss_patience_threshold:
                    print 'train loss patience exceeded: {}'.format(train_loss_patience)
                    return model, params, e, best_train_epoch

            if i % 10 == 0 and i > 0:

                print 'went through {} train batches out of {} ({} examples out of {}, {} batches in total) avg train loss: {}'.format(i, len(train_order),
                                                                                                i * batch_size,
                                                                                                train_len,
                                                                                                total_batches,
                                                                                                avg_train_loss)
            # checkpoint
            if total_batches % eval_after == 0:

                print 'starting checkpoint evaluation'
                dev_bleu, dev_loss = checkpoint_eval(params, batch_size, dev_data, dev_inputs, dev_len, dev_order,
                                                     dev_outputs, int2y, x2int, y2int)

                log_to_file(log_path, e, total_batches, avg_train_loss, dev_loss, train_bleu, dev_bleu)

                if dev_bleu >= best_dev_bleu:
                    best_dev_bleu = dev_bleu
                    best_dev_epoch = e

                    # save best model to disk
                    save_best_model(model, results_file_path)
                    print 'saved new best model'
                    patience = 0
                else:
                    patience += 1

                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss

                print 'epoch: {0} train loss: {1:.4f} dev loss: {2:.4f} dev bleu: {3:.4f} train bleu = {4:.4f} \
            best dev bleu {5:.4f} (epoch {8}) best train bleu: {6:.4f} (epoch {9}) patience = {7}'.format(
                    e,
                    avg_train_loss,
                    dev_loss,
                    dev_bleu,
                    train_bleu,
                    best_dev_bleu,
                    best_train_bleu,
                    patience,
                    best_dev_epoch,
                    best_train_epoch)

                if patience == MAX_PATIENCE:
                    print 'out of patience after {0} checkpoints'.format(str(e))
                    train_progress_bar.finish()
                    if plot:
                        plt.cla()
                    print 'checkpoint patience exceeded'
                    return model, params, e, best_train_epoch

        # epoch evaluation
        if EPOCH_EVAL:
            print 'starting epoch evaluation'
            dev_bleu, dev_loss = checkpoint_eval(params, batch_size, dev_data, dev_inputs, dev_len, dev_order, dev_outputs,
                                                 int2y, x2int, y2int)

            log_to_file(log_path, e, total_batches, avg_train_loss, dev_loss, train_bleu, dev_bleu)

            if dev_bleu >= best_dev_bleu:
                best_dev_bleu = dev_bleu
                best_dev_epoch = e

                # save best model to disk
                save_best_model(model, results_file_path)
                print 'saved new best model'
                patience = 0
            else:
                patience += 1

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss

            print 'epoch: {0} train loss: {1:.4f} dev loss: {2:.4f} dev bleu: {3:.4f} train bleu = {4:.4f} \
best dev bleu {5:.4f} (epoch {8}) best train bleu: {6:.4f} (epoch {9}) patience = {7}'.format(
            e,
            avg_train_loss,
            dev_loss,
            dev_bleu,
            train_bleu,
            best_dev_bleu,
            best_train_bleu,
            patience,
            best_dev_epoch,
            best_train_epoch)

            if patience == MAX_PATIENCE:
                print 'out of patience after {0} checkpoints'.format(str(e))
                train_progress_bar.finish()
                if plot:
                    plt.cla()
                print 'epoch patience exceeded'
                return model, params, e, best_train_epoch

            # update parameters for plotting before ending epoch loop
            epochs_x.append(e)
            train_bleu_y.append(train_bleu)
            train_loss_y.append(avg_train_loss)
            dev_loss_y.append(dev_loss)
            dev_bleu_y.append(dev_bleu)

        # finished epoch
        train_progress_bar.update(e)

        if plot:
            with plt.style.context('fivethirtyeight'):
                p1, = plt.plot(epochs_x, dev_loss_y, label='dev loss')
                p2, = plt.plot(epochs_x, train_loss_y, label='train loss')
                p3, = plt.plot(epochs_x, dev_bleu_y, label='dev acc.')
                p4, = plt.plot(epochs_x, train_bleu_y, label='train acc.')
                plt.legend(loc='upper left', handles=[p1, p2, p3, p4])
            plt.savefig(results_file_path + 'plot.png')

    train_progress_bar.finish()
    if plot:
        plt.cla()
    print 'finished training. average loss: {} best epoch on dev: {} best epoch on train: {}'.format(
        str(avg_train_loss),
        best_dev_epoch,
        best_train_epoch)

    return model, params, e, best_train_epoch


def checkpoint_eval(params, batch_size, dev_data, dev_inputs, dev_len, dev_order, dev_outputs, int2y, x2int, y2int):

    # TODO: could be more efficient - now "encoding" (lookup) the dev set twice (for predictions and loss)
    print 'predicting on dev...'
    # get dev predictions
    dev_predictions = predict_multiple_sequences(params, x2int, y2int, int2y, dev_inputs)
    print 'calculating dev bleu...'
    # get dev accuracy
    dev_bleu = evaluate_model(dev_predictions, dev_inputs, dev_outputs, print_results=True)[1]

    # get dev loss
    print 'computing dev loss...'
    total_dev_loss = 0
    for i, batch_start_index in enumerate(dev_order, start=1):

        # get dev batches
        batch_inputs = [x[0] for x in dev_data[batch_start_index:batch_start_index + batch_size]]
        batch_outputs = [x[1] for x in dev_data[batch_start_index:batch_start_index + batch_size]]

        # skip empty batches
        if len(batch_inputs) == 0 or len(batch_inputs[0]) == 0:
            continue

        loss = compute_batch_loss(params, batch_inputs, batch_outputs, x2int, y2int)

        total_dev_loss += loss.scalar_value()

        if i % 10 == 0 and i > 0:
            print 'went through {} dev batches out of {} ({} examples out of {})'.format(i, len(dev_order),
                                                                                         i * batch_size,
                                                                                         dev_len)

    avg_dev_loss = total_dev_loss / float(len(dev_inputs))

    return dev_bleu, avg_dev_loss


def log_to_file(file_name, epoch, total_updates, train_loss, dev_loss, train_accuracy, dev_accuracy):

    # if first log, add headers
    if epoch == 0:
        log_to_file(file_name, 'epoch', 'update', 'avg_train_loss', 'avg_dev_loss', 'train_accuracy', 'dev_accuracy')

    with open(file_name, "a") as logfile:
        logfile.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(epoch, total_updates, train_loss, dev_loss, train_accuracy,
                                                        dev_accuracy))


def compute_batch_loss(params, input_batch_seqs, output_batch_seqs, x2int, y2int):
    # renew computation graph per batch
    dn.renew_cg()

    # read model parameters
    readout = dn.parameter(params['readout'])
    bias = dn.parameter(params['bias'])
    w_c = dn.parameter(params['w_c'])
    u_a = dn.parameter(params['u_a'])
    v_a = dn.parameter(params['v_a'])
    w_a = dn.parameter(params['w_a'])

    batch_size = len(input_batch_seqs)

    # print 'INPUTS:'
    # for c, inputz in enumerate(input_batch_seqs):
    #     print u' '.join(inputz)
    #     print u' '.join(output_batch_seqs[c])
    #     print '\n'

    # encode batch with bilstm encoder: each element represents one step in time, and is a matrix of 2*h x batch size
    # for example, for sentence length of 12, blstm_outputs wil be: 12 x 2 x 100 x 16
    blstm_outputs = batch_bilstm_encode(x2int, params['input_lookup'], params['encoder_frnn'], params['encoder_rrnn'],
                                        input_batch_seqs)

    # initialize the decoder rnn
    s_0 = params['decoder_rnn'].initial_state()
    s = s_0

    # get output word ids for each step of the decoder
    output_word_ids = get_batch_word_ids(output_batch_seqs, y2int)

    # end symbol ids
    end_seq_vec_ids = [y2int[END_SEQ]] * batch_size

    # add end symbol to output seq
    output_word_ids = output_word_ids + [end_seq_vec_ids]

    # initial "input feeding" vectors to feed decoder - 3*h
    init_input_feeding = dn.lookup_batch(params['init_lookup'], [0] * batch_size)

    # initial feedback embeddings for the decoder, use begin seq symbol embedding
    init_feedback = dn.lookup_batch(params['output_lookup'], [y2int[BEGIN_SEQ]] * batch_size)

    decoder_init = dn.concatenate([init_feedback, init_input_feeding])

    # init decoder
    s = s.add_input(decoder_init)

    # loss per timestep
    losses = []

    # run the decoder through the output sequences and aggregate loss
    for i, step_word_ids in enumerate(output_word_ids):

        # returns h x batch size matrix
        decoder_rnn_output = s.output()

        # compute attention context vector for each sequence in the batch (returns 2h x batch size matrix)
        attention_output_vector, alphas = attend(blstm_outputs, decoder_rnn_output, w_c, v_a, w_a, u_a)

        # compute output scores (returns vocab_size x batch size matrix)
        h = readout * attention_output_vector + bias

        # get batch loss for this timestep
        batch_loss = dn.pickneglogsoftmax_batch(h, step_word_ids)
        losses.append(batch_loss)

        # TODO: enable input feeding approach - input h to the decoder
        # prepare for the next iteration - "feedback"
        feedback_embeddings = dn.lookup_batch(params['output_lookup'], step_word_ids)
        decoder_input = dn.concatenate([feedback_embeddings, attention_output_vector])
        s = s.add_input(decoder_input)

    # sum the loss over the time steps and batch
    total_batch_loss = dn.sum_batches(dn.esum(losses))

    return total_batch_loss


# get list of word ids per each timestep in the batch
def get_batch_word_ids(batch_seqs, x2int):
    output_word_ids = []
    max_seq_len = 0
    for seq in batch_seqs:
        if len(seq) > max_seq_len:
            max_seq_len = len(seq)
    # max_seq_len = len(batch_seqs[0])
    for i in range(max_seq_len):
        output_word_ids.append([])
        for seq in batch_seqs:
            if i > len(seq) - 1:
                output_word_ids[i].append(x2int[END_SEQ])
            else:
                if seq[i] in x2int:
                    output_word_ids[i].append(x2int[seq[i]])
                else:
                    output_word_ids[i].append(x2int[UNK])

    return output_word_ids


# bilstm encode batch, each element in the result is a matrix of 2*h x batch size elements
def batch_bilstm_encode(x2int, input_lookup, encoder_frnn, encoder_rrnn, input_seq_batch):
    f_outputs = []
    r_outputs = []
    final_outputs = []

    # get the word ids for each step
    word_ids = get_batch_word_ids(input_seq_batch, x2int)

    # initialize with BEGIN_SEQ symbol
    init_ids = [x2int[BEGIN_SEQ]] * len(input_seq_batch)

    # finish with END_SEQ
    end_ids = [x2int[END_SEQ]] * len(input_seq_batch)

    # pad with begin seq / end seq
    word_ids = [init_ids] + word_ids + [end_ids]

    # init rnns
    f_state = encoder_frnn.initial_state()
    r_state = encoder_rrnn.initial_state()

    # +2 for begin/end symbols
    max_seq_len = len(input_seq_batch[0]) + 2

    # iterate in both directions
    for i in xrange(max_seq_len):
        f_state = f_state.add_input(dn.lookup_batch(input_lookup, word_ids[i]))
        f_outputs.append(f_state.output())

        r_state = r_state.add_input(dn.lookup_batch(input_lookup, word_ids[max_seq_len - i - 1]))
        r_outputs.append(r_state.output())

    # concatenate forward and backward representations for each step
    for i in xrange(max_seq_len):
        concatenated = dn.concatenate([f_outputs[i], r_outputs[max_seq_len - i - 1]])
        final_outputs.append(concatenated)

    return final_outputs


def predict_output_sequence(params, input_seq, x2int, y2int, int2y):
    dn.renew_cg()
    alphas_mtx = []

    if len(input_seq) == 0:
        return []

    # read model parameters
    readout = dn.parameter(params['readout'])
    bias = dn.parameter(params['bias'])
    w_c = dn.parameter(params['w_c'])
    u_a = dn.parameter(params['u_a'])
    v_a = dn.parameter(params['v_a'])
    w_a = dn.parameter(params['w_a'])

    # encode input sequence
    blstm_outputs = batch_bilstm_encode(x2int, params['input_lookup'], params['encoder_frnn'], params['encoder_rrnn'],
                                        [input_seq])

    # initialize the decoder rnn
    s_0 = params['decoder_rnn'].initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD concatenated with special padding vector
    prev_output_vec = dn.concatenate([params['output_lookup'][y2int[BEGIN_SEQ]], params['init_lookup'][0]])
    predicted_sequence = []
    i = 0

    # run the decoder through the sequence and predict output symbols
    while i < MAX_PREDICTION_LEN:

        # get current h of the decoder
        s = s.add_input(prev_output_vec)
        decoder_rnn_output = s.output()

        # perform attention step
        attention_output_vector, alphas = attend(blstm_outputs, decoder_rnn_output, w_c, v_a, w_a, u_a)

        if plot_param:
            val = alphas.vec_value()
            alphas_mtx.append(val)

        # compute output probabilities
        # print 'computing readout layer...'
        h = readout * attention_output_vector + bias

        # find best candidate output
        probs = dn.softmax(h)
        next_element_index = np.argmax(probs.npvalue())
        predicted_sequence.append(int2y[next_element_index])

        # check if reached end of word
        if predicted_sequence[-1] == END_SEQ:
            break

        # prepare for the next iteration - "feedback"
        prev_output_vec = dn.concatenate([params['output_lookup'][next_element_index], attention_output_vector])
        i += 1

    # remove the end seq symbol
    return predicted_sequence[0:-1], alphas_mtx


# Luong et. al 2015 attention mechanism:
def attend(blstm_outputs, h_t, w_c, v_a, w_a, u_a):
    # blstm_outputs dimension is: seq len x 2*h x batch size, h_t dimension is h x batch size

    # iterate through input states to compute attention scores
    scores = [v_a * dn.tanh(w_a * h_t + u_a * h_input) for h_input in blstm_outputs]

    # normalize scores using softmax - TODO: check if correct
    alphas = dn.softmax(dn.concatenate(scores))

    # compute context vector with weighted sum for each seq in batch
    c = dn.esum([h_input * dn.pick(alphas, j) for j, h_input in enumerate(blstm_outputs)])

    # compute output vector using current decoder state and context vector
    h_output = dn.tanh(w_c * dn.concatenate([h_t, c]))

    return h_output, alphas


def predict_multiple_sequences(params, x2int, y2int, int2y, inputs):
    print 'predicting...'
    predictions = {}
    data_len = len(inputs)
    for i, input_seq in enumerate(inputs):
        if i==0 and plot_param:
            plot_attn_weights(params, input_seq, x2int, y2int, int2y,
                              filename='{}_{}.png'.format(
                                  results_file_path_param, int(time.time())))
        predicted_seq, alphas_mtx = predict_output_sequence(params, input_seq, x2int, y2int, int2y)
        if i % 100 == 0 and i > 0:
            print 'predicted {} examples out of {}'.format(i, data_len)

        index = ' '.join(input_seq)
        predictions[index] = predicted_seq

    return predictions


def evaluate_model(predicted_sequences, inputs, outputs, print_results=False):
    if print_results:
        print 'evaluating model...'

    test_data = zip(inputs, outputs)
    eval_predictions = []
    eval_golds = []

    # go through the parallel sequence pairs
    for i, (input_seq, output_seq) in enumerate(test_data):
        index = ' '.join(input_seq)
        predicted_output = ' '.join(predicted_sequences[index])

        # create strings from sequences for debug prints and evaluation
        enc_in = ' '.join(input_seq).encode('utf8')
        enc_gold = ' '.join(output_seq).encode('utf8')
        enc_out = predicted_output.encode('utf8')

        if print_results:
            print 'input: {}'.format(enc_in)
            print 'gold output: {}'.format(enc_gold)
            print 'prediction: {}\n'.format(enc_out)

        eval_predictions.append(enc_out.decode('utf8'))
        eval_golds.append(enc_gold.decode('utf8'))

    bleu = common.evaluate_bleu(eval_golds, eval_predictions)

    if print_results:
        print 'finished evaluating model. bleu: {}\n\n'.format(bleu)

    return len(predicted_sequences), bleu


def plot_attn_weights(params, input_seq, x2int, y2int, int2y, filename=None):
    # predict
    output_seq, alphas_mtx = predict_output_sequence(params, input_seq, x2int, y2int, int2y)
    fig, ax = plt.subplots()

    image = np.array(alphas_mtx)
    ax.imshow(image, cmap=plt.cm.Blues, interpolation='nearest')

    # fix x axis ticks density - input len (+2)
    ax.xaxis.set_ticks(np.arange(0, len(alphas_mtx[0]), 1))

    # fix y axis ticks density - output len (+1)
    ax.yaxis.set_ticks(np.arange(0, len(alphas_mtx), 1))

    # set tick labels to meaningful symbols
    ax.set_xticklabels([u'begin'] + list(input_seq) + [u'end'])
    ax.set_yticklabels(list(output_seq) + [u'end'])

    # set title
    input_word = u''.join(input_seq)
    output_word = u''.join(output_seq)
    ax.set_title(u'attention-based alignment: {} -> {}'.format(input_word, output_word[0:-1]))
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    arguments = docopt(__doc__)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

    # default values
    if arguments['TRAIN_INPUTS_PATH']:
        train_inputs_path_param = arguments['TRAIN_INPUTS_PATH']
    else:
        train_inputs_path_param = '../data/he_en_clean/train.tags.he-en.en.clean'

    if arguments['TRAIN_OUTPUTS_PATH']:
        train_outputs_path_param = arguments['TRAIN_OUTPUTS_PATH']
    else:
        train_outputs_path_param = '../data/he_en_clean/train.tags.he-en.he.clean'

    if arguments['DEV_INPUTS_PATH']:
        dev_inputs_path_param = arguments['DEV_INPUTS_PATH']
    else:
        dev_inputs_path_param = '../data/he_en_clean/IWSLT14.TED.dev2010.he-en.en.xml.clean'

    if arguments['DEV_OUTPUTS_PATH']:
        dev_outputs_path_param = arguments['DEV_OUTPUTS_PATH']
    else:
        dev_outputs_path_param = '../data/he_en_clean/IWSLT14.TED.dev2010.he-en.en.xml.clean'

    if arguments['TEST_INPUTS_PATH']:
        test_inputs_path_param = arguments['TEST_INPUTS_PATH']
    else:
        test_inputs_path_param = '../data/he_en_clean/IWSLT14.TED.tst2010.he-en.en.xml.clean'

    if arguments['TEST_OUTPUTS_PATH']:
        test_outputs_path_param = arguments['TEST_OUTPUTS_PATH']
    else:
        test_outputs_path_param = '../data/he_en_clean/IWSLT14.TED.tst2010.he-en.he.xml.clean'

    if arguments['RESULTS_PATH']:
        results_file_path_param = arguments['RESULTS_PATH'][0]
    else:
        results_file_path_param = '../results/results_' + st + '.txt'

    if arguments['--input-dim']:
        input_dim_param = int(arguments['--input-dim'])
    else:
        input_dim_param = INPUT_DIM

    if arguments['--hidden-dim']:
        hidden_dim_param = int(arguments['--hidden-dim'])
    else:
        hidden_dim_param = HIDDEN_DIM

    if arguments['--epochs']:
        epochs_param = int(arguments['--epochs'])
    else:
        epochs_param = EPOCHS

    if arguments['--lstm-layers']:
        layers_param = int(arguments['--lstm-layers'])
    else:
        layers_param = LAYERS

    if arguments['--optimization']:
        optimization_param = arguments['--optimization']
    else:
        optimization_param = OPTIMIZATION

    if arguments['--reg']:
        regularization_param = float(arguments['--reg'])
    else:
        regularization_param = REGULARIZATION

    if arguments['--learning']:
        learning_rate_param = float(arguments['--learning'])
    else:
        learning_rate_param = LEARNING_RATE

    if arguments['--plot']:
        plot_param = True
    else:
        plot_param = False

    if arguments['--override']:
        override_param = True
    else:
        override_param = False

    if arguments['--eval']:
        eval_param = True
    else:
        eval_param = False

    if arguments['--ensemble']:
        ensemble_param = arguments['--ensemble']
    else:
        ensemble_param = False

    if arguments['--batch-size']:
        batch_param = int(arguments['--batch-size'])
    else:
        batch_param = BATCH_SIZE

    if arguments['--beam-size']:
        beam_param = arguments['--beam-size']
    else:
        beam_param = 1

    if arguments['--vocab-size']:
        # noinspection PyUnresolvedReferences
        vocab_param = int(arguments['--vocab-size'])
    else:
        # noinspection PyUnresolvedReferences
        vocab_param = MAX_VOCAB_SIZE

    if arguments['--eval-after']:
        eval_after_param = int(arguments['--eval-after'])
    else:
        eval_after_param = EVAL_AFTER

    print arguments

    main(train_inputs_path_param, train_outputs_path_param, dev_inputs_path_param, dev_outputs_path_param,
         test_inputs_path_param, test_outputs_path_param, results_file_path_param, input_dim_param, hidden_dim_param,
         epochs_param, layers_param, optimization_param, regularization_param, learning_rate_param, plot_param,
         override_param, eval_param, ensemble_param, batch_param, beam_param, vocab_param, eval_after_param)
