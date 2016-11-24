"""Sequence to Sequence with Attention using dynet + minibatch support

Usage:
  dynet-seq2seq-attn.py [--dynet-mem MEM] [--dynet-gpu-ids IDS] [--input-dim=INPUT] [--hidden-dim=HIDDEN] [--epochs=EPOCHS]
  [--lstm-layers=LAYERS] [--optimization=OPTIMIZATION] [--reg=REGULARIZATION] [--batch-size=BATCH] [--beam-size=BEAM]
  [--learning=LEARNING] [--plot] [--override] [--eval] [--ensemble=ENSEMBLE] [--gpu] TRAIN_INPUTS_PATH TRAIN_OUTPUTS_PATH
  DEV_INPUTS_PATH DEV_OUTPUTS_PATH TEST_INPUTS_PATH TEST_OUTPUTS_PATH RESULTS_PATH...

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
  --gpu                         enable gpu support
"""

import numpy as np
import random
import prepare_data
import progressbar
import datetime
import time
import os
import common



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

# consts
UNK = 'UNK'
BEGIN_SEQ = '<s>'
END_SEQ = '</s>'


# finish data preproc - DONE
# add support for large vocab/UNKs (limit vocab size) - DONE
# debug on cpu - DONE
# moses BLEU evaluation - DONE
# debug on gpu - DONE

# TODO: minibatch support
# TODO: beamsearch support
# TODO: BPE support
# TODO: ensembling support (by interpolating probabilties)
# TODO: multi-checkpoint support
# TODO: data preproc with moses scripts?


# noinspection PyPep8
def main(train_inputs_path, train_outputs_path, dev_inputs_path, dev_outputs_path, test_inputs_path, test_outputs_path,
         results_file_path, input_dim, hidden_dim, epochs, layers, optimization, regularization, learning_rate, plot,
         override, eval_only, ensemble, batch, beam):
    hyper_params = {'INPUT_DIM': input_dim, 'HIDDEN_DIM': hidden_dim, 'EPOCHS': epochs, 'LAYERS': layers,
                    'MAX_PREDICTION_LEN': MAX_PREDICTION_LEN, 'OPTIMIZATION': optimization, 'PATIENCE': MAX_PATIENCE,
                    'REGULARIZATION': regularization, 'LEARNING_RATE': learning_rate}

    print 'train input path = {}'.format(str(train_inputs_path))
    print 'train output path = {}'.format(str(train_outputs_path))
    print 'test inputs path = {}'.format(str(test_inputs_path))
    print 'test inputs path = {}\n'.format(str(test_outputs_path))
    for param in hyper_params:
        print param + '=' + str(hyper_params[param])

    # load train, dev and test data
    train_inputs, input_vocabulary = prepare_data.load_data(train_inputs_path, MAX_VOCAB_SIZE)
    train_outputs, output_vocabulary = prepare_data.load_data(train_outputs_path, MAX_VOCAB_SIZE)

    dev_inputs, dev_in_vocab = prepare_data.load_data(dev_inputs_path, MAX_VOCAB_SIZE)
    dev_outputs, dev_out_vocab = prepare_data.load_data(dev_outputs_path, MAX_VOCAB_SIZE)

    test_inputs, test_in_vocab = prepare_data.load_data(test_inputs_path, MAX_VOCAB_SIZE)
    test_outputs, test_out_vocab = prepare_data.load_data(test_outputs_path, MAX_VOCAB_SIZE)  # REMOVE

    # TODO: remove
    # train_inputs = train_inputs[:100]
    # train_outputs = train_outputs[:100]
    #
    # dev_inputs = dev_inputs[:100]
    # dev_outputs = dev_outputs[:100]
    #
    # test_inputs = test_inputs[:100]
    # test_outputs = test_outputs[:100]

    # add unk symbols to vocabularies
    input_vocabulary.append(UNK)
    output_vocabulary.append(UNK)

    # add begin/end sequence symbols to vocabularies
    input_vocabulary.append(BEGIN_SEQ)
    input_vocabulary.append(END_SEQ)
    output_vocabulary.append(BEGIN_SEQ)
    output_vocabulary.append(END_SEQ)

    # element 2 int and int 2 element
    x2int = dict(zip(input_vocabulary, range(0, len(input_vocabulary))))

    y2int = dict(zip(output_vocabulary, range(0, len(output_vocabulary))))
    int2y = {index: x for x, index in y2int.items()}

    # try to load existing model
    model_file_name = '{}_bestmodel.txt'.format(results_file_path)
    if os.path.isfile(model_file_name) and not override:
        print '\nloading existing model from {}'.format(model_file_name)
        model, input_lookup, output_lookup, encoder_frnn, encoder_rrnn, decoder_rnn, readout, bias, w_c, w_a, u_a, v_a \
            = load_best_model(input_vocabulary, output_vocabulary, results_file_path, input_dim, hidden_dim, layers)
        print 'loaded existing model successfully'
    else:
        print 'could not find existing model or explicit override was requested. started training from scratch...'
        model, input_lookup, output_lookup, encoder_frnn, encoder_rrnn, decoder_rnn, readout, bias, w_c, w_a, u_a, v_a \
            = build_model(input_vocabulary, output_vocabulary, input_dim, hidden_dim, layers)

    # train the model
    if not eval_only:
        model, input_lookup, encoder_frnn, encoder_rrnn, decoder_rnn, readout, bias, w_c, w_a, u_a, v_a, last_epoch, \
        best_epoch = train_model(model, input_lookup, output_lookup, encoder_frnn, encoder_rrnn, decoder_rnn, readout,
                                 bias, w_c, w_a, u_a, v_a, train_inputs, train_outputs, dev_outputs, dev_inputs, x2int,
                                 y2int, int2y, epochs, optimization, results_file_path, plot)

        print 'last epoch is {}'.format(last_epoch)
        print 'best epoch is {}'.format(best_epoch)
        print 'finished training'
    else:
        print 'skipped training, evaluating on test set...'

    if ensemble:
        # predict test set using ensemble majority
        predicted_sequences = predict_with_ensemble_majority(input_vocabulary, output_vocabulary, x2int, y2int,
                                                             int2y, ensemble, hidden_dim, input_dim, layers,
                                                             test_inputs, test_outputs)
    else:
        # predict test set using a single model
        predicted_sequences = predict_sequences(input_lookup, output_lookup, encoder_frnn, encoder_rrnn, decoder_rnn,
                                                readout, bias, w_c, w_a, u_a, v_a, x2int,
                                                y2int, int2y, test_inputs)
    if len(predicted_sequences) > 0:

        # evaluate the test predictions
        amount, accuracy = evaluate_model(predicted_sequences, test_outputs, test_inputs, print_results=False)
        print 'initial eval: {}% accuracy'.format(accuracy)

        final_results = []
        for i in xrange(len(test_outputs)):
            index = ' '.join(test_inputs[i])
            final_output = ' '.join(predicted_sequences[index])
            final_results.append(final_output)

        # evaluate best models
        predictions_path = common.write_results_files(hyper_params, train_inputs_path, train_outputs_path,
                                                      dev_inputs_path, dev_outputs_path, test_inputs_path,
                                                      test_outputs_path, results_file_path, final_results)

        common.evaluate_bleu(test_outputs_path, predictions_path)
    return


def predict_with_ensemble_majority(input_vocabulary, output_vocabulary, x2int, y2int, int2y, ensemble,
                                   hidden_dim, input_dim, layers, test_inputs, test_outputs):
    ensemble_model_names = ensemble.split(',')
    print 'ensemble paths:\n {}'.format('\n'.join(ensemble_model_names))
    ensemble_models = []

    # load ensemble models
    for ens in ensemble_model_names:
        model, input_lookup, output_lookup, encoder_frnn, encoder_rrnn, decoder_rnn, readout, bias, w_c, w_a, u_a, v_a \
            = load_best_model(input_vocabulary, output_vocabulary, ens, input_dim, hidden_dim, layers)

        ensemble_models.append(
            (model, input_lookup, output_lookup, encoder_frnn, encoder_rrnn, decoder_rnn, readout, bias, w_c, w_a, u_a,
             v_a))

    # predict the entire test set with each model in the ensemble
    ensemble_predictions = []
    for em in ensemble_models:
        model, input_lookup, output_lookup, encoder_frnn, encoder_rrnn, decoder_rnn, readout, bias, w_c, w_a, u_a, v_a \
            = em
        predicted_sequences = predict_sequences(input_lookup, output_lookup, encoder_frnn, encoder_rrnn, decoder_rnn,
                                                readout, bias, w_c, w_a, u_a, v_a, x2int, y2int, int2y, test_inputs)
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


def save_model(model, results_file_path):
    tmp_model_path = results_file_path + '_bestmodel.txt'
    print 'saving to ' + tmp_model_path
    model.save(tmp_model_path)
    print 'saved to {0}'.format(tmp_model_path)


def load_best_model(input_vocabulary, output_vocabulary, results_file_path, input_dim, hidden_dim, layers):
    tmp_model_path = results_file_path + '_bestmodel.txt'
    model, input_lookup, output_lookup, encoder_frnn, encoder_rrnn, decoder_rnn, readout, bias, w_c, w_a, u_a, v_a = \
        build_model(input_vocabulary, output_vocabulary, input_dim, hidden_dim, layers)

    print 'trying to load model from: {}'.format(tmp_model_path)
    model.load(tmp_model_path)
    return model, input_lookup, output_lookup, encoder_frnn, encoder_rrnn, decoder_rnn, readout, bias, w_c, w_a, u_a, \
           v_a


# noinspection PyUnusedLocal
def build_model(input_vocabulary, output_vocabulary, input_dim, hidden_dim, layers):
    print 'creating model...'

    model = dn.Model()

    # input embeddings
    input_lookup = model.add_lookup_parameters((len(input_vocabulary), input_dim))

    # output embeddings
    output_lookup = model.add_lookup_parameters((len(output_vocabulary), input_dim))

    # used in softmax output
    readout = model.add_parameters((len(input_vocabulary), 3 * hidden_dim))
    bias = model.add_parameters(len(input_vocabulary))

    # rnn's
    encoder_frnn = dn.LSTMBuilder(layers, input_dim, hidden_dim, model)
    encoder_rrnn = dn.LSTMBuilder(layers, input_dim, hidden_dim, model)

    # attention MLPs - Luong-style with extra v_a from Bahdanau

    # concatenation layer for h (hidden dim), c (2 * hidden_dim)
    w_c = model.add_parameters((3 * hidden_dim, 3 * hidden_dim))

    # concatenation layer for h_input (2*hidden_dim), h_output (hidden_dim)
    w_a = model.add_parameters((hidden_dim, hidden_dim))

    # concatenation layer for h (hidden dim), c (2 * hidden_dim)
    u_a = model.add_parameters((hidden_dim, 2 * hidden_dim))

    # concatenation layer for h_input (2*hidden_dim), h_output (hidden_dim)
    v_a = model.add_parameters((1, hidden_dim))

    # 1 * HIDDEN_DIM - gets only the feedback input
    decoder_rnn = dn.LSTMBuilder(layers, input_dim, hidden_dim, model)

    print 'finished creating model'

    return model, input_lookup, output_lookup, encoder_frnn, encoder_rrnn, decoder_rnn, readout, bias, w_c, w_a, u_a, \
           v_a


def train_model(model, input_lookup, output_lookup, encoder_frnn, encoder_rrnn, decoder_rnn, readout, bias, w_c, w_a,
                u_a, v_a,
                train_inputs, train_outputs, dev_outputs, dev_inputs, x2int, y2int, int2y, epochs, optimization,
                results_file_path, plot):
    print 'training...'

    np.random.seed(17)
    random.seed(17)

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

    total_loss = 0
    best_avg_dev_loss = 999
    best_dev_bleu = -1
    best_train_bleu = -1
    best_dev_epoch = 0
    best_train_epoch = 0
    patience = 0
    train_len = len(train_outputs)
    # train_sanity_set_size = 100
    train_bleu = -1
    epochs_x = []
    train_loss_y = []
    dev_loss_y = []
    train_bleu_y = []
    dev_bleu_y = []

    # progress bar init
    widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
    train_progress_bar = progressbar.ProgressBar(widgets=widgets, maxval=epochs).start()
    avg_loss = -1
    e = 0

    for e in xrange(epochs):

        # randomize the training set
        indices = range(train_len)
        random.shuffle(indices)
        train_set = zip(train_inputs, train_outputs)
        train_set = [train_set[i] for i in indices]

        # compute loss for each example and update
        for i, example in enumerate(train_set):
            input_seq, output_seq = example
            loss = compute_loss(encoder_frnn, encoder_rrnn, decoder_rnn, input_lookup, output_lookup, readout, bias,
                                w_c, w_a, u_a, v_a, input_seq, output_seq, x2int, y2int)
            loss_value = loss.value()
            total_loss += loss_value
            loss.backward()
            trainer.update()
            if i > 0:
                avg_loss = total_loss / float(i + e * train_len)
            else:
                avg_loss = total_loss

            if i % 100 == 0 and i > 0:
                print 'went through {} examples out of {}'.format(i, train_len)

        if EARLY_STOPPING:
            print 'starting epoch evaluation'

            # get train accuracy
            # print 'train sanity prediction:'
            # train_predictions = predict_sequences(input_lookup, output_lookup, encoder_frnn, encoder_rrnn,
            # decoder_rnn,
            #                                       readout, bias, w_c, w_a, u_a, v_a, x2int,
            #                                       int2x, y2int, int2y, train_inputs[:train_sanity_set_size])
            # print 'train sanity evaluation:'
            # train_accuracy = evaluate_model(train_predictions, train_inputs[:train_sanity_set_size],
            #                                 train_outputs[:train_sanity_set_size], False)[1]

            # if train_accuracy > best_train_accuracy:
            #     best_train_accuracy = train_accuracy
            #     best_train_epoch = e

            dev_bleu = 0
            avg_dev_loss = 0

            if len(dev_inputs) > 0:

                print 'dev prediction:'
                # get dev accuracy
                dev_predictions = predict_sequences(input_lookup, output_lookup, encoder_frnn, encoder_rrnn,
                                                    decoder_rnn, readout, bias, w_c, w_a, u_a, v_a, x2int, y2int,
                                                    int2y, dev_inputs)
                print 'dev evaluation:'
                # get dev accuracy
                dev_bleu = evaluate_model(dev_predictions, dev_inputs, dev_outputs, print_results=True)[1]

                if dev_bleu >= best_dev_bleu:
                    best_dev_bleu = dev_bleu
                    best_dev_epoch = e

                    # save best model to disk
                    save_model(model, results_file_path)
                    print 'saved new best model'
                    patience = 0
                else:
                    patience += 1

                # get dev loss
                total_dev_loss = 0
                for i in xrange(len(dev_inputs)):
                    total_dev_loss += compute_loss(encoder_frnn, encoder_rrnn, decoder_rnn, input_lookup, output_lookup,
                                                   readout, bias, w_c,
                                                   w_a, u_a, v_a, dev_inputs[i], dev_outputs[i], x2int, y2int).value()

                avg_dev_loss = total_dev_loss / float(len(dev_inputs))
                if avg_dev_loss < best_avg_dev_loss:
                    best_avg_dev_loss = avg_dev_loss

                print 'epoch: {0} train loss: {1:.4f} dev loss: {2:.4f} dev bleu: {3:.4f} train bleu = {4:.4f} \
 best dev bleu {5:.4f} (epoch {8}) best train bleu: {6:.4f} (epoch {9}) patience = {7}'.format(
                    e,
                    avg_loss,
                    avg_dev_loss,
                    dev_bleu,
                    train_bleu,
                    best_dev_bleu,
                    best_train_bleu,
                    patience,
                    best_dev_epoch,
                    best_train_epoch)

                log_to_file(results_file_path + '_log.txt', e, avg_loss, train_bleu, dev_bleu)

                if patience == MAX_PATIENCE:
                    print 'out of patience after {0} epochs'.format(str(e))
                    train_progress_bar.finish()
                    if plot:
                        plt.cla()
                    return model, e
            else:

                # if no dev set is present, optimize on train set
                print 'no dev set for early stopping, running all epochs until perfectly fitting or patience was \
                reached on the train set'

                if train_bleu > best_train_bleu:
                    best_train_bleu = train_bleu

                    # save best model to disk
                    save_model(model, results_file_path)
                    print 'saved new best model'
                    patience = 0
                else:
                    patience += 1

                print 'epoch: {0} train loss: {1:.4f} train bleu = {2:.4f} best train bleu: {3:.4f} \
                patience = {4}'.format(e, avg_loss, train_bleu, best_train_bleu, patience)

                # patience was reached
                if patience == MAX_PATIENCE:
                    train_progress_bar.finish()
                    if plot:
                        plt.cla()
                    return model, e

            # update lists for plotting
            train_bleu_y.append(train_bleu)
            epochs_x.append(e)
            train_loss_y.append(avg_loss)
            dev_loss_y.append(avg_dev_loss)
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
    print 'finished training. average loss: {} best epoch on dev: {} best epoch on train: {}'.format(str(avg_loss),
                                                                                                     best_dev_epoch,
                                                                                                     best_train_epoch)

    return model, input_lookup, encoder_frnn, encoder_rrnn, decoder_rnn, readout, bias, w_c, w_a, u_a, v_a, e, \
           best_train_epoch


def log_to_file(file_name, epoch, avg_loss, train_accuracy, dev_accuracy):
    # if first write, add headers
    if epoch == 0:
        log_to_file(file_name, 'epoch', 'avg_loss', 'train_accuracy', 'dev_accuracy')

    with open(file_name, "a") as logfile:
        logfile.write("{}\t{}\t{}\t{}\n".format(epoch, avg_loss, train_accuracy, dev_accuracy))


# noinspection PyPep8Naming
def compute_loss(encoder_frnn, encoder_rrnn, decoder_rnn, input_lookup, output_lookup, readout, bias, w_c, w_a, u_a,
                 v_a, input_seq, output_seq, x2int, y2int):
    dn.renew_cg()

    readout = dn.parameter(readout)
    bias = dn.parameter(bias)
    w_c = dn.parameter(w_c)
    u_a = dn.parameter(u_a)
    v_a = dn.parameter(v_a)
    w_a = dn.parameter(w_a)

    blstm_outputs = encode_input(x2int, input_lookup, encoder_frnn, encoder_rrnn, input_seq)

    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD
    prev_output_vec = output_lookup[y2int[BEGIN_SEQ]]
    loss = []
    padded_output = output_seq + [END_SEQ]

    # run the decoder through the output sequence and aggregate loss
    for i, output_element in enumerate(padded_output):

        # get current state of the decoder LSTM
        s = s.add_input(prev_output_vec)
        decoder_rnn_output = s.output()

        attention_output_vector, alphas, W = attend(blstm_outputs, decoder_rnn_output, w_c, v_a, w_a, u_a)

        # compute output probabilities
        # print 'computing readout layer...'
        h = readout * attention_output_vector + bias

        if output_element in y2int:
            current_loss = dn.pickneglogsoftmax(h, y2int[output_element])
        else:
            current_loss = dn.pickneglogsoftmax(h, y2int[UNK])

        # print 'computed readout layer'
        loss.append(current_loss)

        # prepare for the next iteration - "feedback"
        if output_element in y2int:
            prev_output_vec = output_lookup[y2int[output_element]]
        else:
            prev_output_vec = output_lookup[y2int[UNK]]

    total_sequence_loss = dn.esum(loss)
    # loss = average(loss)

    return total_sequence_loss


def bilstm_transduce(encoder_frnn, encoder_rrnn, input_vecs):
    # BiLSTM forward pass
    s_0 = encoder_frnn.initial_state()
    s = s_0
    frnn_outputs = []
    for c in input_vecs:
        s = s.add_input(c)
        frnn_outputs.append(s.output())

    # BiLSTM backward pass
    s_0 = encoder_rrnn.initial_state()
    s = s_0
    rrnn_outputs = []
    for c in reversed(input_vecs):
        s = s.add_input(c)
        rrnn_outputs.append(s.output())

    # BiLTSM outputs
    blstm_outputs = []
    for i in xrange(len(input_vecs)):
        blstm_outputs.append(dn.concatenate([frnn_outputs[i], rrnn_outputs[len(input_vecs) - i - 1]]))

    return blstm_outputs


# noinspection PyPep8Naming
def predict_output_sequence(encoder_frnn, encoder_rrnn, decoder_rnn, input_lookup, output_lookup, readout, bias, w_c,
                            w_a, u_a, v_a, input_seq, x2int, y2int, int2y):
    dn.renew_cg()

    readout = dn.parameter(readout)
    bias = dn.parameter(bias)
    w_c = dn.parameter(w_c)
    u_a = dn.parameter(u_a)
    v_a = dn.parameter(v_a)
    w_a = dn.parameter(w_a)

    blstm_outputs = encode_input(x2int, input_lookup, encoder_frnn, encoder_rrnn, input_seq)

    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD
    prev_output_vec = output_lookup[y2int[BEGIN_SEQ]]
    i = 0
    predicted_sequence = []

    # run the decoder through the sequence and predict output symbols
    while i < MAX_PREDICTION_LEN:

        # get current h of the decoder
        s = s.add_input(prev_output_vec)
        decoder_rnn_output = s.output()

        # perform attention step
        attention_output_vector, alphas, W = attend(blstm_outputs, decoder_rnn_output, w_c, v_a, w_a, u_a)

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
        prev_output_vec = output_lookup[next_element_index]
        i += 1

    # remove the end word symbol
    return predicted_sequence[0:-1]


def encode_input(vocabulary, lookup, encoder_frnn, encoder_rrnn, input_seq):
    # initialize sequence with begin symbol
    input_vecs = [lookup[vocabulary[BEGIN_SEQ]]]

    # convert symbols to matching embeddings, if UNK handle properly
    for symbol in input_seq:
        try:
            input_vecs.append(lookup[vocabulary[symbol]])
        except KeyError:
            # handle UNK symbol
            input_vecs.append(lookup[vocabulary[UNK]])

    # add feats in the beginning of the input sequence and terminator symbol
    input_vecs.append(lookup[vocabulary[END_SEQ]])

    # create bidirectional representation
    blstm_outputs = bilstm_transduce(encoder_frnn, encoder_rrnn, input_vecs)
    return blstm_outputs


# Loung-style attention mechanism:
def attend(blstm_outputs, h_t, w_c, v_a, w_a, u_a):
    # iterate through input states to compute attention scores
    scores = [v_a * dn.tanh(w_a * h_t + u_a * h_input) for h_input in blstm_outputs]

    # normalize scores using softmax
    alphas = dn.softmax(dn.concatenate(scores))

    # compute context vector with weighted sum
    c = dn.esum([h_input * dn.pick(alphas, j) for j, h_input in enumerate(blstm_outputs)])

    # compute output vector using current decoder state and context vector
    h_output = dn.tanh(w_c * dn.concatenate([h_t, c]))

    # TODO: w_a.value() may be expensive to compute
    return h_output, alphas, w_a.value()


def predict_sequences(input_lookup, output_lookup, encoder_frnn, encoder_rrnn, decoder_rnn, readout, bias, w_c, w_a,
                      u_a, v_a, x2int, y2int, int2y, inputs):
    print 'predicting...'
    predictions = {}
    data_len = len(inputs)
    for i, input_seq in enumerate(inputs):
        predicted_template = predict_output_sequence(encoder_frnn, encoder_rrnn, decoder_rnn, input_lookup,
                                                     output_lookup, readout, bias, w_c, w_a, u_a, v_a,
                                                     input_seq, x2int, y2int, int2y)
        if i % 100 == 0 and i > 0:
            print 'predicted {} examples out of {}'.format(i, data_len)

        joint_index = ' '.join(input_seq)
        predictions[joint_index] = predicted_template

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
        enc_out = ' '.join(output_seq).encode('utf8')
        enc_gold = predicted_output.encode('utf8')

        if print_results:
            print 'input: {}'.format(enc_in)
            print 'gold output: {}'.format(enc_out)
            print 'prediction: {}\n'.format(enc_gold)

        eval_predictions.append(enc_in)
        eval_golds.append(enc_gold)

    bleu = common.evaluate_bleu(eval_golds, eval_predictions)

    if print_results:
        print 'finished evaluating model. bleu: {}\n\n'.format(bleu)

    return len(predicted_sequences), bleu


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
        batch_param = arguments['--batch-size']
    else:
        batch_param = 1

    if arguments['--beam-size']:
        beam_param = arguments['--beam-size']
    else:
        beam_param = 1

    if arguments['--gpu']:
        # noinspection PyUnresolvedReferences
        import _gdynet as dn
    else:
        # noinspection PyUnresolvedReferences
        import dynet as dn

    print arguments


    main(train_inputs_path_param, train_outputs_path_param, dev_inputs_path_param, dev_outputs_path_param,
         test_inputs_path_param, test_outputs_path_param, results_file_path_param, input_dim_param, hidden_dim_param,
         epochs_param, layers_param, optimization_param, regularization_param, learning_rate_param, plot_param,
         override_param, eval_param, ensemble_param, batch_param, beam_param)
