import dynet as dn
import BiLSTMEncoder

class MaxPoolEncoder:
    def __init__(self, x2int, params):
        self.bilstm = BiLSTMEncoder.BiLSTMEncoder(x2int, params)

    # bilstm encode batch, then for each seq return max pooled vector of encoded inputs
    def encode_batch(self, input_seq_batch):
        # TODO: test with batching
        encoded_inputs, masks = self.bilstm.encode_batch(input_seq_batch)
        max_output = dn.emax(encoded_inputs)

        # one mask per step, all are [1]'s since only one state from max pool encoder
        max_masks = [[1]] * (len(input_seq_batch[0]) + 2)
        print 'len masks'
        print len(masks)
        print 'masks'
        print masks
        print 'len max_masks'
        print len(max_masks)
        print 'max_masks'
        print max_masks
        print 'encoded_inputs[0].dim'
        print encoded_inputs[0].dim()
        print 'max_output.dim'
        print max_output.dim()
        return [max_output], max_masks