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
        max_masks = [[1]] * len(input_seq_batch)
        return [max_output], max_masks