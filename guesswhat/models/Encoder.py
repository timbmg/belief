import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size,
                 cell=nn.LSTM, bidirectional=False):

        super().__init__()

        if cell not in [nn.RNN, nn.GRU, nn.LSTM]:
            raise ValueError("{} cell not supported.".format(str(cell)))

        self.rnn = cell(input_size, hidden_size,
                        bidirectional=bidirectional, batch_first=True)

    def forward(self, input, lengths, hx=None):

        if input.size(0) != lengths.size(0):
            raise AssertionError("Expected first dimension of input and " +
                                 "lengths to be the same. But got {} and {}."
                                 .format(input.size(0), lengths.size(0)))

        # pack sequence
        sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
        input = input[sorted_idx]
        packed_input = nn.utils.rnn.pack_padded_sequence(input, sorted_lengths,
                                                         batch_first=True)

        # rnn forward
        packed_outputs, last_hidden = self.rnn(packed_input, hx)

        # re-pad sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs,
                                                      batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx)
        outputs = outputs[reversed_idx]
        if isinstance(last_hidden, tuple):
            last_hidden = (last_hidden[0][:, reversed_idx],
                           last_hidden[1][:, reversed_idx])
        else:
            last_hidden = last_hidden[:, reversed_idx]

        return outputs, last_hidden
