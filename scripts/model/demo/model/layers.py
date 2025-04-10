from math import sqrt
from torch import nn

class LSTM(nn.LSTM):
    def forward(self, x, lengths=None):
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths.detach().cpu(), batch_first=self.batch_first, enforce_sorted=False)
        self.flatten_parameters()
        y, _ = super().forward(x)
        if lengths is not None:
            y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=self.batch_first)
        return y

class Embedding(nn.Embedding):
    def reset_parameters(self):
        std = sqrt(2.0 / (self.num_embeddings + self.embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.weight.data.uniform_(-val, val)
        self._fill_padding_idx_with_zero()
