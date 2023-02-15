import torch.nn as nn

class Baseline(nn.Module):
    
    def __init__(self, in_dim, hidden_dim, out_dim, sequence_length, mode):
        super(Baseline, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.sequence_length = sequence_length
        self.mode = mode

        self.layer_in = nn.Linear(sequence_length, hidden_dim)


    def forward (self, input):
        return input[:, -1]