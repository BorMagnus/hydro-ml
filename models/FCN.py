import torch.nn as nn

class FCN(nn.Module):
    
    def __init__(self,
                 in_dim,
                 sequence_length,
                 lstm_in_dim,
                 lstm_hidden_dim,
                 out_dim,
                 mode):

        super(FCN, self).__init__()
        self.in_dim = in_dim
        self.sequence_length = sequence_length
        
        self.lstm_in_dim = lstm_in_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        self.out_dim = out_dim
        self.mode = mode

        self.batch_norm = nn.BatchNorm1d(in_dim)
        self.layer_in = nn.Linear(in_dim, in_dim,bias=False)
        self.fcn = nn.Linear(in_dim, lstm_hidden_dim)
        self.layer_out = nn.Linear(lstm_hidden_dim, out_dim,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward (self,input):
        out = self.batch_norm(input)
        out = self.layer_in(out)
        out = self.sigmoid(out)
        out = self.fcn(out)
        out = self.sigmoid(out)
        out = self.layer_out(out)
      
        return out