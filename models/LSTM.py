import torch.nn as nn
import torch

class LSTM(nn.Module):
    
    def __init__(self,
                 in_dim,
                 sequence_length,
                 lstm_in_dim,
                 lstm_hidden_dim,
                 out_dim,
                 mode):

        super(LSTM,self).__init__()
        self.in_dim = in_dim
        self.sequence_length = sequence_length
        
        self.lstm_in_dim = lstm_in_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.out_dim = out_dim
        self.mode = mode

        self.batch_norm = nn.BatchNorm1d(in_dim)
        self.layer_in = nn.Linear(in_dim, in_dim)
        self.lstmcell = nn.LSTMCell(lstm_in_dim, lstm_hidden_dim)
        self.layer_out = nn.Linear(lstm_hidden_dim, out_dim,bias=False)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward (self,input):
        out = self.batch_norm(input)
        out = self.layer_in(out)
        
        h_t_1 = torch.zeros(out.size(0), self.lstm_hidden_dim).to(self.mode["device"])
        c_t_1 = torch.zeros(out.size(0), self.lstm_hidden_dim).to(self.mode["device"])
        
        for i in range(self.sequence_length):
            x_t = out[:,i*self.lstm_in_dim:(i+1)*(self.lstm_in_dim)]
            h_t,c_t = self.lstmcell(x_t,(h_t_1,c_t_1)) 
            h_t_1,c_t_1 = h_t,c_t
        out = self.layer_out(h_t)
        return out