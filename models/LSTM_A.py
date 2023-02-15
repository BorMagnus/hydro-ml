import torch.nn as nn
import torch

class LSTM_A(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, sequence_length, mode):
        super(LSTM_A, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.sequence_length = sequence_length
        self.mode = mode

        self.lstm1 = nn.LSTMCell(in_dim, hidden_dim)
        self.lstm2 = nn.LSTMCell(hidden_dim, hidden_dim)

        self.T_A = nn.Linear(sequence_length*hidden_dim, sequence_length)
        
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, out_dim)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, y):
        outputs = []
        h_t = torch.zeros(y.size(0), self.hidden_dim).to(self.mode["device"])
        c_t = torch.zeros(y.size(0), self.hidden_dim).to(self.mode["device"])
        h_t2 = torch.zeros(y.size(0), self.hidden_dim).to(self.mode["device"])
        c_t2 = torch.zeros(y.size(0), self.hidden_dim).to(self.mode["device"])
        
        for time_step in y.split(1, dim=1):
            h_t, c_t = self.lstm1(time_step, (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
            output = self.linear(h_t2) # output from the last FC layer
            outputs.append(output)
            
        total_ht = outputs[0]
        for i in range(1, len(outputs)):
            total_ht = torch.cat((total_ht, outputs[i]), 1)

        beta_t =  self.relu(self.T_A(total_ht))
        beta_t = self.softmax(beta_t)

        out = torch.zeros(y.size(0), self.hidden_dim).to(self.mode["device"])

        for i in range(len(outputs)):
                      
            out = out + outputs[i]*beta_t[:,i].reshape(out.size(0), 1)

        out = self.linear_out(out)
        
        return out