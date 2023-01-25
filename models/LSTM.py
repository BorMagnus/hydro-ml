class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, sequence_length):
        super(LSTM, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.sequence_length = sequence_length
        
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(in_dim, hidden_dim)
        self.lstm2 = nn.LSTMCell(hidden_dim, hidden_dim)
        
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim*sequence_length, out_dim)

        
    def forward(self, y):
        outputs = []
        h_t = torch.zeros(y.size(0), self.hidden_dim, dtype=torch.float32)
        c_t = torch.zeros(y.size(0), self.hidden_dim, dtype=torch.float32)
        h_t2 = torch.zeros(y.size(0), self.hidden_dim, dtype=torch.float32)
        c_t2 = torch.zeros(y.size(0), self.hidden_dim, dtype=torch.float32)
        
        for time_step in y.split(1, dim=1):
            # N, 1
            h_t, c_t = self.lstm1(time_step, (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
            output = self.linear(h_t2) # output from the last FC layer
            outputs.append(output)
        # transform list to tensor    
        outputs = torch.cat(outputs, dim=1)
        out = self.linear_out(outputs)
        return out