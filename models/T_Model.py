import torch.nn as nn
import torch

class TA_LSTM(nn.Module):

    def __init__(self,
                 in_dim,
                 sequence_length,
                 lstm_in_dim,
                 lstm_hidden_dim,
                 out_dim,
                 use_gpu=False):

        super(TA_LSTM,self).__init__()

        self.in_dim = in_dim
        self.sequence_length = sequence_length
        
        self.lstm_in_dim = lstm_in_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.out_dim = out_dim
        self.use_gpu = use_gpu

        self.batch_norm = nn.BatchNorm1d(in_dim)
        
        self.layer_in = nn.Linear(in_dim, in_dim,bias=False)
        
        self.lstmcell = nn.LSTMCell(lstm_in_dim, lstm_hidden_dim)
        
        self.T_A = nn.Linear(sequence_length*lstm_hidden_dim, sequence_length)
        
        self.layer_out = nn.Linear(lstm_hidden_dim, out_dim,bias=False)
         
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward (self,input):

        out = self.batch_norm(input)

        out = self.layer_in(out)
         
        h_t_1 = torch.zeros(out.size(0), self.lstm_hidden_dim) 
        c_t_1 = torch.zeros(out.size(0), self.lstm_hidden_dim) 
      
        h_list = []

        for i in range(self.sequence_length):
            
            x_t = out[:,i*self.lstm_in_dim:(i+1)*(self.lstm_in_dim)]
           
            h_t,c_t = self.lstmcell(x_t,(h_t_1,c_t_1)) 
            
            h_list.append(h_t)

            h_t_1,c_t_1 = h_t,c_t
        
        total_ht = h_list[0]
        for i in range(1,len(h_list)):
            total_ht = torch.cat((total_ht,h_list[i]),1)    
        
        beta_t =  self.relu(self.T_A(total_ht))
        beta_t = self.softmax(beta_t)
        
        out = torch.zeros(out.size(0), self.lstm_hidden_dim)

        for i in range(len(h_list)):
                      
            out = out + h_list[i]*beta_t[:,i].reshape(out.size(0),1)

        out = self.layer_out(out)
        
        return out