import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # define the linear input layer
        self.linear_in = nn.Linear(input_size, hidden_size)

        # define the LSTM layer
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # define the batch normalization layer
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # define the linear output layer
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # apply the linear input layer
        x = self.linear_in(x)

        # apply batch normalization
        x = self.batch_norm(x.transpose(1,2)).transpose(1,2)

        # apply the LSTM layer
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))

        # apply the linear output layer
        out = self.linear_out(lstm_out[:, -1, :])

        # squeeze the output tensor to shape [batch_size]
        out = out.squeeze()

        return out


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden_states):
        # hidden_states shape: (seq_len, batch_size, hidden_size)
        energy = torch.tanh(self.attn(hidden_states))
        attention_weights = torch.softmax(self.v(energy), dim=0)
        context_vector = torch.sum(attention_weights * hidden_states, dim=0)
        return context_vector
    

class SpatialAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SpatialAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_len, hidden_size)
        energy = torch.sigmoid(self.attn(hidden_states))
        attention_weights = torch.softmax(self.v(energy), dim=2)
        context_vector = torch.sum(attention_weights * hidden_states, dim=2, keepdim=True)
        return context_vector * hidden_states
        

class LSTMTemporalAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMTemporalAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # define the linear input layer
        self.linear_in = nn.Linear(input_size, hidden_size)

        # define the LSTM layer
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # define the temporal attention layer
        self.temporal_attention = TemporalAttention(hidden_size)

        # define the batch normalization layer
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # define the linear output layer
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # apply the linear input layer
        x = self.linear_in(x)

        # apply batch normalization
        x = self.batch_norm(x.transpose(1,2)).transpose(1,2)

        # apply the LSTM layer
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))

        # apply temporal attention
        attention_out = self.temporal_attention(lstm_out.transpose(0, 1))

        # apply the linear output layer
        out = self.linear_out(attention_out)

        # squeeze the output tensor to shape [batch_size]
        out = out.squeeze()

        return out


class LSTMSpatialTemporalAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMSpatialTemporalAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # define the spatial attention layer
        self.spatial_attention = SpatialAttention(hidden_size)

        # define the linear input layer
        self.linear_in = nn.Linear(input_size, hidden_size)

        # define the LSTM layer
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # define the temporal attention layer
        self.temporal_attention = TemporalAttention(hidden_size)

        # define the batch normalization layer
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # define the linear output layer
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Input shape (batch_size, seq_len, in_dim)
        # apply the linear input layer
        x = self.linear_in(x)

        # apply batch normalization
        x = self.batch_norm(x.transpose(1,2)).transpose(1,2)
        
        # apply the spatial attention layer
        spatial_out = self.spatial_attention(x)
        
        # apply the LSTM layer
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(spatial_out, (h0, c0))

        # apply temporal attention
        attention_out = self.temporal_attention(lstm_out.transpose(0, 1))

        # apply the linear output layer
        out = self.linear_out(attention_out)

        # squeeze the output tensor to shape [batch_size]
        out = out.squeeze()

        return out
    

class LSTMSpatialAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMSpatialTemporalAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # define the spatial attention layer
        self.spatial_attention = SpatialAttention(hidden_size)

        # define the linear input layer
        self.linear_in = nn.Linear(input_size, hidden_size)

        # define the LSTM layer
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # define the batch normalization layer
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # define the linear output layer
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # apply the linear input layer
        x = self.linear_in(x)

        # apply batch normalization
        x = self.batch_norm(x.transpose(1,2)).transpose(1,2)

        # apply the spatial attention layer
        spatial_out = self.spatial_attention(x)

        # apply the LSTM layer
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(spatial_out.unsqueeze(1), (h0, c0))

        # apply the linear output layer
        out = self.linear_out(lstm_out.transpose(0, 1))

        # squeeze the output tensor to shape [batch_size]
        out = out.squeeze()

        return out


class FCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(FCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # define the linear input layer
        self.linear_in = nn.Linear(input_size, hidden_size)

        # define the batch normalization layer
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # define the fully connected layers
        self.fc_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])

        # define the linear output layer
        self.linear_out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # apply the linear input layer
        x = self.linear_in(x)

        # apply the batch normalization layer
        x = self.batch_norm(x.transpose(1,2)).transpose(1,2)
        
        # reshape the input to (seq_len, batch_size, hidden_size)
        x = x.transpose(0, 1)
        
        # apply the fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        # apply the linear output layer
        x = self.linear_out(x[-1])

        # squeeze the output tensor to shape [batch_size]
        x = x.squeeze()
        
        return x
  

class FCNTemporalAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(FCNTemporalAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # define the linear input layer
        self.linear_in = nn.Linear(input_size, hidden_size)

        # define the batch normalization layer
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # define the fully connected layers
        self.fc_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])

        # define the temporal attention layer
        self.attention = TemporalAttention(hidden_size)
        
        # define the linear output layer
        self.linear_out = nn.Linear(hidden_size, output_size)

    
    def forward(self, x):
        # apply the linear input layer
        x = self.linear_in(x)

        # apply the batch normalization layer
        x = self.batch_norm(x.transpose(1,2)).transpose(1,2)
        
        # reshape the input to (seq_len, batch_size, hidden_size)
        x = x.transpose(0, 1)
        
        # apply the fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        # apply the temporal attention layer
        attention_out = self.attention(x)
        
        # apply the linear output layer
        x = self.linear_out(attention_out)

        # squeeze the output tensor to shape [batch_size]
        x = x.squeeze()
        
        return x
