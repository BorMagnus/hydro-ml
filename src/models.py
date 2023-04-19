import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.hidden_size = hidden_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, lstm_out, return_weights=False):
        # input to temporal attention (batch_size, seq_len, hidden_size)
        query = self.query(lstm_out)  
        key = self.key(lstm_out)  
        value = self.value(lstm_out)  

        attention_logits = torch.bmm(query.transpose(0, 1), key.transpose(0, 1).transpose(1, 2))
        attention_weights = self.softmax(attention_logits)

        attention_out = torch.bmm(attention_weights, value.transpose(0, 1))
        attention_out = attention_out.transpose(0, 1)

        if return_weights:
            return attention_out, attention_weights
        else:
            return attention_out
        

class SpatialAttention(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(SpatialAttention, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.query = nn.Linear(input_size, input_size)  # (input_size, input_size)
        self.key = nn.Linear(input_size, input_size)    # (input_size, input_size)
        self.value = nn.Linear(input_size, hidden_size) # (input_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, return_weights=False):
        #print()
        #print("Attention")
        #print("input", x.shape)
        query = self.query(x)
        #print("query", query.shape)
        key = self.key(x)
        #print("key", key.shape)

        attention_logits = torch.bmm(query.transpose(1, 2), key)
        #print("attention_logits", attention_logits.shape)

        attention_weights = self.softmax(attention_logits)
        #print("attention_weights", attention_weights.shape)

        attention_out = torch.bmm(x, attention_weights)
        #print(f"attention_out: {attention_out.shape}")
        attention_out = self.value(attention_out)
        #print(f"attention_out: {attention_out.shape}")
        #print()

        if return_weights:
            return attention_out, attention_weights
        else:
            return attention_out


class SpatioTemporalAttention (nn.Module):
    def __init__(self, hidden_size, input_size):
        super(SpatioTemporalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.query = nn.Linear(input_size, input_size)  #           (input_size, input_size)
        self.key = nn.Linear(input_size, 25)            #           (input_size, seq_len)
        self.value = nn.Linear(25, hidden_size) #                   (seq_len, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, return_weights=False):
        #print()
        #print("Attention")
        #print("input", x.shape)                      # Input shape (batch_size, seq_len, input_size)
        query = self.query(x)  # Shape                              (batch_size, seq_len, input_size)
        #print("query", query.shape)
        key = self.key(x)      # Shape                              (batch_size, seq_len, seq_len)
        #print("key", key.shape)
        value = self.value(x.transpose(1, 2))  # Shape              (batch_size, input_size, hidden_size)
        #print("value", value.shape)

        attention_logits = torch.bmm(query.transpose(1, 2), key)  # (batch_size, input_size, seq_len)
        #print("attention_logits", attention_logits.shape)

        attention_weights = self.softmax(attention_logits)#         (batch_size, input_size, seq_len)
        #print("attention_weights", attention_weights.shape)
        
        attention_out = torch.bmm(
            attention_weights.transpose(1, 2), value)#              (batch_size, seq_len, hidden_size)
        #print(f"attention_out: {attention_weights.transpose(1, 2).shape}x{value.shape}={attention_out.shape}")
        #print()
        
        if return_weights:
            return attention_out, attention_weights
        else:
            return attention_out


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
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # define the batch normalization layer
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # define the linear output layer
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # apply the linear input layer
        x = self.linear_in(x)

        # apply batch normalization
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)

        # apply the LSTM layer
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))

        # apply the linear output layer
        out = self.linear_out(lstm_out[:, -1, :])

        # squeeze the output tensor to shape [batch_size]
        out = out.squeeze()

        return out


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
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # define the temporal attention layer
        self.temporal_attention = TemporalAttention(hidden_size)

        # define the batch normalization layer
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # define the linear output layer
        self.linear_out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, return_weights=False): # Input shape (batch_size, seq_len, input_size)
        #print("input: \t\t\t",x.shape)
        # apply the linear input
        x = self.linear_in(x) # linear_in shape (batch_size, seq_len, hidden_size)
        #print("linear_in: \t\t",x.shape)

        # apply batch normalization
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2) # batch_norm shape (batch_size, seq_len, hidden_size)
        #print("batch norm: \t\t",x.shape)

        # apply the LSTM layer
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0)) # lstm_out shape (batch_size, seq_len, hidden_size)
        #print("lstm out: \t\t",lstm_out.shape)

        # apply temporal attention. attention_out shape (seq_len, batch_size, hidden_size)
        if return_weights:
            attention_out, temporal_attention_weights  = self.temporal_attention(lstm_out.transpose(0, 1), return_weights)
        else:
            attention_out = self.temporal_attention(lstm_out.transpose(0, 1))
        #print("attention out: \t\t",attention_out.shape)

        # select the last time step from the attention_out tensor
        attention_out_last = attention_out[-1] # attention_out_last shape (batch_size, hidden_size)
        #print("attention_out_last: \t",attention_out_last.shape)

        # apply the linear output layer
        out = self.linear_out(attention_out_last) # linear_out shape (batch_size, output_size)
        #print("linear_out: \t\t",out.shape)

        # squeeze the output tensor to shape [batch_size]
        out = out.squeeze()
        #print("output: \t\t",out.shape)

        if return_weights:
            return out, (None, temporal_attention_weights)
        else:
            return out
    

class LSTMSpatialTemporalAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMSpatialTemporalAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # define the spatial attention layer
        self.spatial_attention = SpatialAttention(hidden_size, input_size)

        # define the linear input layer
        self.linear_in = nn.Linear(input_size, input_size)

        # define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # define the temporal attention layer
        self.temporal_attention = TemporalAttention(hidden_size)

        # define the batch normalization layer
        self.batch_norm = nn.BatchNorm1d(input_size)

        # define the linear output layer
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, return_weights=False):
        # Input shape (batch_size, seq_len, in_dim)
        #print("input: ", x.shape)

        # linear in shape (batch_size, seq_len, input_size)
        x = self.linear_in(x)
        #print("linear_in: ", x.shape)

        # apply batch normalization shape (batch_size, seq_len, input_size)
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        #print("batch_norm: ", x.shape)

        # spatial input shape (batch_size, seq_len, input_size)
        # spatial attention layer output is shape (batch_size, seq_len, hidden_size)
        # spatial attention weights is shape (batch_size, input_size, input_size)
        if return_weights:
            spatial_out, spatial_attention_weights = self.spatial_attention(x, return_weights)
        else:
            spatial_out = self.spatial_attention(x)
        #print("spatial_out: ", spatial_out.shape)

        # apply the LSTM layer 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(spatial_out, (h0, c0))
        #print("lstm_out: ", lstm_out.shape)

        # temporal input shape (batch_size, seq_len, hidden_size)
        # apply temporal attention output is shape (seq_len, batch_size, hidden_size)
        # temporal attention weights is shape (batch_size, sequence_length, sequence_length)
        if return_weights:
            attention_out, temporal_attention_weights  = self.temporal_attention(lstm_out.transpose(0, 1), return_weights)
        else:
            attention_out = self.temporal_attention(lstm_out.transpose(0, 1))
        #print("attention_out: ", attention_out.shape)

        # select the last time step from the attention_out tensor. shape (batch_size, hidden_size)
        attention_out_last = attention_out[-1]
        #print("attention_out_last: ", attention_out_last.shape)

        # apply the linear output layer. shape (batch_size, output_size)
        out = self.linear_out(attention_out_last)
        #print("linear_out: ", out.shape)

        # squeeze the output tensor to shape [batch_size]
        out = out.squeeze()
        #print("output: ", out.shape)

        if return_weights:
            return out, (spatial_attention_weights, temporal_attention_weights)
        else:
            return out


class LSTMSpatialAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMSpatialAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # define the spatial attention layer
        self.spatial_attention = SpatialAttention(hidden_size, input_size)

        # define the linear input layer
        self.linear_in = nn.Linear(input_size, input_size)

        # define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # define the batch normalization layer
        self.batch_norm = nn.BatchNorm1d(input_size)

        # define the linear output layer
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, return_weights=False):
        # Input shape (batch_size, seq_len, in_dim)
        #print("input: ", x.shape)

        # linear in shape (batch_size, seq_len, hidden_size)
        x = self.linear_in(x)
        #print("linear_in: ", x.shape)

        # apply batch normalization shape (batch_size, seq_len, hidden_size)
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        #print("batch_norm: ", x.shape)

        # spatial input shape (batch_size, seq_len, hidden_size)
        # spatial attention layer output is shape (batch_size, seq_len, hidden_size)
        # spatial attention weights is shape (batch_size, input_size, input_size)
        if return_weights:
            spatial_out, spatial_attention_weights = self.spatial_attention(x, return_weights)
        else:
            spatial_out = self.spatial_attention(x)
        #print("spatial_out: ", spatial_out.shape)

        # apply the LSTM layer 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(spatial_out, (h0, c0))
        #print("lstm_out: ", lstm_out.shape)

        # select the last time step from the lstm_out tensor. shape (batch_size, hidden_size)
        lstm_out_last = lstm_out[:, -1, :]
        #print("lstm_out_last: ", lstm_out_last.shape)

        # apply the linear output layer. shape (batch_size, output_size)
        out = self.linear_out(lstm_out_last)
        #print("linear_out: ", out.shape)

        # squeeze the output tensor to shape [batch_size]
        out = out.squeeze()
        #print("output: ", out.shape)

        if return_weights:
            return out, (spatial_attention_weights, None)
        else:
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
        self.fc_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )

        # define the linear output layer
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # apply the linear input layer
        x = self.linear_in(x)

        # apply the batch normalization layer
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)

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
        self.fc_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )

        # define the temporal attention layer
        self.attention = TemporalAttention(hidden_size)

        # define the linear output layer
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # apply the linear input layer
        x = self.linear_in(x)

        # apply the batch normalization layer
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)

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
