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

        self.batch_norm = nn.BatchNorm1d(input_size)

        self.layer_in = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers,batch_first=True, dropout=0.3 if num_layers > 1 else 0)

        self.layer_out = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, input):  # Input shape (batch_size, sequence_length, input_size)
        sequence_length = input.shape[1]

        x = input.transpose(1, 2)  # Swap sequence_length and input_size dimensions
        x = self.batch_norm(x)
        x = x.transpose(1, 2)  # Swap back to the original dimensions

        x = self.layer_in(x)
        x = self.dropout(x)
        # apply the LSTM layer
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        h_t, _ = self.lstm(x, (h0, c0))

        out = self.layer_out(h_t[:, -1, :])

        # squeeze the output tensor to shape [batch_size]
        out = out.squeeze()

        return out
    

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(TemporalAttention, self).__init__()
        self.hidden_to_input = nn.Linear(hidden_size, input_size)
        self.T_A = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h_list, return_weights=False):
        total_ht = self.hidden_to_input(h_list)  # Change hidden_size to input_size

        beta_t = self.relu(self.T_A(total_ht))
        beta_t = self.softmax(beta_t)

        context_vector = torch.sum(beta_t * total_ht, dim=1)

        if return_weights:
            return context_vector, beta_t
        else:
            return context_vector


class LSTMTemporalAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMTemporalAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.batch_norm = nn.BatchNorm1d(input_size)

        self.layer_in = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=0.3 if num_layers > 1 else 0)

        self.temporal_attention = TemporalAttention(hidden_size, input_size)

        self.layer_out = nn.Linear(input_size, output_size, bias=False)

    def forward(self, input, return_weights=False):  # Input shape (batch_size, sequence_length, input_size)
        sequence_length = input.shape[1]

        x = input.transpose(1, 2)  # Swap sequence_length and input_size dimensions
        x = self.batch_norm(x)
        x = x.transpose(1, 2)  # Swap back to the original dimensions

        x = self.layer_in(x)
        x = self.dropout(x)
        # apply the LSTM layer
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        h_t, _ = self.lstm(x, (h0, c0))
        
        if return_weights:
            context_vector, beta_t = self.temporal_attention(h_t, return_weights)
            out = self.layer_out(context_vector)
            return out.squeeze(), beta_t
        else:
            context_vector = self.temporal_attention(h_t)
            out = self.layer_out(context_vector)
            return out.squeeze()


class SpatialAttention(nn.Module):
    def __init__(self, input_size):
        super(SpatialAttention, self).__init__()

        self.S_A = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_t, return_weights=False):
        alpha_t = self.sigmoid(self.S_A(x_t))
        alpha_t = self.softmax(alpha_t)

        if return_weights:
            return x_t * alpha_t, alpha_t
        else:
            return x_t * alpha_t
        

class LSTMSpatioTemporalAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMSpatioTemporalAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.batch_norm = nn.BatchNorm1d(input_size)

        self.spatial_attention = SpatialAttention(input_size)

        self.layer_in = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers,batch_first=True, dropout=0.3 if num_layers > 1 else 0)

        self.temporal_attention = TemporalAttention(hidden_size, input_size)

        self.layer_out = nn.Linear(input_size, output_size, bias=False)

    def forward(self, input, return_weights=False):  # Input shape (batch_size, sequence_length, input_size)
        sequence_length = input.shape[1]

        x = input.transpose(1, 2)  # Swap sequence_length and input_size dimensions
        x = self.batch_norm(x)
        x = x.transpose(1, 2)  # Swap back to the original dimensions

        x_new = torch.zeros_like(x)  # Create a new tensor to store the results

        if return_weights:
            alpha_list = []
            for t in range(sequence_length):
                x_t = x[:, t, :]
                x_t, alpha_t = self.spatial_attention(x_t, return_weights=True)
                x_new[:, t, :] = x_t
                alpha_list.append(alpha_t)
        else:
            for t in range(sequence_length):
                x_t = x[:, t, :]
                x_t = self.spatial_attention(x_t)
                x_new[:, t, :] = x_t

        x = x_new

        x = self.layer_in(x)
        x = self.dropout(x)

        # apply the LSTM layer
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        h_t, _ = self.lstm(x, (h0, c0))

        if return_weights:
            context_vector, beta_t = self.temporal_attention(h_t, return_weights)
            out = self.layer_out(context_vector)
            alpha_list = torch.stack(alpha_list, dim=1)  # Stack along the sequence_length dimension
            return out.squeeze(), alpha_list, beta_t
        else:
            context_vector = self.temporal_attention(h_t)
            out = self.layer_out(context_vector)
            return out.squeeze()


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